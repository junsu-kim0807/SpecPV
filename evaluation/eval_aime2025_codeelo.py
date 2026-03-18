#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

from specpv import SpecConfig, Speculator

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--draft_path", type=str, required=True)
    p.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["aime2025", "codeelo", "all"],
    )
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["specpv", "full", "naive"],
    )
    p.add_argument("--partial_length", type=str, default=None)
    p.add_argument("--partial_spec_tokens", type=str, default="20")

    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Absolute or relative path to a single local JSONL file.",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Directory containing local JSONL files such as aime2025/aime2025.jsonl.",
    )
    p.add_argument("--output_root", type=str, default="outputs")

    # For quick smoke tests / CI-like runs.
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--max_input_tokens", type=int, default=60000)
    return p.parse_args(args)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_chat(tokenizer, user_prompt: str, model_name: str, system_prompt: str | None) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    if "qwen3" in model_name:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resolve_dataset_file(dataset_name: str, dataset_path: str | None, dataset_root: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    candidates: list[Path] = []
    if dataset_root:
        root = Path(dataset_root).expanduser()
        candidates.extend(
            [
                root / f"{dataset_name}.jsonl",
                root / dataset_name / f"{dataset_name}.jsonl",
            ]
        )

    env_root = os.environ.get("SPECPV_DATA_ROOT") or os.environ.get("SPECPV_DATASETS_ROOT")
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend(
            [
                root / f"{dataset_name}.jsonl",
                root / dataset_name / f"{dataset_name}.jsonl",
            ]
        )

    hf_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_cache:
        root = Path(hf_cache).expanduser()
        candidates.extend(
            [
                root / f"{dataset_name}.jsonl",
                root / dataset_name / f"{dataset_name}.jsonl",
                root / "longbenchv1" / f"{dataset_name}.jsonl",  # harmless fallback
            ]
        )

    for p in candidates:
        p = p.resolve()
        if p.exists():
            return p

    searched = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find dataset file for {dataset_name}.\nSearched:\n{searched}")


def _normalize_intish(s: str) -> str:
    s = s.strip().replace(",", "")
    # Extract first integer-like token.
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else s


def extract_aime_answer(text: str) -> str:
    """
    Heuristic: AIME answers are usually integers. Prefer \\boxed{...}, otherwise
    take the last integer in the text.
    """
    # Common: \boxed{70}
    m = re.search(r"\\boxed\s*{([^}]*)}", text)
    if m:
        return _normalize_intish(m.group(1))

    # Sometimes users omit braces: boxed 70
    m = re.search(r"boxed\s*([^\\\n]+)", text, flags=re.IGNORECASE)
    if m:
        return _normalize_intish(m.group(1))

    # Fallback: last integer token.
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return ints[-1] if ints else ""


def _strip_code_fences(text: str) -> str:
    # Remove first fenced block if present.
    m = re.search(r"```(?:[a-zA-Z0-9_-]+)?\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def extract_codeelo_output(text: str) -> str:
    # Prefer a fenced block; otherwise take whole text.
    return _strip_code_fences(text)


def normalize_codeelo_text(s: str) -> str:
    s = _strip_code_fences(s)
    # Normalize newlines/whitespace to reduce accidental formatting mismatch.
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_user_prompt(dataset_name: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if dataset_name == "aime2025":
        question = str(row.get("question", row.get("problem", row.get("input", "")))).strip()
        system_prompt = (
            "You are a helpful mathematical problem solver. "
            "Solve the given AIME 2025 problem. "
            "Provide the final answer only, preferably in the form \\boxed{...}."
        )
        user_prompt = f"{question}\n\nFinal answer:"
        return user_prompt, system_prompt

    if dataset_name == "codeelo":
        desc = str(row.get("description", "")).strip()
        inp = str(row.get("input", "")).strip()
        note = str(row.get("note", "")).strip()

        interaction = str(row.get("interaction", "")).strip()

        system_prompt = (
            "You are a competitive programming assistant. "
            "Given the problem statement and the sample input, "
            "write the exact sample output. Output only the output text, no explanations."
        )

        parts: list[str] = []
        if desc:
            parts.append(desc)
        if note:
            parts.append(f"Note:\n{note}")
        if inp:
            parts.append(f"Sample Input:\n{inp}")
        elif interaction:
            parts.append(f"Interaction:\n{interaction}")

        parts.append("Sample Output (exactly):")
        return "\n\n".join(parts), system_prompt

    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def get_gold_answer(dataset_name: str, row: dict[str, Any]) -> str:
    if dataset_name == "aime2025":
        return str(row.get("answer", "")).strip()
    if dataset_name == "codeelo":
        # CodeElo row typically has `output` as the reference sample output.
        # (The dataset is used here as "output text accuracy".)
        return str(row.get("output", row.get("answer", ""))).strip()
    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def get_pred(
    rank: int,
    data: list[dict[str, Any]],
    dataset_name: str,
    max_gen: int,
    max_input_tokens: int,
    prompt_system: bool,
    model_path: str,
    draft_path: str,
    out_path: str,
    model_name: str,
    spec_config: SpecConfig,
    method: str,
) -> None:
    device = torch.device(f"cuda:{rank}")
    model, tokenizer = load_model_and_tokenizer(model_path, draft_path, device)

    for json_obj in tqdm(data, desc=f"{dataset_name}/rank{rank}"):
        user_prompt, system_prompt = build_user_prompt(dataset_name, json_obj)

        # Keep chat template creation cheap.
        prompt = build_chat(
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            model_name=model_name,
            system_prompt=system_prompt if prompt_system else None,
        )

        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if tokenized_prompt.numel() > max_input_tokens:
            half = max_input_tokens // 2
            # Match the same trimming strategy as other scripts:
            # keep prefix+suffix tokens as a single prompt string.
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        context_length = model_inputs.input_ids.shape[-1]
        is_llama3 = "llama3" in model_name

        if method == "naive":
            output, _metrics = model.naive_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                log=True,
            )
        else:
            output, _metrics = model.spec_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                spec_config=spec_config,
                log=True,
            )

        pred_raw = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
        gold = get_gold_answer(dataset_name, json_obj)

        evaluated = bool(gold)
        if dataset_name == "aime2025":
            pred_answer = extract_aime_answer(pred_raw)
            gold_norm = _normalize_intish(gold)
            pred_norm = _normalize_intish(pred_answer)
            correct = evaluated and (pred_norm == gold_norm)
        elif dataset_name == "codeelo":
            pred_output = extract_codeelo_output(pred_raw)
            correct = evaluated and (normalize_codeelo_text(pred_output) == normalize_codeelo_text(gold))
        else:
            raise ValueError(dataset_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "id": json_obj.get("_id", json_obj.get("id", None)),
                    "pred_raw": pred_raw,
                    "pred": pred_answer if dataset_name == "aime2025" else pred_output,
                    "gold": gold,
                    "evaluated": evaluated,
                    "correct": correct,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_and_tokenizer(model_path: str, draft_path: str, device: torch.device):
    model = Speculator.from_pretrained(
        base_model_path=model_path,
        ea_model_path=draft_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        total_token=-1,
    )
    model.eval()
    tokenizer = model.tokenizer
    return model, tokenizer


def main() -> None:
    seed_everything(42)
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. This evaluation script expects at least 1 GPU.")

    mp.set_start_method("spawn", force=True)

    datasets = ["aime2025", "codeelo"] if args.dataset_name == "all" else [args.dataset_name]

    spec_config = SpecConfig()
    if args.method == "full":
        spec_config.enable_partial_kv = False
    elif args.method == "specpv":
        spec_config.partial_spec_tokens = int(args.partial_spec_tokens)
        spec_config.enable_partial_kv = True
        if args.partial_length is None:
            raise SystemExit("--partial_length is required for method=specpv")
        spec_config.n_retrieval_blocks = int(args.partial_length) // spec_config.block_size
    else:
        # naive does not use spec_config
        pass

    model_name = os.path.basename(args.model_path).lower()
    output_root = Path(args.output_root)
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in datasets:
        dataset_file = resolve_dataset_file(
            dataset_name=dataset_name,
            dataset_path=args.dataset_path,
            dataset_root=args.dataset_root,
        )
        print(f"[dataset] {dataset_name} using local file: {dataset_file}")

        data = load_dataset("json", data_files=str(dataset_file))["train"]
        if args.max_samples is not None:
            data = data.select(range(min(args.max_samples, len(data))))

        out_path = model_output_dir / f"{dataset_name}-{args.method}.jsonl"
        if out_path.exists():
            out_path.unlink()

        # Slice dataset across ranks.
        data_all = [dict(x) for x in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        processes: list[mp.Process] = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    data_subsets[rank],
                    dataset_name,
                    args.max_new_tokens,
                    args.max_input_tokens,
                    True,
                    args.model_path,
                    args.draft_path,
                    str(out_path),
                    model_name,
                    spec_config,
                    args.method,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Compute accuracy from the generated jsonl.
        correct = 0
        evaluated = 0
        total = 0
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                total += 1
                obj = json.loads(line)
                if obj.get("evaluated"):
                    evaluated += 1
                    if obj.get("correct"):
                        correct += 1

        acc = (correct / evaluated * 100.0) if evaluated else 0.0
        print(f"[metrics] dataset={dataset_name} evaluated={evaluated}/{total} accuracy={acc:.3f}")

        metrics_path = model_output_dir / f"{dataset_name}-{args.method}-metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "method": args.method,
                    "evaluated": evaluated,
                    "total": total,
                    "accuracy": acc,
                    "correct": correct,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

