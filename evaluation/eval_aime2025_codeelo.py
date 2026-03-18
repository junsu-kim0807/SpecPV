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
from transformers import AutoModelForCausalLM, AutoTokenizer

from specpv import SpecConfig, Speculator
from specpv.speculate.naive_sd import vanilla_speculative_decode as vanilla_speculative_decode_sd

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
        choices=["specpv", "full", "naive", "ar"],
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
    p.add_argument(
        "--spec_k",
        type=int,
        default=8,
        help="Vanilla speculative decoding: draft propose length K (used when --method=naive).",
    )
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


def extract_first_code_block(text: str) -> tuple[str | None, str]:
    """
    Returns (lang_tag, code).
    If no code fence exists, returns (None, whole_text).
    """
    # Match ```lang\n...\n``` or ```\n...\n```
    m = re.search(r"```(\w+)?\n(.*?)\n```", text, flags=re.DOTALL)
    if not m:
        return None, text.strip()
    lang = (m.group(1) or "").strip().lower() or None
    code = (m.group(2) or "").strip()
    return lang, code


def extract_codeelo_lang_tag(model_output: str) -> str | None:
    lang, _code = extract_first_code_block(model_output)
    return lang


def get_codeelo_code_and_lang(model_output: str) -> tuple[str | None, str]:
    lang, code = extract_first_code_block(model_output)
    return lang, code


def extract_contest_id_from_prob(prob: str) -> str | None:
    # Common format: "2000A" / "1980B" etc.
    m = re.match(r"(\d+)", prob.strip())
    return m.group(1) if m else None


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
        # CodeElo expects generated code that can be submitted to Codeforces.
        # We store a "submission-ready response" by forcing the model to output
        # complete source code in one fenced code block labeled `cpp`.
        desc = str(row.get("description", "")).strip()
        inp = str(row.get("input", "")).strip()
        note = str(row.get("note", "")).strip()
        interaction = str(row.get("interaction", "")).strip()

        system_prompt = (
            "You are a competitive programming assistant. "
            "Write a complete C++ solution for the given problem. "
            "Output only the complete code inside a single fenced block with language tag `cpp`. "
            "Do not include any explanations or additional text."
        )

        parts: list[str] = []
        if desc:
            parts.append(desc)
        if note:
            parts.append(f"Note:\n{note}")
        if inp:
            parts.append(f"Sample Input:\n{inp}")
        if interaction:
            parts.append(f"Interaction:\n{interaction}")

        parts.append("Now write the complete C++ code for this problem.")
        return "\n\n".join(parts), system_prompt

    raise ValueError(f"Unsupported dataset_name={dataset_name}")


def get_gold_answer(dataset_name: str, row: dict[str, Any]) -> str:
    if dataset_name == "aime2025":
        return str(row.get("answer", "")).strip()
    if dataset_name == "codeelo":
        # Reference sample output (used only for optional local sanity checks).
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
    spec_k: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    target_model, draft_model, tokenizer = load_models_and_tokenizer(
        model_path, draft_path, device, method=method
    )

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
        is_llama3 = ("llama3" in model_name) or ("llama-3" in model_name) or ("llama_3" in model_name)

        if method == "ar":
            output = ar_generate_transformers(
                model=target_model,
                tokenizer=tokenizer,
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                is_llama3=is_llama3,
            )
            # draftless decoding: keep list shape consistent for aggregation.
            acceptance_rate_per_pos = [0.0 for _ in range(spec_k)]
        elif method == "naive":
            if draft_model is None:
                raise RuntimeError("method=naive requires draft_model.")
            output, _metrics = vanilla_speculative_decode_sd(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                is_llama3=is_llama3,
                spec_k=spec_k,
            )
            acceptance_rate_per_pos = _metrics.get("acceptance_rate_per_pos", [])
        elif method in ("specpv", "full"):
            # specpv/full: require Speculator (Eagle draft adapters).
            output, _metrics = target_model.spec_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                spec_config=spec_config,
                log=True,
            )
            accept_lengths = _metrics.get("accept_lengths", []) or []
            K = int(getattr(spec_config, "partial_spec_tokens", 20) or 20)
            n = len(accept_lengths)
            acceptance_rate_per_pos = [
                (sum(1 for al in accept_lengths if int(al) >= pos) / n) if n > 0 else 0.0
                for pos in range(K)
            ]
        else:
            raise ValueError(f"Unknown method={method}")

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
            # Official CodeElo uses code execution on Codeforces.
            # We do NOT compute official accuracy here; this is submission-prep stage.
            evaluated = False
            correct = False

            codeelo_lang_tag, codeelo_code = get_codeelo_code_and_lang(pred_raw)
            prob = json_obj.get("_id", json_obj.get("id", json_obj.get("prob", None)))
            if prob is None:
                prob = ""
            prob_str = str(prob)
            contest_id = extract_contest_id_from_prob(prob_str) if prob_str else None
        else:
            raise ValueError(dataset_name)

        with open(out_path, "a", encoding="utf-8") as f:
            prob_id = json_obj.get("_id", json_obj.get("id", json_obj.get("prob", None)))
            json.dump(
                {
                    "id": json_obj.get("_id", json_obj.get("id", None)),
                    "pred_raw": pred_raw,
                    "pred": pred_answer if dataset_name == "aime2025" else pred_output,
                    "gold": gold,
                    "evaluated": evaluated,
                    "correct": correct,
                    "acceptance_rate_per_pos": acceptance_rate_per_pos,
                    "codeelo_submission": (
                        {
                            "prob": prob_id,
                            "contest_id": extract_contest_id_from_prob(str(prob_id)) if prob_id else None,
                            "lang_tag": codeelo_lang_tag if dataset_name == "codeelo" else None,
                            "code": codeelo_code if dataset_name == "codeelo" else None,
                        }
                        if dataset_name == "codeelo"
                        else None
                    ),
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    if dist.is_initialized():
        dist.destroy_process_group()


def ar_generate_transformers(*, model, tokenizer, input_ids, max_new_tokens: int, max_length: int, is_llama3: bool):
    import time

    # Greedy decoding only (matches temperature=0 behaviour).
    eos_ids: list[int] | None
    if is_llama3:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [eot_id]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(tokenizer.eos_token_id)
    else:
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": int(max_new_tokens),
        "max_length": int(max_length),
        "do_sample": False,
        "temperature": 0.0,
        "use_cache": True,
        "attention_mask": attention_mask,
    }
    if eos_ids:
        gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
    if tokenizer.pad_token_id is not None:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    else:
        gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

    _start = time.time()
    outputs = model.generate(input_ids=input_ids, **gen_kwargs)
    return outputs


def vanilla_speculative_decode_local(
    *,
    target_model,
    draft_model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_length: int,
    is_llama3: bool,
    spec_k: int = 8,
):
    """
    Vanilla speculative decoding (accept/reject) using generic HF causal LMs.
    Returns the full generated token ids tensor (prompt + generated).
    """
    import time

    start_time = time.time()
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    output_ids = input_ids.clone()

    stop_ids: list[int] = []
    if is_llama3:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        stop_ids.append(eot_id)
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    stop_ids = sorted(set(stop_ids))

    generated = 0
    while generated < max_new_tokens and output_ids.shape[1] < max_length:
        base_ids = output_ids
        temp_ids = output_ids
        proposed: list[int] = []
        q_token_probs: list[float] = []

        with torch.no_grad():
            for _ in range(spec_k):
                if temp_ids.shape[1] >= max_length:
                    break
                draft_out = draft_model(input_ids=temp_ids)
                next_logits = draft_out.logits[:, -1, :]
                q_probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.argmax(q_probs, dim=-1, keepdim=True)  # [1,1]
                tok = int(next_token.item())
                proposed.append(tok)
                q_token_probs.append(float(q_probs[0, tok].item()))
                temp_ids = torch.cat([temp_ids, next_token], dim=1)

        if not proposed:
            break

        proposed_ids = torch.tensor(proposed, device=device, dtype=torch.long).unsqueeze(0)  # [1,K]
        verify_ids = torch.cat([base_ids, proposed_ids], dim=1)

        with torch.no_grad():
            target_out = target_model(input_ids=verify_ids)
            target_logits = target_out.logits  # [1, L+K, vocab]

            stop_reached = False
            for i, tok in enumerate(proposed):
                if generated >= max_new_tokens or output_ids.shape[1] >= max_length:
                    break

                pos = base_ids.shape[1] + i - 1
                p_logits = target_logits[:, pos, :]
                p_probs = torch.softmax(p_logits, dim=-1)
                p_tok_prob = float(p_probs[0, tok].item())
                q_tok_prob = q_token_probs[i]

                if q_tok_prob <= 0:
                    a = 1.0
                else:
                    a = min(1.0, p_tok_prob / q_tok_prob)

                if float(torch.rand(1).item()) <= a:
                    next_token_tensor = torch.tensor([[tok]], device=device, dtype=torch.long)
                    output_ids = torch.cat([output_ids, next_token_tensor], dim=1)
                    generated += 1
                    if tok in stop_ids:
                        stop_reached = True
                        break
                else:
                    next_token = torch.argmax(p_probs, dim=-1, keepdim=True)  # [1,1]
                    next_tok = int(next_token.item())
                    output_ids = torch.cat([output_ids, next_token], dim=1)
                    generated += 1
                    if next_tok in stop_ids:
                        stop_reached = True
                    break

            if stop_reached:
                break

    _ = time.time() - start_time
    return output_ids


def load_models_and_tokenizer(
    model_path: str, draft_path: str, device: torch.device, *, method: str
) -> tuple[Any, Any | None, Any]:
    """
    Returns (target_model, draft_model, tokenizer)
    """
    if method in ("ar", "naive"):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        target_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
        target_model.eval()
        if method == "ar":
            return target_model, None, tokenizer

        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map={"": str(device)},
        )
        draft_model.eval()
        return target_model, draft_model, tokenizer

    # specpv/full: require Speculator (Eagle draft adapters).
    spec_model = Speculator.from_pretrained(
        base_model_path=model_path,
        ea_model_path=draft_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device,
        total_token=-1,
    )
    spec_model.eval()
    tokenizer = spec_model.tokenizer
    return spec_model, None, tokenizer


def main() -> None:
    seed_everything(42)
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. This evaluation script expects at least 1 GPU.")

    mp.set_start_method("spawn", force=True)

    datasets = ["aime2025", "codeelo"] if args.dataset_name == "all" else [args.dataset_name]

    spec_config = SpecConfig()
    # We do not use KV cache offload for these experiments.
    spec_config.enable_offload = False
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
                    args.spec_k,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Compute accuracy from the generated jsonl.
        total = 0
        correct = 0
        evaluated = 0

        code_submitted = 0
        if not out_path.exists():
            print(f"[metrics] WARNING: output jsonl not found: {out_path} (worker likely crashed).")
        else:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    total += 1
                    obj = json.loads(line)
                    if obj.get("evaluated"):
                        evaluated += 1
                        if obj.get("correct"):
                            correct += 1
                    if dataset_name == "codeelo":
                        sub = obj.get("codeelo_submission") or {}
                        if sub.get("code"):
                            code_submitted += 1

        metrics_path = model_output_dir / f"{dataset_name}-{args.method}-metrics.json"
        if dataset_name == "aime2025":
            acc = (correct / evaluated * 100.0) if evaluated else 0.0
            print(f"[metrics] dataset={dataset_name} evaluated={evaluated}/{total} accuracy={acc:.3f}")
            metrics = {
                "dataset": dataset_name,
                "method": args.method,
                "evaluated": evaluated,
                "total": total,
                "accuracy": acc,
                "correct": correct,
            }
        else:
            # For CodeElo, official scoring requires submission+execution on Codeforces.
            print(f"[metrics] dataset={dataset_name} code_extracted={code_submitted}/{total} (no local accuracy)")
            metrics = {
                "dataset": dataset_name,
                "method": args.method,
                "total": total,
                "code_extracted": code_submitted,
                "note": "Official CodeElo scoring requires CodeElo submission; use scripts/codeelo_submit_and_score.py.",
            }

        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

