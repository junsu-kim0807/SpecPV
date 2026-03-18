#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--draft_path", type=str, required=False, default=None)
    p.add_argument(
        "--dataset_name",
        type=str,
        default="longbench_v2",
        choices=["longbench_v2"],
    )
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["ar"],
        help="Only `ar` is supported for longbench_v2 in this matrix.",
    )
    p.add_argument("--spec_k", type=int, default=8)
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Absolute/relative path to exported JSONL.",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Directory containing exported JSONL.",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="outputs",
    )
    p.add_argument("--max_samples", type=int, default=5)
    return p.parse_args(args)


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def resolve_dataset_file(dataset_root: str | None, dataset_path: str | None) -> Path:
    if dataset_path:
        p = Path(dataset_path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        return p

    candidates: list[Path] = []
    if dataset_root:
        candidates.append(Path(dataset_root).expanduser().resolve() / "longbenchv2" / "longbench_v2.jsonl")
        candidates.append(Path(dataset_root).expanduser().resolve() / "longbench_v2.jsonl")

    hf_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_cache:
        candidates.append(Path(hf_cache).expanduser().resolve() / "longbenchv2" / "longbench_v2.jsonl")
        candidates.append(Path(hf_cache).expanduser().resolve() / "longbench_v2.jsonl")

    candidates.append(REPO_ROOT / "data" / "longbenchv2" / "longbench_v2.jsonl")

    for c in candidates:
        if c.exists():
            return c

    searched = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(f"Could not find longbench_v2 dataset.\nSearched:\n{searched}")


def build_chat(tokenizer, prompt_text: str, model_name: str, cot: str | None = None) -> str:
    """
    Keep the logic aligned with the previous `longbenchv2_pred.py` behavior:
    - llama3 family uses a two-turn chat when `cot` is provided.
    - qwen3 family uses enable_thinking=True; `cot` is handled by token concatenation outside.
    - deepseek uses the llama-style chat formatting as a best-effort fallback.
    """

    model_name_l = model_name.lower()
    if (
        ("llama3" in model_name_l)
        or ("llama-3" in model_name_l)
        or ("llama_3" in model_name_l)
        or ("deepseek" in model_name_l)
    ):
        if cot is not None:
            messages = [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": cot},
                {
                    "role": "user",
                    "content": "Based on the above, what is the single, most likely answer choice? "
                    "Format your response as follows: 'The correct answer is (insert answer here)'.",
                },
            ]
        else:
            messages = [{"role": "user", "content": prompt_text}]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    if "qwen3" in model_name_l:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return prompt

    # Fallback: treat as llama-style chat.
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def clean_cot(text: str) -> str:
    import re

    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_content = text
    if match:
        think_content = match.group(1)
        return f"<think>{think_content}</think>"

    match2 = re.search(r"<think>", text)
    if match2:
        return f"<think>{text[match2.end():]}</think>"
    return text


def extract_answer(response: str) -> str | None:
    import re

    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", response)
    if match:
        return match.group(1)
    return None


def ar_generate(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_length: int,
    is_llama3: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    start_time = time.time()
    device = input_ids.device

    eos_ids: list[int] | None
    if is_llama3:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        eos_ids = [int(eot_id)]
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_ids:
            eos_ids.append(int(tokenizer.eos_token_id))
    else:
        eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else None

    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    gen_kwargs: dict[str, Any] = {
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

    outputs = model.generate(input_ids=input_ids, **gen_kwargs)

    total_time = time.time() - start_time
    new_token = int(outputs.shape[-1] - input_ids.shape[-1])
    throughput = (new_token / total_time) if total_time > 0 else 0.0

    return outputs, {
        "new_token": new_token,
        "total_time": float(total_time),
        "throughput": float(throughput),
    }


def get_pred(
    rank: int,
    data,
    model_path: str,
    out_path: str,
    model_name: str,
    spec_k: int,
) -> None:
    device = torch.device(f"cuda:{rank}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
    )
    model.eval()

    is_llama3 = "llama3" in model_name or "llama-3" in model_name or "llama_3" in model_name
    is_qwen3 = "qwen3" in model_name

    # Pick the best available template dir.
    template_dir = "qwen3-8b" if is_qwen3 else "llama3.1-8b-instruct"
    template_path = REPO_ROOT / "evaluation" / "config" / "prompts" / template_dir / "0shot_cot_longbench_v2.txt"
    template_0shot_cot = template_path.read_text(encoding="utf-8")

    with open(out_path, "a", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"rank{rank}"):
            context = item["context"]
            prompt_text = template_0shot_cot
            prompt_text = (
                prompt_text.replace("$DOC$", context.strip())
                .replace("$Q$", item["question"].strip())
                .replace("$C_A$", item["choice_A"].strip())
                .replace("$C_B$", item["choice_B"].strip())
                .replace("$C_C$", item["choice_C"].strip())
                .replace("$C_D$", item["choice_D"].strip())
            )

            # Stage 1: generate CoT/thinking.
            prompt = build_chat(tokenizer, prompt_text, model_name)
            model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
            input_ids = model_inputs["input_ids"]

            max_new_tokens_cot = 1024
            max_length = int(input_ids.shape[1]) + max_new_tokens_cot + 200
            outputs_cot, m1 = ar_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens_cot,
                max_length=max_length,
                is_llama3=is_llama3,
            )

            output_cot_raw = tokenizer.decode(outputs_cot[0, input_ids.shape[1] :], skip_special_tokens=False)
            output_think_cot = clean_cot(output_cot_raw)

            # Stage 2: generate final answer.
            if is_qwen3:
                # For qwen3: concatenate the <think> tokens and continue decoding.
                output_q_ids = outputs_cot[0, : input_ids.shape[1]].unsqueeze(0).to(device)
                model_think_ids = tokenizer(output_think_cot, return_tensors="pt")["input_ids"].to(device)
                input_ans_ids = torch.cat([output_q_ids, model_think_ids], dim=1)
                max_new_tokens_ans = 128
                outputs_ans, m2 = ar_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ans_ids,
                    max_new_tokens=max_new_tokens_ans,
                    max_length=int(input_ans_ids.shape[1]) + max_new_tokens_ans + 50,
                    is_llama3=False,
                )
                output = tokenizer.decode(outputs_ans[0, input_ans_ids.shape[1] :], skip_special_tokens=True)
            else:
                prompt_ans = build_chat(tokenizer, prompt_text, model_name, cot=output_cot_raw)
                model_ans_inputs = tokenizer([prompt_ans], return_tensors="pt").to(device)
                input_ans_ids = model_ans_inputs["input_ids"]
                outputs_ans, m2 = ar_generate(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ans_ids,
                    max_new_tokens=128,
                    max_length=int(input_ans_ids.shape[1]) + 128 + 50,
                    is_llama3=is_llama3,
                )
                output = tokenizer.decode(outputs_ans[0, input_ans_ids.shape[1] :], skip_special_tokens=True)

            pred = extract_answer(output)
            evaluated = True
            correct = pred == item["answer"]

            total_time = float(m1["total_time"] + m2["total_time"])
            new_token = int(m1["new_token"] + m2["new_token"])
            throughput = (new_token / total_time) if total_time > 0 else 0.0

            rec = {
                "id": item.get("_id"),
                "pred": pred,
                "pred_raw": output,
                "gold": item.get("answer"),
                "evaluated": evaluated,
                "correct": bool(correct),
                "length": item.get("length"),
                "difficulty": item.get("difficulty"),
                "domain": item.get("domain"),
                "avg_acc_length": 0,
                "acceptance_rate_per_pos": [0.0 for _ in range(spec_k)],
                "new_token": float(new_token),
                "total_time": float(total_time),
                "throughput": float(throughput),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    seed_everything(42)
    args = parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices found. This evaluation script expects at least 1 GPU.")

    mp.set_start_method("spawn", force=True)

    dataset_file = resolve_dataset_file(args.dataset_root, args.dataset_path)
    data = load_dataset("json", data_files=str(dataset_file))["train"]
    data_all = [dict(x) for x in data]

    tokenizer_probe = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    max_context_len = 16000

    selected: list[dict[str, Any]] = []
    for item in data_all:
        # Filter by `context` token length (requirement: context length < 16K).
        ctx = item["context"]
        ctx_len = len(tokenizer_probe(ctx, add_special_tokens=False).input_ids)
        if ctx_len < max_context_len:
            selected.append(item)
        if len(selected) >= args.max_samples:
            break

    data_all = selected
    data_subsets = [data_all[i::world_size] for i in range(world_size)]

    output_root = Path(args.output_root)
    model_name = os.path.basename(args.model_path).lower()
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    out_path = model_output_dir / f"{args.dataset_name}-{args.method}.jsonl"
    if out_path.exists():
        out_path.unlink()

    processes: list[mp.Process] = []
    for rank in range(world_size):
        p = mp.Process(
            target=get_pred,
            args=(rank, data_subsets[rank], args.model_path, str(out_path), model_name, args.spec_k),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

