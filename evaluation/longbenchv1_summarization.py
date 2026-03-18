#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from specpv import Speculator, SpecConfig
from specpv.speculate.naive_sd import vanilla_speculative_decode as vanilla_speculative_decode_sd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--draft_path", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gov_report",
        choices=["gov_report", "qmsum", "multi_news", "all"],
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["specpv", "full", "naive", "ar"],
    )
    parser.add_argument("--partial_length", type=str, default=None)
    parser.add_argument("--partial_spec_tokens", type=str, default="20")
    parser.add_argument(
        "--spec_k",
        type=int,
        default=8,
        help="Vanilla speculative decoding: draft propose length K (used when --method=naive).",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Absolute or relative path to a single local JSONL file.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Directory containing local JSONL files such as gov_report.jsonl.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="For smoke tests: limit number of samples per dataset.",
    )
    return parser.parse_args(args)


def build_chat(tokenizer, prompt, model_name):
    messages = [{"role": "user", "content": prompt}]
    if "qwen3" in model_name:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def resolve_dataset_file(dataset_name: str, dataset_path: str | None, dataset_root: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    candidates = []

    if dataset_root:
        candidates.append(Path(dataset_root).expanduser() / f"{dataset_name}.jsonl")

    env_root = os.environ.get("SPECPV_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser() / "longbenchv1" / f"{dataset_name}.jsonl")
        candidates.append(Path(env_root).expanduser() / f"{dataset_name}.jsonl")

    hf_cache = os.environ.get("HF_DATASETS_CACHE")
    if hf_cache:
        # download_dataset.py exports LongBench v1 under:
        #   $HF_DATASETS_CACHE/longbenchv1/{subset}.jsonl
        candidates.append(Path(hf_cache).expanduser() / "longbenchv1" / f"{dataset_name}.jsonl")
        candidates.append(Path(hf_cache).expanduser() / f"{dataset_name}.jsonl")
        # Legacy fallback (older experiments)
        candidates.append(Path(hf_cache).expanduser() / "specpv_eval" / "longbenchv1" / f"{dataset_name}.jsonl")

    candidates.append(REPO_ROOT / "data" / "longbenchv1" / f"{dataset_name}.jsonl")

    for path in candidates:
        path = path.resolve()
        if path.exists():
            return path

    searched = "\n".join(str(p.resolve()) for p in candidates)
    raise FileNotFoundError(
        f"Could not find dataset file for {dataset_name}.\nSearched:\n{searched}"
    )


def get_pred(
    rank,
    data,
    max_gen,
    prompt_format,
    dataset,
    model_path,
    draft_path,
    out_path,
    model_name,
    spec_config,
    method,
    spec_k: int,
):
    device = torch.device(f"cuda:{rank}")
    max_length = 60000
    target_model, draft_model, tokenizer = load_models_and_tokenizer(
        model_path, draft_path, device, method=method
    )

    for json_obj in tqdm(data, desc=f"rank{rank}"):
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)

        model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
        context_length = model_inputs.input_ids.shape[-1]
        # model_name is typically a basename like "Llama-3.1-8B-Instruct" (not "llama3"),
        # so detect both variants.
        is_llama3 = ("llama3" in model_name) or ("llama-3" in model_name) or ("llama_3" in model_name)

        if method == "ar":
            output, metrics = ar_generate_transformers(
                model=target_model,
                tokenizer=tokenizer,
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                is_llama3=is_llama3,
            )
            # draftless decoding: acceptance per speculative position is not defined.
            # Still keep the list shape consistent for downstream aggregation.
            metrics["acceptance_rate_per_pos"] = [0.0 for _ in range(spec_k)]
        elif method == "naive":
            if draft_model is None:
                raise RuntimeError("method=naive requires draft_model.")
            output, metrics = vanilla_speculative_decode_sd(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                is_llama3=is_llama3,
                spec_k=spec_k,
            )
        else:
            output, metrics = target_model.spec_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                spec_config=spec_config,
                log=True,
            )
            # spec_generate returns per-iteration accept lengths.
            # In Speculator, `accept_length` corresponds to accepted_count - 1,
            # so position `pos` is accepted iff `pos <= accept_length`.
            accept_lengths = metrics.get("accept_lengths", []) or []
            K = int(getattr(spec_config, "partial_spec_tokens", 20) or 20)
            n = len(accept_lengths)
            acceptance_rate_per_pos: list[float] = []
            for pos in range(K):
                if n <= 0:
                    acceptance_rate_per_pos.append(0.0)
                else:
                    accepted = sum(1 for al in accept_lengths if int(al) >= pos)
                    acceptance_rate_per_pos.append(accepted / n)
            metrics["acceptance_rate_per_pos"] = acceptance_rate_per_pos

        pred = tokenizer.decode(output[0][context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj.get("all_classes"),
                    "length": context_length,
                    "avg_acc_length": metrics["avg_accept_length"],
                    "acceptance_rate_per_pos": metrics.get("acceptance_rate_per_pos", []),
                    "new_token": metrics["new_token"],
                    "total_time": metrics["total_time"],
                    "throughput": metrics["throughput"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

    if dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def ar_generate_transformers(*, model, tokenizer, input_ids, max_new_tokens: int, max_length: int, is_llama3: bool):
    """
    Transformers-based greedy decoding for `method=naive`.
    This path intentionally does NOT require Eagle draft adapters.
    """
    import time

    start_time = time.time()

    # Stop conditions: llama3 uses special <|eot_id|> token, otherwise rely on eos_token_id.
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

    outputs = model.generate(input_ids=input_ids, **gen_kwargs)

    total_time = time.time() - start_time
    new_token = outputs.shape[-1] - input_ids.shape[-1]
    throughput = (new_token / total_time) if total_time > 0 else 0.0

    return outputs, {
        "avg_accept_length": 0,
        "acceptance_rate_per_pos": [],
        "new_token": int(new_token),
        "total_time": float(total_time),
        "throughput": float(throughput),
    }


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
    This does not use Eagle adapters; it only needs target_model + draft_model.
    """
    import time

    start_time = time.time()
    device = input_ids.device

    prompt_len = input_ids.shape[1]
    output_ids = input_ids.clone()

    # Stop ids
    stop_ids: list[int] = []
    if is_llama3:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        stop_ids.append(eot_id)
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    stop_ids = sorted(set(stop_ids))

    generated = 0
    accept_lengths: list[int] = []
    while generated < max_new_tokens and output_ids.shape[1] < max_length:
        # 1) Draft propose K tokens greedily
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

        # 2) Verify using target model logits for each proposed token position
        with torch.no_grad():
            target_out = target_model(input_ids=verify_ids)
            target_logits = target_out.logits  # [1, L+K, vocab]

            stop_reached = False
            accept_len = 0
            for i, tok in enumerate(proposed):
                if generated >= max_new_tokens or output_ids.shape[1] >= max_length:
                    break

                pos = base_ids.shape[1] + i - 1  # predicts token at base_len + i
                p_logits = target_logits[:, pos, :]
                p_probs = torch.softmax(p_logits, dim=-1)
                p_tok_prob = float(p_probs[0, tok].item())
                q_tok_prob = q_token_probs[i]

                # Acceptance probability a = min(1, p/q)
                if q_tok_prob <= 0:
                    a = 1.0
                else:
                    a = min(1.0, p_tok_prob / q_tok_prob)

                if float(torch.rand(1).item()) <= a:
                    # accept proposed token
                    next_token_tensor = torch.tensor([[tok]], device=device, dtype=torch.long)
                    output_ids = torch.cat([output_ids, next_token_tensor], dim=1)
                    generated += 1
                    accept_len += 1
                    if tok in stop_ids:
                        stop_reached = True
                        break
                else:
                    # reject: sample next token from target distribution (deterministic argmax)
                    next_token = torch.argmax(p_probs, dim=-1, keepdim=True)  # [1,1]
                    next_tok = int(next_token.item())
                    output_ids = torch.cat([output_ids, next_token], dim=1)
                    generated += 1
                    if next_tok in stop_ids:
                        stop_reached = True
                        break
                    break

            # If the model rejected early, loop continues from new output_ids state.
            accept_lengths.append(accept_len)
            if stop_reached:
                break

    total_time = time.time() - start_time
    new_token = output_ids.shape[-1] - prompt_len
    throughput = (new_token / total_time) if total_time > 0 else 0.0

    avg_accept_length = (sum(accept_lengths) / len(accept_lengths)) if accept_lengths else 0.0
    return output_ids, {
        "avg_accept_length": float(avg_accept_length),
        "new_token": int(new_token),
        "total_time": float(total_time),
        "throughput": float(throughput),
    }



def load_models_and_tokenizer(model_path, draft_path, device, *, method: str):
    """
    Returns (target_model, draft_model, tokenizer).
    - method=ar: draft_model=None
    - method=naive: draft_model is required (vanilla speculative decoding)
    - method=specpv/full: draft_model is None; target_model is a Speculator
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

    # specpv/full: require Eagle3 draft adapter.
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


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    if args.dataset_name == "all":
        datasets = ["gov_report", "qmsum"]
    else:
        datasets = [args.dataset_name]

    dataset2prompt = json.load(open(REPO_ROOT / "evaluation" / "config" / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(REPO_ROOT / "evaluation" / "config" / "dataset2maxlen.json", "r"))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    spec_config = SpecConfig()
    # We do not use KV cache offload for these experiments.
    spec_config.enable_offload = False
    if args.method == "full":
        spec_config.enable_partial_kv = False
    elif args.method == "specpv":
        spec_config.partial_spec_tokens = int(args.partial_spec_tokens)
        spec_config.enable_partial_kv = True
        spec_config.n_retrieval_blocks = int(args.partial_length) // spec_config.block_size
    else:
        print("Naive generation, no spec config needed.")

    print(spec_config)

    model_name = os.path.basename(args.model_path).lower()
    model_output_dir = output_root / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        dataset_file = resolve_dataset_file(
            dataset_name=dataset,
            dataset_path=args.dataset_path,
            dataset_root=args.dataset_root,
        )
        print(f"[dataset] using local file: {dataset_file}")

        data = load_dataset("json", data_files=str(dataset_file))["train"]

        data_all = [data_sample for data_sample in data]
        if args.max_samples is not None:
            data_all = data_all[: args.max_samples]

        out_path = model_output_dir / f"{dataset}-{args.method}.jsonl"
        if args.method == "specpv":
            out_path = model_output_dir / f"{dataset}-{args.method}-{args.partial_length}-{args.partial_spec_tokens}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        if out_path.exists():
            out_path.unlink()

        processes = []
        for rank in range(world_size):
            p = mp.Process(
                target=get_pred,
                args=(
                    rank,
                    data_subsets[rank],
                    max_gen,
                    prompt_format,
                    dataset,
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

        print(f"[done] wrote {out_path}")