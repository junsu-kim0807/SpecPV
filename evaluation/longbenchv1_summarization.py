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

from specpv import Speculator, SpecConfig


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
        choices=["specpv", "full", "naive"],
    )
    parser.add_argument("--partial_length", type=str, default=None)
    parser.add_argument("--partial_spec_tokens", type=str, default="20")

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


def get_pred(rank, data, max_gen, prompt_format, dataset, model_path, draft_path, out_path, model_name, spec_config, method):
    device = torch.device(f"cuda:{rank}")
    max_length = 60000
    model, tokenizer = load_model_and_tokenizer(model_path, draft_path, device)

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
        is_llama3 = "llama3" in model_name

        if method == "naive":
            output, metrics = model.naive_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                log=True,
            )
            metrics["avg_accept_length"] = 0
        else:
            output, metrics = model.spec_generate(
                input_ids=model_inputs.input_ids,
                max_new_tokens=max_gen,
                max_length=context_length + max_gen + 100,
                temperature=0.0,
                is_llama3=is_llama3,
                spec_config=spec_config,
                log=True,
            )

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


def load_model_and_tokenizer(model_path, draft_path, device):
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
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        print(f"[done] wrote {out_path}")