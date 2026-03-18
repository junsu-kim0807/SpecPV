#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset

from specpv import SpecConfig, Speculator
from specpv.speculate.profile import print_time_stats, reset_time_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile accept-length/time for several sequence lengths.")
    p.add_argument(
        "--data_file",
        type=str,
        default=os.environ.get("SPECPV_PG19_DATA_FILE", "data/pg19_test/pg19_test_60k.parquet"),
    )
    p.add_argument(
        "--base-model-path",
        type=str,
        default=os.environ.get("SPECPV_BASE_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
    )
    p.add_argument(
        "--ea-model-path",
        type=str,
        default=os.environ.get("SPECPV_EA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct"),
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples_per_len", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_model_path or not args.ea_model_path:
        raise SystemExit("Set --base-model-path/--ea-model-path or env SPECPV_BASE_MODEL_PATH/SPECPV_EA_MODEL_PATH.")

    dataset = load_dataset("parquet", data_files=args.data_file)["train"]
    text0 = dataset[0]["text"]

    model = Speculator.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        total_token=-1,
    )
    model.eval()

    tokenizer = model.tokenizer
    system_prompt = "You are a creative writing assistant. Continue the following story in a coherent way."
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text0}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]

    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    spec_config = SpecConfig(
        enable_offload=False,
        enable_partial_kv=False,
        n_retrieval_blocks=512,
        partial_spec_tokens=20,
    )

    # warm up
    for _ in range(args.warmup):
        _output_ids, _metrics = model.spec_generate(
            input_ids[:, :20000],
            temperature=0,
            max_new_tokens=64,
            max_length=65000,
            log=True,
            is_llama3=True,
            spec_config=spec_config,
        )

    for seqlen in seq_lens:
        print(f"\n==============================\n### Testing sequence length: {seqlen}\n==============================")
        reset_time_stats()

        for idx in range(args.samples_per_len):
            text = dataset[idx]["text"]
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([chat_text], return_tensors="pt").to(device)
            input_ids = model_inputs["input_ids"]

            trunc_len = min(seqlen, input_ids.shape[-1])
            input_ids = input_ids[:, :trunc_len]

            _output_ids, metrics = model.spec_generate(
                input_ids,
                temperature=0,
                max_new_tokens=512,
                max_length=65000,
                log=True,
                is_llama3=True,
                spec_config=spec_config,
            )
            print(f"Sample {idx} avg_accept_length = {metrics.get('avg_accept_length')}")

        print_time_stats()


if __name__ == "__main__":
    main()

