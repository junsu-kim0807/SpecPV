#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset

from specpv import SpecConfig, Speculator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partial KV smoke test.")
    p.add_argument(
        "--data_file",
        type=str,
        default=os.environ.get("SPECPV_PG19_DATA_FILE", "data/pg19_test/pg19_test_30k.parquet"),
        help="Parquet file path with a 'text' column.",
    )
    p.add_argument(
        "--base-model-path",
        type=str,
        default=os.environ.get("SPECPV_BASE_MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct"),
        help="Target model path or HF repo id (base model).",
    )
    p.add_argument(
        "--ea-model-path",
        type=str,
        default=os.environ.get("SPECPV_EA_MODEL_PATH", "meta-llama/Llama-3.2-1B-Instruct"),
        help="Draft model path or HF repo id (Eagle adapter).",
    )
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_model_path or not args.ea_model_path:
        raise SystemExit("Set --base-model-path/--ea-model-path or env SPECPV_BASE_MODEL_PATH/SPECPV_EA_MODEL_PATH.")

    dataset = load_dataset("parquet", data_files=args.data_file)["train"]
    text = dataset[0]["text"]

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
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer([chat_text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]

    spec_config = SpecConfig(
        enable_offload=True,
        enable_partial_kv=True,
        n_retrieval_blocks=512,
        partial_spec_tokens=20,
    )
    output_ids, metrics = model.spec_generate(
        input_ids,
        temperature=0,
        max_new_tokens=args.max_new_tokens,
        max_length=35000,
        log=True,
        is_llama3=True,
        spec_config=spec_config,
    )
    output = model.tokenizer.decode(output_ids[0][input_ids.shape[-1] :])
    print(output)
    print(metrics.get("avg_accept_length"))


if __name__ == "__main__":
    main()

