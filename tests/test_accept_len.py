#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import pandas as pd
import torch

from specpv import Speculator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure avg accept length for a few samples.")
    p.add_argument(
        "--base-model-path",
        type=str,
        default=os.environ.get("SPECPV_BASE_MODEL_PATH", ""),
        help="Base model path or HF repo id.",
    )
    p.add_argument(
        "--ea-model-path",
        type=str,
        default=os.environ.get("SPECPV_EA_MODEL_PATH", ""),
        help="EAGLE draft model path or HF repo id.",
    )
    p.add_argument(
        "--data_file",
        type=str,
        default=os.environ.get("SPECPV_PG19_DATA_FILE", "data/pg-19/test_pg19_10k_to_60k.parquet"),
        help="Parquet file path with a 'text' column.",
    )
    p.add_argument("--id_start", type=int, default=6)
    p.add_argument("--id_end", type=int, default=6)
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_model_path or not args.ea_model_path:
        raise SystemExit("Set --base-model-path/--ea-model-path or env SPECPV_BASE_MODEL_PATH/SPECPV_EA_MODEL_PATH.")

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

    df = pd.read_parquet(args.data_file)
    if "text" not in df.columns:
        raise SystemExit(f"Expected a 'text' column in {args.data_file}.")

    for seq_id in range(args.id_start, args.id_end + 1):
        sum_accept = 0.0
        n_samples = 20
        start = (seq_id - 1) * n_samples
        end = seq_id * n_samples
        for i in range(start, end):
            prompt = str(df["text"][i])
            system_prompt = "Please help me continue this story."
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = model_inputs["input_ids"]

            output_ids, accept_len = model.spec_generate(
                input_ids,
                temperature=0.7,
                max_new_tokens=args.max_new_tokens,
                max_length=10240 * seq_id + 1000,
            )

            avg_accept = sum(accept_len) / len(accept_len)
            sum_accept += avg_accept

        print(f"For {seq_id}0k tokens, average accept length: {(sum_accept / n_samples):.4f} tokens")


if __name__ == "__main__":
    main()

