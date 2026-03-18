#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch

from specpv import Speculator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick smoke test for Speculator (Llama3 + EAGLE3).")
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
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
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
    prompt = (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, "
        "mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate "
        "meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her "
        "chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final "
        "meal of the day if the size of Wendi's flock is 20 chickens?"
    )
    system_prompt = (
        "Solve the following math problem efficiently and clearly:\n\n"
        "- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n"
        "- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n"
        "## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n"
        "## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n"
        "...\n\n"
        "Regardless of the approach, always conclude with:\n\n"
        "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
        "Where [answer] is just the final number or expression that solves the problem."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]

    # generate
    output_ids, _metrics = model.spec_generate(
        input_ids, temperature=args.temperature, max_new_tokens=args.max_new_tokens, max_length=64000
    )
    output = model.tokenizer.decode(output_ids[0])
    print(output)


if __name__ == "__main__":
    main()

