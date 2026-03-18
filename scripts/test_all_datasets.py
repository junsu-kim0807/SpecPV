#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test all supported datasets.")
    p.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="(Optional) Target model path or HF repo id. If not provided, --families presets are used.",
    )
    p.add_argument(
        "--draft_path",
        type=str,
        default=None,
        help="(Optional) Draft model path or HF repo id. If not provided, --families presets are used.",
    )
    p.add_argument(
        "--method",
        type=str,
        default="naive",
        choices=["specpv", "full", "naive"],
        help="Generation method.",
    )
    p.add_argument("--datasets", nargs="*", default=["qmsum", "gov_report", "aime2025", "codeelo"])
    p.add_argument("--max_samples", type=int, default=5, help="Limit samples per dataset.")
    p.add_argument("--output_root", type=str, default="outputs_smoke")

    p.add_argument(
        "--families",
        nargs="*",
        default=["llama", "qwen", "deepseek"],
        choices=["llama", "qwen", "deepseek"],
        help="If set, runs family presets: (llama draft=3.2-1B -> target=3.1-8B), etc.",
    )

    # specpv options (only used when --method=specpv)
    p.add_argument("--partial_length", type=str, default="2048")
    p.add_argument("--partial_spec_tokens", type=str, default="20")

    # aime/codeelo options
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()


def family_to_models(family: str) -> tuple[str, str]:
    """
    Returns (target_model_path, draft_model_path) for the given family.
    """
    if family == "llama":
        # draft: Llama 3.2 1B Instruct
        # target: Llama 3.1 8B Instruct
        return (
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
        )
    if family == "qwen":
        # draft: Qwen3 0.6B
        # target: Qwen3 8B
        return (
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-0.6B",
        )
    if family == "deepseek":
        # draft: deepseek-coder-instruct 1.3B
        # target: deepseek-coder-instruct 6.7B
        return (
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
        )
    raise ValueError(f"Unknown family: {family}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/scratch/.cache/datasets"))

    families = args.families or []
    if not families and not (args.model_path and args.draft_path):
        raise SystemExit("Provide either --families or both --model_path and --draft_path.")

    # If user provides explicit model/draft, run once with them (ignores families).
    if args.model_path and args.draft_path:
        families_to_run = [None]
    else:
        families_to_run = families

    for fam in families_to_run:
        if fam is None:
            target_model_path = args.model_path
            draft_model_path = args.draft_path
            output_root = args.output_root
        else:
            target_model_path, draft_model_path = family_to_models(fam)
            output_root = str(Path(args.output_root) / fam)

        for dataset in args.datasets:
            if dataset in ("aime2025", "codeelo"):
                script = "evaluation/eval_aime2025_codeelo.py"
                cmd = [
                    python,
                    "-u",
                    str(repo_root / script),
                    "--model_path",
                    target_model_path,
                    "--draft_path",
                    draft_model_path,
                    "--dataset_name",
                    dataset,
                    "--method",
                    args.method,
                    "--output_root",
                    output_root,
                    "--max_samples",
                    str(args.max_samples),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                ]
                if args.method == "specpv":
                    cmd += [
                        "--partial_length",
                        str(args.partial_length),
                        "--partial_spec_tokens",
                        str(args.partial_spec_tokens),
                    ]
            else:
                # LongBench v1-style summarization datasets.
                script = "evaluation/longbenchv1_summarization.py"
                cmd = [
                    python,
                    "-u",
                    str(repo_root / script),
                    "--model_path",
                    target_model_path,
                    "--draft_path",
                    draft_model_path,
                    "--dataset_name",
                    dataset,
                    "--method",
                    args.method,
                    "--output_root",
                    output_root,
                    "--max_samples",
                    str(args.max_samples),
                ]
                if args.method == "specpv":
                    cmd += [
                        "--partial_length",
                        str(args.partial_length),
                        "--partial_spec_tokens",
                        str(args.partial_spec_tokens),
                    ]

            print("\n==== Running dataset ====")
            if fam is None:
                print(f"dataset={dataset} method={args.method}")
            else:
                print(f"family={fam} dataset={dataset} method={args.method}")
            print(" ".join(cmd))
            subprocess.check_call(cmd, env=env, cwd=str(repo_root))


if __name__ == "__main__":
    main()

