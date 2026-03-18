#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
        default="all",
        choices=["all", "specpv", "full", "naive", "ar"],
        help="Generation method (use `all` to run the full matrix).",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["qmsum", "gov_report", "multi_news", "longbench_v2", "aime2025", "codeelo"],
    )
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
    p.add_argument(
        "--spec_k",
        type=int,
        default=8,
        help="Vanilla speculative decoding: draft propose length K (used when --method=naive).",
    )
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


def slugify_model_path(path_or_repo: str) -> str:
    # Keep it stable for directory names.
    base = str(path_or_repo).rstrip("/").split("/")[-1]
    base = base.lower()
    out = []
    for ch in base:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_", "."):
            out.append("-")
        else:
            out.append("-")
    slug = "".join(out)
    slug = "-".join([s for s in slug.split("-") if s])
    return slug or "model"


def compute_profile_from_response(response_jsonl_path: Path, *, dataset: str, method: str) -> dict:
    records = []
    with open(response_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    n = len(records)
    k = 0
    for r in records:
        rates = r.get("acceptance_rate_per_pos", [])
        if rates:
            k = max(k, len(rates))
    if k == 0:
        avg_rates: list[float] = []
    else:
        sums = [0.0 for _ in range(k)]
        for r in records:
            rates = r.get("acceptance_rate_per_pos", [])
            for i in range(min(k, len(rates))):
                sums[i] += float(rates[i])
        avg_rates = [s / n if n > 0 else 0.0 for s in sums]

    return {
        "dataset": dataset,
        "method": method,
        "n_samples": n,
        "acceptance_rate_per_pos": avg_rates,
        "K": len(avg_rates),
    }


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    python = sys.executable

    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/scratch/.cache/datasets"))

    families = args.families or []
    if not families and not (args.model_path and args.draft_path):
        raise SystemExit("Provide either --families or both --model_path and --draft_path.")

    # "all" expands to a fixed matrix matching the project requirements:
    # - longbenchv1 (gov_report/qmsum/multi_news): `ar` everywhere, `naive/specpv` only for `qmsum`
    # - aime2025/codeelo: `ar` only
    run_methods = ["ar"] if args.method == "ar" else (["naive", "specpv", "ar"] if args.method == "all" else [args.method])

    # If user provides explicit model/draft, run once with them (ignores families).
    if args.model_path and args.draft_path:
        families_to_run = [None]
    else:
        families_to_run = families

    llama_target, llama_naive_draft = family_to_models("llama")
    # Eagle adapter path for Speculator-based speculative decoding.
    llama_eagle_draft = "/home/lthpc/nvmessd/zhendong/models/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K"

    for fam in families_to_run:
        if fam is None:
            target_model_path = args.model_path
            draft_model_path = args.draft_path
            output_root = args.output_root
        else:
            target_model_path, draft_model_path = family_to_models(fam)
            output_root = args.output_root

        for dataset in args.datasets:
            # Choose which methods are applicable for this dataset.
            if dataset in ("aime2025", "codeelo"):
                methods_to_run = [m for m in run_methods if m == "ar"]
            elif dataset in ("gov_report", "multi_news"):
                methods_to_run = [m for m in run_methods if m == "ar"]
            elif dataset == "longbench_v2":
                methods_to_run = ["ar"]
            elif dataset == "qmsum":
                methods_to_run = [m for m in run_methods if m in ("ar", "naive", "specpv")]
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            for method in methods_to_run:
                if fam is not None and method in ("naive", "specpv") and fam != "llama":
                    # Only llama has a working Eagle adapter for specpv/naive in this smoke matrix.
                    continue

                # For method=specpv we need the Eagle adapter (ea_model_path), not the vanilla small draft.
                effective_draft_path = draft_model_path
                if fam == "llama" and method == "specpv":
                    effective_draft_path = llama_eagle_draft

                if dataset in ("aime2025", "codeelo"):
                    script = "evaluation/eval_aime2025_codeelo.py"
                elif dataset == "longbench_v2":
                    script = "evaluation/longbenchv2_ar.py"
                else:
                    script = "evaluation/longbenchv1_summarization.py"

                cmd = [
                    python,
                    "-u",
                    str(repo_root / script),
                    "--model_path",
                    target_model_path,
                    "--draft_path",
                    effective_draft_path,
                    "--dataset_name",
                    dataset,
                    "--method",
                    method,
                    "--output_root",
                    str(Path(args.output_root) / "_runs"),
                    "--max_samples",
                    str(args.max_samples),
                    "--spec_k",
                    str(args.spec_k),
                ]
                if dataset in ("aime2025", "codeelo"):
                    cmd += ["--max_new_tokens", str(args.max_new_tokens)]
                # longbench_v2 is fixed to ar in this matrix; no extra args.
                if method == "specpv":
                    cmd += [
                        "--partial_length",
                        str(args.partial_length),
                        "--partial_spec_tokens",
                        str(args.partial_spec_tokens),
                    ]

                # Run evaluation.
                print("\n==== Running dataset ====")
                if fam is None:
                    print(f"dataset={dataset} method={method}")
                else:
                    print(f"family={fam} dataset={dataset} method={method}")
                print(" ".join(cmd))
                subprocess.check_call(cmd, env=env, cwd=str(repo_root))

                # Postprocess: rename to requested structure and generate profile.jsonl.
                model_name = os.path.basename(target_model_path).lower().strip()
                partial_suffix = ""
                if method == "specpv":
                    partial_suffix = f"-{args.partial_length}-{args.partial_spec_tokens}"

                # Source file names match the evaluation scripts.
                if dataset in ("aime2025", "codeelo"):
                    src = Path(args.output_root) / "_runs" / model_name / f"{dataset}-{method}.jsonl"
                else:
                    if method == "specpv":
                        src = Path(args.output_root) / "_runs" / model_name / f"{dataset}-{method}-{partial_suffix.lstrip('-')}.jsonl"
                    else:
                        src = Path(args.output_root) / "_runs" / model_name / f"{dataset}-{method}.jsonl"
                if not src.exists():
                    raise FileNotFoundError(f"Expected evaluation output not found: {src}")

                # Destination directory name: method/$dataset/$draft_$target
                if method == "ar":
                    draft_slug = "draftless"
                else:
                    draft_slug = slugify_model_path(effective_draft_path)
                target_slug = slugify_model_path(target_model_path)
                dest_dir = Path(args.output_root) / method / dataset / f"{draft_slug}_{target_slug}"
                dest_dir.mkdir(parents=True, exist_ok=True)

                response_dst = dest_dir / "response.jsonl"
                profile_dst = dest_dir / "profile.jsonl"
                if response_dst.exists():
                    response_dst.unlink()
                if profile_dst.exists():
                    profile_dst.unlink()

                response_dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

                profile = compute_profile_from_response(response_dst, dataset=dataset, method=method)
                with open(profile_dst, "w", encoding="utf-8") as f:
                    f.write(json.dumps(profile, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

