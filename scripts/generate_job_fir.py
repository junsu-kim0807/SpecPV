#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "longbenchv1_summarization"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "longbenchv1_summarization"
RESULTS_ROOT = REPO_ROOT / "results" / "longbenchv1_summarization"

# ============================================
# FIR environment
# ============================================
REPO_DIR = "/home/jhwoo36/scratch/specdec/SpecPV"
VENV_DIR = "/home/jhwoo36/scratch/venvs/vllm"

# code 1 스타일을 최대한 유지하면서, HF hub도 확실히 같은 캐시를 쓰게 한다.
HF_HOME = "/home/jhwoo36/scratch/.cache"
TRANSFORMERS_CACHE = "/home/jhwoo36/scratch/.cache/transformers"
HUGGINGFACE_HUB_CACHE = "/home/jhwoo36/scratch/.cache/huggingface_hub"
HF_DATASETS_CACHE = os.path.expanduser("~/scratch/.cache/datasets")

# 필요하면 생성 전에 쉘에서 export HF_TOKEN=... 해두면 된다.
HF_TOKEN = os.environ.get("HF_TOKEN", "")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_ref: str
    gpu_count: int = 1
    cpus_per_task: int = 16
    mem_per_gpu: str = "80G"
    time_limit: str = "08:00:00"


@dataclass(frozen=True)
class DatasetConfig:
    name: str


@dataclass(frozen=True)
class MethodConfig:
    name: str
    extra_args: tuple[str, ...] = ()


# ============================================
# Models
# model_ref는 "로컬 경로 또는 HF repo id" 둘 다 가능하다.
# 현재는 다른 모델들과 동일하게 HF repo id를 그대로 넘기도록 구성.
# ============================================
TARGET_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="llama31_8b_instruct",
        model_ref="meta-llama/Llama-3.1-8B-Instruct",
        gpu_count=4,
        cpus_per_task=16,
        mem_per_gpu="80G",
        time_limit="08:00:00",
    ),
]

DRAFT_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="eagle3_llama31_8b_yarn_64k",
        model_ref="TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K",
        gpu_count=4,
        cpus_per_task=16,
        mem_per_gpu="80G",
        time_limit="03:00:00",
    ),
]

DATASETS: list[DatasetConfig] = [
    DatasetConfig(name="qmsum"),
    DatasetConfig(name="gov_report"),
    DatasetConfig(name="aime2025"),
    DatasetConfig(name="codeelo"),
]

METHODS: list[MethodConfig] = [
    MethodConfig(name="full"),
    MethodConfig(name="ar"),
    MethodConfig(name="naive"),
    MethodConfig(
        name="specpv",
        extra_args=(
            "--partial_length", "2048",
            "--partial_spec_tokens", "20",
        ),
    ),
]


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def shquote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def pair_slug(draft: ModelConfig, target: ModelConfig) -> str:
    return f"{sanitize_for_path(draft.name)}__TO__{sanitize_for_path(target.name)}"


def ensure_dirs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def filter_by_attr(items: list, wanted: set[str], attr: str) -> list:
    if not wanted:
        return items
    return [item for item in items if getattr(item, attr) in wanted]


def job_name(
    method: MethodConfig,
    dataset: DatasetConfig,
    draft: ModelConfig,
    target: ModelConfig,
) -> str:
    return f"lb_{method.name}_{dataset.name}_{draft.name}_to_{target.name}"


def job_header(
    *,
    method: MethodConfig,
    dataset: DatasetConfig,
    draft: ModelConfig,
    target: ModelConfig,
    log_dir: Path,
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_count = max(draft.gpu_count, target.gpu_count)
    cpus_per_task = max(draft.cpus_per_task, target.cpus_per_task)
    time_limit = target.time_limit
    mem_per_gpu = target.mem_per_gpu
    jname = job_name(method, dataset, draft, target)

    hf_token_export = ""
    if HF_TOKEN:
        hf_token_export = f"export HF_TOKEN={shquote(HF_TOKEN)}"

    return f"""#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:{gpu_count}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --time={time_limit}
#SBATCH --output={log_dir / (jname + ".out")}
#SBATCH --error={log_dir / (jname + ".err")}

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shquote(REPO_DIR)}
VENV_DIR={shquote(VENV_DIR)}

cd "${{REPO_DIR}}"
source "${{VENV_DIR}}/bin/activate"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD="${{VLLM_WORKER_MULTIPROC_METHOD:-spawn}}"

{hf_token_export}

export HF_HOME={shquote(HF_HOME)}
export TRANSFORMERS_CACHE={shquote(TRANSFORMERS_CACHE)}
export HUGGINGFACE_HUB_CACHE={shquote(HUGGINGFACE_HUB_CACHE)}
export HF_HUB_CACHE={shquote(HUGGINGFACE_HUB_CACHE)}
export HF_DATASETS_CACHE={shquote(HF_DATASETS_CACHE)}

mkdir -p \\
  "${{HF_HOME}}" \\
  "${{TRANSFORMERS_CACHE}}" \\
  "${{HUGGINGFACE_HUB_CACHE}}" \\
  "${{HF_DATASETS_CACHE}}" \\
  "{RESULTS_ROOT}"
"""


def build_python_command(
    *,
    method: MethodConfig,
    dataset: DatasetConfig,
    draft: ModelConfig,
    target: ModelConfig,
) -> str:
    if dataset.name in ("aime2025", "codeelo"):
        entry = "evaluation/eval_aime2025_codeelo.py"
    else:
        entry = "evaluation/longbenchv1_summarization.py"

    args = [
        f"python -u {entry}",
        f"--model_path {shquote(target.model_ref)}",
        f"--draft_path {shquote(draft.model_ref)}",
        f"--dataset_name {shquote(dataset.name)}",
        f"--method {shquote(method.name)}",
    ]

    extra = list(method.extra_args)
    for i in range(0, len(extra), 2):
        key = extra[i]
        value = extra[i + 1]
        args.append(f"{key} {shquote(value)}")

    return " \\\n  ".join(args)


def render_job_script(
    *,
    method: MethodConfig,
    dataset: DatasetConfig,
    draft: ModelConfig,
    target: ModelConfig,
) -> str:
    slug = pair_slug(draft, target)
    jname = job_name(method, dataset, draft, target)

    log_dir = LOGS_ROOT / method.name / dataset.name / slug
    result_dir = RESULTS_ROOT / method.name / dataset.name / slug

    header = job_header(
        method=method,
        dataset=dataset,
        draft=draft,
        target=target,
        log_dir=log_dir,
    )

    cmd = build_python_command(
        method=method,
        dataset=dataset,
        draft=draft,
        target=target,
    )

    body = f"""
METHOD_NAME={shquote(method.name)}
DATASET_NAME={shquote(dataset.name)}
DRAFT_NAME={shquote(draft.name)}
TARGET_NAME={shquote(target.name)}
PAIR_SLUG={shquote(slug)}
JOB_NAME_STR={shquote(jname)}

DRAFT_MODEL_REF={shquote(draft.model_ref)}
TARGET_MODEL_REF={shquote(target.model_ref)}

RESULT_DIR={shquote(str(result_dir))}
mkdir -p "${{RESULT_DIR}}"

echo "============================================================"
echo "JOB_NAME            : ${{JOB_NAME_STR}}"
echo "METHOD              : ${{METHOD_NAME}}"
echo "DATASET             : ${{DATASET_NAME}}"
echo "DRAFT_NAME          : ${{DRAFT_NAME}}"
echo "DRAFT_MODEL_REF     : ${{DRAFT_MODEL_REF}}"
echo "TARGET_NAME         : ${{TARGET_NAME}}"
echo "TARGET_MODEL_REF    : ${{TARGET_MODEL_REF}}"
echo "PAIR_SLUG           : ${{PAIR_SLUG}}"
echo "RESULT_DIR          : ${{RESULT_DIR}}"
echo "HF_HOME             : ${{HF_HOME}}"
echo "HF_HUB_CACHE        : ${{HF_HUB_CACHE}}"
echo "TRANSFORMERS_CACHE  : ${{TRANSFORMERS_CACHE}}"
echo "HF_DATASETS_CACHE   : ${{HF_DATASETS_CACHE}}"
echo "HOSTNAME            : $(hostname)"
echo "PWD                 : $(pwd)"
echo "PYTHON              : $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${{CUDA_VISIBLE_DEVICES:-unset_by_shell}}"
echo "============================================================"

env | sort > "${{RESULT_DIR}}/env.txt"
cp "$0" "${{RESULT_DIR}}/job.slurm"

{cmd} 2>&1 | tee "${{RESULT_DIR}}/run.log"

echo "Saved outputs under: ${{RESULT_DIR}}"
"""
    return header + "\n" + body.strip() + "\n"


def write_job_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--drafts", nargs="*", default=[])
    parser.add_argument("--targets", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    methods = filter_by_attr(METHODS, set(args.methods), "name")
    datasets = filter_by_attr(DATASETS, set(args.datasets), "name")
    drafts = filter_by_attr(DRAFT_MODELS, set(args.drafts), "name")
    targets = filter_by_attr(TARGET_MODELS, set(args.targets), "name")

    num_written = 0

    for method in methods:
        for dataset in datasets:
            for draft in drafts:
                for target in targets:
                    slug = pair_slug(draft, target)
                    path = JOBS_ROOT / method.name / dataset.name / f"{slug}.slurm"
                    text = render_job_script(
                        method=method,
                        dataset=dataset,
                        draft=draft,
                        target=target,
                    )
                    write_job_script(path, text)
                    num_written += 1
                    print(f"Wrote {path}")

    print(f"Generated {num_written} job scripts.")


if __name__ == "__main__":
    main()