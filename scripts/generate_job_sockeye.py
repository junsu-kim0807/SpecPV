#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "longbenchv1_summarization"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "longbenchv1_summarization"

# =========================
# Sockeye-specific paths
# =========================
ROOT_DIR = "/scratch/st-prashnr-1/junsu/specdec/SpecPV"
REPO_DIR = "/home/junsuk87/scratch/specdec/SpecPV"
VENV_DIR = "/home/junsuk87/scratch/venvs/vllm"

HF_HOME = "/scratch/st-prashnr-1/junsu/hf_cache"
UV_CACHE_DIR = "/home/junsuk87/scratch/.uv-cache"
PIP_CACHE_DIR = "/home/junsuk87/scratch/.pip-cache"

# 모델이 저장된 루트 경로만 여기서 바꾸면 된다.
MODEL_ROOT = "/scratch/st-prashnr-1/junsu/models"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    path: str
    gpu_count: int = 4
    cpus_per_task: int = 12
    mem: str = "64G"
    time_limit: str = "08:00:00"


@dataclass(frozen=True)
class DatasetConfig:
    name: str


@dataclass(frozen=True)
class MethodConfig:
    name: str
    extra_args: tuple[str, ...] = ()


# =========================
# Edit these lists as needed
# =========================
TARGET_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="llama31_8b_instruct",
        path=f"{MODEL_ROOT}/LLAMA3.1-8B-Instruct",
        gpu_count=4,
        cpus_per_task=12,
        mem="64G",
        time_limit="08:00:00",
    ),
]

DRAFT_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="eagle3_llama31_8b_yarn_64k",
        path=f"{MODEL_ROOT}/eagle/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K",
        gpu_count=4,
        cpus_per_task=12,
        mem="64G",
        time_limit="03:00:00",
    ),
]

DATASETS: list[DatasetConfig] = [
    DatasetConfig(name="qmsum"),
    DatasetConfig(name="gov_report"),
]

METHODS: list[MethodConfig] = [
    MethodConfig(name="full"),
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


def filter_by_attr(items: list, wanted: set[str], attr: str) -> list:
    if not wanted:
        return items
    return [item for item in items if getattr(item, attr) in wanted]


def job_name(method: MethodConfig, dataset: DatasetConfig, draft: ModelConfig, target: ModelConfig) -> str:
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

    # target 기준으로 자원 할당
    gpu_count = max(draft.gpu_count, target.gpu_count)
    cpus_per_task = max(draft.cpus_per_task, target.cpus_per_task)
    mem = target.mem
    time_limit = target.time_limit

    jname = job_name(method, dataset, draft, target)

    return f"""#!/bin/bash
#SBATCH --job-name={jname}
#SBATCH --nodes=1
#SBATCH --account=st-prashnr-1-gpu
#SBATCH --gpus={gpu_count}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --time={time_limit}
#SBATCH --output={log_dir / "%x_%j.out"}
#SBATCH --error={log_dir / "%x_%j.err"}

set -euo pipefail

ROOT_DIR={shquote(ROOT_DIR)}
REPO_DIR={shquote(REPO_DIR)}
VENV_DIR={shquote(VENV_DIR)}

source "${{VENV_DIR}}/bin/activate"

export UV_CACHE_DIR={shquote(UV_CACHE_DIR)}
export PIP_CACHE_DIR={shquote(PIP_CACHE_DIR)}

cd "${{REPO_DIR}}"

unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

export HF_HOME={shquote(HF_HOME)}
export HF_HUB_CACHE="${{HF_HOME}}/hub"
export HF_XET_CACHE="${{HF_HOME}}/xet"
export TRANSFORMERS_CACHE="${{HF_HOME}}/transformers"
export HF_MODULES_CACHE="${{HF_HOME}}/modules"
export XDG_CACHE_HOME="${{HF_HOME}}"

mkdir -p \\
  "${{HF_HOME}}" \\
  "${{HF_HUB_CACHE}}" \\
  "${{HF_XET_CACHE}}" \\
  "${{TRANSFORMERS_CACHE}}" \\
  "${{HF_MODULES_CACHE}}" \\
  "${{ROOT_DIR}}/experiments/longbenchv1_summarization"
"""


def build_python_command(
    *,
    method: MethodConfig,
    dataset: DatasetConfig,
    draft: ModelConfig,
    target: ModelConfig,
) -> str:
    args = [
        "python evaluation/longbenchv1_summarization.py",
        f"--model_path {shquote(target.path)}",
        f"--draft_path {shquote(draft.path)}",
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
    log_dir = LOGS_ROOT / method.name / dataset.name / slug
    header = job_header(
        method=method,
        dataset=dataset,
        draft=draft,
        target=target,
        log_dir=log_dir,
    )

    run_dir = (
        f"{ROOT_DIR}/experiments/longbenchv1_summarization/"
        f"{method.name}/{dataset.name}/{slug}"
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

RUN_DIR={shquote(run_dir)}
mkdir -p "${{RUN_DIR}}"

echo "============================================================"
echo "JOB_NAME       : {job_name(method, dataset, draft, target)}"
echo "METHOD         : ${{METHOD_NAME}}"
echo "DATASET        : ${{DATASET_NAME}}"
echo "DRAFT_NAME     : ${{DRAFT_NAME}}"
echo "DRAFT_PATH     : {draft.path}"
echo "TARGET_NAME    : ${{TARGET_NAME}}"
echo "TARGET_PATH    : {target.path}"
echo "PAIR_SLUG      : ${{PAIR_SLUG}}"
echo "RUN_DIR        : ${{RUN_DIR}}"
echo "HOSTNAME       : $(hostname)"
echo "PWD            : $(pwd)"
echo "PYTHON         : $(which python)"
echo "CUDA_VISIBLE_DEVICES : ${{CUDA_VISIBLE_DEVICES:-unset_by_shell}}"
echo "============================================================"

env | sort > "${{RUN_DIR}}/env.txt"
cp "$0" "${{RUN_DIR}}/job.slurm"

{cmd} 2>&1 | tee "${{RUN_DIR}}/run.log"
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