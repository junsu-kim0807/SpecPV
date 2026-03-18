#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    from datasets import load_dataset
except ImportError as e:
    print("Missing dependency:", e, file=sys.stderr)
    print(
        'Install first: pip install -U "huggingface_hub[cli]" datasets',
        file=sys.stderr,
    )
    sys.exit(1)


# ============================================================
# Sockeye-specific paths
# ============================================================
ROOT_DIR = Path("/scratch/st-prashnr-1/junsu/specdec/SpecPV")
MODEL_ROOT = Path("/scratch/st-prashnr-1/junsu/models")
HF_HOME = Path("/scratch/st-prashnr-1/junsu/hf_cache")
DATASET_ROOT = ROOT_DIR / "datasets"

# Optional: if you want to use another venv or cache root later, edit above only.


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo_id: str
    repo_type: str = "model"


@dataclass(frozen=True)
class DatasetSpec:
    alias: str
    repo_id: str
    config_name: str | None = None
    split: str | None = None
    repo_type: str = "dataset"


MODEL_SPECS: dict[str, ModelSpec] = {
    "llama32_1b_instruct": ModelSpec(
        alias="Llama-3.2-1B-Instruct",
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
    ),
    "llama32_3b_instruct": ModelSpec(
        alias="Llama-3.2-3B-Instruct",
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
    ),
    "llama31_8b_instruct": ModelSpec(
        alias="Llama-3.1-8B-Instruct",
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
    ),
    "llama31_70b_instruct": ModelSpec(
        alias="Llama-3.1-70B-Instruct",
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
    ),
    "deepseek_coder_1p3b_instruct": ModelSpec(
        alias="deepseek-coder-1.3b-instruct",
        repo_id="deepseek-ai/deepseek-coder-1.3b-instruct",
    ),
    "deepseek_coder_6p7b_instruct": ModelSpec(
        alias="deepseek-coder-6.7b-instruct",
        repo_id="deepseek-ai/deepseek-coder-6.7b-instruct",
    ),
    "deepseek_coder_33b_instruct": ModelSpec(
        alias="deepseek-coder-33b-instruct",
        repo_id="deepseek-ai/deepseek-coder-33b-instruct",
    ),
    "qwen3_0p6b": ModelSpec(
        alias="Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
    ),
    "qwen3_4b": ModelSpec(
        alias="Qwen3-4B",
        repo_id="Qwen/Qwen3-4B",
    ),
    "qwen3_8b": ModelSpec(
        alias="Qwen3-8B",
        repo_id="Qwen/Qwen3-8B",
    ),
    "qwen3_30b_a3b": ModelSpec(
        alias="Qwen3-30B-A3B",
        repo_id="Qwen/Qwen3-30B-A3B",
    ),
    # 현재 네 예시 longbench job이 이것도 필요하면 주석 해제해서 쓰면 된다.
    # "eagle3_llama31_8b_yarn_64k": ModelSpec(
    #     alias="EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K",
    #     repo_id="TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K",
    # ),
}

DATASET_SPECS: dict[str, DatasetSpec] = {
    "aime2025": DatasetSpec(
        alias="aime2025",
        repo_id="opencompass/AIME2025",
        config_name=None,
        split=None,
    ),
    "codeelo": DatasetSpec(
        alias="codeelo",
        repo_id="Qwen/CodeElo",
        config_name=None,
        split=None,
    ),
    "longbench_v2": DatasetSpec(
        alias="longbench_v2",
        repo_id="zai-org/LongBench-v2",
        config_name=None,
        split=None,
    ),
    "longbench_v1_govreport": DatasetSpec(
        alias="longbench_v1_govreport",
        repo_id="zai-org/LongBench",
        config_name="gov_report",
        split=None,
    ),
    # 네 generator 예시에서 qmsum도 쓸 거면 이것도 같이 prefetch하면 된다.
    # "longbench_v1_qmsum": DatasetSpec(
    #     alias="longbench_v1_qmsum",
    #     repo_id="zai-org/LongBench",
    #     config_name="qmsum",
    #     split=None,
    # ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prefetch models and datasets from Hugging Face on a login node."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Subset of model keys to download. Default: all models.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Subset of dataset keys to download. Default: all datasets.",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Do not download any models.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Do not download any datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload / overwrite local dataset save_to_disk copies if they exist.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="HF token. Defaults to HF_TOKEN env var.",
    )
    return parser.parse_args()


def ensure_env() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
    os.environ["HF_XET_CACHE"] = str(HF_HOME / "xet")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
    os.environ["HF_MODULES_CACHE"] = str(HF_HOME / "modules")
    os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
    os.environ["XDG_CACHE_HOME"] = str(HF_HOME)

    for p in [
        ROOT_DIR,
        MODEL_ROOT,
        HF_HOME,
        HF_HOME / "hub",
        HF_HOME / "xet",
        HF_HOME / "transformers",
        HF_HOME / "modules",
        HF_HOME / "datasets",
        DATASET_ROOT,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def select_items[T](all_items: dict[str, T], keys: list[str]) -> dict[str, T]:
    if not keys:
        return all_items
    out: dict[str, T] = {}
    for k in keys:
        if k not in all_items:
            valid = ", ".join(sorted(all_items.keys()))
            raise SystemExit(f"Unknown key: {k}\nValid keys: {valid}")
        out[k] = all_items[k]
    return out


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def download_model(spec: ModelSpec, hf_token: str) -> dict[str, Any]:
    dst = MODEL_ROOT / spec.alias
    dst.mkdir(parents=True, exist_ok=True)

    print_header(f"MODEL: {spec.alias}")
    print(f"repo_id      : {spec.repo_id}")
    print(f"destination  : {dst}")

    kwargs: dict[str, Any] = {
        "repo_id": spec.repo_id,
        "repo_type": spec.repo_type,
        "local_dir": str(dst),
        "max_workers": 8,
    }
    if hf_token:
        kwargs["token"] = hf_token

    snapshot_download(**kwargs)

    return {
        "kind": "model",
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "local_path": str(dst),
        "status": "ok",
    }


def _load_dataset_with_fallback(spec: DatasetSpec):
    """
    Try a few loading patterns because HF datasets can differ in whether they
    prefer split=None or an explicit split.
    """
    attempts: list[dict[str, Any]] = []

    # Preferred path
    attempts.append(
        {
            "path": spec.repo_id,
            "name": spec.config_name,
            "split": spec.split,
            "cache_dir": str(HF_HOME / "datasets"),
            "trust_remote_code": True,
        }
    )

    # Fallbacks
    if spec.split is None:
        attempts.append(
            {
                "path": spec.repo_id,
                "name": spec.config_name,
                "split": "test",
                "cache_dir": str(HF_HOME / "datasets"),
                "trust_remote_code": True,
            }
        )
        attempts.append(
            {
                "path": spec.repo_id,
                "name": spec.config_name,
                "split": "train",
                "cache_dir": str(HF_HOME / "datasets"),
                "trust_remote_code": True,
            }
        )

    last_err = None
    for kwargs in attempts:
        try:
            clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            print("load_dataset kwargs:", clean_kwargs)
            ds = load_dataset(**clean_kwargs)
            return ds, clean_kwargs
        except Exception as e:
            last_err = e
            print(f"Load failed with kwargs={kwargs}: {e}", file=sys.stderr)

    raise last_err  # type: ignore[misc]


def download_dataset(spec: DatasetSpec, force: bool) -> dict[str, Any]:
    dst = DATASET_ROOT / spec.alias

    print_header(f"DATASET: {spec.alias}")
    print(f"repo_id      : {spec.repo_id}")
    print(f"config_name  : {spec.config_name}")
    print(f"split        : {spec.split}")
    print(f"cache_root   : {HF_HOME / 'datasets'}")
    print(f"save_to_disk : {dst}")

    ds, used_kwargs = _load_dataset_with_fallback(spec)

    if dst.exists() and force:
        if dst.is_dir():
            for child in sorted(dst.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
            for child in sorted(dst.rglob("*"), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            try:
                dst.rmdir()
            except OSError:
                pass

    if not dst.exists():
        ds.save_to_disk(str(dst))

    split_info: str
    if hasattr(ds, "keys"):
        split_info = ",".join(list(ds.keys()))
    else:
        split_info = used_kwargs.get("split", "unknown")

    return {
        "kind": "dataset",
        "alias": spec.alias,
        "repo_id": spec.repo_id,
        "config_name": spec.config_name,
        "local_path": str(dst),
        "status": "ok",
        "loaded_with": used_kwargs,
        "splits": split_info,
    }


def main() -> None:
    args = parse_args()
    ensure_env()

    selected_models = select_items(MODEL_SPECS, args.models)
    selected_datasets = select_items(DATASET_SPECS, args.datasets)

    manifest: list[dict[str, Any]] = []

    print_header("ENVIRONMENT")
    print(f"ROOT_DIR          : {ROOT_DIR}")
    print(f"MODEL_ROOT        : {MODEL_ROOT}")
    print(f"HF_HOME           : {HF_HOME}")
    print(f"HF_DATASETS_CACHE : {HF_HOME / 'datasets'}")
    print(f"DATASET_ROOT      : {DATASET_ROOT}")
    print(f"HF_TOKEN set      : {'yes' if bool(args.hf_token) else 'no'}")

    if not args.skip_models:
        for key, spec in selected_models.items():
            try:
                result = download_model(spec, args.hf_token)
                result["key"] = key
                manifest.append(result)
            except GatedRepoError as e:
                manifest.append(
                    {
                        "kind": "model",
                        "key": key,
                        "alias": spec.alias,
                        "repo_id": spec.repo_id,
                        "status": "failed",
                        "error": f"Gated repo access denied: {e}",
                    }
                )
                print(
                    f"[FAILED] gated repo access for {spec.repo_id}. "
                    f"Check HF_TOKEN and license approval.",
                    file=sys.stderr,
                )
            except (RepositoryNotFoundError, HfHubHTTPError, Exception) as e:
                manifest.append(
                    {
                        "kind": "model",
                        "key": key,
                        "alias": spec.alias,
                        "repo_id": spec.repo_id,
                        "status": "failed",
                        "error": repr(e),
                    }
                )
                print(f"[FAILED] model {spec.repo_id}: {e}", file=sys.stderr)
                traceback.print_exc()

    if not args.skip_datasets:
        for key, spec in selected_datasets.items():
            try:
                result = download_dataset(spec, args.force)
                result["key"] = key
                manifest.append(result)
            except Exception as e:
                manifest.append(
                    {
                        "kind": "dataset",
                        "key": key,
                        "alias": spec.alias,
                        "repo_id": spec.repo_id,
                        "status": "failed",
                        "error": repr(e),
                    }
                )
                print(f"[FAILED] dataset {spec.repo_id}: {e}", file=sys.stderr)
                traceback.print_exc()

    manifest_path = ROOT_DIR / "prefetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_header("DONE")
    print(f"Manifest written to: {manifest_path}")
    print("Summary:")
    ok = 0
    fail = 0
    for item in manifest:
        status = item["status"]
        if status == "ok":
            ok += 1
        else:
            fail += 1
        print(f"  [{status.upper():6}] {item['kind']:7} {item['key']} -> {item['repo_id']}")
    print(f"ok={ok}, failed={fail}")

    if fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()