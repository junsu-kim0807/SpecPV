#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

AIME25_REPO = "opencompass/AIME2025"
CODEELO_REPO = "Qwen/CodeElo"
LONGBENCH_V1_REPO = "THUDM/LongBench"
LONGBENCH_V2_REPO = "zai-org/LongBench-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        default=os.environ.get(
            "HF_DATASETS_CACHE",
            os.path.expanduser("~/scratch/.cache/datasets"),
        ),
        help="Export root. Default: HF_DATASETS_CACHE",
    )
    p.add_argument(
        "--longbench-v1-subsets",
        nargs="*",
        default=["gov_report", "qmsum", "multi_news"],
    )
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[saved] {out_path}  rows={len(rows)}")


def ds_to_rows(ds) -> list[dict[str, Any]]:
    return [dict(x) for x in ds]


def load_aime25():
    ds_i = load_dataset(AIME25_REPO, "AIME2025-I", split="test")
    ds_ii = load_dataset(AIME25_REPO, "AIME2025-II", split="test")
    return concatenate_datasets([ds_i, ds_ii])


def load_codeelo():
    last_err = None
    for split in ("test", "validation", "train"):
        try:
            return load_dataset(CODEELO_REPO, split=split)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load {CODEELO_REPO}: {last_err}") from last_err


def ensure_longbench_v1_unzipped(cache_root: Path) -> Path:
    zip_path = Path(
        hf_hub_download(
            repo_id=LONGBENCH_V1_REPO,
            repo_type="dataset",
            filename="data.zip",
            cache_dir=str(cache_root / "_hf_cache"),
        )
    )
    extract_root = cache_root / "_longbench_v1_raw"
    marker = extract_root / ".unzipped_ok"

    if marker.exists():
        return extract_root

    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    marker.write_text("ok\n", encoding="utf-8")
    return extract_root


def find_subset_jsonl(root: Path, subset: str) -> Path:
    candidates = [
        root / f"{subset}.jsonl",
        root / subset / "test.jsonl",
        root / subset / f"{subset}.jsonl",
        root / "data" / f"{subset}.jsonl",
        root / "data" / subset / "test.jsonl",
        root / "LongBench" / f"{subset}.jsonl",
        root / "LongBench" / subset / "test.jsonl",
    ]

    for p in candidates:
        if p.exists():
            return p

    all_matches = list(root.rglob(f"{subset}.jsonl"))
    if all_matches:
        return all_matches[0]

    raise FileNotFoundError(
        f"Could not find JSONL for subset={subset} under extracted LongBench root: {root}"
    )


def load_longbench_v1_subset(subset: str, cache_root: Path):
    extracted_root = ensure_longbench_v1_unzipped(cache_root)
    subset_file = find_subset_jsonl(extracted_root, subset)
    print(f"[longbench-v1] subset={subset} file={subset_file}")
    return load_dataset("json", data_files=str(subset_file))["train"]


def load_longbench_v2():
    return load_dataset(LONGBENCH_V2_REPO, split="train")


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "root": str(root),
        "exports": {},
    }

    for subset in args.longbench_v1_subsets:
        print(f"[download] LongBench v1 subset={subset}")
        ds = load_longbench_v1_subset(subset, root)
        out_path = root / "longbenchv1" / f"{subset}.jsonl"
        rows = ds_to_rows(ds)
        save_jsonl(rows, out_path)
        manifest["exports"][f"longbenchv1/{subset}"] = {
            "repo": LONGBENCH_V1_REPO,
            "rows": len(rows),
            "path": str(out_path),
        }

    print("[download] LongBench v2")
    ds_v2 = load_longbench_v2()
    out_path_v2 = root / "longbenchv2" / "longbench_v2.jsonl"
    rows_v2 = ds_to_rows(ds_v2)
    save_jsonl(rows_v2, out_path_v2)
    manifest["exports"]["longbenchv2"] = {
        "repo": LONGBENCH_V2_REPO,
        "split": "train",
        "rows": len(rows_v2),
        "path": str(out_path_v2),
    }

    print("[download] AIME2025")
    ds_aime = load_aime25()
    out_path_aime = root / "aime2025" / "aime2025.jsonl"
    rows_aime = ds_to_rows(ds_aime)
    save_jsonl(rows_aime, out_path_aime)
    manifest["exports"]["aime2025"] = {
        "repo": AIME25_REPO,
        "split": "test",
        "rows": len(rows_aime),
        "path": str(out_path_aime),
    }

    print("[download] CodeElo")
    ds_codeelo = load_codeelo()
    out_path_codeelo = root / "codeelo" / "codeelo.jsonl"
    rows_codeelo = ds_to_rows(ds_codeelo)
    save_jsonl(rows_codeelo, out_path_codeelo)
    manifest["exports"]["codeelo"] = {
        "repo": CODEELO_REPO,
        "rows": len(rows_codeelo),
        "path": str(out_path_codeelo),
    }

    write_manifest(root, manifest)
    print(f"\nAll dataset exports are under: {root}")


if __name__ == "__main__":
    main()