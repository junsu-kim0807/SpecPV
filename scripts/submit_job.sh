#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS_ROOT="$REPO_ROOT/scripts/jobs/longbenchv1_summarization"

if [[ ! -d "$JOBS_ROOT" ]]; then
    echo "Jobs root not found: $JOBS_ROOT"
    exit 1
fi

declare -a METHODS=()

if [[ "$#" -eq 0 ]]; then
    while IFS= read -r -d '' method_dir; do
        METHODS+=("$(basename "$method_dir")")
    done < <(find "$JOBS_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
else
    METHODS=("$@")
fi

if [[ "${#METHODS[@]}" -eq 0 ]]; then
    echo "No methods found under: $JOBS_ROOT"
    exit 1
fi

total_submitted=0

for method in "${METHODS[@]}"; do
    method_dir="$JOBS_ROOT/$method"

    if [[ ! -d "$method_dir" ]]; then
        echo "Skipping unknown method: $method"
        continue
    fi

    echo "============================================================"
    echo "Submitting method: $method"
    echo "Method dir: $method_dir"
    echo "============================================================"

    submitted_for_method=0

    while IFS= read -r -d '' job_file; do
        echo "sbatch $job_file"
        sbatch "$job_file"
        submitted_for_method=$((submitted_for_method + 1))
        total_submitted=$((total_submitted + 1))
    done < <(find "$method_dir" -type f -name "*.slurm" -print0 | sort -z)

    echo "Submitted $submitted_for_method jobs for method=$method"
done

echo "============================================================"
echo "Total submitted jobs: $total_submitted"
echo "============================================================"