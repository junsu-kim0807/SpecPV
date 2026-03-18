#!/bin/bash
#SBATCH --job-name=lb_naive_qmsum_eagle3_llama31_8b_yarn_64k_to_llama31_8b_instruct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=80G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/jhwoo36/specdec/SpecPV/scripts/logs/longbenchv1_summarization/naive/qmsum/eagle3_llama31_8b_yarn_64k__TO__llama31_8b_instruct/lb_naive_qmsum_eagle3_llama31_8b_yarn_64k_to_llama31_8b_instruct.out
#SBATCH --error=/scratch/jhwoo36/specdec/SpecPV/scripts/logs/longbenchv1_summarization/naive/qmsum/eagle3_llama31_8b_yarn_64k__TO__llama31_8b_instruct/lb_naive_qmsum_eagle3_llama31_8b_yarn_64k_to_llama31_8b_instruct.err

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR='/home/jhwoo36/scratch/specdec/SpecPV'
VENV_DIR='/home/jhwoo36/scratch/venvs/vllm'

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

export HF_TOKEN='$HUGGINGFACE_HUB_TOKEN'

export HF_HOME='/home/jhwoo36/scratch/.cache'
export TRANSFORMERS_CACHE='/home/jhwoo36/scratch/.cache/transformers'
export HUGGINGFACE_HUB_CACHE='/home/jhwoo36/scratch/.cache/huggingface_hub'
export HF_HUB_CACHE='/home/jhwoo36/scratch/.cache/huggingface_hub'
export HF_DATASETS_CACHE='/home/jhwoo36/scratch/.cache/datasets'

mkdir -p \
  "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "/scratch/jhwoo36/specdec/SpecPV/results/longbenchv1_summarization"

METHOD_NAME='naive'
DATASET_NAME='qmsum'
DRAFT_NAME='eagle3_llama31_8b_yarn_64k'
TARGET_NAME='llama31_8b_instruct'
PAIR_SLUG='eagle3_llama31_8b_yarn_64k__TO__llama31_8b_instruct'
JOB_NAME_STR='lb_naive_qmsum_eagle3_llama31_8b_yarn_64k_to_llama31_8b_instruct'

DRAFT_MODEL_REF='TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K'
TARGET_MODEL_REF='meta-llama/Llama-3.1-8B-Instruct'

RESULT_DIR='/scratch/jhwoo36/specdec/SpecPV/results/longbenchv1_summarization/naive/qmsum/eagle3_llama31_8b_yarn_64k__TO__llama31_8b_instruct'
mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo "JOB_NAME            : ${JOB_NAME_STR}"
echo "METHOD              : ${METHOD_NAME}"
echo "DATASET             : ${DATASET_NAME}"
echo "DRAFT_NAME          : ${DRAFT_NAME}"
echo "DRAFT_MODEL_REF     : ${DRAFT_MODEL_REF}"
echo "TARGET_NAME         : ${TARGET_NAME}"
echo "TARGET_MODEL_REF    : ${TARGET_MODEL_REF}"
echo "PAIR_SLUG           : ${PAIR_SLUG}"
echo "RESULT_DIR          : ${RESULT_DIR}"
echo "HF_HOME             : ${HF_HOME}"
echo "HF_HUB_CACHE        : ${HF_HUB_CACHE}"
echo "TRANSFORMERS_CACHE  : ${TRANSFORMERS_CACHE}"
echo "HF_DATASETS_CACHE   : ${HF_DATASETS_CACHE}"
echo "HOSTNAME            : $(hostname)"
echo "PWD                 : $(pwd)"
echo "PYTHON              : $(which python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset_by_shell}"
echo "============================================================"

env | sort > "${RESULT_DIR}/env.txt"
cp "$0" "${RESULT_DIR}/job.slurm"

python -u evaluation/longbenchv1_summarization.py \
  --model_path 'meta-llama/Llama-3.1-8B-Instruct' \
  --draft_path 'TanBaby/EAGLE3-LLaMA3.1-Instruct-8B-YARN-64K' \
  --dataset_name 'qmsum' \
  --method 'naive' 2>&1 | tee "${RESULT_DIR}/run.log"

echo "Saved outputs under: ${RESULT_DIR}"
