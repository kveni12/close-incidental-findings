#!/bin/bash
#SBATCH --job-name=close_capture
#SBATCH --partition=dgx-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/close_%j.out

set -euo pipefail

find_repo_root() {
  local candidate
  for candidate in "${REPO_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    while [[ -n "${candidate}" && "${candidate}" != "/" ]]; do
      if [[ -f "${candidate}/scripts/run_close_pipeline.py" ]]; then
        echo "${candidate}"
        return 0
      fi
      candidate="$(dirname "${candidate}")"
    done
  done
  return 1
}

REPO_ROOT="$(find_repo_root)" || { echo "Could not locate CLOSE repo root. Submit from the repo or set REPO_ROOT."; exit 1; }
cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs" logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export GPU_COUNT="${GPU_COUNT:-1}"
export DATA_DIR="${DATA_DIR:-data}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/slurm_${SLURM_JOB_ID:-manual}}"
export CONFIG="${CONFIG:-configs/capture_vllm.yaml}"

if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

bash "${REPO_ROOT}/scripts/run_close_gpu.sh"
