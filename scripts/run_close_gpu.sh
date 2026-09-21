#!/bin/bash
#SBATCH --job-name=close_gpu
#SBATCH --partition=dgx-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/close_gpu_%j.out

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

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-32B-Instruct}"
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gpu_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/capture_vllm.yaml}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_COUNT="${GPU_COUNT:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="${LOG_DIR:-logs}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES
export CLOSE_INFERENCE_MODE="${CLOSE_INFERENCE_MODE:-LOCAL_VLLM}"
export CLOSE_LLM_MODEL="${MODEL_NAME}"
export CLOSE_LLM_BATCH_SIZE="${BATCH_SIZE}"
export CLOSE_GPU_COUNT="${GPU_COUNT}"

if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

python - <<'PY'
import importlib.util
missing = [pkg for pkg in ("torch",) if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(f"Missing GPU dependencies: {missing}")
PY

python "${REPO_ROOT}/scripts/run_close_pipeline.py" \
  --config "${CONFIG}" \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${LOG_DIR}/close_gpu_$(date +%Y%m%d_%H%M%S).log"
