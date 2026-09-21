#!/bin/bash
#SBATCH -J close_hybrid
#SBATCH -p qtim
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --array=0-2
#SBATCH -o logs/close_hybrid_%A_%a.out

# Hybrid CLOSE: ACR recalibration + Link/Orchestrate rerun (no LLM / no GPU).
#
# Uses existing LLM Capture JSONL outputs, then rebuilds full end-to-end results
# with improved Capture agreement and updated downstream agents.
#
# Submit:
#   sbatch scripts/run_hybrid_downstream_slurm.sh
#
# Optional overrides:
#   LLAMA_INPUT=... QWEN_INPUT=... MISTRAL_INPUT=... sbatch ...
set -euo pipefail

CLOSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLOSE_ROOT"

# shellcheck source=scripts/lib/setup_python_env.sh
source "${CLOSE_ROOT}/scripts/lib/setup_python_env.sh"
export PYENV_VERSION="${PYENV_VERSION:-xray-drift-nlp-new}"
setup_python_env || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CLOSE_ROOT}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-${CLOSE_ROOT}/data}"
LOG_DIR="${CLOSE_ROOT}/logs"
mkdir -p "$LOG_DIR"

LLAMA_INPUT="${LLAMA_INPUT:-${CLOSE_ROOT}/outputs/model_comparison_9283993/llama/close_results.jsonl}"
QWEN_INPUT="${QWEN_INPUT:-${CLOSE_ROOT}/outputs/model_comparison_9500762/qwen/close_results.jsonl}"
MISTRAL_INPUT="${MISTRAL_INPUT:-${CLOSE_ROOT}/outputs/model_comparison_9500762/mistral/close_results.jsonl}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL_SLOT="qwen"
    INPUT_PATH="$QWEN_INPUT"
    ;;
  1)
    MODEL_SLOT="llama"
    INPUT_PATH="$LLAMA_INPUT"
    ;;
  2)
    MODEL_SLOT="mistral"
    INPUT_PATH="$MISTRAL_INPUT"
    ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Missing input JSONL for ${MODEL_SLOT}: ${INPUT_PATH}"
  exit 1
fi

OUTPUT_PATH="$(dirname "$INPUT_PATH")/close_results_hybrid.jsonl"
PIPELINE_LOG="${LOG_DIR}/hybrid_${MODEL_SLOT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

echo "Hybrid downstream for ${MODEL_SLOT}"
echo "  Input:  ${INPUT_PATH}"
echo "  Output: ${OUTPUT_PATH}"

python "${CLOSE_ROOT}/scripts/run_hybrid_downstream.py" \
  --input "${INPUT_PATH}" \
  --output "${OUTPUT_PATH}" \
  --data-dir "${DATA_DIR}" \
  2>&1 | tee "${PIPELINE_LOG}"

echo "============================================================"
echo "Hybrid CLOSE complete for ${MODEL_SLOT}"
echo "  Output: ${OUTPUT_PATH}"
echo "  Log:    ${PIPELINE_LOG}"
echo "============================================================"
