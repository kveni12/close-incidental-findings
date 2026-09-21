#!/bin/bash
#SBATCH -J close_ollama_models
#SBATCH -p qtim
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH -t 7-00:00:00
#SBATCH --array=0-2
#SBATCH -o logs/close_ollama_%A_%a.out

# Three Ollama models in parallel via SLURM array (one GPU per task).
#   0=qwen     (qwen3.6:35b)           — uses native /api/chat + think=false
#   1=llama    (llama4:scout)
#   2=mistral  (mistral-small3.2:24b)  — cached; no 80GB pull
#
# Submit from a Slurm login host:
#   LIMIT=1000 sbatch scripts/run_close_all_models_parallel.sh
#   LIMIT=1000 sbatch --array=0,2 scripts/run_close_all_models_parallel.sh   # qwen+mistral only
set -euo pipefail

CLOSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLOSE_ROOT"

# =================== ENV ===================
# Pipeline Python only (Ollama is a separate binary; no vLLM needed).
# shellcheck source=scripts/lib/setup_python_env.sh
source "${CLOSE_ROOT}/scripts/lib/setup_python_env.sh"
export PYENV_VERSION="${PYENV_VERSION:-xray-drift-nlp-new}"
setup_python_env || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CLOSE_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Thorough LLM-primary Capture (no keyword-rule contamination by default).
export CLOSE_ALLOW_RULE_FALLBACK="${CLOSE_ALLOW_RULE_FALLBACK:-0}"
export CLOSE_LLM_RETRIES="${CLOSE_LLM_RETRIES:-3}"
export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
export CLOSE_LLM_REPORT_CHARS="${CLOSE_LLM_REPORT_CHARS:-12000}"

# =================== PATHS ===================
DATA_DIR="${DATA_DIR:-${CLOSE_ROOT}/data}"
CONFIG="${CONFIG:-${CLOSE_ROOT}/configs/capture_ollama.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLOSE_ROOT}/outputs/model_comparison_${SLURM_ARRAY_JOB_ID:-manual}}"
LOG_DIR="${CLOSE_ROOT}/logs"
LIMIT="${LIMIT:-1000}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Ensure RAG index exists before pipeline runs.
RAG_INDEX="${CLOSE_ROOT}/data/guideline_rag/chunks.jsonl"
if [[ ! -f "$RAG_INDEX" ]]; then
  echo "Building white-paper RAG index..."
  python "${CLOSE_ROOT}/scripts/build_guideline_rag.py"
fi

# shellcheck source=scripts/lib/ollama.inc.sh
source "${CLOSE_ROOT}/scripts/lib/ollama.inc.sh"
# shellcheck source=scripts/lib/ollama_models.inc.sh
source "${CLOSE_ROOT}/scripts/lib/ollama_models.inc.sh"

PORT="$(ollama_assign_port "${SLURM_ARRAY_TASK_ID}")"
echo "Using Ollama port ${PORT} (array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID})"

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL_SLOT="qwen" ;;
  1) MODEL_SLOT="llama" ;;
  2) MODEL_SLOT="mistral" ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

close_configure_ollama_model "$MODEL_SLOT"

JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MODEL_OUT="${OUTPUT_DIR}/${MODEL_SLOT}"
OLLAMA_LOG="${LOG_DIR}/ollama_${MODEL_SLOT}_${JOB_TAG}.log"
PIPELINE_LOG="${LOG_DIR}/close_${MODEL_SLOT}_${JOB_TAG}.log"
mkdir -p "$MODEL_OUT"

OLLAMA_PID="$(start_ollama_serve "$PORT" "$OLLAMA_LOG")"
echo "Ollama PID: $OLLAMA_PID"

cleanup() {
  echo "Stopping Ollama (${MODEL_SLOT})..."
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_ollama_api "$PORT" "$OLLAMA_PID" "$OLLAMA_LOG"; then
  cleanup
  trap - EXIT
  exit 1
fi

# Pull inside the job if not already cached (shared OLLAMA_MODELS).
export OLLAMA_HOST="127.0.0.1:${PORT}"
ollama_pull_model "$OLLAMA_MODEL"

echo "Warming up ${MODEL_SLOT} (${OLLAMA_MODEL})..."
ollama_warmup "$PORT" "$OLLAMA_MODEL"

echo "Running CLOSE pipeline for ${MODEL_SLOT}..."
export CLOSE_INFERENCE_MODE=LOCAL_OLLAMA
export CLOSE_LLM_MODEL="$OLLAMA_MODEL"
export CLOSE_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export CLOSE_LLM_API_KEY="${CLOSE_LLM_API_KEY:-ollama}"
export CLOSE_LLM_BATCH_SIZE="${BATCH_SIZE:-8}"

ARGS=(--config "${CONFIG}" --data-dir "${DATA_DIR}" --output-dir "${MODEL_OUT}")
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python "${CLOSE_ROOT}/scripts/run_close_pipeline.py" "${ARGS[@]}" \
  2>&1 | tee "$PIPELINE_LOG"

echo "============================================================"
echo "CLOSE outputs for ${MODEL_SLOT} (job ${JOB_TAG}):"
echo "  Model:         ${OLLAMA_MODEL}"
echo "  Output dir:    ${MODEL_OUT}"
echo "  Pipeline log:  ${PIPELINE_LOG}"
echo "  Ollama log:    ${OLLAMA_LOG}"
echo "============================================================"
echo "${MODEL_SLOT} task done."
