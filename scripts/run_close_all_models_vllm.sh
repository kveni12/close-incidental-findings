#!/bin/bash
#SBATCH -J close_all_models_par
#SBATCH -p qtim
#SBATCH -A qtim
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --gpus=1
#SBATCH -t 7-00:00:00
#SBATCH --array=0-3
#SBATCH -o logs/close_all_models_vllm_%A_%a.out

# Legacy vLLM lineup (medgemma/qwen/llama/mistral). Prefer:
#   sbatch scripts/run_close_all_models_parallel.sh
# which uses Ollama + newer model tags.
# Submit:
#   LIMIT=100 sbatch scripts/run_close_all_models_vllm.sh
set -euo pipefail

CLOSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLOSE_ROOT"

# =================== ENV ===================
# Prefer dedicated vLLM>=0.17 venv (shared with ivc_filter); fall back to pyenv.
IVC_VLLM_VENV="${IVC_VLLM_VENV:-}"
if [[ -x "${IVC_VLLM_VENV}/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${IVC_VLLM_VENV}/bin/activate"
  unset PYENV_VERSION || true
  export PATH="${IVC_VLLM_VENV}/bin:${PATH}"
  export CLOSE_VLLM_HAS_LM_ONLY=1
  echo "[close] Using ${IVC_VLLM_VENV}"
else
  # shellcheck source=scripts/lib/setup_python_env.sh
  source "${CLOSE_ROOT}/scripts/lib/setup_python_env.sh"
  export PYENV_VERSION="${PYENV_VERSION:-xray-drift-nlp-new}"
  setup_python_env || exit 1
  export CLOSE_VLLM_HAS_LM_ONLY=0
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CLOSE_ROOT}/src:${PYTHONPATH:-}"
export VLLM_USE_RAY=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Thorough LLM-primary Capture (no keyword-rule contamination by default).
export CLOSE_ALLOW_RULE_FALLBACK="${CLOSE_ALLOW_RULE_FALLBACK:-0}"
export CLOSE_LLM_RETRIES="${CLOSE_LLM_RETRIES:-3}"
export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
export CLOSE_LLM_REPORT_CHARS="${CLOSE_LLM_REPORT_CHARS:-12000}"

# =================== PATHS ===================
DATA_DIR="${DATA_DIR:-${CLOSE_ROOT}/data}"
CONFIG="${CONFIG:-${CLOSE_ROOT}/configs/capture_vllm.yaml}"
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

# shellcheck source=scripts/lib/vllm_wait.inc.sh
source "${CLOSE_ROOT}/scripts/lib/vllm_wait.inc.sh"
# shellcheck source=scripts/lib/models.inc.sh
source "${CLOSE_ROOT}/scripts/lib/models.inc.sh"

PORT="$(vllm_assign_port "${SLURM_ARRAY_TASK_ID}")"
echo "Using vLLM port ${PORT} (array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID})"

case "${SLURM_ARRAY_TASK_ID}" in
  0) MODEL_SLOT="medgemma" ;;
  1) MODEL_SLOT="qwen" ;;
  2) MODEL_SLOT="llama" ;;
  3) MODEL_SLOT="mistral" ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
    exit 1
    ;;
esac

close_configure_model "$MODEL_SLOT"

JOB_TAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MODEL_OUT="${OUTPUT_DIR}/${MODEL_SLOT}"
VLLM_LOG="${LOG_DIR}/vllm_${MODEL_SLOT}_${JOB_TAG}.log"
PIPELINE_LOG="${LOG_DIR}/close_${MODEL_SLOT}_${JOB_TAG}.log"
mkdir -p "$MODEL_OUT"

echo "Launching vLLM for ${MODEL_SLOT} (${VLLM_MODEL}) on port ${PORT}..."
python -u -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL" \
  --download-dir "$MODEL_DIR" \
  --tensor-parallel-size 1 \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enforce-eager \
  --host 127.0.0.1 \
  --port "$PORT" \
  "${VLLM_MODEL_EXTRA_ARGS[@]}" \
  > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

cleanup() {
  echo "Stopping vLLM (${MODEL_SLOT})..."
  kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

if ! wait_for_vllm_api "$PORT" "$VLLM_PID" "$VLLM_LOG"; then
  cleanup
  trap - EXIT
  exit 1
fi

echo "Warming up ${MODEL_SLOT}..."
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"${VLLM_MODEL}\",
    \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],
    \"max_tokens\": 1
  }" >/dev/null || true

echo "Running CLOSE pipeline for ${MODEL_SLOT}..."
export CLOSE_INFERENCE_MODE=LOCAL_VLLM
export CLOSE_LLM_MODEL="$VLLM_MODEL"
export CLOSE_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export CLOSE_LLM_API_KEY="${CLOSE_LLM_API_KEY:-EMPTY}"
export CLOSE_LLM_BATCH_SIZE="${BATCH_SIZE:-8}"

ARGS=(--config "${CONFIG}" --data-dir "${DATA_DIR}" --output-dir "${MODEL_OUT}")
if [[ -n "${LIMIT}" && "${LIMIT}" != "0" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

python "${CLOSE_ROOT}/scripts/run_close_pipeline.py" "${ARGS[@]}" \
  2>&1 | tee "$PIPELINE_LOG"

echo "============================================================"
echo "CLOSE outputs for ${MODEL_SLOT} (job ${JOB_TAG}):"
echo "  Output dir:    ${MODEL_OUT}"
echo "  Pipeline log:  ${PIPELINE_LOG}"
echo "  vLLM log:      ${VLLM_LOG}"
echo "============================================================"
echo "${MODEL_SLOT} task done."
