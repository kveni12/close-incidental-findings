#!/bin/bash
#SBATCH --job-name=close_vllm
#SBATCH --partition=dgx-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/close_vllm_%j.out

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
OUTPUT_DIR="${OUTPUT_DIR:-outputs/vllm_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-configs/capture_vllm.yaml}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_COUNT="${GPU_COUNT:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-logs}"
VLLM_ARGS="${VLLM_ARGS:-}"
MODEL_DOWNLOAD_DIR="${MODEL_DOWNLOAD_DIR:-}"
LIMIT="${LIMIT:-}"
PYENV_VERSION="${PYENV_VERSION:-xray-drift-nlp-new}"
export PYENV_VERSION

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES
export CLOSE_INFERENCE_MODE=LOCAL_VLLM
export CLOSE_LLM_MODEL="${MODEL_NAME}"
export CLOSE_LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export CLOSE_LLM_API_KEY=EMPTY
export CLOSE_LLM_BATCH_SIZE="${BATCH_SIZE}"
export VLLM_USE_RAY="${VLLM_USE_RAY:-False}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# shellcheck source=scripts/lib/setup_python_env.sh
source "${REPO_ROOT}/scripts/lib/setup_python_env.sh"
setup_python_env || exit 1

PYTHON="$(command -v python3 2>/dev/null || command -v python)"

check_vllm() {
  "${PYTHON}" -m vllm.entrypoints.openai.api_server --help >/dev/null 2>&1 || {
    echo "vLLM is not available in this Python environment (${PYTHON})."
    echo "Set PYENV_VERSION=xray-drift-nlp-new (cluster default) or PYTHON_BIN=/path/to/python"
    exit 1
  }
}

check_vllm

count_visible_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '
    return 0
  fi
  echo "0"
}

check_gpu_count() {
  local visible
  visible="$(count_visible_gpus)"
  if [[ "${visible}" -gt 0 && "${GPU_COUNT}" -gt "${visible}" ]]; then
    echo "ERROR: GPU_COUNT=${GPU_COUNT} but only ${visible} GPU(s) visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})."
    echo "Request more GPUs in sbatch (--gres=gpu:${GPU_COUNT}) or lower GPU_COUNT."
    exit 1
  fi
  echo "tensor_parallel_size=${GPU_COUNT} (${visible:-unknown} GPU(s) visible)"
}

check_gpu_count

resolve_download_dir() {
  local dir="${1:-}"
  if [[ -z "${dir}" ]]; then
    return 1
  fi
  case "${dir}" in
    /path/*|/base/path/*|*to/your/models*|*placeholder*)
      echo "Ignoring placeholder MODEL_DOWNLOAD_DIR=${dir}" >&2
      return 1
      ;;
  esac
  if [[ ! -d "${dir}" ]]; then
    echo "MODEL_DOWNLOAD_DIR does not exist: ${dir}" >&2
    return 1
  fi
  if [[ ! -r "${dir}" ]]; then
    echo "MODEL_DOWNLOAD_DIR is not readable: ${dir}" >&2
    return 1
  fi
  echo "${dir}"
  return 0
}

VLLM_LOG="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
DOWNLOAD_ARGS=()
if resolved_dir="$(resolve_download_dir "${MODEL_DOWNLOAD_DIR}")"; then
  DOWNLOAD_ARGS+=(--download-dir "${resolved_dir}")
  echo "Using model download dir: ${resolved_dir}"
elif [[ -n "${MODEL_DOWNLOAD_DIR}" ]]; then
  echo "Proceeding without --download-dir (HF will use default cache)."
fi

if [[ -z "${VLLM_STARTUP_TIMEOUT:-}" ]]; then
  if [[ "${MODEL_NAME}" == *"72B"* || "${MODEL_NAME}" == *"70B"* ]]; then
    VLLM_STARTUP_TIMEOUT=1800
  else
    VLLM_STARTUP_TIMEOUT=600
  fi
fi

wait_for_vllm() {
  echo "Waiting for vLLM on port ${PORT} (timeout ${VLLM_STARTUP_TIMEOUT}s)..."
  timeout "${VLLM_STARTUP_TIMEOUT}" bash -c "until grep -q 'Uvicorn running on' '${VLLM_LOG}'; do sleep 1; done"
}

"${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size "${GPU_COUNT}" \
  --port "${PORT}" \
  "${DOWNLOAD_ARGS[@]}" \
  ${VLLM_ARGS} \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
sleep "${VLLM_STARTUP_SECONDS:-30}"
if ! wait_for_vllm; then
  echo "vLLM did not start properly. Last log lines:"
  tail -n 80 "${VLLM_LOG}" || true
  exit 1
fi
echo "vLLM started successfully; running CLOSE pipeline."

ARGS=(--config "${CONFIG}" --data-dir "${DATA_DIR}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${LIMIT}" ]]; then
  ARGS+=(--limit "${LIMIT}")
fi

"${PYTHON}" "${REPO_ROOT}/scripts/run_close_pipeline.py" "${ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/close_vllm_$(date +%Y%m%d_%H%M%S).log"
