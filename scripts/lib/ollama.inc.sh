# shellcheck shell=bash
# Ollama helpers for CLOSE Slurm jobs.
# Expects CLOSE_ROOT to be set.

OLLAMA_BIN="${OLLAMA_BIN:-${CLOSE_ROOT}/.tools/bin/ollama}"
OLLAMA_MODELS="${OLLAMA_MODELS:-${CLOSE_ROOT}/data/ollama_models}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

ollama_assign_port() {
  local task_offset="${1:-0}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
  echo $((29000 + (job_id % 7000) + task_offset))
}

ensure_ollama_binary() {
  if [[ -x "$OLLAMA_BIN" ]]; then
    return 0
  fi
  echo "Ollama binary missing at ${OLLAMA_BIN}" >&2
  echo "Run: bash ${CLOSE_ROOT}/scripts/install_ollama.sh" >&2
  return 1
}

start_ollama_serve() {
  local port="${1:?}"
  local log_path="${2:?}"

  ensure_ollama_binary || return 1
  mkdir -p "$OLLAMA_MODELS" "$(dirname "$log_path")"

  export OLLAMA_HOST="127.0.0.1:${port}"
  export OLLAMA_MODELS
  export OLLAMA_KEEP_ALIVE
  export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
  export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"
  # Prefer the Slurm-assigned GPU only.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
  fi

  echo "Starting Ollama serve on ${OLLAMA_HOST} (models=${OLLAMA_MODELS}, ctx=${OLLAMA_CONTEXT_LENGTH})" >&2
  "$OLLAMA_BIN" serve >"$log_path" 2>&1 &
  echo $!
}

wait_for_ollama_api() {
  local port="${1:?}"
  local pid="${2:?}"
  local log_path="${3:?}"
  local max_iters="${OLLAMA_WAIT_MAX_ITERS:-200}"
  local sleep_s="${OLLAMA_WAIT_SLEEP_S:-2}"

  echo "Waiting for Ollama on port ${port} (pid=${pid})..."
  local i=0
  while (( i < max_iters )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Ollama process exited before the API was ready."
      echo "---- tail ${log_path} ----"
      tail -n 80 "$log_path" 2>/dev/null || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:${port}/api/tags" >/dev/null \
      || curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      echo "Ollama API ready (after ~$((i * sleep_s))s)"
      return 0
    fi
    ((i++)) || true
    if (( i % 15 == 0 )); then
      echo "  ... still waiting (${i}/${max_iters})"
      tail -n 20 "$log_path" 2>/dev/null || true
    fi
    sleep "$sleep_s"
  done
  echo "Ollama API did not respond in time."
  tail -n 80 "$log_path" 2>/dev/null || true
  return 1
}

ollama_pull_model() {
  local model="${1:?}"
  ensure_ollama_binary || return 1
  export OLLAMA_MODELS
  echo "Ensuring Ollama model is present: ${model}"
  if "$OLLAMA_BIN" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fxq "$model"; then
    echo "Model already local: ${model}"
    return 0
  fi
  # Also match name without :latest alias differences.
  if "$OLLAMA_BIN" list 2>/dev/null | awk 'NR>1 {print $1}' | grep -Fq "$model"; then
    echo "Model already local (fuzzy): ${model}"
    return 0
  fi
  "$OLLAMA_BIN" pull "$model"
}

ollama_warmup() {
  local port="${1:?}"
  local model="${2:?}"
  curl -s "http://127.0.0.1:${port}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${model}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}],
      \"max_tokens\": 1,
      \"temperature\": 0
    }" >/dev/null || true
}
