# shellcheck shell=bash
# Usage: vllm_assign_port [task_offset]
#        wait_for_vllm_api <port> <vllm_pid> <log_path>

vllm_assign_port() {
  local task_offset="${1:-0}"
  local job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$$}}"
  echo $((28000 + (job_id % 7000) + task_offset))
}

wait_for_vllm_api() {
  local p="${1:?}"
  local vllm_pid="${2:?}"
  local vllm_log="${3:?}"
  local max_iters="${VLLM_WAIT_MAX_ITERS:-1200}"
  local sleep_s="${VLLM_WAIT_SLEEP_S:-3}"
  local initial_sleep="${VLLM_WAIT_INITIAL_SLEEP_S:-8}"

  echo "Waiting for vLLM on port ${p} (pid=${vllm_pid}, up to ~$((max_iters * sleep_s / 60)) min)..."
  echo "  (${initial_sleep}s grace — avoid false-ready from another job on this port)"
  sleep "$initial_sleep"
  local i=0
  while (( i < max_iters )); do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM process exited before the API was ready."
      echo "---- tail ${vllm_log} ----"
      tail -n 150 "$vllm_log" 2>/dev/null || true
      return 1
    fi
    if curl -sf "http://127.0.0.1:${p}/v1/models" >/dev/null \
      && grep -qE "Application startup complete|Uvicorn running on" "$vllm_log" 2>/dev/null; then
      echo "vLLM API ready (after ~$((initial_sleep + i * sleep_s))s)"
      return 0
    fi
    ((i++)) || true
    if (( i % 40 == 0 )); then
      echo "  ... still loading (${i}/${max_iters} checks)"
      if [[ -n "${vllm_log}" && -f "${vllm_log}" ]]; then
        echo "---- tail -n 40 ${vllm_log} ----"
        tail -n 40 "$vllm_log" 2>/dev/null || true
        echo "---- end tail ----"
      fi
    fi
    sleep "$sleep_s"
  done
  echo "vLLM API did not respond in time."
  echo "---- tail ${vllm_log} ----"
  tail -n 150 "$vllm_log" 2>/dev/null || true
  return 1
}
