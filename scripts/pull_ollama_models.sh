#!/usr/bin/env bash
# Prefetch Ollama models into the shared CLOSE model store (run once before array jobs).
# Usage:
#   bash scripts/pull_ollama_models.sh
#   CLOSE_OLLAMA_QWEN=qwen3.6:27b bash scripts/pull_ollama_models.sh
set -euo pipefail

CLOSE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/ollama.inc.sh
source "${CLOSE_ROOT}/scripts/lib/ollama.inc.sh"
# shellcheck source=scripts/lib/ollama_models.inc.sh
source "${CLOSE_ROOT}/scripts/lib/ollama_models.inc.sh"

ensure_ollama_binary || exit 1
mkdir -p "$OLLAMA_MODELS"
export OLLAMA_MODELS

# Start a temporary serve if needed so `pull` can talk to a daemon.
NEED_STOP=0
PID=""
PORT="${OLLAMA_PULL_PORT:-11434}"
if ! curl -sf "http://127.0.0.1:${PORT}/api/tags" >/dev/null 2>&1; then
  LOG="${CLOSE_ROOT}/logs/ollama_pull_serve.log"
  mkdir -p "${CLOSE_ROOT}/logs"
  PID="$(start_ollama_serve "$PORT" "$LOG")"
  NEED_STOP=1
  wait_for_ollama_api "$PORT" "$PID" "$LOG"
fi
export OLLAMA_HOST="127.0.0.1:${PORT}"

for model in "$CLOSE_OLLAMA_QWEN" "$CLOSE_OLLAMA_LLAMA" "$CLOSE_OLLAMA_MISTRAL"; do
  ollama_pull_model "$model"
done

if [[ "$NEED_STOP" == "1" && -n "$PID" ]]; then
  kill "$PID" 2>/dev/null || true
fi

echo "All requested Ollama models are present under ${OLLAMA_MODELS}"
"$OLLAMA_BIN" list || true
