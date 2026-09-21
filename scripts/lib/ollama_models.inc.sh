# shellcheck shell=bash
# Ollama model lineup for CLOSE array jobs.
# Prefer already-cached, fast single-GPU tags by default.

# Cached & fast defaults (override before sbatch if needed):
CLOSE_OLLAMA_QWEN="${CLOSE_OLLAMA_QWEN:-qwen3.6:35b}"
CLOSE_OLLAMA_LLAMA="${CLOSE_OLLAMA_LLAMA:-llama4:scout}"
# small3.2 is already local (~15GB). medium-3.5 is ~80GB and slow to pull.
CLOSE_OLLAMA_MISTRAL="${CLOSE_OLLAMA_MISTRAL:-mistral-small3.2:24b}"

close_configure_ollama_model() {
  local slot="$1"
  export CLOSE_ALLOW_RULE_FALLBACK="${CLOSE_ALLOW_RULE_FALLBACK:-0}"
  export CLOSE_LLM_RETRIES="${CLOSE_LLM_RETRIES:-3}"
  export CLOSE_LLM_REPORT_CHARS="${CLOSE_LLM_REPORT_CHARS:-12000}"
  export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
  export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-8192}"

  case "$slot" in
    qwen)
      OLLAMA_MODEL="$CLOSE_OLLAMA_QWEN"
      export CLOSE_DISABLE_THINKING=1
      ;;
    llama)
      OLLAMA_MODEL="$CLOSE_OLLAMA_LLAMA"
      unset CLOSE_DISABLE_THINKING || true
      ;;
    mistral)
      OLLAMA_MODEL="$CLOSE_OLLAMA_MISTRAL"
      unset CLOSE_DISABLE_THINKING || true
      ;;
    *)
      echo "Unknown Ollama model slot: $slot" >&2
      return 1
      ;;
  esac
}
