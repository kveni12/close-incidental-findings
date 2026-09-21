# shellcheck shell=bash
# Central vLLM model lineup for CLOSE array jobs.
# Override before sbatch, e.g. CLOSE_MODEL_QWEN=Qwen/Qwen2.5-7B-Instruct
#
# Llama-4 Scout FP8 OOMs on a single 96GB GPU for this workload; default to Llama-3.1-8B.
# To try Scout solo: CLOSE_MODEL_LLAMA=RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic LLAMA_MAX_MODEL_LEN=2048

MODEL_DIR_BASE="${MODEL_DIR_BASE:-/autofs/space/crater_001/projects/xray-drift-nlp/models}"
MODEL_DIR_QWEN="${MODEL_DIR_QWEN:-/autofs/space/crater_001/projects/xray-drift-nlp/qwen}"
MODEL_DIR_LLAMA="${MODEL_DIR_LLAMA:-/autofs/space/crater_001/projects/xray-drift-nlp/llama}"
MODEL_DIR_MEDGEMMA="${MODEL_DIR_MEDGEMMA:-/autofs/space/crater_001/projects/xray-drift-nlp/medgemma}"

CLOSE_MODEL_MEDGEMMA="${CLOSE_MODEL_MEDGEMMA:-google/medgemma-4b-it}"
CLOSE_MODEL_QWEN="${CLOSE_MODEL_QWEN:-Qwen/Qwen3.5-35B-A3B}"
CLOSE_MODEL_LLAMA="${CLOSE_MODEL_LLAMA:-meta-llama/Llama-3.1-8B-Instruct}"
CLOSE_MODEL_MISTRAL="${CLOSE_MODEL_MISTRAL:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"

_close_vllm_text_only_mm() {
  if [[ "${CLOSE_VLLM_HAS_LM_ONLY:-0}" == "1" ]]; then
    VLLM_MODEL_EXTRA_ARGS+=(--language-model-only)
  else
    VLLM_MODEL_EXTRA_ARGS+=(--limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0)
  fi
}

close_configure_model() {
  local slot="$1"
  VLLM_MODEL_EXTRA_ARGS=()
  # Thorough Capture defaults (overridable).
  export CLOSE_ALLOW_RULE_FALLBACK="${CLOSE_ALLOW_RULE_FALLBACK:-0}"
  export CLOSE_LLM_RETRIES="${CLOSE_LLM_RETRIES:-3}"
  export CLOSE_LLM_REPORT_CHARS="${CLOSE_LLM_REPORT_CHARS:-12000}"

  case "$slot" in
    medgemma)
      VLLM_MODEL="$CLOSE_MODEL_MEDGEMMA"
      MODEL_DIR="$MODEL_DIR_MEDGEMMA"
      DTYPE="${MEDGEMMA_DTYPE:-bfloat16}"
      MAX_MODEL_LEN="${MEDGEMMA_MAX_MODEL_LEN:-4096}"
      GPU_MEM_UTIL="${MEDGEMMA_GPU_MEM_UTIL:-0.60}"
      VLLM_MODEL_EXTRA_ARGS+=(--max-num-batched-tokens "${MEDGEMMA_MAX_NUM_BATCHED_TOKENS:-2048}")
      export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
      # Keep report + completion inside 4k context (was causing 400 Bad Request).
      export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-768}"
      export CLOSE_LLM_REPORT_CHARS="${CLOSE_LLM_REPORT_CHARS:-5500}"
      ;;
    qwen)
      VLLM_MODEL="$CLOSE_MODEL_QWEN"
      MODEL_DIR="$MODEL_DIR_QWEN"
      DTYPE="bfloat16"
      MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
      GPU_MEM_UTIL="${QWEN_GPU_MEM_UTIL:-0.90}"
      export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
      if [[ "$VLLM_MODEL" == *AWQ* ]] || [[ "$VLLM_MODEL" == *awq* ]]; then
        DTYPE="float16"
        VLLM_MODEL_EXTRA_ARGS+=(--quantization awq)
        GPU_MEM_UTIL="${QWEN_GPU_MEM_UTIL:-0.92}"
      fi
      if [[ "$VLLM_MODEL" == *FP8* ]] || [[ "$VLLM_MODEL" == *fp8* ]]; then
        VLLM_MODEL_EXTRA_ARGS+=(--quantization fp8)
      fi
      if [[ "$VLLM_MODEL" == Qwen/Qwen3.5* ]] || [[ "$VLLM_MODEL" == Qwen/Qwen3.6* ]]; then
        _close_vllm_text_only_mm
        export CLOSE_DISABLE_THINKING=1
      fi
      ;;
    llama)
      VLLM_MODEL="$CLOSE_MODEL_LLAMA"
      MODEL_DIR="$MODEL_DIR_LLAMA"
      DTYPE="bfloat16"
      if [[ "$VLLM_MODEL" == meta-llama/Llama-3.1* ]]; then
        MODEL_DIR="$MODEL_DIR_BASE"
        MAX_MODEL_LEN="${LLAMA_MAX_MODEL_LEN:-8192}"
        GPU_MEM_UTIL="${LLAMA_GPU_MEM_UTIL:-0.85}"
      else
        # Scout / larger Llama-4: keep context short to fit one 96GB GPU.
        MAX_MODEL_LEN="${LLAMA_MAX_MODEL_LEN:-2048}"
        GPU_MEM_UTIL="${LLAMA_GPU_MEM_UTIL:-0.92}"
      fi
      export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
      if [[ "$VLLM_MODEL" == *Llama-4* ]] || [[ "$VLLM_MODEL" == *llama-4* ]] || [[ "$VLLM_MODEL" == RedHatAI/Llama-4* ]]; then
        _close_vllm_text_only_mm
        VLLM_MODEL_EXTRA_ARGS+=(--trust-remote-code)
      fi
      if [[ "$VLLM_MODEL" == *NVFP4* ]]; then
        VLLM_MODEL_EXTRA_ARGS+=(--quantization nvfp4)
      elif [[ "$VLLM_MODEL" == RedHatAI/* ]]; then
        : # compressed-tensors baked into checkpoint; do not pass --quantization
      elif [[ "$VLLM_MODEL" == *FP8* ]] || [[ "$VLLM_MODEL" == *fp8* ]]; then
        VLLM_MODEL_EXTRA_ARGS+=(--quantization fp8)
      fi
      ;;
    mistral)
      VLLM_MODEL="$CLOSE_MODEL_MISTRAL"
      MODEL_DIR="$MODEL_DIR_BASE"
      DTYPE="bfloat16"
      MAX_MODEL_LEN="${MISTRAL_MAX_MODEL_LEN:-8192}"
      GPU_MEM_UTIL="${MISTRAL_GPU_MEM_UTIL:-0.85}"
      export CLOSE_LLM_MAX_TOKENS="${CLOSE_LLM_MAX_TOKENS:-1536}"
      if [[ "$VLLM_MODEL" == *Mistral-Small* ]] || [[ "$VLLM_MODEL" == *Mistral-Large* ]]; then
        _close_vllm_text_only_mm
      fi
      ;;
    *)
      echo "Unknown model slot: $slot" >&2
      return 1
      ;;
  esac
}
