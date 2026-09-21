#!/bin/bash
# Thin wrapper: submit the CLOSE multi-model Ollama array job.
# Prefer submitting the job script directly:
#   LIMIT=100 sbatch scripts/run_close_all_models_parallel.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec sbatch "$@" "${ROOT}/scripts/run_close_all_models_parallel.sh"
