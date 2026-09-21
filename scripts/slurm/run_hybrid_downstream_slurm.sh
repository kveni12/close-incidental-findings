#!/bin/bash
# Thin wrapper: submit hybrid downstream (no LLM) array job.
#   sbatch scripts/slurm/run_hybrid_downstream_slurm.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec sbatch "$@" "${ROOT}/scripts/run_hybrid_downstream_slurm.sh"
