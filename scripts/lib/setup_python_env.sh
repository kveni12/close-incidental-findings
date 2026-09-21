# Shared Python/pyenv setup for Slurm GPU jobs (non-interactive shells lack login PATH).
# Usage: source "${REPO_ROOT}/scripts/lib/setup_python_env.sh"

setup_python_env() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    echo "Using PYTHON_BIN=${PYTHON_BIN}"
    return 0
  fi

  export PYENV_VERSION="${PYENV_VERSION:-xray-drift-nlp-new}"
  export PYENV_ROOT="${PYENV_ROOT:-/autofs/space/crater_001/tools/pyenv}"
  if [[ -d "${PYENV_ROOT}/bin" ]]; then
    export PATH="${PYENV_ROOT}/bin:${PYENV_ROOT}/shims:${PATH}"
    echo "Using pyenv version ${PYENV_VERSION} (root=${PYENV_ROOT})"
  fi

  if command -v conda >/dev/null 2>&1 && [[ -n "${CONDA_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "Activated conda env: ${CONDA_ENV}"
  fi

  local py=""
  py="$(command -v python3 2>/dev/null || true)"
  if [[ -z "${py}" ]]; then
    py="$(command -v python 2>/dev/null || true)"
  fi
  if [[ -z "${py}" ]]; then
    echo "No python found after pyenv/conda setup."
    echo "Set PYENV_VERSION (default: xray-drift-nlp-new) or export PYTHON_BIN=/path/to/python"
    return 1
  fi
  echo "Python executable: ${py}"
  "${py}" -V || true
}
