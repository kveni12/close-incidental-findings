# Source from any script under scripts/ or scripts/slurm/ to cd to repository root.

if [[ "${#BASH_SOURCE[@]}" -gt 1 ]]; then
  _start_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
else
  _start_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ "$(basename "${_start_dir}")" == "slurm" && "$(basename "$(dirname "${_start_dir}")")" == "scripts" ]]; then
  REPO_ROOT="$(cd "${_start_dir}/../.." && pwd)"
elif [[ "$(basename "${_start_dir}")" == "scripts" ]]; then
  REPO_ROOT="$(cd "${_start_dir}/.." && pwd)"
elif [[ -f "${_start_dir}/run_close_pipeline.py" ]]; then
  REPO_ROOT="${_start_dir}"
else
  REPO_ROOT="$(cd "${_start_dir}/.." && pwd)"
fi
cd "${REPO_ROOT}"
