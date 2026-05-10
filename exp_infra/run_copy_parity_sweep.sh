#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-ltf}"
RECIPE="${RECIPE:-exp_infra/configs/recipes/copy_parity_fixed_ponder_pe_seeds.yaml}"
SUMMARY_DIR=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  exp_infra/run_copy_parity_sweep.sh [--dry-run] [--summary-dir DIR]

Environment overrides:
  CONDA_ENV=ltf      Conda environment to activate.
  RECIPE=path.yaml   Recipe file to run.

Examples:
  exp_infra/run_copy_parity_sweep.sh --dry-run
  CUDA_VISIBLE_DEVICES=0 exp_infra/run_copy_parity_sweep.sh
  exp_infra/run_copy_parity_sweep.sh --summary-dir exp_infra/results/recipes/copy_parity_sweep
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --summary-dir)
      SUMMARY_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -n "${CONDA_EXE:-}" ]]; then
  # shellcheck disable=SC1091
  source "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh. Activate ${CONDA_ENV} manually or set CONDA_EXE." >&2
  exit 1
fi

conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_ROOT}/exp_infra${PYTHONPATH:+:${PYTHONPATH}}"

cmd=(python -m ltf.cli.recipe --recipe "${RECIPE}")
if [[ "${DRY_RUN}" -eq 1 ]]; then
  cmd+=(--dry-run)
fi
if [[ -n "${SUMMARY_DIR}" ]]; then
  cmd+=(--summary-dir "${SUMMARY_DIR}")
fi

echo "Repo: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "Recipe: ${RECIPE}"
echo "Mode: $([[ "${DRY_RUN}" -eq 1 ]] && echo dry-run || echo run)"
echo "Command: ${cmd[*]}"

exec "${cmd[@]}"
