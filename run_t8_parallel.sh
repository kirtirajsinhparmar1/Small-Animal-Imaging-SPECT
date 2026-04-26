#!/bin/bash
#SBATCH --job-name=ppdf_t8
#SBATCH --cluster=ub-hpc
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/out/ppdf_%A_%a.out
#SBATCH --error=slurm_logs/err/ppdf_%A_%a.err

set -euo pipefail

mkdir -p slurm_logs/out slurm_logs/err

echo "=========================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Start: $(date)"
echo "=========================================================="

# --- Repo-local defaults (override via env vars if needed) ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

# Parallelism defaults (override via env vars)
SLURM_CPUS="${SLURM_CPUS_PER_TASK:-16}"
POSE_WORKERS="${POSE_WORKERS:-8}"
if (( POSE_WORKERS > SLURM_CPUS )); then
  POSE_WORKERS="${SLURM_CPUS}"
fi

# Cap torch threads per worker to avoid oversubscribing CPU threads.
# Example: 16 CPUs, 8 pose workers -> 2 torch threads/worker.
TORCH_THREADS="${TORCH_THREADS:-$(( SLURM_CPUS / POSE_WORKERS ))}"
if (( TORCH_THREADS < 1 )); then
  TORCH_THREADS=1
fi
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"

# Optional: Activate venv if provided
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_ACTIVATE}"
fi

# Make local modules importable by default
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# If LAYOUT_FILE isn't provided, attempt to pick the newest local layout under ./data
if [[ -z "${LAYOUT_FILE:-}" ]]; then
  LAYOUT_FILE="$(ls -t "${DATA_DIR}"/scanner_layouts_*.tensor 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "${LAYOUT_FILE:-}" ]]; then
  echo "FATAL: LAYOUT_FILE not set and no ${DATA_DIR}/scanner_layouts_*.tensor found."
  echo "Generate one via: python ${REPO_ROOT}/generate_mph_scanner_circularfov.py --output_dir ${DATA_DIR}"
  exit 1
fi

echo "Running T8 for layout=${SLURM_ARRAY_TASK_ID}"
python "${REPO_ROOT}/arg_ppdf_t8.py" \
  ${SLURM_ARRAY_TASK_ID} \
  --layout_file "${LAYOUT_FILE}" \
  --output_dir "${DATA_DIR}" \
  --pose-workers "${POSE_WORKERS}" \
  --torch-threads "${TORCH_THREADS}" \
  --torch-interop-threads "${TORCH_INTEROP_THREADS}" \
  --skip-existing \
  --a_mm 0.8 --b_mm 0.8

echo "=========================================================="
echo "End: $(date)"
echo "Exit code: $?"
echo "=========================================================="
