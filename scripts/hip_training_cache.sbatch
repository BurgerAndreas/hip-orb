#!/usr/bin/env bash
#SBATCH --job-name=hip-step-cache
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/hip_step_cache_%j.out
#SBATCH --error=logs/hip_step_cache_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_data --prepare-only \
  --source "$HIP_LOADER_SOURCE" --assignments "$HIP_LOADER_ASSIGNMENTS" \
  --cache "$HIP_LOADER_CACHE" --report "$HIP_LOADER_REPORT"
