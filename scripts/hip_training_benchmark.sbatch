#!/usr/bin/env bash
#SBATCH --job-name=hip-step-bench
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/hip_step_%j.out
#SBATCH --error=logs/hip_step_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HIP_LOADER_PYTHON:?Set an existing Python environment}"
: "${HIP_LOADER_PACKAGE:?Set the HIP package folder}"
: "${HIP_LOADER_EXTRAS:?Set the optional dependency folder}"
: "${HIP_LOADER_CACHE:?Set the prepared benchmark cache}"
: "${HIP_STEP_CHECKPOINT:?Set the collection teacher checkpoint}"
: "${HIP_STEP_REPORT:?Set the training benchmark report}"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_training \
  --cache "$HIP_LOADER_CACHE" --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT" \
  --local-cache-parent "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}" --batch-size "${HIP_STEP_BATCH_SIZE:-4}"
