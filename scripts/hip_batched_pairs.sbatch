#!/usr/bin/env bash
#SBATCH --job-name=hip-batched-pairs
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/hip_batched_%j.out
#SBATCH --error=logs/hip_batched_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HIP_LOADER_PYTHON:?Set Python}"
: "${HIP_LOADER_PACKAGE:?Set HIP package}"
: "${HIP_LOADER_EXTRAS:?Set dependencies}"
: "${HIP_LOADER_CACHE:?Set cache}"
: "${HIP_STEP_CHECKPOINT:?Set checkpoint}"
: "${HIP_STEP_REPORT:?Set private report}"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_training --cache "$HIP_LOADER_CACHE" \
  --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT" \
  --batch-pairs-comparison --steps 60 --warmup 10 --repeats 2 --batch-size 16
