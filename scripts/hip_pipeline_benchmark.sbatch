#!/usr/bin/env bash
#SBATCH --job-name=hip-pipeline
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=logs/hip_pipeline_%j.out
#SBATCH --error=logs/hip_pipeline_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HIP_LOADER_PYTHON:?Set Python}"
: "${HIP_LOADER_PACKAGE:?Set the HIP package folder}"
: "${HIP_LOADER_EXTRAS:?Set optional dependencies}"
: "${HIP_LOADER_CACHE:?Set the benchmark cache}"
: "${HIP_STEP_CHECKPOINT:?Set the teacher checkpoint}"
: "${HIP_STEP_REPORT:?Set a fresh private report path}"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_training \
  --cache "$HIP_LOADER_CACHE" --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT" \
  --pipeline --profile-stages --steps 80 --warmup 10 --repeats 2 --batch-size 16
