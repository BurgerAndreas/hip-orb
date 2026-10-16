#!/usr/bin/env bash
#SBATCH --job-name=hip-batch-sizes
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
export PYTHONPATH="$HIP_LOADER_PACKAGE:$HIP_LOADER_EXTRAS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_batch_sizes --index "$HIP_BATCH_INDEX" \
  --assignments "$HIP_ASSIGNMENTS" --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT"
