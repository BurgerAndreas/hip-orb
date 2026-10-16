#!/usr/bin/env bash
#SBATCH --job-name=hip-readiness
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.readiness --index "$HIP_LOADER_CACHE/2_350_gzip.npz" \
  --checkpoint "$HIP_STEP_CHECKPOINT" --output "$HIP_READINESS_OUTPUT"
