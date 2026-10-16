#!/usr/bin/env bash
#SBATCH --job-name=hip-randomness
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
: "${HIP_LOADER_PYTHON:?Set Python}"
: "${HIP_LOADER_PACKAGE:?Set HIP package}"
: "${HIP_LOADER_EXTRAS:?Set dependencies}"
: "${HIP_RANDOMNESS_INDEX:?Set sample index}"
: "${HIP_STEP_CHECKPOINT:?Set teacher checkpoint}"
: "${HIP_STEP_REPORT:?Set private report}"
export PYTHONPATH="$HIP_LOADER_PACKAGE:$HIP_LOADER_EXTRAS${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.audit_randomness --index "$HIP_RANDOMNESS_INDEX" \
  --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT"
