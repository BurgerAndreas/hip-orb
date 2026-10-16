#!/usr/bin/env bash
#SBATCH --job-name=hip-graphs
#SBATCH --time=00:15:00
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
: "${HIP_LOADER_EXTRAS:?Set optional dependencies}"
: "${HIP_LOADER_CACHE:?Set cache}"
: "${HIP_STEP_CHECKPOINT:?Set checkpoint}"
: "${HIP_STEP_REPORT:?Set private report}"
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -m hip_orb.check_graphs --index "$HIP_LOADER_CACHE/2_350_gzip.npz" \
  --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT"
