#!/usr/bin/env bash
#SBATCH --job-name=hip-compile-audit
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
export PYTHONPATH="$HIP_LOADER_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HIP_LOADER_PYTHON" -m hip_orb.check_compilation --index "$HIP_TRAIN_INDEX" \
 --checkpoint "$HIP_STEP_CHECKPOINT" --report "${HIP_STEP_REPORT:?Set the private audit report path}"
