#!/usr/bin/env bash
#SBATCH --job-name=hip-compile-e2e
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/compiler_cache"
export HIP_E2E_VARIANTS="${HIP_E2E_VARIANTS:-runner2 compile2 triton2 compile_triton2}"
export HIP_E2E_STEPS="${HIP_E2E_STEPS:-512}"
bash "${HIP_E2E_SCRIPT:-$PWD/scripts/hip_training_e2e.sbatch}"
export PYTHONPATH="$HIP_LOADER_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$HIP_LOADER_PYTHON" -m hip_orb.check_pair_inputs
"$HIP_LOADER_PYTHON" -m hip_orb.check_compilation --index "$HIP_TRAIN_INDEX" \
 --checkpoint "$HIP_STEP_CHECKPOINT" --report "$PWD/audit.json" \
 --variants shape2 compile2 fusion2 triton2 compile_triton2
