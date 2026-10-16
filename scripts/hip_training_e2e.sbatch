#!/usr/bin/env bash
#SBATCH --job-name=hip-training-e2e
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
: "${HIP_LOADER_PACKAGE:?Set private package snapshot}"
: "${HIP_TRAIN_INDEX:?Set original training index}"
: "${HIP_VALIDATION_INDEX:?Set separate validation index}"
: "${HIP_STEP_CHECKPOINT:?Set teacher checkpoint}"
: "${HIP_STEP_REPORT:?Set report path}"
export PYTHONPATH="$HIP_LOADER_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
if [[ -n "${HIP_E2E_PREPARE:-}" ]]; then "$HIP_LOADER_PYTHON" "$HIP_E2E_PREPARE"; fi
config_args=()
if [[ -n "${HIP_TRAIN_CONFIG:-}" ]]; then config_args=(--config "$HIP_TRAIN_CONFIG"); fi
variant_args=()
if [[ -n "${HIP_E2E_VARIANTS:-}" ]]; then
  read -r -a variants <<< "$HIP_E2E_VARIANTS"
  variant_args=(--variants "${variants[@]}")
fi
"$HIP_LOADER_PYTHON" -m hip_orb.profile_training_e2e \
  --train-index "$HIP_TRAIN_INDEX" --validation-index "$HIP_VALIDATION_INDEX" \
  --checkpoint "$HIP_STEP_CHECKPOINT" --report "$HIP_STEP_REPORT" "${config_args[@]}" \
  --steps "${HIP_E2E_STEPS:-192}" "${variant_args[@]}"
