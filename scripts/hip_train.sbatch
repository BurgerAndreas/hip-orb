#!/usr/bin/env bash
#SBATCH --job-name=hip-orb-train
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
: "${HIP_LOADER_PYTHON:?Set the training Python interpreter}"
: "${HIP_TRAIN_INDEX:?Set the packed training index}"
: "${HIP_VALIDATION_INDEX:?Set the separate validation index}"
: "${HIP_TRAIN_OUTPUT:?Set the output directory}"
if [[ -n "${HIP_TRAIN_RUNTIME_POINTER:-}" ]]; then
  IFS= read -r HIP_TRAIN_RUNTIME_PACKAGE < "$HIP_TRAIN_RUNTIME_POINTER"
  : "${HIP_TRAIN_RUNTIME_PACKAGE:?The runtime pointer must contain a package directory}"
  export HIP_TRAIN_RUNTIME_PACKAGE
fi
if [[ -n "${HIP_TRAIN_RUNTIME_PACKAGE:-}" ]]; then
  export HIP_LOADER_PACKAGE="$HIP_TRAIN_RUNTIME_PACKAGE"
fi
export PYTHONPATH="$HIP_LOADER_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
# Expandable segments reuse memory across variable molecular batch shapes.
export PYTORCH_CUDA_ALLOC_CONF="${HIP_CUDA_ALLOC_CONF-${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
if [[ -f "$HIP_LOADER_PACKAGE/manifest.json" ]]; then
  "$HIP_LOADER_PYTHON" -P -m hip_orb.runtime_snapshot --output "$HIP_LOADER_PACKAGE" --verify
fi
hip_train_args=(--train-index "$HIP_TRAIN_INDEX" --validation-index "$HIP_VALIDATION_INDEX"
                --weights-path "$HIP_STEP_CHECKPOINT" --output "$HIP_TRAIN_OUTPUT")
if [[ -n "${HIP_TRAIN_CONFIG:-}" ]]; then hip_train_args+=(--config "$HIP_TRAIN_CONFIG"); fi
if [[ -n "${HIP_RESUME_CHECKPOINT:-}" && -f "$HIP_RESUME_CHECKPOINT" ]]; then
  hip_train_args+=(--checkpoint "$HIP_RESUME_CHECKPOINT")
fi
if [[ -n "${HIP_SESSION_SECONDS:-}" ]]; then
  hip_train_args+=(--session-duration-seconds "$HIP_SESSION_SECONDS")
fi
if [[ -n "${HIP_TRAIN_EXECUTION_MODE:-}" ]]; then
  hip_train_args+=(--execution-mode "$HIP_TRAIN_EXECUTION_MODE")
fi
printf 'Training runtime: %s\n' "$HIP_LOADER_PACKAGE" >&2
"$HIP_LOADER_PYTHON" -P -m hip_orb.train "${hip_train_args[@]}"
