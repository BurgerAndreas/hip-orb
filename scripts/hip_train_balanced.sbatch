#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
# Initial packed-path check, then the standard checkpoint/continuation chain.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
: "${HIP_TRAIN_RUNTIME_PACKAGE:?Set an immutable balanced-training runtime}"
export PYTHONPATH="$HIP_TRAIN_RUNTIME_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${HIP_CUDA_ALLOC_CONF-expandable_segments:True}"
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
"$HIP_LOADER_PYTHON" -P -m hip_orb.runtime_snapshot --output "$HIP_TRAIN_RUNTIME_PACKAGE" --verify
hip_preflight_output="$HIP_TRAIN_OUTPUT/preflight"
mkdir -p "$hip_preflight_output"
if [[ -f "$HIP_TRAIN_OUTPUT/batch_plan.npz" ]]; then
  cp "$HIP_TRAIN_OUTPUT/batch_plan.npz" "$hip_preflight_output/batch_plan.npz"
fi
"$HIP_LOADER_PYTHON" -P -m hip_orb.train \
  --config "$HIP_TRAIN_CONFIG" --train-index "$HIP_TRAIN_INDEX" \
  --validation-index "$HIP_VALIDATION_INDEX" --weights-path "$HIP_STEP_CHECKPOINT" \
  --output "$hip_preflight_output" --batching balanced \
  --max-steps 2 --session-steps 2 --validation-every 0 --validation-samples 10
# The full run starts from initialization, with its own output and schedule.
bash "$PWD/scripts/hip_train_steps.sbatch"
