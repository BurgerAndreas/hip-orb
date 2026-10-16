#!/usr/bin/env bash
#SBATCH --job-name=hip-speed-benchmark
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --nice=10000
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
: "${HIP_LOADER_PYTHON:?Set the Python interpreter}"
: "${HIP_LOADER_PACKAGE:?Set an immutable package snapshot}"
: "${HIP_TRAIN_INDEX:?Set the packed training index}"
: "${HIP_STEP_CHECKPOINT:?Set the frozen Orb checkpoint}"
: "${HIP_SPEED_CONFIG:?Set the model configuration}"
: "${HIP_SPEED_REPORT:?Set the private report path}"
export PYTHONPATH="$HIP_LOADER_PACKAGE:${HIP_LOADER_EXTRAS:-}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONSAFEPATH=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${HIP_CUDA_ALLOC_CONF-${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}}"
if [[ -n "${HIP_STEP_SETUP:-}" ]]; then source "$HIP_STEP_SETUP"; fi
hip_speed_args=(--index "$HIP_TRAIN_INDEX" --weights "$HIP_STEP_CHECKPOINT"
  --configs "$HIP_SPEED_CONFIG" --streaming --warmup "${HIP_SPEED_WARMUP:-16}"
  --steps "${HIP_SPEED_STEPS:-512}" --repeats "${HIP_SPEED_REPEATS:-2}" --profile-steps 0)
if [[ -n "${HIP_SPEED_HEAD_CHECKPOINT:-}" ]]; then
  hip_speed_args+=(--head-checkpoint "$HIP_SPEED_HEAD_CHECKPOINT")
fi
read -r -a hip_speed_candidates <<< "${HIP_SPEED_CANDIDATES:-eager production}"
hip_speed_case=0
for hip_speed_candidate in "${hip_speed_candidates[@]}"; do
  "$HIP_LOADER_PYTHON" -P -m hip_orb.benchmark_speed "${hip_speed_args[@]}" \
    --candidate "$hip_speed_candidate" --report "${HIP_SPEED_REPORT%.json}_${hip_speed_case}_${hip_speed_candidate}.json"
  hip_speed_case=$((hip_speed_case + 1))
done
