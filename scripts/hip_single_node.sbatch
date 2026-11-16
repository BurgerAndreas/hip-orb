#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail

# All site paths come from the environment. Supply hardware/account/partition
# overrides to sbatch; initialize modules/ROCm in the submitting environment.
: "${HIP_ORB_RUNTIME:?set the immutable runtime/repository directory}"
: "${HIP_ORB_PYTHON:?set the target environment Python executable}"
: "${HIP_ORB_OUTPUT:?set the run output directory}"
: "${HIP_ORB_CONFIG:?set the training config JSON}"
: "${HIP_ORB_TRAIN_INDEX:?set the relocated training index}"
: "${HIP_ORB_VALIDATION_INDEX:?set the relocated validation index}"
: "${HIP_ORB_CACHE_ROOT:?set the shared cache directory}"
: "${HIP_ORB_CHECKSUMS:?set the pinned shard checksum mapping}"
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || { echo 'Local cache locking requires one node' >&2; exit 2; }
nproc=${HIP_ORB_NPROC:-1}
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] || { echo 'HIP_ORB_NPROC must be positive' >&2; exit 2; }
if (( nproc > 1 )) && [[ -z "${HIP_ORB_DISTRIBUTED_ENTRYPOINT:-}" ]]; then
  echo 'Multiple ranks require an explicit synchronized distributed entrypoint; the base runner is single-process' >&2
  exit 2
fi
cd "$HIP_ORB_RUNTIME"
export PYTHONPATH="$HIP_ORB_RUNTIME${PYTHONPATH:+:$PYTHONPATH}"
export HIP_ORB_CACHE_LOCAL_LOCK_ROOT="${HIP_ORB_CACHE_LOCAL_LOCK_ROOT:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/hip-orb-cache-locks-$(id -u)}"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mkdir -p "$HIP_ORB_OUTPUT" "$HIP_ORB_CACHE_ROOT" "$HIP_ORB_CACHE_LOCAL_LOCK_ROOT"
train=("$HIP_ORB_PYTHON" -m hip_orb.train)
if (( nproc > 1 )); then
  train=("$HIP_ORB_PYTHON" -m torch.distributed.run --standalone --nnodes=1
         --nproc_per_node="$nproc" "$HIP_ORB_DISTRIBUTED_ENTRYPOINT")
fi
args=(--config "$HIP_ORB_CONFIG" --train-index "$HIP_ORB_TRAIN_INDEX"
      --validation-index "$HIP_ORB_VALIDATION_INDEX" --output "$HIP_ORB_OUTPUT"
      --workers "${HIP_ORB_WORKERS:-2}" --prefetch-factor "${HIP_ORB_PREFETCH:-2}"
      --loader-cache-mode "${HIP_ORB_LOADER_CACHE_MODE:-arrays}"
      --loader-open-files "${HIP_ORB_OPEN_FILES:-8}"
      --session-duration-seconds "${HIP_ORB_SESSION_SECONDS:-10800}")
if [[ -n "${HIP_ORB_CHECKPOINT:-}" ]]; then args+=(--checkpoint "$HIP_ORB_CHECKPOINT"); fi
if [[ -n "${HIP_ORB_WEIGHTS:-}" ]]; then args+=(--weights-path "$HIP_ORB_WEIGHTS"); fi
if [[ -n "${HIP_ORB_STOP_REQUEST_FILE:-}" ]]; then args+=(--stop-request-file "$HIP_ORB_STOP_REQUEST_FILE"); fi
# Run this one wrapper on the allocated node, outside torchrun, so the lease
# host and local lock root are identical for all children. No auto-submission.
srun --nodes=1 --ntasks=1 flock -n "$HIP_ORB_OUTPUT/training.lock" \
  "$HIP_ORB_PYTHON" -m hip_orb.cache_lease -- "${train[@]}" "${args[@]}" "$@"
