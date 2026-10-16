#!/usr/bin/env bash
# Run from the repository root: bash scripts/submit_job.sh SCRIPT [SBATCH_OPTIONS...]
set -euo pipefail
source "$PWD/scripts/load_env.sh"
: "${HIP_SLURM_ACCOUNT:?Set HIP_SLURM_ACCOUNT in .env}"
: "${HIP_SLURM_PARTITION:?Set HIP_SLURM_PARTITION in .env}"
: "${HIP_SLURM_LOG_DIR:?Set HIP_SLURM_LOG_DIR in .env}"
if (( $# < 1 )); then
  echo 'Usage: bash scripts/submit_job.sh SCRIPT [SBATCH_OPTIONS...]' >&2
  exit 2
fi
hip_job_script="$1"
shift
# A pending job can move partitions after sbatch captures its environment.
# Training continuations follow the actual allocation rather than that stale value.
if [[ "${hip_job_script##*/}" == hip_train_steps.sbatch && -n "${SLURM_JOB_PARTITION:-}" ]]; then
  export HIP_SLURM_PARTITION="$SLURM_JOB_PARTITION"
fi
mkdir -p "$HIP_SLURM_LOG_DIR"
hip_submit_args=(--parsable --account="$HIP_SLURM_ACCOUNT" --partition="$HIP_SLURM_PARTITION"
  --chdir="$PWD" --export=ALL --output="$HIP_SLURM_LOG_DIR/%x_%j.out"
  --error="$HIP_SLURM_LOG_DIR/%x_%j.err")
if [[ -n "${HIP_SLURM_QOS:-}" ]]; then hip_submit_args+=(--qos="$HIP_SLURM_QOS"); fi
if [[ -n "${HIP_SLURM_TIME_LIMIT:-}" ]]; then hip_submit_args+=(--time="$HIP_SLURM_TIME_LIMIT"); fi
if [[ -n "${HIP_SLURM_CPUS_PER_TASK:-}" ]]; then
  hip_submit_args+=(--cpus-per-task="$HIP_SLURM_CPUS_PER_TASK")
fi
sbatch "${hip_submit_args[@]}" "$@" "$hip_job_script"
