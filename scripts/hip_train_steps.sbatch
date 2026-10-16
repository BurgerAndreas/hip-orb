#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
export HIP_RESUME_CHECKPOINT="$HIP_TRAIN_OUTPUT/head.pt"
unset HIP_SESSION_SECONDS
bash "$PWD/scripts/hip_train.sbatch"
if "$HIP_LOADER_PYTHON" -c 'import json,os,pathlib,sys; r=json.loads((pathlib.Path(os.environ["HIP_TRAIN_OUTPUT"])/"report.json").read_text()); sys.exit(0 if r["training_complete"] else 1)'; then
  echo 'Training and final validation completed.'
else
  next_job=$(bash "$PWD/scripts/submit_job.sh" "$PWD/scripts/hip_train_steps.sbatch" \
    --job-name="$SLURM_JOB_NAME" --dependency="afterok:$SLURM_JOB_ID")
  printf '%s\n' "${next_job%%;*}" >> "$HIP_TRAIN_OUTPUT/job_ids.txt"
  echo "Submitted continuation: ${next_job%%;*}"
fi
