#!/usr/bin/env bash
#SBATCH --job-name=hip-orb-pilot
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=outputs/pilot_%j.log
#SBATCH --error=outputs/pilot_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${HIP_PYTHON:?Set a Python interpreter with collection dependencies}"
: "${HIP_SHARDS:?Set the OrbMol collection shard directory}"
: "${HIP_ASSIGNMENTS:?Set the materialized assignments file}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HIP_ORB_REAL_TESTS=1
"$HIP_PYTHON" -m pytest -q -s --confcutdir=tests/hip tests/hip
"$HIP_PYTHON" -m hip_orb.train --shards "$HIP_SHARDS" --output outputs/overfit \
  --stage overfit --train-count 4 --validation-count 1 --epochs 500 \
  --assignments "$HIP_ASSIGNMENTS"
"$HIP_PYTHON" -m hip_orb.train --shards "$HIP_SHARDS" --output outputs/pilot \
  --stage pilot --train-count 96 --validation-count 32 --epochs 20 \
  --assignments "$HIP_ASSIGNMENTS" \
  --checkpoint outputs/overfit/head.pt
"$HIP_PYTHON" -m hip_orb.benchmark --shards "$HIP_SHARDS" \
  --assignments "$HIP_ASSIGNMENTS" --checkpoint outputs/pilot/head.pt \
  --output outputs/pilot/benchmark.json
