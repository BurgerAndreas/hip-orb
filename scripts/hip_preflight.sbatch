#!/usr/bin/env bash
#SBATCH --job-name=hip-orb-preflight
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --output=outputs/preflight_%j.log
#SBATCH --error=outputs/preflight_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${HIP_PYTHON:?Set a Python interpreter with Orb dependencies}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HIP_ORB_REAL_TESTS=1
export OMP_NUM_THREADS=2
# Only generated inputs and temporary synthetic files. No train command or data paths.
"$HIP_PYTHON" -m pytest -q -s --confcutdir=tests/hip tests/hip
