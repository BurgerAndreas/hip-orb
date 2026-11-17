#!/usr/bin/env bash
#SBATCH --job-name=hip-neighbor-e2e
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HIP_NEIGHBOR_PYTHON:?Set isolated benchmark Python}"
: "${HIP_NEIGHBOR_WEIGHTS:?Set frozen Orb checkpoint}"
: "${HIP_NEIGHBOR_INPUT:?Set source PDB or XYZ}"
: "${HIP_NEIGHBOR_OUTPUT:?Set new metrics directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$HIP_NEIGHBOR_PYTHON" -m pytest -q tests/hip/test_neighbor_search.py tests/hip/test_backbone_chunks.py
"$HIP_NEIGHBOR_PYTHON" -m hip_orb.benchmark_neighbor_search \
  --weights "$HIP_NEIGHBOR_WEIGHTS" --input "$HIP_NEIGHBOR_INPUT" \
  --output "$HIP_NEIGHBOR_OUTPUT" "$@"
