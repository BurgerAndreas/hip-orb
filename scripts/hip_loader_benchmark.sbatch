#!/usr/bin/env bash
#SBATCH --job-name=hip-loader-bench
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$PWD/scripts/load_env.sh"
: "${HIP_LOADER_PYTHON:?Set an existing Python environment}"
: "${HIP_LOADER_PACKAGE:?Set the HIP package folder}"
: "${HIP_LOADER_UV:?Set an installed uv executable}"
: "${HIP_LOADER_EXTRAS:?Set a private optional dependency folder}"
: "${HIP_LOADER_SOURCE:?Set the packed training release}"
: "${HIP_LOADER_ASSIGNMENTS:?Set the verified shard assignments}"
: "${HIP_LOADER_CACHE:?Set a fresh benchmark cache folder}"
: "${HIP_LOADER_REPORT:?Set a private benchmark report path}"
"$HIP_LOADER_UV" pip install --python "$HIP_LOADER_PYTHON" --target "$HIP_LOADER_EXTRAS" hdf5plugin
export PYTHONPATH="$HIP_LOADER_EXTRAS:$HIP_LOADER_PACKAGE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$HIP_LOADER_PYTHON" -m hip_orb.benchmark_data \
  --source "$HIP_LOADER_SOURCE" --assignments "$HIP_LOADER_ASSIGNMENTS" \
  --cache "$HIP_LOADER_CACHE" --report "$HIP_LOADER_REPORT" \
  --mmap-local "${HIP_LOCAL_CACHE_PARENT:?Set the private local cache path}"
