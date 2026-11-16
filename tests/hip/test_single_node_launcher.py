"""Exercise shell argument/path handling without a scheduler or training run."""
import json
import os
from pathlib import Path
import subprocess

import pytest


@pytest.mark.parametrize("ranks", [1, 3])
def test_launcher_command_and_paths_with_spaces(tmp_path, ranks):
    runtime = tmp_path / "runtime with spaces"
    runtime.mkdir()
    bins = tmp_path / "bin"
    bins.mkdir()
    for name, body in {
        "srun": 'shift 2; exec "$@"',
        "flock": 'shift 2; exec "$@"',
        "python": 'exec "' + os.sys.executable + '" -c \'import json,os,sys; print(json.dumps({"args":sys.argv[1:],"locks":os.environ["HIP_ORB_CACHE_LOCAL_LOCK_ROOT"]}))\' "$@"',
    }.items():
        p = bins / name
        p.write_text("#!/bin/bash\n" + body + "\n")
        p.chmod(0o755)
    env = dict(os.environ, PATH=str(bins) + os.pathsep + os.environ["PATH"],
               HIP_ORB_RUNTIME=str(runtime), HIP_ORB_PYTHON=str(bins / "python"),
               HIP_ORB_OUTPUT=str(tmp_path / "run with spaces"), HIP_ORB_CONFIG="config path",
               HIP_ORB_TRAIN_INDEX="train path", HIP_ORB_VALIDATION_INDEX="valid path",
               HIP_ORB_CACHE_ROOT=str(tmp_path / "cache"), HIP_ORB_CHECKSUMS="checks path",
               HIP_ORB_NPROC=str(ranks), HIP_ORB_DISTRIBUTED_ENTRYPOINT="distributed path.py",
               HIP_ORB_WORKERS="4", HIP_ORB_PREFETCH="8", HIP_ORB_LOADER_CACHE_MODE="chunks")
    script = Path(__file__).parents[2] / "scripts/hip_single_node.sbatch"
    p = subprocess.run(["bash", str(script), "--lr", "0.001"], env=env, capture_output=True, text=True, check=True)
    args = json.loads(p.stdout)["args"]
    assert args[:3] == ["-m", "hip_orb.cache_lease", "--"]
    assert args[args.index("--output") + 1] == env["HIP_ORB_OUTPUT"]
    assert args[args.index("--workers") + 1] == "4"
    assert args[args.index("--prefetch-factor") + 1] == "8"
    assert args[-2:] == ["--lr", "0.001"]
    if ranks == 3:
        assert "--nproc_per_node=3" in args
        assert "distributed path.py" in args
    env["SLURM_JOB_NUM_NODES"] = "2"
    assert subprocess.run(["bash", str(script)], env=env, capture_output=True).returncode == 2
