"""Check private environment loading and submission without using Slurm."""
import os
from pathlib import Path
import shutil
import subprocess
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("HIP_", "HF_", "FAIRCHEM_"))}


def test_loader_preserves_exported_run_settings_and_loads_private_values(tmp_path):
    (tmp_path / ".env").write_text('HIP_TRAIN_CONFIG="default.json"\nHIP_STEP_REPORT="private-report.json"\nHF_TOKEN="dummy-secret"\n')
    env = environment()
    env["HIP_TRAIN_CONFIG"] = "override.json"
    env["HF_TOKEN"] = "existing-secret"
    result = subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1"; '
        'test "$HIP_TRAIN_CONFIG" = override.json; test "$HIP_STEP_REPORT" = private-report.json; '
        'test "$HF_TOKEN" = existing-secret; case "$-" in *a*) exit 1;; esac',
        "test", str(SCRIPTS / "load_env.sh")], cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_loader_works_without_existing_environment_variables(tmp_path):
    (tmp_path / ".env").write_text('HIP_SLURM_ACCOUNT="test-account"\n')
    result = subprocess.run(["bash", "-c", 'set -euo pipefail; source "$1"; test "$HIP_SLURM_ACCOUNT" = test-account',
                             "test", str(SCRIPTS / "load_env.sh")],
                            cwd=tmp_path, env=environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("allocation", [None, "allocated-short"])
def test_submission_reads_account_partition_logs_and_keeps_options(tmp_path, allocation):
    (tmp_path / "scripts").mkdir()
    for name in ("submit_job.sh", "load_env.sh"):
        shutil.copy(SCRIPTS / name, tmp_path / "scripts" / name)
    (tmp_path / ".env").write_text('HIP_SLURM_ACCOUNT="test-account"\nHIP_SLURM_PARTITION="test-gpu"\n'
        'HIP_SLURM_LOG_DIR="$PWD/private logs"\nHIP_SLURM_QOS="test-qos"\nHIP_SLURM_TIME_LIMIT="02:00:00"\nHF_TOKEN="dummy-secret"\n')
    (tmp_path / "bin").mkdir()
    fake = tmp_path / "bin" / "sbatch"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    fake.chmod(0o755)
    env = environment()
    env.pop("SLURM_JOB_PARTITION", None)
    if allocation:
        env["SLURM_JOB_PARTITION"] = allocation
    env["PATH"] = f'{tmp_path / "bin"}:{env["PATH"]}'
    result = subprocess.run(["bash", "scripts/submit_job.sh", "scripts/hip_train_steps.sbatch",
                             "--dependency=afterok:123", "--job-name=test-run"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert "--account=test-account" in args
    assert f"--partition={allocation or 'test-gpu'}" in args
    assert "--time=02:00:00" in args
    assert "--qos=test-qos" in args
    assert "--dependency=afterok:123" in args
    assert f"--output={tmp_path}/private logs/%x_%j.out" in args
    assert args[-1] == "scripts/hip_train_steps.sbatch"
    assert "dummy-secret" not in result.stdout
    assert (tmp_path / "private logs").is_dir()


@pytest.mark.parametrize("use_pointer", [False, True])
def test_training_launcher_selects_runtime_without_cwd_shadowing(tmp_path, use_pointer):
    (tmp_path / "scripts").mkdir()
    for name in ("hip_train.sbatch", "load_env.sh"):
        shutil.copy(SCRIPTS / name, tmp_path / "scripts" / name)
    fake = tmp_path / "python"
    fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n'
                    'printf "package=%s\\nallocator=%s\\nsafepath=%s\\n" '
                    '"$PYTHONPATH" "$PYTORCH_CUDA_ALLOC_CONF" "$PYTHONSAFEPATH"\n')
    fake.chmod(0o755)
    runtime = tmp_path / "new runtime"
    (tmp_path / ".env").write_text(
        f'HIP_LOADER_PYTHON="{fake}"\nHIP_TRAIN_INDEX="train.npz"\n'
        'HIP_VALIDATION_INDEX="val.npz"\nHIP_TRAIN_OUTPUT="outputs/run"\n'
        f'HIP_TRAIN_RUNTIME_PACKAGE="{runtime}"\nHIP_STEP_CHECKPOINT="teacher.pt"\n'
        'HIP_TRAIN_EXECUTION_MODE="optimized"\nHIP_CUDA_ALLOC_CONF="expandable_segments:True"\n')
    env = environment()
    env.update(SLURM_SUBMIT_DIR=str(tmp_path), HIP_LOADER_PACKAGE="stale-package")
    if use_pointer:
        pointer = tmp_path / ".env.runtime"
        pointer.write_text(str(runtime) + "\n")
        env.update(HIP_TRAIN_RUNTIME_POINTER=str(pointer), HIP_TRAIN_RUNTIME_PACKAGE="previous-runtime")
    result = subprocess.run(["bash", "scripts/hip_train.sbatch"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:3] == ["-P", "-m", "hip_orb.train"]
    assert lines[lines.index("--execution-mode") + 1] == "optimized"
    assert any(line.startswith(f"package={runtime}:") for line in lines)
    assert "allocator=expandable_segments:True" in lines
    assert "safepath=1" in lines
    assert f"Training runtime: {runtime}" in result.stderr
