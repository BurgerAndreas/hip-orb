"""Real HDF5/spawn integration and cross-process lease ownership."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from hip_orb.data import OrbMolDataset, make_loader
from test_loader import fixture


@pytest.mark.parametrize("local", [False, True])
def test_spawned_workers_and_checksums(tmp_path, monkeypatch, local):
    index = fixture(tmp_path)
    sums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.h5")}
    checks = tmp_path / "checks.json"
    checks.write_text(json.dumps(sums))
    monkeypatch.setenv("HIP_ORB_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("HIP_ORB_CHECKSUMS", str(checks))
    monkeypatch.setenv("HIP_ORB_CACHE_BYTES", "1048576")
    if local:
        monkeypatch.setenv("HIP_ORB_CACHE_LOCAL_LOCK_ROOT", str(tmp_path / "locks"))
        monkeypatch.setenv("HIP_ORB_CACHE_GLOBAL_LEASE", "1")
        monkeypatch.setenv("HIP_ORB_CACHE_LEASE_HOST", socket.gethostname())
    ds = OrbMolDataset(index, cache_mode="chunks", max_open=1)
    loader = make_loader(ds, workers=2, pin_memory=False)
    try:
        assert sorted(i for batch in loader for i in batch["omol_index"].tolist()) == list(range(15))
    finally:
        loader._iterator._shutdown_workers()
        ds.close()


def test_lease_excludes_legacy_users_and_releases_on_exit(tmp_path):
    env = dict(os.environ, HIP_ORB_CACHE_ROOT=str(tmp_path),
               HIP_ORB_CACHE_LOCAL_LOCK_ROOT=str(tmp_path / "locks"))
    command = [sys.executable, "-m", "hip_orb.cache_lease", "--", sys.executable, "-c",
               "import os,socket,sys; assert os.environ['HIP_ORB_CACHE_LEASE_HOST']==socket.gethostname(); print('ready',flush=True); sys.stdin.readline()"]
    process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "ready"
        with (tmp_path / ".cache-lock").open("a+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process.communicate("done\n", timeout=10)
        assert process.returncode == 0
        with (tmp_path / ".cache-lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
