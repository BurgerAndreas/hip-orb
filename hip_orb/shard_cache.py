"""Bounded immutable-shard cache with explicit dataset ownership.

Shared mode retains the legacy coarse lock. Local mode splits fetch/index locks
under a launcher-held global lease and supports one host only.
"""
import fcntl
import hashlib
import json
import os
import sqlite3
import socket
import time
from pathlib import Path

REPO = "andreasburger/HIP-OrbMol-OMol25"
REVISION = "8fdc9f01395e926d166cd0eb5d2d4bd275843bb6"
CACHE_IMPLEMENTATION = "split-fetch-index-lock-local-v2"
from contextlib import closing, contextmanager


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ShardCache:
    def __init__(self, root, checksums, limit=64 << 30, local_lock_root=None):
        root = Path(root).resolve()
        limit = int(limit)
        if limit <= 0:
            raise ValueError("HIP_ORB_CACHE_BYTES must be positive")
        root.mkdir(parents=True, exist_ok=True)
        lock_root = root
        if local_lock_root:
            # The launcher holds root/.cache-lock for the entire training session,
            # excluding legacy and remote consumers. Every rank/worker must share
            # this host and local lock directory; the index stays on shared storage.
            if (os.environ.get("HIP_ORB_CACHE_GLOBAL_LEASE") != "1" or
                    os.environ.get("HIP_ORB_CACHE_LEASE_HOST") != socket.gethostname()):
                raise ValueError("Node-local cache locks require the launcher's global lease on this host")
            lock_root = Path(local_lock_root) / hashlib.sha256(str(root).encode()).hexdigest()
            lock_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = lock_root / ".cache-lock"
        fetch_lock_path = lock_root / ".cache-fetch-lock"
        db_path = root / ".cache-index.sqlite3"
        self.root, self.limit = root, limit
        self.lock_path, self.fetch_lock_path, self.db_path = lock_path, fetch_lock_path, db_path
        self.local = bool(local_lock_root)
        self.checks = json.loads(Path(checksums).read_text())

    @contextmanager
    def _index_lock(self, held=False):
        # In shared mode the caller holds this for the complete transaction.
        if held:
            yield
            return
        with self.lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield

    @staticmethod
    def _download(*args, **kwargs):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(*args, **kwargs)

    @classmethod
    def from_environment(cls):
        root = os.environ.get("HIP_ORB_CACHE_ROOT")
        if not root:
            return None
        return cls(root, os.environ["HIP_ORB_CHECKSUMS"],
                   int(os.environ.get("HIP_ORB_CACHE_BYTES", str(64 << 30))),
                   os.environ.get("HIP_ORB_CACHE_LOCAL_LOCK_ROOT"))

    def open(self, path, opener):
        # Revalidate affinity in spawned/forked workers as well as the parent.
        if self.local and (os.environ.get("HIP_ORB_CACHE_GLOBAL_LEASE") != "1" or
                           os.environ.get("HIP_ORB_CACHE_LEASE_HOST") != socket.gethostname()):
            raise ValueError("Node-local cache locks require the launcher's global lease on this host")
        if self.local:
            return self._open(path, opener)
        with self._index_lock():
            return self._open(path, opener, held=True)

    def _open(self, path, opener, held=False):
        root, limit = self.root, self.limit
        fetch_lock_path, db_path = self.fetch_lock_path, self.db_path
        path = Path(path).resolve()
        relative = path.relative_to(root).as_posix()
        expected = self.checks[relative]
        def open_cached(*, register=False):
            # Return a tagged result because dataset payloads need not be truthy.
            with self._index_lock(held):
                with closing(sqlite3.connect(db_path)) as db, db:
                    db.execute("CREATE TABLE IF NOT EXISTS shards (path TEXT PRIMARY KEY, bytes INTEGER NOT NULL, used INTEGER NOT NULL)")
                    row = db.execute("SELECT bytes FROM shards WHERE path=?", (relative,)).fetchone()
                    if not path.is_file():
                        # A stale index entry must not make a replacement file
                        # look verified while its download/checksum is pending.
                        db.execute("DELETE FROM shards WHERE path=?", (relative,))
                        db.commit()
                        return False, None
                    if row is None and not register:
                        return False, None
                    size = path.stat().st_size
                    if size > limit:
                        raise ValueError(f"Shard exceeds cache limit: {relative} ({size} bytes)")
                    db.execute("INSERT OR REPLACE INTO shards VALUES (?, ?, ?)", (relative, size, time.time_ns()))
                    # This stays inside the same lock as every eviction.
                    result = opener()
                    total = db.execute("SELECT COALESCE(SUM(bytes),0) FROM shards").fetchone()[0]
                    while total > limit:
                        victim = db.execute("SELECT path, bytes FROM shards WHERE path != ? ORDER BY used LIMIT 1", (relative,)).fetchone()
                        if victim is None:
                            raise RuntimeError("Cannot reduce shard cache below its limit")
                        (root / victim[0]).unlink(missing_ok=True)
                        db.execute("DELETE FROM shards WHERE path=?", (victim[0],))
                        total -= victim[1]
                    db.commit()
                    return True, result

        hit, result = open_cached()
        if hit:
            return result
        # Keep one downloader, as before, but let cached opens proceed while it
        # waits on the network or verifies a large file. Recheck after waiting:
        # another worker may already have fetched this exact shard.
        with fetch_lock_path.open("a+") as fetch_lock:
            fcntl.flock(fetch_lock.fileno(), fcntl.LOCK_EX)
            hit, result = open_cached()
            if hit:
                return result
            if not path.is_file():
                self._download(REPO, relative, repo_type="dataset", revision=REVISION, local_dir=root)
            if _sha256(path) != expected:
                raise ValueError(f"Shard checksum mismatch: {relative}")
            hit, result = open_cached(register=True)
            if not hit:
                raise FileNotFoundError(path)
            return result
