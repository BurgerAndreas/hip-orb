"""Real locks/files/SQLite, deterministic stalled-download regression tests.

Torch and network calls are stubbed; no environment or production cache needed.
Run with python3 test_cache.py [ondemand_lru.py].
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import socket
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

TARGET = Path(__file__).parents[2] / 'hip_orb' / 'shard_cache.py'


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'cache'
        self.payloads = {'a': b'aaaaaa', 'b': b'bbbbbb', 'bad': b'wrong', 'large': b'x' * 21}
        sums = {k: hashlib.sha256(v).hexdigest() for k, v in self.payloads.items()}
        sums['bad'] = hashlib.sha256(b'expected').hexdigest()
        checks = Path(self.tmp.name) / 'checks.json'
        checks.write_text(json.dumps(sums))
        env = patch.dict(os.environ, HIP_ORB_CACHE_ROOT=str(self.root),
                         HIP_ORB_CACHE_BYTES='10', HIP_ORB_CHECKSUMS=str(checks))
        env.start()
        self.addCleanup(env.stop)
        if os.environ.get('HIP_ORB_TEST_LOCAL_LOCKS') == '1':
            local_env = patch.dict(os.environ, HIP_ORB_CACHE_LOCAL_LOCK_ROOT=str(Path(self.tmp.name) / 'locks'),
                                   HIP_ORB_CACHE_GLOBAL_LEASE='1', HIP_ORB_CACHE_LEASE_HOST=socket.gethostname())
            local_env.start()
            self.addCleanup(local_env.stop)
        self.fetches = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_name = None

        def download(repo, name, *, repo_type, revision, local_dir):
            self.assertEqual(repo, self.module.REPO)
            self.assertEqual(revision, self.module.REVISION)
            self.assertEqual(repo_type, 'dataset')
            self.fetches.append(name)
            if name == self.block_name:
                self.started.set()
                if not self.release.wait(timeout=5):
                    raise TimeoutError('Test failed to release its fake download')
            path = Path(local_dir) / name
            path.write_bytes(self.payloads[name])
            return str(path)

        class Dataset:
            def __init__(inner, names):
                inner.paths = [str(self.root / n) for n in names]
                inner._files = {}
                inner._pid = os.getpid()

            def _open_datasets(inner, shard):
                if shard not in inner._files:
                    inner._files[shard] = Path(inner.paths[shard]).read_bytes()
                return inner._files[shard]

            def _datasets(inner, shard):
                if shard in inner._files:
                    return inner._open_datasets(shard)
                return self.cache.open(inner.paths[shard], lambda: inner._open_datasets(shard))

        self.Dataset = Dataset
        spec = importlib.util.spec_from_file_location('tested_cache', TARGET)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.cache = self.module.ShardCache.from_environment()
        self.cache._download = download

    @unittest.skipUnless(os.environ.get("HIP_ORB_TEST_LOCAL_LOCKS") == "1", "Shared mode retains coarse legacy locking")
    def test_cached_open_is_not_blocked_by_download(self):
        self.Dataset(['a'])._datasets(0)
        self.block_name = 'b'
        with ThreadPoolExecutor(max_workers=2) as pool:
            missing = pool.submit(self.Dataset(['b'])._datasets, 0)
            self.assertTrue(self.started.wait(timeout=2))
            hit = pool.submit(self.Dataset(['a'])._datasets, 0)
            try:
                self.assertEqual(hit.result(timeout=0.5), self.payloads['a'])
                self.assertFalse(missing.done())
            finally:
                self.release.set()
            self.assertEqual(missing.result(timeout=2), self.payloads['b'])

    def test_duplicate_misses_download_once(self):
        self.block_name = 'b'
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.Dataset(['b'])._datasets, 0)
            self.assertTrue(self.started.wait(timeout=2))
            second = pool.submit(self.Dataset(['b'])._datasets, 0)
            self.release.set()
            self.assertEqual(first.result(timeout=2), self.payloads['b'])
            self.assertEqual(second.result(timeout=2), self.payloads['b'])
        self.assertEqual(self.fetches, ['b'])

    @unittest.skipUnless(os.environ.get("HIP_ORB_TEST_LOCAL_LOCKS") == "1", "Shared mode retains coarse legacy locking")
    def test_cached_open_is_not_blocked_by_checksum(self):
        self.Dataset(['a'])._datasets(0)
        (self.root / 'b').write_bytes(self.payloads['b'])
        original_sha = self.module._sha256
        def slow_sha(path):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError('Test failed to release its checksum')
            return original_sha(path)
        self.module._sha256 = slow_sha
        with ThreadPoolExecutor(max_workers=2) as pool:
            verifying = pool.submit(self.Dataset(['b'])._datasets, 0)
            self.assertTrue(self.started.wait(timeout=2))
            hit = pool.submit(self.Dataset(['a'])._datasets, 0)
            try:
                self.assertEqual(hit.result(timeout=0.5), self.payloads['a'])
                self.assertFalse(verifying.done())
            finally:
                self.release.set()
            self.assertEqual(verifying.result(timeout=2), self.payloads['b'])

    def test_eviction_bound_and_open_inode(self):
        # A complete shard is temporarily present during download, as before;
        # committed index bytes must be within the configured limit afterwards.
        self.Dataset(['a'])._datasets(0)
        with (self.root / 'a').open('rb') as held:
            self.Dataset(['b'])._datasets(0)
            self.assertFalse((self.root / 'a').exists())
            self.assertEqual(held.read(), self.payloads['a'])
        self.assertTrue((self.root / 'b').exists())
        with sqlite3.connect(self.root / '.cache-index.sqlite3') as db:
            self.assertLessEqual(db.execute('SELECT SUM(bytes) FROM shards').fetchone()[0], 10)

    def test_bad_checksum_is_not_registered(self):
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.Dataset(['bad'])._datasets(0)
        with sqlite3.connect(self.root / '.cache-index.sqlite3') as db:
            self.assertIsNone(db.execute("SELECT path FROM shards WHERE path='bad'").fetchone())

    @unittest.skipUnless(os.environ.get("HIP_ORB_TEST_LOCAL_LOCKS") == "1", "Shared mode retains coarse legacy locking")
    def test_stale_index_cannot_bypass_replacement_checksum(self):
        self.Dataset(['a'])._datasets(0)
        (self.root / 'a').unlink()
        original_sha = self.module._sha256
        def slow_sha(path):
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError('Test failed to release its checksum')
            return original_sha(path)
        self.module._sha256 = slow_sha
        with ThreadPoolExecutor(max_workers=2) as pool:
            replacement = pool.submit(self.Dataset(['a'])._datasets, 0)
            self.assertTrue(self.started.wait(timeout=2))
            with sqlite3.connect(self.root / '.cache-index.sqlite3') as db:
                try:
                    self.assertIsNone(db.execute("SELECT path FROM shards WHERE path='a'").fetchone())
                finally:
                    self.release.set()
            self.assertEqual(replacement.result(timeout=2), self.payloads['a'])

    def test_oversize_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exceeds cache limit'):
            self.Dataset(['large'])._datasets(0)

    def test_unregistered_existing_file_is_verified(self):
        (self.root / 'bad').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.Dataset(['bad'])._datasets(0)
        self.assertEqual(self.fetches, [])

    def test_outside_path_is_rejected(self):
        dataset = self.Dataset(['a'])
        dataset.paths[0] = str(Path(self.tmp.name) / 'outside')
        with self.assertRaises(ValueError):
            dataset._datasets(0)

    def test_local_locks_require_global_lease(self):
        with patch.dict(os.environ, HIP_ORB_CACHE_LOCAL_LOCK_ROOT=str(self.root / 'locks'),
                        HIP_ORB_CACHE_GLOBAL_LEASE='0'):
            with self.assertRaisesRegex(ValueError, 'global lease'):
                self.cache = self.module.ShardCache.from_environment()

    def test_local_locks_require_same_host(self):
        with patch.dict(os.environ, HIP_ORB_CACHE_LOCAL_LOCK_ROOT=str(self.root / 'locks'),
                        HIP_ORB_CACHE_GLOBAL_LEASE='1', HIP_ORB_CACHE_LEASE_HOST='another-host'):
            with self.assertRaisesRegex(ValueError, 'global lease'):
                self.cache = self.module.ShardCache.from_environment()


if __name__ == '__main__':
    unittest.main()
