# Optional bounded shard cache

Set `HIP_ORB_CACHE_ROOT`, `HIP_ORB_CHECKSUMS` (relative path to SHA256 mapping)
and optionally `HIP_ORB_CACHE_BYTES` (default 64 GiB). Dataset index paths must
resolve under the cache root. Missing files download from dataset
`andreasburger/HIP-OrbMol-OMol25` at pinned revision
`8fdc9f01395e926d166cd0eb5d2d4bd275843bb6`. Install `orb-models[cache]`.
No cache environment means ordinary local loading, without downloads.

Shared mode retains the legacy `.cache-lock` for the complete transaction.
For one-node sessions, set `HIP_ORB_CACHE_LOCAL_LOCK_ROOT` to a node-local directory
shared by every rank/worker, then run:

```sh
python3 -m hip_orb.cache_lease -- python3 -m hip_orb.train ...
```

The wrapper acquires the exclusive global `.cache-lock` before setting lease/host
environment variables and keeps it until the command exits. Start this wrapper
**on the compute node**, outside torchrun, once per session. Never export the lease
flag manually. All consumers must honor this protocol; multi-node sessions are
unsupported. The same cache root hashes to one local worker-lock namespace.
Local mode splits fetch/checksum from short index/open/eviction transactions.
Misses recheck after waiting. Unregistered files are verified, stale missing rows
are invalidated, oversized shards are rejected, and committed index bytes are
bounded. Existing open Linux inodes survive eviction. As in the original cache,
registered immutable shards are trusted; external file replacement is unsupported.
A download can temporarily exceed the committed index bound.

Provenance: loader study final v2, job249485 (15 November 2026), 11 deterministic
tests in each original study lock mode, and all-rank content/loss/gradient/head/Adam
checks. The original-to-final +48.3% short comparison spans different trials and
is descriptive, not a repeated production estimate. This integration replaces the
external global dataset monkey-patch with an explicit per-dataset cache object;
shared mode deliberately retains coarse locking for legacy compatibility.
Production cache/checkpoint/deployment are untouched.
