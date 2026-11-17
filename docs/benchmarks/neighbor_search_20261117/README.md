# Cutoff neighbour search: full inference and sparse training steps

Measured on Vulcan, NVIDIA L40S (44.39 GiB visible), 17 November 2026.
Cutoff: 30 Å. Frozen OrbMol-v2 backbone; deterministic random HIP head with
hidden width 128. Precision is `float32-high` (TF32 allowed). All compared
backends use 65,536 backbone MLP rows and 65,536 Hessian pairs per chunk.
Each result is the median of five raw repeats after two warmup steps.

The hypothesis was that spatial search removes quadratic rejection work, and
that CPU search could overlap with the GPU backbone. GPU cells win on large
structures. CPU overlap hides some search work but leaves a longer full step.
The head and backbone remain the main cost. Small systems favour brute force;
the default uses brute force through 512 atoms, then GPU cells on CUDA.
The crossover is a heuristic, not a guarantee for every cutoff and batch.

## Inference wall time

| Atoms | Retained pairs | Brute force (s) | Async CPU (s) | GPU cells (s) |
|---:|---:|---:|---:|---:|
| 100 | 4,950 | 0.00914 | 0.00933 | 0.00967 |
| 350 | 46,875 | 0.01396 | 0.01426 | 0.01419 |
| 1,204 | 488,981 | 0.10064 | 0.09978 | 0.09900 |
| 10,000 | 9,146,502 | 1.63959 | 1.70350 | 1.56331 |
| 30,000 | 32,229,619 | 5.94657 | 6.40613 | 5.41538 |
| 100,000 | 116,977,262 | 24.72955 | 23.81057 | 19.39362 |

At 100,000 atoms GPU cells give a 1.23× speedup over async CPU, and 1.28×
over brute force. The original frozen backbone runs out of GPU memory on this
input. Row chunking permits the full run, with a 32.96 GiB peak allocation.

## Sparse training-step wall time

| Atoms | Brute force (s) | Async CPU (s) | GPU cells (s) |
|---:|---:|---:|---:|
| 100 | 0.01677 | 0.01754 | 0.02837 |
| 350 | 0.03040 | 0.02806 | 0.03161 |
| 1,204 | 0.22962 | 0.22890 | 0.22705 |
| 10,000 | 4.25421 | 4.23063 | 4.10789 |

The 10,000-atom GPU step is about 3% faster than async CPU. Search is not the
main training cost for this head. Peak GPU allocation is 11.87 GiB for this
GPU case. Larger training steps were not measured.

## Measurement boundary and scope

Timing starts with CPU ASE atoms already in memory. It includes message graph
construction, transfers, the frozen backbone, fresh radius search, local
context pooling, and block readout. Async CPU work starts before graph
construction and is joined before pair readout. CUDA is synchronized at each
timing boundary. Backbone features and pair lists are not cached across steps.

Training also includes `zero_grad`, the loss, backward, and AdamW. The loss is
`mean(blocks**2) + mean(diagonal**2)`, with zero targets. This measures a full
sparse update, not model accuracy, supervised convergence, production file
loading, or the dense trainer's rotation-consistency objective. The head uses
random initial weights; the frozen backbone uses its real checkpoint. Model
loading, input file reads, and warmup are excluded.

Inputs are fixed atom prefixes of PDB 1AON. The 100,000-atom case uses a second,
translated copy and is a scaling proxy, not a single experimental structure.
Every atom identity and float32 coordinate is saved in [final/inputs](final/inputs).
The source PDB is saved as [source_1AON.pdb.gz](source_1AON.pdb.gz).

## Correctness and numerical limits

- CPU checks: 137 passed, one CUDA-only case skipped.
- GPU checks: 50 passed, one CPU/GPU combination skipped.
- Complete 10,000- and 100,000-atom pair sets match across all backends:
  zero missing pairs and zero extra pairs. There is no neighbour cap.
- CPU, GPU, and brute paths use the CUDA three-coordinate reduction order,
  `sqrt((dx² + dz²) + dy²) < cutoff`. The initial x,y,z order missed six pairs
  near 30 Å at 100,000 atoms. Raw atom IDs and distances are retained in
  [validation-before.jsonl](checks/validation-before.jsonl); the corrected
  [validation-final.jsonl](checks/validation-final.jsonl) compares every pair.
- Tests cover strict boundaries, molecule separation, empty inputs, uncapped
  enumeration, replayed chunks, Hessian blocks, parameter gradients, checkpoint
  recomputation, and CPU/GPU streaming sinks. A dense head is an independent
  block-value reference when all pairs fit within the search radius.

Backbone row chunking keeps the same weights and graph reductions, but GEMM
row shapes change rounding. On the 10,000-atom input, the timed TF32 setting
gives maximum block/diagonal differences of 6.73e-4 / 7.67e-4 eV/Å² relative
to the unchunked backbone. `float32-highest` reduces those differences to
1.01e-6 / 1.55e-6 eV/Å². These are measured bounds for this input, not an
accuracy guarantee for other structures. Both raw comparisons are saved:
[timed precision](checks/backbone-validation.json),
[highest precision](checks/backbone-validation-highest.json).

The cell index is O(N). Its tile schedule uses O(T) descriptors for T candidate
tiles. Materialized indices and blocks cost O(M) for M retained pairs. The
streaming sink avoids retaining that full output; the caller must consume or
store each chunk. CPU tree queries use a 4 GiB native-pair-array budget and
fall back to bounded row queries beyond it.

## Raw evidence and reproduction

- [Per-structure, per-repeat metrics](final/per_structure.jsonl): 150 successful
  records, with stable structure IDs, method version, units, pair counts,
  output checksums, raw seconds, and peak GPU memory. No clipping is applied.
- [Provenance](final/provenance.json): input hashes, checkpoint hash, source
  hashes, architecture, seed, timing boundary, and exact arguments.
- [Summary](final/summary.json): medians, minima, maxima, and memory, derived
  without changing the raw records.
- [Environment versions](requirements-frozen.txt), [GPU](final/gpu.txt),
  [CPU hardware](checks/hardware-cpu.txt), [CPU checks](checks/cpu-tests-final.log),
  [GPU checks](checks/gpu-tests-final.log).
- [Trial records](trials): all earlier per-structure timings and failures,
  including the zero-volume backbone box error and the unchunked 100k OOM.
  Trial input arrays are represented by the same structure IDs in final/inputs.
  The archive in trials is an intermediate source snapshot. Final measured
  source hashes match implementation commit `75ef1fe` exactly.

Use the repo's Python environment with [requirements-frozen.txt](requirements-frozen.txt).
The compute node needed the matching Python 3.12 `pyconfig.h` include from its
login node for Triton compilation; this was supplied through `CPATH` in a
private benchmark directory. No shared environment was changed.

The backbone checkpoint is available from the [official model URL](https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orbmol-v2-teqabfhg-20260523.ckpt).
Its SHA-256 is `3dbef70b5fdc9124392181b0d8686c79f1c02bf5fefcb878a4eaa51d03e4f5e8`.

```bash
gzip -dc docs/benchmarks/neighbor_search_20261117/source_1AON.pdb.gz > /tmp/1AON.pdb
python -m hip_orb.benchmark_neighbor_search \
  --weights /path/to/orbmol-v2.ckpt --input /tmp/1AON.pdb \
  --output /path/to/new-results --counts 100 350 1204 10000 30000 100000 \
  --train-max-atoms 10000 --backbone-chunk 65536 --pair-chunk 65536 \
  --warmup 2 --repeats 5
python scripts/summarize_neighbor_benchmark.py /path/to/new-results
```

Each cluster allocation used an explicit time limit. GPU allocation 2400373
was released after the checks and measurements finished.
