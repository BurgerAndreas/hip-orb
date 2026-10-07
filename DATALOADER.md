# Packed OrbMol training loader

## Primary performance criterion

Choose the loader by complete training-step time, not isolated loading throughput.
`scripts/hip_training_cache.sbatch` prepares bit-verified caches on CPUs.
`scripts/hip_training_benchmark.sbatch` runs bounded GPU training comparisons.
Each strategy runs two warmup steps and ten measured optimizer steps, repeated twice.
It uses the same batch limit, sample sequence, rotation seeds, and initial head weights.
Timing includes loader waits, graph construction, transfers, frozen backbone inference,
analytical Coulomb curvature, head prediction, loss, backward, and optimizer updates.
No backbone features are cached.
Reports include measured milliseconds per step and total time with worker startup and warmup.
One-time cache preparation, local staging, and model loading stay outside step timing.
Local staging time is reported separately.
Shared GPU priming runs before the comparisons and has a separate timer.
Each comparison loads only fields needed for Hessian training.
Bounded prefetch stops at the requested number of batches.
The iterator reaches its end before worker shutdown.
The benchmark verifies that the frozen backbone state stays unchanged.
The wrapper reads Orb 0.7 feature dimensions from its message-passing processors.
It rejects inconsistent dimensions and checks actual node and edge outputs before head inference.
No learned checkpoint is saved, and the full-training gate stays closed.

The earlier CPU loader benchmark is only a diagnostic tool.

`hip_orb.data` streams the checked upper-triangle teacher labels.
It supports the public HDF5 shards and private memory-mapped NumPy caches.
It does not change the source files or enable full-dataset model training.

## Index and batches

Build a private index once on CPUs with `build_index(paths, output)`.
The index stores file locations, shard boundaries, atom counts, and parent split names.
Do not publish the private index: it contains deployment paths.

```python
from hip_orb.data import OrbMolDataset, make_loader, unpack_hessian

dataset = OrbMolDataset("private_index.npz", cache_mode="arrays", training_only=True)
loader = make_loader(dataset, workers=0, order="shard", pin_memory=True)
for epoch in range(epochs):
    loader.batch_sampler.set_epoch(epoch)
    for batch in loader:
        upper = batch["hessian_upper"].to("cuda", non_blocking=True)
        positions = batch["coords"].to("cuda", non_blocking=True)
        # Expand only the current batch when rotations or dense metrics need it.
        dense = unpack_hessian(upper, positions.shape[1])
```

Batches contain equal atom counts and no padding.
The sampler limits both molecule count and dense Hessian elements.
A molecule larger than the element budget forms a single-molecule batch.
The caller must handle its memory requirements.
The default element budget is 2,000,000 elements, not a full-model GPU memory limit.

Shard order shuffles files and rows within each file.
Random order shuffles samples across files within each atom-count bucket.
Both orders visit every indexed sample exactly once per epoch.
Call `set_epoch` to change the order reproducibly.

`source_dataset` and `omol_index` identify samples together.
Parent indices alone cannot identify samples across train and validation releases.

## Cache choices

- `none`: opens and closes HDF5 files for each sample. Use as a baseline only.
- `chunks`: retains worker-local file handles and HDF5 dataset chunk caches.
- `arrays`: decompresses complete shards into a bounded worker-local array cache.
- `backend="mmap"`: reads uncompressed, packed `.npy` arrays through memory mapping.

The default array budget is 256 MiB per worker.
Oversized shards fall back to chunk reads.
The default limit is eight open files per worker.
HDF5 caches, Python objects, prefetch buffers, and tensors use additional memory.
Workers use the spawn start method; they do not share inherited HDF5 handles.
Persistent workers preserve caches between epochs.
Zstandard files require `hdf5plugin` in every worker environment.
The default uses no workers, based on short end-to-end tests.
Set `max_batches` for bounded tests to prevent unused batch prefetch.
Leave it unset for complete epochs.
`training_only=True` omits energy and force labels without changing source files.

## Preliminary training results

The first comparison completed the 50–70 and 70–110 atom ranges.
Cached gzip with no workers took about 21 ms per training step.
Uncached gzip took 34–38 ms per step with the same sample sequences.
Worker startup increased total time for these short tests.
The run failed during worker shutdown before it completed the 2–350 atom range.
These results do not establish throughput for the complete dataset.

The next comparison tests batch limits of four and sixteen molecules.
It compares uncached gzip, chunk-cached gzip, array-cached gzip, Zstandard, and memory mapping.
Array-cached gzip uses zero, two, and four workers.
Zstandard and memory mapping use zero and two workers.
Common GPU priming removes first-strategy startup from the comparison.

## Implementation readiness checks

`scripts/hip_readiness.sbatch` runs bounded real-model GPU checks.
It selects available molecules near 2, 50, 70, 110, 160, 250, and 350 atoms.
The report records actual atom counts, complete optimizer steps, and peak GPU memory.
Checks require finite symmetric Hessians, unchanged backbone weights, and unchanged original energy and force predictions.
No backbone parameter may receive a gradient.
A four-molecule diagnostic tests head fitting with fixed backbone features.
Its pass criterion requires relative Hessian RMSE below 0.05 after 500 updates.
This diagnostic tests head fitting, not end-to-end training throughput.
A private checkpoint test verifies identical predictions after saving and restoring the head.

Passing these checks does not authorize full training.
External validation and a molecularly disjoint split remain separate requirements.
The readiness report keeps the full-training gate closed.

## CPU preparation and graph comparison

`scripts/hip_pipeline_benchmark.sbatch` runs one bounded GPU comparison.
It tests synchronous preparation, fused pair calls, CPU preparation workers, and CPU graph construction.
CPU preparation uses zero, two, and four workers.
CPU graph construction uses two workers and SciPy neighbor searches.
Learned node and edge features always run on GPU.
Augmentation uses a local seed derived from sample identities.
Worker scheduling cannot change the rotations used in a comparison.
Change the preparation seed for new augmentation draws in future training epochs.

The comparison uses cached gzip labels and a batch limit of sixteen molecules.
Each strategy runs ten warmup steps and eighty measured steps, repeated twice.
The three ranges remain 50–70, 70–110, and 2–350 atoms.
Reports include actual sample counts by atom count and aggregate samples per second.
Main-process CPU time excludes worker CPU time and does not replace wall-clock training time.
Startup, warmup, and peak GPU memory remain separate measurements.
These timings include CPU preparation, graph transfer or construction, and every optimizer update.

Before timing CPU graphs, the benchmark checks one molecule from every representative shard.
It requires identical sorted message edges and matching predicted Hessians within the recorded tolerance.
The benchmark excludes CPU graph construction if this check fails.
This sampled check does not establish agreement for the entire dataset.
Fused pair calls preserve the architecture and parameter names.
Unit tests check both predictions and gradients against separate pair calls.
The head retains at most eight cached pair-index tensors.
Known batch sizes stay on CPU to avoid a GPU-to-CPU size transfer during training.
The control disables pair-call fusion, pair-index caching, and CPU batch-size reuse.
Benchmark batches alternate available atom counts to ensure coverage through 350 atoms.
This benchmark mixture does not reproduce the dataset's atom-count distribution.
Use the per-atom counts when interpreting throughput; do not project full-dataset time from its aggregate alone.
After timing, a separate profiler run checks the fastest wide-range variant.
It records CPU and GPU time for graph preparation, backbone inference, labels, the head, and optimizer updates.
Profiler overhead never enters the throughput comparison.
The main-process trace cannot measure CPU work inside loader workers.
The full-training gate stays closed regardless of measured throughput.

## CPU graph correction and verification

Message graphs and Hessian graphs have separate roles.
CPU message graphs use Orb's configured radius and neighbor limit.
Both preparation paths sort message edges by sender and receiver before backbone inference.
The sort applies the same permutation to all edge features and targets.
It changes edge order, not connectivity or learned parameters.
After transfer, both paths recompute message vectors from the final device positions.
This avoids comparing geometry computed with different device arithmetic.
Nonperiodic preparation rejects nonzero image shifts and periodic inputs.

The CPU Hessian graph contains every unordered atom pair, independent of message connectivity.
Each pair stores forward and reverse message-edge indices, or minus one for missing edges.
The CPU builder rejects duplicate directed edges and edges between molecules.
The head uses these indices without rebuilding a dense message-edge lookup on GPU.
Node readouts, learned edge features, pair geometry, and analytical Coulomb curvature remain on GPU.
The device-only Hessian topology remains available as a reference path.

`scripts/hip_graph_check.sbatch` checks real molecules through 350 atoms.
It tests single molecules and batches of four from every representative shard.
The report separates native edge-order differences from corrected graph differences.
It also records repeated GPU prediction differences under high and highest float32 matrix precision.
Corrected CPU and GPU geometry must match exactly.
Hessian predictions retain the existing absolute and relative tolerance of 0.0001.
CPU Hessian topology must also match the device topology's head gradients.
The check verifies that backbone weights stay unchanged.
The dependent pipeline benchmark runs only if the graph check succeeds.
These checks do not replace held-out accuracy evaluation.

The first corrected audit found matching geometry but varying predictions from the same GPU graph.
At 20 atoms, repeated GPU Hessians differed by up to 0.00042 in that diagnostic.
The CPU/GPU difference was smaller than this repeated-prediction difference.
Graph agreement now uses deterministic reductions and highest float32 matrix precision.
The absolute and relative tolerances remain unchanged.
The audit restores the original deterministic and precision settings even when it fails.
Training benchmarks still use normal training precision and reduction settings.
Reports retain the native repeated-prediction differences for diagnosis.

## Detailed Hessian-head profile

`scripts/hip_head_profile.sbatch` uses one GPU for a bounded diagnostic.
It tests real molecules near 50, 70, 110, 250, and 350 atoms.
Each size starts with the same initial head weights and a fresh optimizer.
Complete-step timing uses two CPU preparation workers, ten warmup steps, and thirty measured steps.
These timings include label loading, graph construction, the frozen backbone, and optimizer updates.
The sampler retains the existing molecule and Hessian-element budgets.

A separate three-step profile reuses frozen features to isolate head execution.
That diagnostic does not measure complete training throughput or claim a feature-cache optimization.
Layer ranges isolate endpoint/edge input projection, hidden layers, diagonal readout, and block readout.
Forward, backward, and optimizer ranges remain separate.
Reports include operator shapes, CPU time, GPU time, and allocation measurements.
Detailed traces remain private and must not enter a public release without manual review.
Interpret self GPU time separately from inclusive ranges; do not add overlapping ranges together.

## Batched pair readout

The batched head adapts HIP's cumulative atom and flattened-Hessian offset pattern.
Reference: https://github.com/BurgerAndreas/hip/blob/main/nets/equiformer_v2/hessian_pred_utils.py
It creates pairs within each molecule, then concatenates them with global atom offsets.
It never creates a Cartesian product of all atoms across the batch.
Missing message edges remain explicit through minus-one lookup indices.
Without CPU lookup indices, sorted directed message keys provide sparse edge lookup.
This avoids allocating a dense message lookup matrix across the whole batch.

The pair adapter and block readout process both directions for every molecule in one call.
The diagonal readout also uses one call across all nodes.
Flat Cartesian indices assemble disjoint per-molecule Hessian segments without cross-molecule entries.
Unequal atom counts and optional Hessian cutoffs remain supported.
Analytical Coulomb curvature still runs separately for each molecule and remains unchanged.
The model retains all parameters, directed information, and output matrix shapes.
Set `head.batch_pairs=False` to use the original loop reference.
Double-precision tests compare predictions and all parameter gradients across both paths.

`scripts/hip_batched_pairs.sbatch` compares complete training steps with identical samples and initial weights.
Both variants use two CPU preparation workers, cached gzip labels, and normal training precision.
Each range runs ten warmup steps and sixty measured steps, repeated twice.
An untimed deterministic GPU check verifies predictions and gradients before timing begins.
Deploy the comparison in a separate private package copy to protect running tests.
No full training starts, and no head checkpoint is saved.

## Current defaults

The head batches pairs across molecules and processes both directions in one adapter call.
Pair-index caching remains enabled for the loop reference.
Complete-step benchmarks use these defaults unless a comparison explicitly selects the loop reference.
Standard AdamW remains the optimizer. Model parameters and checkpoint keys remain unchanged.

Batched pairs reduced complete-step time by 1.76–2.35 times versus the loop reference.
A rejected projection experiment slowed complete steps by 1.71–2.82 times.
It reduced peak memory by 5–27%, but the speed loss outweighed this benefit.
Fused AdamW gave no meaningful speed improvement in that experiment.
The rejected implementation, benchmark options, tests, and submission script were removed.

Profiling limits step counts to available batches and excludes overlapping CUDA annotations from GPU rankings.
Each atom size includes a separate complete-step trace with real loader waits and fresh backbone features.
Persistent workers retain their caches. Complete-step traces include graph construction, transfers, loss, backward, and optimizer updates.
Head-only traces isolate readout costs. Neither trace contributes to the uninstrumented training timings.
Held-out validation remains required before full training.

## Dense block defaults

The default batched path computes fixed-charge Coulomb blocks across molecules in one call.
An eight-entry cache stores geometry-free pair and assembly indices by batch sizes and device.
It never caches coordinates, charges, message-edge lookup, distances, or features.
CUDA validity checks use asynchronous assertions. Missing edges use a zero-row gather without dynamic boolean indexing.
The dense API remains available for analysis.

Parameters, checkpoint keys, directed pair information, and Coulomb damping remain unchanged.
GPU output and gradient checks passed through 350 atoms.
Measured dense step times were 17.6, 23.0, and 26.9 milliseconds for 50–70, 70–110, and 2–350 atoms.
The 50–70 baseline varied strongly between repeats, so its exact speedup remains uncertain.
Packed training gave no clear extra speed benefit. Its output API, loss helper, cache indices, and benchmark options were removed.
The loop reference remains available through `head.batch_pairs=False` for correctness tests.

The sampler can partition shards by distributed rank without overlap.
Different ranks can receive different batch counts.
Use DDP's uneven-input join support or implement equal-step scheduling before distributed training.
Shard partitioning alone does not prove a molecularly disjoint validation split.

## Loss and rotation

Saved labels remain float32 packed upper triangles, including the diagonal.
Training reconstructs dense targets, rotates them, and compares them with dense predictions using matrix MSE.
`unpack_hessian` reconstructs only the requested batch on the tensor's device.
Rotation augmentation must transform positions and Hessians consistently.
It must also rerun the frozen backbone.

The bounded pilot reader now accepts packed and dense shards, including nested release directories.
The full-training gate remains unchanged.

## CPU benchmark

`scripts/hip_loader_benchmark.sbatch` tests real training shards in three atom ranges:
50–70, 70–110, and 2–350 atoms.
The benchmark selects up to three shards near each representative atom count.
It tests gzip, Zstandard 3, and node-local memory-mapped arrays.
It compares random sampling, shard-local sampling, chunk caching, and whole-shard caching.
Worker counts are zero, two, four, and eight for the gzip sweep.
The Zstandard and memory-mapped sweeps test zero, two, and four workers.

Each strategy runs three epochs and checks exact sample coverage after every epoch.
The report separates startup time from warm-epoch throughput.
It records preparation time, payload throughput, and cache storage size.
It verifies cache arrays bit for bit before timing the loader.
The gzip storage size includes the full source files; cache sizes include training fields only.

No GPU is allocated for this CPU-only benchmark.
The benchmark does not measure backbone inference, head training, or host-to-device transfers.
It does not evict operating-system caches.
Warm results describe the selected working set, not the full 10-million-sample dataset.
Do not infer end-to-end training speed from loader-only measurements.

Set the `HIP_LOADER_*` paths in a private environment before submission.
Use a fresh benchmark cache folder.
Optional dependencies install into a separate private folder, not the production environment.
Node-local benchmark caches are temporary and disappear after the benchmark.
Preserve the gzip archive regardless of the training-cache choice.
