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

## Exact-batch preparation profiles

`scripts/hip_step_profile.sbatch` profiles large batches on one GPU.
`hip_orb.profile_batch_steps` compares synchronous preparation, prebuilt CPU graphs, and asynchronous preparation with two or four workers.
The default cases cover 20, 50, 110, 250, and 350 atoms.
Batch sizes range from 64 to 128.
Use `--cases ATOMS:BATCH` to select other cases.
Each variant uses identical records, rotations, initial head weights, and a fresh optimizer.
The benchmark reuses a bounded record pool and reports repeated records.
Source reads stay outside timing, as in the batch-capacity benchmark.
Worker timings include collation, process transfer, and loader waits.
The prebuilt CPU graph case excludes preparation and cannot represent complete training throughput.
All cases recompute frozen backbone features and update the head.
Sampled CPU graph checks compare message edges, predictions, and head gradients before timing.
Separate traces record complete steps and pair adapter layers.
Profiler overhead stays outside the throughput timings.
CPU annotation times and GPU annotation times have different meanings.
GPU annotations span elapsed device ranges; CPU operator GPU times measure attributed device work.
Do not sum overlapping ranges or infer worker execution time from the main-process trace.
Detailed traces remain private.

The October 7 comparison completed on an H100 80GB.
Each variant used five warmup steps and two runs of twelve measured steps.
Six sampled graph checks passed, including prediction and gradient agreement.
The frozen backbone stayed unchanged.
The private results are in `outputs/step_profile/report.json` and `outputs/step_profile/summary.json`.

| Atoms | Batch | Synchronous step | Two preparation workers | Two graph workers | Four graph workers |
| --- | --- | --- | --- | --- | --- |
| 20 | 128 | 27.2 ms | 28.2 ms | 104.0 ms | 79.5 ms |
| 50 | 128 | 51.7 ms | 52.7 ms | 185.7 ms | 111.7 ms |
| 110 | 64 | 66.1 ms | 63.8 ms | 128.8 ms | 80.2 ms |
| 110 | 128 | 123.4 ms | 122.1 ms | 430.3 ms | 244.3 ms |
| 250 | 64 | 204.8 ms | 202.1 ms | 442.2 ms | 224.7 ms |
| 350 | 64 | 346.1 ms | 333.8 ms | 558.9 ms | 338.7 ms |

These capacity timings exclude source reads and reuse batches.
They cannot establish a speedup for streaming training.
Two preparation workers provide small improvements within this diagnostic.
CPU graph workers provide no consistent improvement within this diagnostic.
At 350 atoms and batch 64, the diagnostic GPU backbone range spans about 192 ms.
The head forward and backward ranges span about 125 ms combined.
The graph range spans about 13 ms.
These profile ranges include device gaps and cannot establish hardware occupancy or memory bandwidth limits.
The directed pair input alone occupies 22.57 GiB in this case.
Backbone compilation and fusion of pair gathers are useful next experiments.
This comparison does not test those optimizations or changes to numerical precision.

## Streaming training comparison

`scripts/hip_training_e2e.sbatch` compares bounded runs on the original packed training files.
It uses the production shard sampler, batch limit, Hessian memory limit, and update function.
The runner and benchmark both call `hip_orb.runner.train_update`.
This function includes augmentation, graph preparation, transfer, backbone features, loss, backward, optimization, and scalar loss reporting.
The benchmark keeps the runner's per-step loss synchronization.
It never preloads a repeated record pool or reuses frozen features.
Every batch uses fresh sampler rows within an epoch.
Paired strategies must use identical source identities, molecule identities, sampler rows, and batch boundaries.
The current benchmark excludes its duplicate row-plan checks and identity hashing from all training timings.
It retains only small CPU identity metadata while updates run.

The default comparison uses four CPUs, one GPU, and 32 GiB of host memory.
The production batch limit is 128, and the dense Hessian limit is four million elements.
The actual batch size decreases as atom count increases.
Each strategy uses ten warmup steps and 192 measured steps, repeated twice.
The second repeat reverses strategy order to reduce ordering bias.
Worker-local caches reset between cases, but the benchmark does not evict operating-system caches.
Reports include actual atom counts, batch sizes, source records per second, loader waits, and GPU memory.
Worker startup and warmup remain separate from steady streaming throughput.
Training throughput including startup counts the warmup records too.

Each bounded run writes a real checkpoint and evaluates 256 records from the separate validation collection.
Checkpoint and validation time remain separate from update time.
This bounded validation limit differs from the long-run configuration's 10,000-record limit.
Do not extrapolate its cost to the full validation cadence.
Three additional fresh batches provide a separate diagnostic trace before validation pauses the pipeline.
The benchmark restores the measured head before validation.
Profiler overhead and diagnostic updates stay outside all reported throughput timings.
The benchmark does not measure a complete epoch or justify a dataset-wide time estimate.

The first streaming comparison completed on an H100 80GB.
Each strategy read 8,800 measured records in the first repeat and 10,296 in the second repeat.
All paired record sequences matched, and the frozen backbone stayed unchanged.
Eight sampled CPU graph checks passed, including head gradients.

| Strategy | Streaming records per second | Records per second including startup and warmup |
| --- | --- | --- |
| Production runner, two workers | 1,158 | 778 |
| Production runner, four workers | 1,012 | 585 |
| Worker augmentation, two workers | 1,144 | 783 |
| Worker augmentation, four workers | 1,018 | 579 |
| CPU graph construction, two workers | 486 | 321 |

Four workers reduced throughput under the four-CPU allocation.
CPU graph construction reduced throughput by about 58%.
Two augmentation workers were slower in one repeat and faster in the other.
This comparison does not establish a repeatable augmentation-worker speedup.
The private results are in `outputs/training_e2e/report.json` and `outputs/training_e2e/summary.json`.
Those initial results include small identity checks in the update loop and extra sampler checks during startup.
Use `outputs/clean_e2e` for measurements after the benchmark bookkeeping correction.

The corrected comparison used 512 measured steps per repeat.
Each strategy used 26,759 measured records in the first repeat and 28,178 in the second.
Shared update code preserves the production augmentation, graph path, precision, optimizer, and scalar reporting.
The measured snapshot uses MSE, AdamW, and float32 high precision.
Future comparisons read the configured loss type, head shape, and memory limits.
Record checks, profiler overhead, and diagnostic updates remain outside the training measurements.
Peak training memory stayed near 3.5 GiB under the production Hessian limit.

| Strategy | Streaming records per second |
| --- | --- |
| Production runner, two workers | 1,247 |
| Worker augmentation, two workers | 1,258 |
| Deferred loss reporting, two workers | 1,229 |

The augmentation-worker gain is about 0.9% in this comparison.
Earlier paired repeats gave mixed results, so this does not establish a useful repeatable improvement.
Production keeps two source workers, synchronous augmentation, GPU graph construction, and per-step scalar reporting.
Model initialization is shared and reported separately from case pipeline totals.
The first case includes some common first-use costs, so startup differences cannot establish an optimization benefit.

The separate `deferred2` benchmark candidate accumulates detached losses in float64 on the GPU.
It reads the aggregate at reporting boundaries and preserves model precision, gradients, and optimizer updates.
A 512-step comparison tested this candidate twice with two source workers.
It increased throughput by 3.4% in one repeat and reduced throughput by 3.6% in the other.
Combined streaming throughput was nearly unchanged: 1,274 records per second for the runner and 1,271 for deferred reporting.
Production retains per-step loss reporting.
The private results are in `outputs/loss_reporting_e2e/report.json`.

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

Short and long training use `hip_orb.train` with `--train-index` and `--validation-index`.
Both use `OrbMolDataset`, `make_loader`, and the same batched optimizer step.
The pilot changes configuration values only. It does not cache frozen features or preload dense labels.
Inference benchmarks retain a bounded reader for packed and dense shards.
Do not launch full training until the documented training checks pass.

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
Copy `.env.example` to `.env` to configure those paths and the scheduler account.
Use `bash scripts/submit_job.sh SCRIPT` to submit a benchmark or preparation job.
Set its private report path through `HIP_STEP_REPORT` or `HIP_LOADER_REPORT`.
Use a fresh benchmark cache folder.
Optional dependencies install into a separate private folder, not the production environment.
Node-local benchmark caches are temporary and disappear after the benchmark.
Preserve the gzip archive regardless of the training-cache choice.

### Backbone compilation and gather fusion

`hip_orb.profile_training_e2e` also accepts these explicit compiler candidates:

- `shape2`: supply known node and edge counts to charge/spin conditioning.
- `compile2`: compile the frozen backbone with dynamic shapes and full-graph capture.
- `fusion2`: compile the head's directed pair input assembly. This region joins both
  pair directions, gathers endpoint and edge features, and concatenates geometry
  and charges. It does not cache frozen features.
- `compile_fusion2`: combine backbone compilation and pair input fusion.

All candidates retain the same source loader, optimizer, loss, and numerical
precision. `compile2` includes the known conditioning size change. Compiler setup
costs remain in startup and warmup timing. Recompilation after warmup remains in
streaming timing. Full-graph capture prevents silent fallback in these compiled
regions. Candidate contexts restore eager execution after each case.

An initial full-graph attempt failed at conditioning's data-dependent
`repeat_interleave`. The known output size resolves that capture obstacle without
changing conditioning values. `hip_orb.check_compilation` compares eager and
candidate node features, edge features, charges, Hessians, and head gradients on
real molecular graphs. Its checks run separately from throughput measurements.

`triton2` tests a dedicated float32 gather-and-concatenate kernel.
`compile_triton2` combines it with backbone compilation. The kernel writes the
joined tensor directly. It avoids endpoint gather buffers, index concatenations,
and the padded edge-feature copy. It requires detached backbone inputs; head
parameter gradients still use normal autograd. `hip_orb.check_pair_inputs` checks
empty pairs, missing message edges, both directions, partial tiles, and two
latent dimensions on CUDA. These are explicit candidates, not runner defaults.

The current compiler candidate also sets `force_same_precision=True`. The initial
compiler measurement did not use that option. Its uncontrolled numerical audit
exceeded the selected tolerances. Eager calls also differed because GPU reductions
were nondeterministic. The corrected audit uses deterministic reductions. Head-only
checks use identical frozen features. Throughput runs retain normal reduction behavior.
The eager repeat control passes for 2 and 50 atoms. It remains unstable for 350
atoms at the training precision. Backbone conclusions therefore use the smaller
controlled cases. Both head-fusion candidates pass all three shared-feature checks.
Keep the initial measurement separate from the later precision-controlled comparison.
The benchmark now reads the configured learning-rate schedule and applies the
production schedule before each measured optimizer update. Earlier snapshots
used a constant learning rate. Report that distinction with their measurements.

The final confirmation uses MAE and the configured cosine learning-rate schedule.
Job 7239618 measures 512 fresh training batches per repeat, with two repeats.
The eager runner reaches 1,242.6 records/second. The dedicated gather kernel reaches
1,240.2 records/second. Its paired changes are +2.3% and -2.6%. The earlier
constant-rate MAE comparison shows roughly 2–3% slower training in both repeats.
These results do not establish a consistent end-to-end benefit.

Backbone compilation improves some warm comparisons. Initial shape compilation
costs reduce its bounded throughput. It also exceeds the selected numerical
tolerance in the 2- and 50-atom checks, where the eager repeat control passes.
Leave the backbone and head eager. Compiler candidates remain explicit benchmark
options. See `outputs/compile_e2e_scheduled/summary.json` for the final result and
`outputs/compile_e2e_mae/deterministic_audit.json` for numerical checks.

### Complete update benchmark

`hip_orb.benchmark_speed` measures complete training updates on real molecular batches.
Streaming cases include data loading, rotation, graph construction, frozen backbone inference, loss, backward, and optimizer updates.
They apply the configured learning-rate schedule and retain the checkpoint's optimizer state.
Each report records sample fingerprints, implementation hashes, memory, and backbone integrity.
Startup and warm-up remain separate from steady update timing.
Fixed-batch cases exclude source reads; do not describe them as streaming throughput.

Set `HIP_SPEED_CONFIG`, `HIP_SPEED_HEAD_CHECKPOINT`, and `HIP_SPEED_REPORT` in the private environment.
Set `HIP_LOADER_PACKAGE` to a verified complete snapshot.
Submit with `bash scripts/submit_job.sh scripts/hip_speed_benchmark.sbatch`.
Use one fixed head checkpoint for all comparisons.
The default candidates are `eager production`. An A-B-B-A sequence checks timing drift:

```bash
HIP_SPEED_CANDIDATES="eager production production eager"
```

The script adds a case index to each report name. Repeated candidate names therefore retain separate reports.

`hip_orb.audit_speed` compares predictions, losses, and all trainable gradients against eager execution.
Use `--full-backbone --production-batches --precision float32-high` to test the current training precision.
Exact graph tests also compare packed graphs with the original Orb adapter.
`hip_orb.compare_execution` compares held-out metrics without updating weights.
Numeric audits and throughput measurements run separately.

The first profile identifies absent-edge gradient accumulation as the largest cost in sparse Orb variants.
One 350-atom case spends about 224 ms per update in this accumulation.
The masked-edge kernel removes the shared zero-row contention.
Large fixed-batch updates improve by about 3–4 times in these variants.
Use the streaming comparison for estimates across the real atom-count distribution.

The allocator test compares default and expandable allocation on the same global-context stream.
Reserved memory drops from 77.7 to 34.5 GiB. Active tensors use about 33.7 GiB in both cases.
Update time stays near 109 ms. The speed comparison therefore uses expandable allocation on both paths.

### H100 measurements, 2026-10-16

Each comparison runs A-B-B-A, with 1,024 fresh updates per case and the same saved checkpoint.
All cases use the same 56,316 samples. They recompute frozen backbone features at every update.
Timings include loading, augmentation, graph construction, backward, and optimizer updates. They exclude validation and startup.
Both execution paths use expandable allocation.

| Model | Reference ms/update | Optimized ms/update | Speedup |
|---|---:|---:|---:|
| Context, MLP width 1024, all Hessian pairs | 109.8 | 102.6 | 1.07× |
| Random Orb round, backbone graph | 220.5 | 92.3 | 2.39× |
| Random Orb round, Hessian graph | 141.5 | 131.6 | 1.08× |
| Pretrained Orb round and context | 194.3 | 67.2 | 2.89× |
| Pretrained Orb round, context and translation projection | 195.0 | 67.2 | 2.90× |

Sparse Orb variants benefit most from the masked-edge backward kernel.
Dense Orb graphs contain no missing edges. Frozen context inputs also avoid the original backward bottleneck.
The extra runtime changes provide smaller gains in those models.

The final GPU suite passes 285 tests, with three skips.
All 72 controlled prediction, loss, and gradient checks pass at the current and highest float32 precisions.
These checks cover six heads and three atom counts. They also check complete backbone evaluation.
Checkpoint tests resume all five active architectures for 100 updates, save, and resume for 100 further updates.
Frozen weights stay unchanged. Six launcher cases also pass on the local shell.

Affine factorization changes TF32 rounding at the current training precision. Bfloat16 worsens held-out metrics.
These experiments remain outside the training path. The final execution modes retain float32 precision.
Detailed private reports stay in the ignored `outputs/speed_20261016` directory.

### Batch capacity and element budget, 2026-10-16

`hip_orb.benchmark_capacity` measures batch capacity without changing training checkpoints or live configurations.
It tests complete float32 updates with the production optimizations and saved optimizer state.
Frozen backbone inference runs at every update. No backbone features are cached.

The sampler sets the molecule cap to `min(batch_size, max(1, limit // (3 * N)**2))`.
With batch 128 and a four-million-element budget, the caps are:

| Atoms per geometry | 20 | 50 | 70 | 100 | 200 | 350 |
|---|---:|---:|---:|---:|---:|---:|
| Molecules per batch | 128 | 128 | 90 | 44 | 11 | 3 |

The exact ten-million-geometry shard plan gives 167,791 updates per epoch and 59.60 geometries per update.
The element budget reduces the cap below 128 for 56.49% of updates.
Full batches at the reduced cap account for 42.58% of updates.
A separate 1,024-update sample reaches 3,999,699 elements and fills the reduced cap in 46.68% of updates.
Shard tails also produce smaller batches. The current shard sampler does not combine records across shards.
Each source shard contains at most 400 records. Thus, a requested batch of 512 never produces 512 records during streaming.

Keep a memory guard. The dense Hessian alone uses about 16 MB at four million float32 elements.
Pair features and their gradients require much more memory. Their size also grows with the square of the atom count.
The same element limit gives different memory use for different heads and graph choices.
The budget is a proxy for memory, not a hard GPU allocation limit.

The capacity sweep uses real geometries at 20, 40, 70, 150, 250, and 350 atoms.
It tests batch caps 64, 128, 256, and 512, with budgets from two to twelve million elements.
Deduplication leaves 39 distinct fixed shapes per model. Fixed-shape timings exclude source reads.
Each shape has three warmup updates and two timed blocks of twelve updates.
Out-of-memory cases stop larger batches at that atom count.
Streaming comparisons exclude settings that fail capacity checks or exceed 85% of device memory at a tested shape.
These sampled checks do not guarantee capacity for every geometry.

Streaming tests process the same 8,831 geometries from 48 shuffled shards for each setting.
Each setting has two repeats. Baseline controls run first and last.
A larger confirmation uses 192 shards and compares the baseline, a selected candidate, and the baseline again.
Each comparison restores the same head and optimizer before each repeat.
Sample fingerprints verify identical coverage and order across batch settings.
Rebatching changes augmentation groups. These tests measure speed and memory, not model accuracy.
Timing includes loading, rotation, graph construction, frozen inference, head prediction, backward, and optimizer updates.
Worker startup, model loading, validation, and separate profiles remain outside throughput timing.

Use private environment variables for deployment paths:

```bash
python -m hip_orb.benchmark_capacity \
  --index "$HIP_TRAIN_INDEX" --weights "$HIP_STEP_CHECKPOINT" \
  --config "$HIP_SPEED_CONFIG" --head-checkpoint "$HIP_SPEED_HEAD_CHECKPOINT" \
  --report "$HIP_SPEED_REPORT"
```

For confirmation, add `--capacity-report "$HIP_CAPACITY_REPORT" --cohort-shards 192 --profile-steps 0`.
Select cases with `--settings 128:4000000 512:8000000 128:4000000`.
The prior report must match the configuration and checkpoint step.
Run GPU measurements in a Slurm allocation with one GPU and expandable CUDA allocation.

A higher geometry rate does not imply faster completion of a fixed number of updates.
Larger batches increase update time and change geometry exposure and update weighting by atom count.
Keep current matched training runs unchanged unless a new training comparison explicitly changes these settings.

The new profiles identify pair-MLP matrix multiplications and activation operations as the main remaining costs.
At 70 and 350 atoms, the wide context head spends about 50–57% of its device timeline in backward and optimizer updates.
Its frozen backbone accounts for about 9–21%. The head forward pass and loss account for about 25–29%.
The smaller extra-Orb context head spends about 39–49% in backward and optimizer updates.
Its frozen backbone accounts for about 19–36%. Graph construction takes about 4–8% across these sampled cases.
These percentages use annotated device ranges, not hardware utilization counters. Profile overhead stays outside throughput timing.

Large matrix multiplications dominate the wide heads. SiLU forward and backward follow.
Concatenation, reductions, gathers, and scatter operations remain secondary costs.
Loader waits stay below 1% in the initial matched-cohort tests.
More loader workers therefore have little measured benefit under these settings.
The next optimization targets are pair-feature materialization and MLP activation memory, followed by their backward operations.
Earlier fused-MLP experiments do not establish a speedup and remain outside production execution.

#### Matched-cohort results

The initial sweep processes 8,831 geometries twice per setting on one H100.
The baseline is batch 128 with a four-million-element budget.
Baseline rates average the first and last controls. Memory is peak allocated CUDA memory.

| Model | Baseline geometries/s | Larger batch / budget | Geometries/s | Change | Peak GiB, baseline to larger |
|---|---:|---|---:|---:|---:|
| Context, MLP 1024, all pairs | 406.5 | 256 / 5M | 411.7 | +1.3% | 33.7 → 42.0 |
| Random Orb round, backbone graph | 447.8 | 512 / 8M | 457.5 | +2.2% | 29.9 → 61.0 |
| Random Orb round, Hessian graph | 313.3 | 512 / 4M | 311.4 | -0.6% | 43.3 → 43.4 |
| Pretrained Orb round and context | 647.6 | 512 / 8M | 689.1 | +6.4% | 17.7 → 36.8 |
| Pretrained Orb round, context, translation projection | 652.7 | 512 / 8M | 688.1 | +5.4% | 17.7 → 36.8 |

The larger column selects the fastest successful nonbaseline setting from the sweep.
The dense Orb head gains no useful throughput from these larger settings.
Small differences require the longer confirmation below.

The confirmation processes the same 41,282 geometries twice per setting.
It measures the baseline, the selected larger setting, and a final baseline control.

| Model | Larger batch / budget | Baseline geometries/s | Larger geometries/s | Change | Baseline ms/update | Larger ms/update | Peak GiB, baseline to larger |
|---|---|---:|---:|---:|---:|---:|---:|
| Context, MLP 1024, all pairs | 256 / 5M | 529.1 | 537.0 | +1.5% | 102.3 | 151.9 | 33.7 → 42.1 |
| Random Orb round, backbone graph | 512 / 8M | 585.8 | 607.5 | +3.7% | 92.4 | 214.4 | 30.0 → 61.1 |
| Pretrained Orb round and context | 512 / 8M | 844.9 | 920.1 | +8.9% | 64.0 | 141.5 | 17.8 → 37.0 |
| Pretrained Orb round, context, translation projection | 512 / 8M | 844.1 | 910.3 | +7.8% | 64.1 | 143.1 | 17.8 → 37.0 |

The two cohorts have different atom-count distributions. Compare settings within each cohort.
Baseline drift in the confirmation is recorded with the detailed reports.
The dense Orb head has no promising larger setting in the initial sweep; it needs no additional confirmation.

Keep batch 128 and the four-million budget for the current fixed-step comparisons.
For future runs measured by geometry exposure, the smaller context heads can use an eight-million budget.
The larger setting gives more geometries per second but takes roughly twice as long per optimizer update.
The wider heads gain too little throughput to justify their added memory and update cost.
A twelve-million budget causes out-of-memory failures in the wide heads.
It also underperforms eight million in both smaller context heads.

The sweep completes all five models and verifies unchanged frozen weights.
The six CPU benchmark-planning tests pass.
Private measurements and source hashes remain in the ignored `outputs/capacity_20261016` directory.

### Experimental packed mixed-size training

The packed experiment preserves the current equal-size implementation and its default settings.
`hip_orb.packed_training` concatenates coordinates and upper-triangle labels across geometries.
Atom counts and offsets define the separate molecular graphs and Hessian matrices.
The pair MLP evaluates all within-geometry pairs together. No atom or Hessian padding is needed.
Vectorized device indexing builds pair, block, label, and assembly indices from the size metadata.
The packed loss first averages blocks within each geometry, then averages geometries equally.
Translation projection uses segmented block reductions and stays separate for each geometry.
This experimental update supports single-view Block RMSE, MSE, and MAE.
It does not enable consistency loss or translation augmentation.

`hip_orb.benchmark_packed` compares four cases: current, packed equal-size, packed mixed-size, and a final current control.
The equal-size comparison isolates packed execution overhead on identical batches.
The mixed-size plan interleaves records within eight-shard locality windows.
It retains the molecule cap of 128 and the four-million-element budget, summed across the actual geometry sizes.
The plan is computed once before timing. Both plans visit exactly the same geometries once per pass.
Mixed batches change sample order and augmentation groups; the comparison does not establish training accuracy.
The saved optimizer and head reset before each repeat. The configured cosine schedule remains active.
Complete-update timing includes worker waits, rotations, graph construction, frozen inference, loss, backward, and optimizer updates.
Worker startup and the first prefetched source batch remain outside timing.
A separate audit checks exact graph membership and compares predictions, losses, and trainable gradients.
The audit uses deterministic highest float32 precision. Timings use the current high float32 precision.

Use deployment paths from the private environment:

```bash
python -m hip_orb.benchmark_packed \
  --index "$HIP_TRAIN_INDEX" --weights "$HIP_STEP_CHECKPOINT" \
  --config "$HIP_SPEED_CONFIG" --head-checkpoint "$HIP_SPEED_HEAD_CHECKPOINT" \
  --report "$HIP_SPEED_REPORT" --shards 96 --repeats 2
```

The default builds indices with vectorized GPU operations.
`--cpu-indices` instead builds full indices in loader workers and transfers them with the batch.
Run this bounded diagnostic in a GPU allocation. It does not save or replace a training checkpoint.

#### Packed benchmark results, 2026-10-16

Each model processes the same 19,696 geometries on one H100 with float32-high precision.
Each case has two repeats. The current rate averages the first and last controls.
Timings include the complete training update and measured loader waits.

The initial implementation builds vectorized device indices without retaining layouts between batches.
It uses the current 256 MiB array cache per worker and eight open files.

| Model | Current geometries/s | Packed equal-size | Packed mixed-size | Mixed change | Loader wait, current to mixed |
|---|---:|---:|---:|---:|---:|
| Context, MLP 1024, all pairs | 491.4 | 485.9 | 471.1 | -4.1% | 0.4% → 7.9% |
| Random Orb round, backbone graph | 539.4 | 532.8 | 514.2 | -4.7% | 0.5% → 9.4% |
| Random Orb round, Hessian graph | 377.7 | 382.0 | 377.4 | -0.1% | 0.1% → 4.8% |
| Pretrained Orb round and context | 771.1 | 761.8 | 709.2 | -8.0% | 0.6% → 15.8% |
| Pretrained Orb round, context, translation projection | 768.6 | 704.3 | 648.2 | -15.7% | 0.3% → 14.6% |

The follow-up caches eight device layouts and uses 2 GiB of loader array cache per worker, with 32 open files.
Both the current and packed paths use the same larger loader cache in these comparisons.
The projected head also uses segmented projection reductions and a custom self-adjoint backward.
Its backward applies the same linear operation, `P dH P`, to the output gradient.

| Model | Current geometries/s | Packed equal-size | Packed mixed-size | Mixed change | Loader wait, current to mixed |
|---|---:|---:|---:|---:|---:|
| Context, MLP 1024, all pairs | 488.0 | 488.9 | 469.5 | -3.8% | 0.5% → 8.1% |
| Pretrained Orb round and context | 767.0 | 763.3 | 704.7 | -8.1% | 0.4% → 15.9% |
| Pretrained Orb round, context, translation projection | 768.4 | 761.8 | 712.7 | -7.2% | 0.6% → 15.0% |

The larger loader cache does not remove the mixed-size penalty.
Segmented projection removes the earlier projected-head overhead on equal-size batches.
The projected-head comparison changes both projection execution and index caching; it is not a separate projection-only ablation.

The update path excludes only the explicitly measured loader wait.
It still includes CPU preparation, transfers, frozen backbone inference, the head, backward, and optimizer operations.
This calculation is not a separate GPU-only throughput measurement.

| Selected implementation | Update-path throughput change without loader wait | Current ms/update | Mixed ms/update | Peak allocated GiB, current to mixed |
|---|---:|---:|---:|---:|
| Context, MLP 1024, all pairs | +4.1% | 105.9 | 146.7 | 33.7 → 34.4 |
| Random Orb round, backbone graph | +4.7% | 95.8 | 133.9 | 29.9 → 29.6 |
| Random Orb round, Hessian graph | +4.8% | 136.9 | 182.5 | 43.3 → 43.5 |
| Pretrained Orb round and context | +8.8% | 67.4 | 97.7 | 17.7 → 18.1 |
| Pretrained Orb round, context, translation projection | +8.4% | 67.3 | 96.6 | 17.7 → 18.1 |

Mixed packing reduces this cohort from 381 to 286 updates. Mean batch size rises from 51.70 to 68.87 geometries.
Mean use of the Hessian-element budget rises from 61.1% to 81.4%.
The measured update path benefits, but loader waits offset this benefit.
Mixed plans read several source shards per batch; current batches read one shard.
Further work should test I/O locality and prefetch depth.

Metadata planning for all ten million geometries takes 2.26 seconds.
The saved 32-bit index plan and batch offsets use 40.49 MB.
It contains 122,439 batches, with a mean of 81.67 geometries and a maximum of 3,999,996 Hessian elements.
The private plan visits each geometry exactly once. It is not active in training.
A future training sampler must shuffle both batch order and within-batch order each epoch.
This keeps batch-identity-based rotation seeds from repeating unchanged.

The final GPU audit checks all five heads at three size combinations: `(70, 3, 20)`, `(20, 350, 3, 70)`, and `(350,)`.
All 15 checks pass exact native graph agreement and the recorded prediction, loss, and gradient tolerances.
The maximum trainable-gradient difference is 9.4780698e-06.
Frozen backbone states remain unchanged, and no backbone parameter receives a gradient.
The tests establish numerical agreement for these cases. They do not establish accuracy after training with mixed-size batches.

Keep the current training defaults. The packed path remains an explicit experimental implementation.
Private reports, preserved source snapshots, and batch plans stay outside published model artifacts.

### Atom-balanced packed batches

`hip_orb.balanced_sampler.BalancedBatchSampler` builds a reusable batch plan from
`OrbMolDataset.ptr` and `OrbMolDataset.natoms`. It never loads geometry data to
count atoms. No additional metadata file is required.

This sampler balances work between batches on one GPU. FAIRChem instead
redistributes samples between distributed ranks. This implementation does not
support distributed training.

Each batch obeys three limits:

- `max_atoms`: total atoms.
- `batch_size`: maximum geometries, rather than a fixed geometry count.
- `max_hessian_elements`: sum of `9 * natoms**2` across geometries.

The planner shuffles shards and rows with a fixed seed. Within each shard
window, it places large geometries first. It fills remaining atom capacity with
smaller geometries when the other limits permit. Shard windows restrict source
mixing. They can leave partial batches. Smaller windows favor file locality;
larger windows offer more packing choices.

Membership stays fixed. Each epoch shuffles batch order and row order. Every
geometry appears exactly once per complete epoch. There is no padding,
oversampling, or dropped remainder. An oversized geometry raises an error.

Use the sampler with packed collation and `packed_update`:

```python
from hip_orb.balanced_sampler import BalancedBatchSampler
from hip_orb.data import make_loader
from hip_orb.packed_training import collate_packed_compact

sampler = BalancedBatchSampler(
    dataset,
    max_atoms=4096,                 # Starting point; benchmark before adoption.
    batch_size=128,
    max_hessian_elements=4_000_000,
    window_shards=8,
    seed=7,
    cache_path=plan_path,            # Resolve deployment paths from .env.
)
loader = make_loader(
    dataset,
    batch_sampler=sampler,
    collate_fn=collate_packed_compact,
    workers=4,
)
loader.batch_sampler.set_epoch(epoch)
```

`cache_path` is optional. Without it, construction builds an in-memory plan.
With it, the sampler saves or loads a compact NPZ plan. Cache validation checks
source paths, atom counts, shard boundaries, planning settings, coverage, and
budgets. A mismatch raises an error. Choose another path or remove the stale
plan to rebuild it. Use the same metadata and settings on resume. Existing
`start_batch` support skips batches without reading their geometries.

The atom limit does not replace the quadratic Hessian limit. Neither limit
bounds every GPU allocation. Measure memory with the actual model. Atom balance
alone does not guarantee faster loading or equal update time.

The production runner retains equal-size batches. This API is opt-in for the
packed training path. Existing training jobs and default settings stay unchanged.

Local metadata validation used all 10,000,000 training geometries. With the
example limits, planning took 79 seconds and cache loading took 0.28 seconds.
The plan contains 148,437 batches, with exact coverage. Atom-count variation
(standard deviation divided by mean) fell from 0.67 for current batches to
0.36. The new 4,096-atom cap contributes to that reduction. About 69% of batches
mix atom counts. These measurements do not establish GPU throughput.

The HIP CPU suite passed: 298 tests passed and 26 tests skipped.

The training runner also accepts `--batching balanced`, `--batch-max-atoms`,
and `--batch-window-shards`. It saves the plan in the run output directory.
Validation keeps its existing equal-size batches. Checkpoints record the plan
fingerprint and reject a different sampler on resume. Packed training supports
one uniform rotation view and Block RMSE, MSE, or MAE. It rejects unsupported
augmentation and loss settings before model loading.

`configs/train_2m_context_mlp1024_all_balanced_lr_1e-3.json` defines the first
training comparison. It preserves the context-width-1024 all-pairs settings,
with a 4,096-atom cap, 128-geometry cap, and four-million-element Hessian budget.
`scripts/hip_train_balanced.sbatch` checks two updates and ten validation
geometries before it starts a separate full run from initialization. The full
run uses the standard checkpoint continuation chain.

### Balanced training throughput changes, 2026-10-16

New balanced samplers default to a 64-geometry cap and no Hessian-element cap.
Use `max_hessian_elements=None` in Python or `null` in JSON. The atom budget
still controls total atoms. Existing run configurations retain their recorded
limits. The four-million-element comparison remains a separate experiment.

The new default order is `locality`. Each epoch shuffles shard-window groups,
then batches within each group, then rows within each batch. This keeps nearby
updates within a small set of source files. `order="global"` retains the earlier
global batch shuffle for comparisons. Both orders cover every geometry exactly
once. The order change affects the optimizer trajectory.

Plans now include `group_ptr` and a versioned fingerprint. An older plan must
use its original runtime or a separate output directory. The loader does not
silently replace a plan during checkpoint resume.

`OrbMolDataset.__getitems__` now reads consecutive geometry rows together.
It preserves input order and duplicates. The array-cache path gathers complete
rows without per-sample HDF5 calls. Cache eviction does not change returned data.

The runner exposes `loader_cache_mib`, `loader_open_files`, `prefetch_factor`,
and `workers`. Each progress record reports measured loader wait, geometries
per second, updates per second, and mean geometry count. These counters exclude
validation. They include the measured loader and training update time.

Use `HIP_SLURM_CPUS_PER_TASK` in `.env` to reserve CPUs for loader workers.
The submission script passes this setting to checkpoint continuations.

#### Selected defaults and measured results

`configs/train.json` now selects balanced batches with at most 64 geometries
and 2,560 atoms. It disables the Hessian-element cap. It preserves the existing
model architecture and optimizer schedule. Four persistent loader workers use
prefetch depth four, 1 GiB of array cache each, and 16 open files each.
Reserve eight CPUs through the private environment. Validation retains its
existing geometry count, batch cap, and Hessian budget.

The wide all-pairs context-head variant uses the same loader settings in
`configs/train_2m_context_mlp1024_all_balanced_fast_lr_1e-3.json`.
The earlier balanced comparison file retains its original limits.

The matched test uses one H100, 8,831 identical geometries, two repetitions,
and the same model and optimizer checkpoint. Each case verifies exact sample
coverage. Backbone features are recomputed for every update. Timings include
source reads and complete updates. Worker startup and initial warmup are excluded.

| Setting | Geometries/s | ms/update | Loader wait | Peak allocated GiB |
|---|---:|---:|---:|---:|
| Earlier balanced 128 / 4M, initial control | 399.5 | 130.3 | 1.43% | 34.3 |
| Uncapped 64 / 2,048 atoms, two workers, locality | 380.7 | 77.6 | 2.78% | 51.0 |
| Same plan, four workers and prefetch four | 385.8 | 76.6 | 0.22% | 51.0 |
| Same plan, eight workers | 390.6 | 75.6 | 0.22% | 51.0 |
| Uncapped 32 / 2,048 atoms, four workers | 378.4 | 64.8 | 0.23% | 51.0 |
| Uncapped 128 / 2,048 atoms, four workers | 390.4 | 79.9 | 0.18% | 50.9 |
| **Selected: uncapped 64 / 2,560 atoms, four workers** | **403.3** | **87.6** | **0.16%** | **65.0** |
| Earlier balanced 128 / 4M, final control | 415.9 | 124.9 | 0.38% | 34.3 |

The selected setting reduces update duration by 31% against the mean control.
It processes fewer geometries per update: 35.3 versus 51.9 on this cohort.
Geometry throughput is 1.1% lower than the mean control. This difference lies
within the observed control drift. Shorter updates do not establish faster
convergence or better accuracy. Loader waits are effectively removed in this
test. Graph preparation and model updates remain in the timed path.

Eight workers do not reduce loader waits further. The 128-geometry alternative
gains only 1.2% over the corresponding 64-geometry setting at 2,048 atoms.
The selected 2,560-atom limit gives 4.5% more geometries per second than the
2,048-atom limit with the same worker settings.

At 350 atoms, the wide head needs 46.3 GiB for five geometries and 64.6 GiB
for seven. Eight geometries need 73.8 GiB, which fails the benchmark's memory
margin. Eleven geometries exceed GPU memory. Therefore, removing the Hessian
cap does not permit arbitrary atom counts. These probes apply to the tested
wide head. Other architectures need their own capacity checks.

The full ten-million-geometry plan builds once in about 51 seconds. It contains
229,932 batches and averages 43.5 geometries per batch. Its largest Hessian
contains 7,822,476 elements. About 7.6% of batches exceed four million elements.
The plan preserves exact coverage. The loader settings do not change validation.

Validation passed: 302 CPU tests passed and 26 tests skipped. The completed GPU
sweep covers all nine successful streaming cases. The first benchmark attempt
failed during reference-loader worker startup; the repaired attempt completed.
Private reports remain in `outputs/balanced_speed_20261016`.

See [PRECISION.md](PRECISION.md) for the later BF16, FP16, and native FP8 experiments.
Those experiments include backbone checks, backward audits, and streaming measurements.

The later mixed-precision comparison retains the original 128-geometry plan and
four-million-element cap. It uses four workers, prefetch depth four, and direct
chunk reads. See [SPEED.md](SPEED.md) for the complete execution changes and checks.
