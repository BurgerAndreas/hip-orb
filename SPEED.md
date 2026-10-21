# Training speed changes

The selected configuration is
`configs/train_2m_context_mlp1024_all_balanced_fp16_lr_1e-3.json`.
It preserves the original balanced plan, global sample order, batch limits,
model weights, loss definition, learning rate schedule, and two-million-step target.

## Selected changes

1. Use four loader workers and prefetch depth four. Reserve eight CPUs.
2. Read HDF5 chunks directly. Keep sixteen open files per worker.
3. Convert and pad linear inputs with one kernel to a multiple of sixteen channels.
   The 791-channel readout then uses faster matrix kernels.
   Padding adds no model parameters. Backward removes the padded channels.
4. Compute input gradients only for the radial feature block when the other
   feature inputs are frozen. Requests for feature or coordinate gradients
   disable this shortcut automatically.
5. Use fused AdamW with FP32 parameters and optimizer states.
   Clear trainable precision caches after every update. This is required because
   fused AdamW does not increment parameter version counters in this environment.
6. Evaluate the supervised loss on independent pair blocks and diagonal blocks.
   Off-diagonal blocks have weight two. The mathematical loss remains unchanged.
   This path requires balanced batches without translation projection.
7. Fuse frozen pair-feature gathers and input concatenation. Return radial gradients
   directly, without constructing a full frozen-input gradient buffer.

The precision policy remains `fp16-message-head`. The frozen backbone message
MLPs use FP16. Most head matrix products use FP16 operands and FP32 outputs.
The context node-update, pair-update, and diagonal-readout MLPs use FP32 to prevent activation overflow.
Normalization, residual paths, loss calculations, and optimizer states retain FP32.

## Verification

CPU tests compare dense and independent-block losses for MSE, MAE, and block RMSE.
They check values and all head parameter gradients, including trainable radial bases.
Tests include single-atom cases and mixed atom counts.

GPU checks cover padded forward operations, input gradients, parameter gradients,
and fused optimizer cache refresh. Real-model audits use identical frozen features
through 350 atoms. They compare predictions and all head parameter gradients.
The largest measured aggregate gradient difference for the selected block path
was below 0.06%. The frozen Orb state stayed unchanged.

A production runner test resumes the existing optimizer and gradient scaler.
It trains, validates, saves, and resumes again. It verifies unchanged batch-plan
and learning-rate metadata, four successful updates, and preserved scaler state.

## Deployment

Write a complete immutable snapshot path into `runtime_next.txt` inside the
private training output directory. Replace this pointer atomically.
`hip_train_steps.sbatch` reads it at every checkpoint continuation.
The pointer works even when a parent job exports earlier environment values.
Existing allocations keep their loaded runtime.

The mixed-precision comparison receives these changes. A later loader-only rollout
also updates the four FP32 comparisons at their next checkpoint continuation.
Those comparisons retain FP32 arithmetic and their original sampler versions.
No training checkpoint is reset.

## Rejected changes

Combining both pair directions gave little improvement. A custom context-pooling
kernel did not improve complete-step time beyond the simpler optimizer change.
These candidates remain outside the selected implementation.

FP32 input padding improved the bounded context benchmark by about 4%.
It remains a separate experiment. It does not justify changing the control runs
within this comparison.

Initial fused-optimizer timings preceded the cache-invalidation check.
Those timings do not establish correct training performance. The final comparison
uses explicit invalidation and a verified production restart.

## Final matched comparison

The test uses the same 512 production batches, the same initial head weights,
and two repetitions per case. The first eight updates remain outside timing.
Each timed repetition contains 33,881 geometries and 504 complete updates.
Source reads, backbone inference, backward, and optimizer updates remain timed.
The selected optimizer uses the verified cache invalidation fix.

| Settings | ms/update | Loader wait | Peak allocated GiB |
|---|---:|---:|---:|
| Previous settings, initial and final controls | 101.1 | 14.69% | 26.76 |
| Four workers, array cache | 83.9 | 0.26% | 26.76 |
| Four workers, chunk reads | 82.8 | 0.22% | 26.76 |
| Selected loader and execution changes | **72.3** | **0.24%** | **26.02** |

The selected configuration increases average throughput by **40%**. It reduces
update time by **28.5%**. Storage-cache state affects the controls: individual
control means range from 91.5 to 125.9 ms. The selected repetitions take 71.4
and 73.1 ms. The comparison does not change the number of geometries per update.

Verification completed: **324 CPU tests passed**, with 36 CUDA tests skipped
on the laptop. The release suite passed **20 tests on an H100**, including cache
invalidation and low-precision backward checks. The production restart test
also passed. The selected run resumed from **200,000 steps** on one H100.

The live run reached 206,000 steps after deployment. Its latest windows report
12.6–13.0 updates/s and 0.28% loader wait. The last window before deployment
reported 6.83 updates/s and 39.9% loader wait. These windows contain different
samples. They verify the rollout but do not replace the matched comparison.

## FP32 loader continuation checks

All four FP32 model variants passed a restart check with four workers,
prefetch depth four, chunk reads, and sixteen open shard files per worker.
Each check resumed an existing checkpoint and trained for two updates.
It validated sixteen geometries and saved a readable checkpoint.
The checks confirmed unchanged precision, loss, architecture, augmentation,
learning-rate schedule, and batch metadata. Optimizer counters advanced twice.
The frozen backbone stayed unchanged.

The deployment uses separate immutable runtime copies for equal-size and legacy
balanced training. Only their runner loader options and prefetch forwarding differ.
Sampler code and batch-plan files remain unchanged. Continuations request eight CPUs.
The source configurations record the loader settings without private deployment values.

## Updated profile

A new profile uses the deployed mixed-precision execution, including aligned
inputs, radial-only backward, fused AdamW, independent-block loss, and the tuned loader.
The streaming check processes 8,157 timed geometries in 120 updates.
It measures 78.0 ms/update and 0.40% loader wait on this sample sequence.
The separate trace covers sixteen complete updates after warmup.

Actual GPU kernel durations divide as follows. These shares exclude CPU work,
launch gaps, and profiling overhead; they do not measure hardware occupancy.

| Kernel group | Share of summed kernel duration |
|---|---:|
| Matrix multiplication | 37.8% |
| Fused SiLU forward and backward | 16.7% |
| FP32-to-FP16 conversion kernels | 10.9% |
| Concatenation kernels | 4.1% |
| Other kernels | 30.5% |

Stage annotations overlap other events. Do not sum annotation durations with
individual kernel durations. The measured kernel shares identify further fusion targets.

## Further head fusion

The optional `--head-input-fusion` setting accepts `none`, `cast`, `pairs`, or `both`.
The default remains `none`. The runner records the setting in execution history.
These candidates require the balanced mixed-precision training path.

The cast kernel combines conversion and zero padding in one pass.
The pair kernel combines frozen node and edge gathers, geometric inputs,
charges, and radial embedding concatenation. It preserves the radial gradient.
Requests for trainable node, edge, charge, or directional inputs use the existing path.
Unsupported feature layouts and dtypes also use the existing path.

When radial-only backward is active, the pair path connects the radial tensor
directly to the first fused linear operation. Its backward returns seventeen
radial channels without allocating a full readout-input gradient buffer.
It preserves the two directed contributions to the shared radial embedding.

A separate custom matrix-plus-SiLU kernel passed value and gradient checks,
but slowed complete updates to about 84.8 ms versus 74.9 ms for native controls.
That kernel remains outside the production implementation.

The final candidate suite passed 32 tests on a GPU node, including trainable-feature
fallbacks and empty pair sets. Real-model comparisons use identical frozen features
for geometries through 350 atoms. The largest aggregate head-gradient difference
was below 0.002%, relative to the current mixed-precision baseline. Prediction
relative differences stayed below 0.003%. The frozen backbone state stayed unchanged.
The CPU suite passed 324 tests and skipped 48 GPU or optional checks.

## Confirmation and continuation settings

A larger matched comparison starts from one fixed checkpoint and uses 512 production
batches per setting. Each of two timed repeats covers 35,946 geometries in 504 updates.
Eight initial updates stay outside timing. Initial and final native controls bound drift.
The comparison includes source reads and complete optimizer updates.
Every case receives the same batch sequence and initial head parameters.

| Execution | Mean ms/update | Peak allocated GiB |
|---|---:|---:|
| Native controls, initial and final | 70.13 | 26.01 |
| Combined input fusion and direct radial backward | **66.57** | **26.01** |

The combined path increases throughput by **5.35%** and reduces update time by **5.07%**.
Its two repeats take 66.58 and 66.56 ms/update. Native controls range from 69.87 to
70.35 ms/update. Loader wait stays below 0.35%, and no optimizer update is skipped.
These measurements use their own controls; do not compare their absolute times
with earlier sample sequences or GPU allocations.

Two production-runner sessions resume a copied checkpoint, train, validate, save,
and resume again. All restart checks pass. Four successful updates preserve the
optimizer counters, gradient-scaler state, learning-rate schedule, batch-plan metadata,
and frozen backbone state. Each session validates sixteen geometries for this check.
Production validation retains its normal ten-thousand-geometry limit and cadence.

The mixed-precision configuration enables `head_input_fusion: both`.
Its private runtime pointer selects the tested immutable package at the next
checkpoint continuation. Existing allocations keep their loaded code and arguments.
The four FP32 comparisons receive loader changes only.

## Numerical failure diagnostics

Long training can expose ranges that short performance tests do not cover.
The packed runner now checks the loss on the host before backward.
A nonfinite loss raises `FloatingPointError` instead of a CUDA device assertion.
The runner saves its checkpoint and `nonfinite_batch.pt` before it stops.
The saved batch retains its augmentation seed and successful update count.
Resume starts at that batch and preserves optimizer and gradient-scaler states.

`precision_overrides` maps linear layer patterns to `LinearPolicy` settings.
An override changes only the selected matrix operation and its activation policy.
An explicit FP32 override converts its input to FP32 before multiplication.
This permits an FP32 boundary after a half-precision activation.
Unmatched and overlapping patterns raise errors before training starts.
The runner records overrides in its execution history.

The selected production overrides are:

```json
{
  "precision_overrides": {
    "head.node_update.*": {"dtype": "fp32"},
    "head.pair_update.*": {"dtype": "fp32"},
    "head.node_head.[02]": {"dtype": "fp32"}
  }
}
```

Tests check half-input boundaries, forward values, parameter gradients, and scope restoration.
GPU regression tests also cover activation values above the FP16 range.

### Context readout overflow correction

A captured production failure keeps finite backbone features and model parameters.
The context update produces node features near 50,600.
The pair-update and diagonal-readout MLPs then overflow their FP16 activations.
Disabling reduced-precision reductions does not correct the failure.
Changing only the backbone precision also leaves the failure unchanged.

The first correction used FP32 for five matrix operations across these two MLPs:
`head.pair_update.{0,2,4}` and `head.node_head.{0,2}`.
These branches also keep their SiLU activations in FP32.
Other message and head matrix operations retain their existing precision policies.
The model architecture, loss, augmentation, batch plan, and learning-rate schedule remain unchanged.

On the captured batch, the corrected prediction differs from the FP32 baseline by 0.088% in relative L2.
The aggregate parameter-gradient difference is 0.117%, and all gradients remain finite.
The remaining difference includes unchanged FP16 operations outside the corrected branches.

Power-of-two row scaling also corrects the failure while retaining FP16 matrix products.
Its gradient difference is 0.115%, but its extra conversions increase step time.
An early timing comparison omitted GPU completion waits after backward.
Its timing values do not establish complete-step performance.
The corrected comparison synchronizes CUDA before stopping each step timer.
Its measured peak allocated GPU memory increases from 26.00 GiB to 27.57 GiB.
BF16 also prevents overflow, but its prediction and gradient errors are larger.

The selected fix passes the captured failing batch and 1,504 further successful updates.
Four small validation checks and two checkpoint-resume checks also pass.
The checks preserve optimizer state, gradient-scaler state, batch-plan metadata,
and the original learning-rate schedule. Production validation still uses 10,000 geometries.

### Complete context precision boundary

A later captured failure occurs at successful update 297,201.
All inputs, head parameters, and optimizer states remain finite.
The first invalid value appears in `head.node_update.2`, before backward starts.
The FP32 reference produces a positive activation near 66,374.
The FP16 maximum is 65,504, so storing this activation in FP16 produces infinity.
The next context matrix produces values near three million in the FP32 reference.
Changing only the failing activation still permits overflow at the next FP16 input cast.

The corrected boundary therefore keeps the entire context node-update MLP in FP32.
It retains the prior FP32 pair-update and diagonal-readout branches.
The eight explicit overrides cover `head.node_update.{0,2,4}`,
`head.pair_update.{0,2,4}`, and `head.node_head.{0,2}`.
Final block output layers already use FP32.
Other head and backbone precision settings remain unchanged.
The correction does not add normalization or change the architecture.

The diagnostic replay checks the same saved geometry batch and rotation seed.
It compares the current policy, FP32 backbone, FP32 head, and complete FP32 reference.
A backbone-only precision change does not correct the failure.
The corrected context boundary produces finite forward values and parameter gradients.
A regression test covers the complete precision boundary and its backward gradients.

A bounded training audit records layer maxima and RMS values every 100 updates.
It also records unscaled parameter gradients before clipping and optimizer update norms.
These measurements distinguish FP16 range failures from optimizer or weight divergence.
Loss scaling and gradient clipping operate after forward evaluation and cannot prevent this overflow.
Timing profiles remain useful for performance; value and gradient hooks locate numerical failures.

The production throughput counter now uses complete window wall time.
It includes CUDA completion waits and work outside the update function.
A regression test includes two seconds of loop work and one second of update work.
The reported rate must be one update per three seconds.

The complete-context audit trains 6,000 successful updates, from step 297,201 to 303,201.
It crosses the peak learning rate at step 300,000 and processes 400,832 geometries.
The audit records 61 snapshots, including the failing batch and one snapshot per 100 updates.
All sampled weights, activations, gradients, and optimizer updates remain finite.
The global parameter L2 norm increases by 2.51% across the audit.
The largest sampled unscaled parameter-gradient norm is 6.68.
The largest sampled update-to-parameter L2 ratio is 0.00292.
Remaining FP16 head activations reach 302; frozen FP16 backbone MLP outputs reach 485.5.
The corrected FP32 context and readout branches handle values above the FP16 range.
Two gradient-scaler overflows skip optimizer updates safely; no nonfinite forward loss occurs.

The synchronized comparison uses the same 96 batches and two measured repetitions after warmup:

| Precision boundary | Complete step time | Peak allocated GPU memory |
| --- | ---: | ---: |
| Previous five-operation correction | 78.95 ms | 27.80 GiB |
| Complete context correction | 79.21 ms | 27.57 GiB |
| Entire head in FP32 | 104.42 ms | 33.70 GiB |

The complete correction differs by 0.3% from the previous boundary, within repeat variation.
The entire-head FP32 alternative increases step time by about 32%.
These complete-step measurements replace the earlier asynchronous timing estimates.

The final runner check validates, saves, and resumes twice after the monitored audit.
It retains the existing optimizer, scaler, batch plan, and learning-rate schedule.
Diagnostic validation records stay outside the production validation history.
Production resumes from the verified checkpoint with the complete context precision boundary.

## Dense H100 training, 21 October 2026

The dense Bessel and smooth radial runs use four further execution changes.
These changes preserve model parameters, source batches, training loss, and the learning rate schedule.

- Fused RMS normalization computes each row and its backward reduction in FP32.
- Internal zero padding aligns irregular linear input widths to sixteen channels.
- Direct index calculation replaces sorting and searching for canonical dense molecular edges.
- Vectorized topology construction removes per-molecule GPU launches for new compact batch layouts.
- The existing independent pair-block loss now supports dense Orb extensions without translation projection.

The kernels preserve the existing `float32-high` matrix precision setting.
Parameters, optimizer states, normalization, and output storage remain FP32.
Internal padding does not add trainable parameters or change checkpoint tensor shapes.
The native fallback supports other devices, dtypes, and input layouts.
The custom backward operations retain native paths for higher derivatives.

`configure_execution` enables these changes in optimized mode.
Its `fused_norm=False` and `aligned_inputs=False` options permit separate kernel comparisons.
Eager mode restores the original module methods.
Execution metadata records the selected kernels in each checkpoint.

GPU tests compare normalization values and gradients across empty, irregular, and large batches.
They also check extreme input ranges and higher derivatives.
Linear tests check irregular widths, empty batches, and parameter gradients.
CPU tests check dense edge lookup, encoder gradients, and independent-block loss equivalence.

The initial bounded tests compare mature checkpoints on identical batches through 350 atoms.
They include complete forward, backward, clipping, and optimizer operations.
Initial and final native controls measure timing variation.
These tests show 10–17% higher throughput on medium and large batches.
Small batches show little improvement.

A separate FP16 candidate retains FP32 outputs and uses loss scaling.
It improves large-batch speed, but its gradient differences exceed the native repeat variation.
It remains outside the selected runtime.

The final streaming audit uses 128 production batches per setting.
The first eight updates remain outside timing.
Each selected measurement includes 120 complete updates and source reads.
Initial and final native controls use the same batch sequence, checkpoint, optimizer state, and learning rate schedule.
Batch identity hashes must agree across all three measurements.
The audit computes fresh backbone features for every update.

Each architecture also resumes twice in a private output directory.
Four successful updates must preserve checkpoint architecture, precision, batch metadata, optimizer counters, and schedule metadata.
Each validation check uses sixteen geometries.
These diagnostic checks do not change production validation records or limits.

The largest batch check uses five geometries with 350 atoms each.
All thirteen architectures produce finite losses and gradients with the selected runtime.
The highest measured allocation is 69.9 GiB.
The frozen backbone weights remain unchanged.

Deployment changes each run's `runtime_next.txt` pointer after all checks pass.
Each checkpoint continuation verifies the immutable package manifest before training starts.
Existing allocations keep their loaded runtime until their normal checkpoint continuation.
The deployment record preserves every previous runtime path for rollback.

### Final matched results for all thirteen settings

All thirteen settings pass both checkpoint resume checks and the largest batch check.
The local suite passes 384 tests. The H100 suite passes 144 tests.

| Setting | Native ms/update | Selected ms/update | Throughput gain |
| --- | ---: | ---: | ---: |
| Bessel control | 76.94 | 72.75 | 5.8% |
| Bessel + both adapters | 99.11 | 93.47 | 6.0% |
| Smooth radial baseline | 76.36 | 69.94 | 9.2% |
| Smooth radial + shared geometry | 77.58 | 69.39 | 11.8% |
| Smooth radial + combined changes | 103.99 | 91.51 | 13.6% |
| Bessel + edge adapter | 97.18 | 84.56 | 14.9% |
| Bessel + atom adapter | 75.75 | 67.25 | 12.6% |
| Smooth radial + edge reuse | 82.25 | 74.08 | 11.0% |
| Smooth radial + layer mixture | 80.11 | 70.57 | 13.5% |
| Bessel + pair width 512 | 89.41 | 70.58 | 26.7% |
| Bessel + both adapters + pair width 512 | 104.73 | 93.00 | 12.6% |
| Smooth radial + readout context | 93.51 | 87.96 | 6.3% |
| Smooth radial + translation projection | 76.43 | 70.47 | 8.5% |

Median throughput increases by 11.8%.
The largest relative prediction difference is 0.0437% across 65 fixed validation geometries.
These checks compare numerical predictions. They do not measure final training accuracy after convergence.

All thirteen runtime pointers select the verified package at the next normal checkpoint continuation.
Model weights, optimizer history, batch plans, and the two-million-step target remain unchanged.

## Further H100 optimization, 21 October 2026

This update compares against the optimized runtime from the preceding dense H100 audit.
All comparisons use unchanged checkpoints, source batches, optimizer states, and learning rate schedules.

Custom kernels pack dense pairs and messages directly.
They also compute gated neighbor sums and feature gradients without atomic additions.
These kernels retain FP32 arithmetic.

Large MLPs use FP16 matrix operands and FP32 accumulation.
Each backward operation scales its gradient by a local power of two before FP16 conversion.
This preserves small gradients that the earlier FP16 candidate lost.
Parameters, optimizer states, returned gradients, normalization, residuals, geometry, and losses retain FP32.
The final nine-channel Hessian projection retains FP32.

The MLP kernel stores internal activations and saved preactivations in FP16.
These intermediate tensors stay inside one custom backward operation.
Small MLP inputs retain the FP32 path below 16,384 rows.
The frozen backbone message MLPs also use FP16 in the eleven optimized settings.
The execution metadata records all precision boundaries.

Lower activation memory permits the edge adapter to avoid recomputation through 650,000 edges.
Larger edge inputs retain recomputation.
The tested large batch contains 610,750 directed edges.

`configure_execution(model, speed_features=False)` restores the preceding optimized execution path.
Eager mode restores native module methods.
Native higher derivative paths remain available.

The layer-mixture and combined settings retain the preceding FP32 execution path.
This protects sensitive vibration modes.
The other eleven settings use the new kernels.

### Matched training results

Each measurement includes 120 complete optimizer updates after eight warmup updates.
Initial and final controls use the same batches and checkpoint.
Batch identity hashes agree for all thirteen settings.

| Setting | Previous ms/update | Selected ms/update | Throughput gain |
| --- | ---: | ---: | ---: |
| Bessel control | 56.37 | 47.25 | 19.3% |
| Bessel + both adapters | 87.41 | 66.34 | 31.8% |
| Smooth radial baseline | 67.68 | 55.34 | 22.3% |
| Smooth radial + shared geometry | 48.56 | 44.74 | 8.5% |
| Bessel + edge adapter | 87.14 | 65.81 | 32.4% |
| Bessel + atom adapter | 52.03 | 43.86 | 18.6% |
| Smooth radial + edge reuse | 46.89 | 39.13 | 19.8% |
| Bessel + pair width 512 | 72.04 | 59.23 | 21.6% |
| Bessel + both adapters + pair width 512 | 91.49 | 72.19 | 26.7% |
| Smooth radial + readout context | 88.32 | 66.08 | 33.7% |
| Smooth radial + translation projection | 47.91 | 39.14 | 22.4% |
| Smooth radial + layer mixture | 57.95 | 57.23 | unchanged |
| Smooth radial + combined changes | 81.82 | 81.39 | unchanged |

Median throughput increases by 22.3% across the eleven optimized settings.
Gains range from 8.5% to 33.7%.

### Numerical and restart checks

All thirteen settings resume twice and complete four optimizer updates.
Architecture, batch plans, optimizer counters, loss settings, and schedules remain compatible.
The two-million-update target stays unchanged.

All thirteen settings pass the batch test with five 350-atom geometries.
Peak allocation is 69.85 GiB across all thirteen settings.
The frozen backbone weights stay unchanged.

The largest relative prediction difference is 0.0415% across 65 fixed validation geometries.
An additional audit compares eighty validation geometries per setting, with initial and final FP32 controls.
The largest absolute relative change in mean Hessian MAE is 0.0580%.
The largest absolute relative change in eigenvalue MAE is 0.1568%.
This cohort spans small molecules through 350 atoms.
These checks measure numerical agreement. They do not establish equal accuracy after convergence.

Three separate training audits each complete 1,536 updates.
They process 153,392 geometries in total.
All sampled losses, gradients, parameters, and optimizer states stay finite.
No backbone weights change.

CPU tests pass 392 cases. H100 kernel tests pass 143 cases.
The kernel tests cover empty batches, irregular widths, higher derivatives, and gradients down to 10⁻¹⁸.

### Deployment

Each run selects the verified immutable runtime at its next normal checkpoint continuation.
Existing allocations keep their loaded runtime until then.
The deployment record stores every preceding runtime path for rollback.

Local evidence: `cluster_runs/hip_training_speed_round2_20261021/summary.json`.
Full reports include matched batches, validation metrics, restart checks, capacity checks, and training audits.
Selected runtime: `/lustre/fs1/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip-uma/hf_release_tools/training_speed_round2_20261021/runtime_r2_v4`.

## Third H100 speed update: 21 October 2026

This update compares against `runtime_r2_v4` from the preceding round.
Each comparison uses the same checkpoint, optimizer state, schedule, and ordered batches.
Batch identity hashes match in every comparison.

The layer-mixture head keeps its original FP32 forward operations.
Its large MLP backward operations use locally scaled FP16 matrix operands with FP32 accumulation.
The combined head also keeps its FP32 forward operations.
It keeps activation gradients in FP32 and uses scaled FP16 only for weight-gradient matrix products.
Parameters, optimizer states, returned gradients, losses, normalization, and geometry stay in FP32.
Inputs below 16,384 rows use the preceding path.
Native higher derivatives remain available.
For a second-round control, set `backward_fp16`, `shared_projection`, and `fused_reductions` to `False`.
The combined gain is small and remains within short benchmark variation.

A fused reduction kernel reads each output gradient once for its maximum and bias sum.
This removes the full gradient absolute-value tensor and separate full-gradient bias reduction.
Hidden-layer kernels combine activation derivatives, scale restoration, maxima, and bias reductions.

The shared-geometry head projects complete geometry features before it gathers both pair directions.
This replaces two 256-channel feature gathers with two 17-channel gathers.
The gradient paths still reach both uses of the shared encoder.
The large resident batch saves about 0.58 GiB.
Incomplete edge graphs retain the preceding gather order.
FP64 tests compare predictions and every parameter gradient for both graph cases.

Validation batches compatible eigendecompositions by vibrational rank.
It retains each original SVD, projection, and subspace dot product.
The initial fully batched projection changed sensitive mode statistics and was rejected.
A custom FP32 kernel rotates each 3 by 3 Hessian block directly.
It avoids the dense Kronecker matrix and its rounding error.
The block kernel agrees with an FP64 reference.
For five 350-atom geometries, its measured transform is about thirteen times faster.
This ratio describes the rotation operation. It does not describe complete validation.

Bounded CUDA graph replay passes changed-input correctness tests.
Repeated resident small batches improve, but capture adds a startup cost.
Two ordered tests each contain 1,032 real batches and produce no capture reuse.
Graph capture stays disabled by default.
Optional capture limits the graph count, tensor sizes, metadata cache, and capture attempts.
The batch sequence stays unchanged.

### Matched training measurements

Each row includes 120 complete updates after eight warmup updates.
Initial and final controls bracket each selected measurement.

| Setting | Previous ms/update | Selected ms/update | Throughput gain |
| --- | ---: | ---: | ---: |
| Bessel control | 46.57 | 43.67 | 6.65% |
| Bessel + both adapters | 68.71 | 65.92 | 4.22% |
| Smooth radial baseline | 47.43 | 45.46 | 4.32% |
| Smooth + shared geometry | 50.84 | 48.39 | 5.07% |
| Bessel + edge adapter | 68.73 | 65.66 | 4.67% |
| Bessel + atom adapter | 46.35 | 45.96 | 0.85% |
| Smooth + edge reuse | 50.17 | 48.58 | 3.28% |
| Smooth + layer mixture | 53.52 | 50.72 | 5.52% |
| Bessel + pair width 512 | 57.46 | 56.83 | 1.11% |
| Bessel + both adapters + pair width 512 | 69.84 | 67.33 | 3.73% |
| Smooth + readout context | 61.92 | 59.70 | 3.72% |
| Smooth + translation projection | 52.34 | 50.15 | 4.37% |
| Smooth + combined changes | 73.00 | 72.73 | 0.37% |

The median gain across all thirteen settings is 4.22%.

### Complete validation measurements

These measurements include graph preparation, both model evaluations, and all validation metrics.
Each comparison uses eighty geometries through 350 atoms.

| Setting | Previous seconds | Selected seconds | Throughput gain |
| --- | ---: | ---: | ---: |
| Bessel control | 1.421 | 1.372 | 3.6% |
| Smooth + shared geometry | 1.416 | 1.363 | 3.9% |
| Smooth + combined changes | 1.616 | 1.555 | 3.9% |

### Numerical, capacity, and resume checks

All thirteen settings pass fixed-checkpoint checks on eighty validation geometries each.
The largest absolute relative MAE change is 0.0139%.
The largest absolute relative eigenvalue MAE change is 0.0753%.
Each setting resumes twice and completes four updates with the final training runtime.
The final validation update changes only rotation input layout and its block kernel selection.
Checkpoint architecture, optimizer counters, batch settings, augmentation, and schedules remain compatible.
The two-million-update target stays unchanged.

Capacity tests cover five 350-atom geometries in all thirteen settings.
The largest allocated memory is 68.56 GiB.
This batch does not bound every possible mixed-size layout.

Long trials use 1,024 ordered updates per trajectory.
Three settings each run an initial control, a selected trajectory, and a final control.
The nine audit trajectories process 306,051 geometries.
All sampled gradients, parameters, losses, and optimizer states stay finite.
Frozen backbone weights stay unchanged.
The faster combined candidate changes sensitive mode statistics and is rejected.
The selected combined path retains FP32 activation gradients.
Controls also show vibration-metric variation.
These checks test numerical behavior. They do not establish equal accuracy at convergence.

CPU tests pass 410 cases. H100 tests pass 178 cases.
Tests cover empty inputs, irregular widths, higher derivatives, block rotations, and gradients down to 10⁻¹⁸.

### Continuation deployment

Each run selects the immutable runtime at its next normal checkpoint continuation.
Current allocations retain their loaded runtime.
No production job is cancelled.
The deployment record keeps each preceding runtime path for rollback.

Selected runtime: `/lustre/fs1/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip-uma/hf_release_tools/training_speed_round3_20261021/runtime_r3_v2`.
Local evidence: `cluster_runs/hip_training_speed_round3_20261021/summary.json`.

## Cleanup after the third H100 update

New runs use the verified execution policy in `configs/train.json`.
The precision policy is `float32`, and the execution mode is `optimized`.
Selected kernels still use the tested local FP16 matrix operations.
The batch limit is 1,792 atoms and 64 geometries.
Legacy head input fusion, manual input alignment, and radial-only backward flags stay off.
Fused AdamW, chunk loading, and the existing model architecture remain enabled.
Use the original configuration to resume an existing run.

The production package excludes bounded CUDA graph capture and the unused individual-linear backward prototype.
Their source remains under `experiments/training_speed_20261021` for reference.
The selected MLP backward kernels and the eager reference path remain in the package.
Production context names no longer call selected kernels “candidates”.
Execution metadata describes the FP32 mixture forward guard and active gradient reductions.
Older precision configurations retain their compatibility paths.

The core kernel suite uses 74 representative cases instead of 178 combinations.
It retains all numerical assertions and tolerances.
It covers zero and extreme gradients, empty inputs, irregular widths, missing edges, shared inputs, and higher derivatives.
Checkpoint, batching, schedule, data, and physical-metric tests remain in the maintained suite.
See `tests/hip/README.md` for the test policy.

The cleanup preserves previously deployed immutable runtime snapshots and existing run configurations.

Cleanup verification passes 405 CPU tests and 74 kernel tests on H100.
The control and combined models each resume twice and complete four updates.
A fresh default run completes two updates, validation checks, and checkpoint writing.
Evidence: `cluster_runs/hip_training_cleanup_20261021/summary.json`.
