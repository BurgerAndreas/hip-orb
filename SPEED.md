# Training speed changes

The selected configuration is
`configs/train_2m_context_mlp1024_all_balanced_fp16_lr_1e-3.json`.
It preserves the original balanced plan, global sample order, batch limits,
model weights, loss definition, learning rate schedule, and two-million-step target.

## Selected changes

1. Use four loader workers and prefetch depth four. Reserve eight CPUs.
2. Read HDF5 chunks directly. Keep sixteen open files per worker.
3. Pad linear inputs with zeros to a multiple of sixteen channels.
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

The precision policy remains `fp16-message-head`. The frozen backbone message
MLPs use FP16. Selected head matrix products use FP16 operands and FP32 outputs.
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

The mixed-precision comparison receives these changes. The FP32 comparisons keep
their existing execution. No training checkpoint is reset.

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
