# Mixed precision experiments

These experiments do not change training defaults or restart training jobs.
The current baseline keeps tensors in FP32 and permits TF32 matrix multiplication.
Strict FP32 uses `torch.set_float32_matmul_precision("highest")`.

## Scope

`hip_orb.mixed_precision.linear_precision` temporarily replaces selected `nn.Linear` forwards.
It preserves parameter objects, state keys, FP32 parameters, and optimizer state.
It restores the original forwards after normal exit or an exception.

Selections cover the Hessian head, Orb backbone, both regions, and optional final Hessian projections.
The internal selection covers square matrices with widths of at least 512.
Geometry, radial functions, normalization, pooling, Coulomb curvature, and the loss retain FP32 operations.
The charge head stays FP32.

Policies include conventional BF16 and FP16 matrix multiplication.
The `bf16_f32` and `fp16_f32` policies use half inputs and FP32 matrix outputs.
Their backward matrix multiplications also return FP32 outputs.
They avoid rounding each matrix output to a half value.

FP8 uses native H100 `torch._scaled_mm`, with E4M3 forward operands and dynamic tensor scales.
Backward can use BF16, E4M3, or E5M2 operands.
Unaligned matrices receive zero padding to multiples of 16.
FP8 weight caches refresh after parameter updates or replacement.

Custom matrix operations support first-order gradients only.
They reject requests for higher derivatives.
Do not use these custom operations to obtain Hessians through automatic differentiation.
Direct Hessian prediction only needs first-order parameter gradients.

## Correctness checks

The operator audit checks forward results, input gradients, weight gradients, and bias gradients.
It covers irregular dimensions, zero values, padding, and all three FP8 backward policies.
Independent FP32 matrix expressions use the same quantized operands as the tested kernels.
The relative error threshold is 0.03% for each matrix result.

These checks establish arithmetic correctness for the specified quantization surrogate.
They do not establish agreement with the original FP32 model.
Finite differences of discrete FP8 quantization do not test this surrogate derivative.

The model audit compares predictions, loss, and every head parameter gradient with the current baseline.
Three batches cover small molecules, mixed sizes, and a single large molecule.
The head gate requires global gradient cosine >= 0.999 and global relative gradient error <= 5%.
It also requires relative loss change <= 1% and relative prediction error <= 1%.
The audit records separate results for each parameter.

Production training freezes Orb.
An isolated probe temporarily enables Orb parameter and coordinate gradients.
This probe tests backward computation through the backbone on a fixed graph.
Normal training restores the frozen flags and checks the backbone parameter hash.

## Measurement

The fixed trained model has one context round, internal width 1024, and an all-pairs Hessian graph.
Each policy starts from the same checkpoint and optimizer state.
Balanced batches use at most 64 geometries and 2,560 atoms, without a Hessian-element cap.
No backbone features are cached.

The first sweeps time 48 complete updates twice on 1,372 geometries.
Host batches stay resident to isolate model costs.
Timing includes graph construction, rotation, Orb inference, head prediction, loss, backward, clipping, and AdamW.
A final streaming check uses four workers, prefetch four, and three repeats.
Worker startup and the first batch read stay outside the timer.
FP16 uses gradient scaling and reports skipped optimizer updates.

The validation diagnostic contains 48 held-out geometries across six size groups.
Mass weighting, Eckart projection, and eigenvalue analysis use FP64.
This diagnostic does not replace the normal 10,000-geometry validation.
It cannot establish final training accuracy or convergence.

## Initial full sweep

| Policy | ms/update | Peak allocated GiB | Worst head gradient error | Head gate |
|---|---:|---:|---:|---|
| Current FP32/TF32 | 110.5 | 65.6 | 0.63% | Pass |
| Strict FP32 | 352.4 | 65.6 | 1.75% | Pass |
| BF16 head | 115.7 | 50.9 | 10.4% | Fail |
| BF16 backbone | 117.2 | 65.7 | 4.86% | Fail |
| FP8 head, BF16 backward | 175.8 | 45.3 | 72.4% | Fail |
| FP8 head, E4M3 backward | 218.3 | 45.3 | 71.1% | Fail |
| FP8 head, E5M2 backward | 218.1 | 45.3 | 70.7% | Fail |
| FP8 backbone | 134.3 | 65.6 | 269% | Fail |

The head gate includes loss, prediction, and gradient checks.
A small gradient norm error alone does not ensure a pass.
The final FP32 control takes 112.0 ms/update.
Repeated FP32 computations show small differences from atomic reductions.

BF16 activation propagation fails because an existing operation requires matching tensor dtypes.
Conventional FP16 produces nonfinite predictions on held-out validation geometries.
A control lowers the initial gradient scale from 65536 to 1024.
Conventional FP16 then passes the three training-batch gradient checks, but still produces nonfinite validation predictions.
FP16 with FP32 matrix outputs remains finite, passes the checks, and skips zero optimizer updates.
The validation failure therefore does not come only from the initial gradient scale.
The native kernels pass; broad FP8 still changes the model gradients substantially.
FP8 scaling, casting, padding, and backward overhead exceed its matrix speed benefit here.
This result applies to this implementation and workload.

## Selective sweep

All 15 matrix operator audits pass on H100.
The complete HIP CPU suite passes 306 tests and skips 28 GPU or optional tests.
All six precision unit tests pass on H100, including the higher-derivative rejection tests.

| Policy | ms/update | Peak allocated GiB | Worst head gradient error | Minimum gradient cosine | Head gate |
|---|---:|---:|---:|---:|---|
| Current FP32/TF32 | 111.4 | 65.6 | 0.71% | 0.99998 | Pass |
| BF16 internal head matrices | 109.6 | 59.1 | 3.75% | 0.99931 | Fail |
| BF16 head, FP32 matrix outputs | 112.4 | 51.1 | 9.27% | 0.99582 | Fail |
| BF16 internal head matrices, FP32 outputs | 106.3 | 59.1 | 2.84% | 0.99960 | Pass |
| BF16 backbone, FP32 matrix outputs | 116.7 | 65.6 | 3.78% | 0.99933 | Fail |
| FP16 head, FP32 matrix outputs | 111.1 | 51.1 | 0.95% | 0.99995 | Pass |
| FP8 internal head, BF16 backward | 136.7 | 57.3 | 47.0% | 0.88885 | Fail |
| FP8 internal head, E4M3 backward | 155.3 | 56.9 | 45.0% | 0.89885 | Fail |
| FP8 internal backbone matrices | 118.0 | 65.6 | 35.0% | 0.94080 | Fail |
| FP8 internal head and backbone | 165.2 | 56.9 | 42.0% | 0.91058 | Fail |

Both passing reduced-precision policies keep the backbone and final Hessian projections in FP32.
FP32 output means native FP32 matrix results, not a cast after a rounded half result.

On the 48-geometry diagnostic, selective BF16 changes projected Hessian MAE by -0.069%.
It changes projected eigenvalue MAE by -0.167% and absolute mode cosine by -0.011 percentage points.
FP16 with FP32 outputs changes these metrics by +0.012%, +0.003%, and -0.045 percentage points.
These small diagnostic changes do not establish accuracy after training.

The backbone probe fails even when some held-out metrics remain similar.
BF16 backbone gradients have 97% relative error with conventional outputs.
FP32 matrix outputs reduce that error to 59%, but it remains too large.
Broad FP8 backbone gradients have 114% relative error and negative cosine against the current baseline.
Even strict FP32 differs by 8.9% on this probe against the current TF32 baseline.
The probe exposes a sensitive backbone Jacobian; frozen-backbone inference does not measure that Jacobian.

## Streaming confirmation

The confirmation includes source reads through four persistent workers with prefetch four.
Each policy runs the same 48 batches three times.
The sampler limits batches to 64 geometries and 2,560 atoms, without a Hessian-element cap.

| Policy | ms/update | Geometries/s | Peak allocated GiB | Worst head gradient error | Minimum gradient cosine |
|---|---:|---:|---:|---:|---:|
| FP32, initial control | 108.9 | 262.5 | 65.6 | 0.68% | 0.999977 |
| BF16 internal head matrices, FP32 outputs | 104.3 | 274.2 | 59.1 | 2.90% | 0.999580 |
| FP16 head, FP32 outputs | 108.0 | 264.6 | 51.1 | 0.75% | 0.999972 |
| FP32, final control | 109.7 | 260.5 | 65.6 | 0.99% | 0.999952 |

All four cases pass the head checks and skip zero optimizer updates.
Backbone parameter hashes remain unchanged.
The mean control takes 109.3 ms/update.
Selective BF16 increases throughput by 4.85% and reduces allocated peak memory by 10.0%.
FP16 with FP32 matrix outputs increases throughput by 1.18% and reduces peak memory by 22.1%.
The FP16 speed difference is small compared with the control drift.

The gradient gate uses all head gradients together.
Selective BF16 reaches 9.8% relative error on an individual parameter tensor.
FP16 with FP32 outputs reaches 3.5% on an individual tensor.
Therefore, passing the global gate does not mean every parameter has less than 5% error.

Keep the Orb backbone in FP32.
Selective BF16 is the better speed candidate for this workload.
FP16 with FP32 matrix outputs is a memory-saving alternative.
Both remain experimental until a training comparison and normal validation establish accuracy.
No production precision setting changes in this experiment.

## Follow-up experiments, 17 October

The follow-up combines lower precision with fused operations and cached weights.
It keeps FP32 master parameters, optimizer state, normalization, aggregation, residual additions, and final Hessian projections.

The fused head uses FP16 matrix operands and FP32 matrix outputs.
A Triton kernel combines bias addition and SiLU in FP32.
It can store the resulting internal activation in FP16.
The backward kernel combines the SiLU derivative, FP16 conversion, and partial bias reduction.
It accumulates bias gradients in FP32 without a full FP32 activation-gradient buffer.
Both backward matrix products also return FP32 results.
Cached weight conversions refresh when parameter versions or storage addresses change.

Runtime row counts prevent a new compilation for each geometry mixture.
Each final benchmark warms the full batch sequence before each timed repeat.
It reports warmup duration separately.
Early fused reports contain compilation in their first repeats.
Use their later repeats only; the final sweeps remove this timing problem.

Additional experiments cover:

- FP16 matrix operations throughout Orb, with and without FP32 matrix outputs.
- FP16 internal Orb MLP activations, with FP32 LayerNorm boundaries.
- Separate message-passing rounds, encoder, decoder, node MLPs, and edge MLPs.
- FP32 attention gates with lower precision in the surrounding MLPs.
- Power-of-two row scaling to protect FP16 range.
- Row-scaled FP8 operands with FP32 outputs and FP16 backward products.
- Three BF16 matrix products with correction terms.

The scaled kernels pass inputs with row magnitudes from 1e-6 to 1e6.
Large backward transposes use a Torch reduction above 4,096 columns.
An initial attempt exceeded Triton's reduction width and stopped that sweep.
The repaired sweep includes an 8,193-row arithmetic test.
Torch 2.8 only supports BF16 outputs for native row-scaled FP8 products.
The experimental FP8 path therefore uses unit-scaled products followed by a fused FP32 row rescale.

All 32 native arithmetic audits pass, including forward, input, weight, and bias gradients.
All 14 precision unit tests pass on H100.
The CPU HIP suite passes 310 tests and skips 32 optional or GPU tests.
First-order custom backward remains the only supported differentiation mode.

The strongest initial result comes from the fused FP16 head with FP16 internal activations.
Its warmed updates take about 87 ms, against 109 ms for FP32 controls.
Peak allocated memory falls from 65.6 GiB to about 50.7 GiB.
These initial results use 48 matched batches, three repeats, and the four-worker streaming loader.
The final sweeps also test the full shape warmup and 10,000 held-out geometries.

The 10,000-geometry check samples the separate validation index uniformly without replacement.
All precision cases use the same sample set and seed 914.
Validation batches cap geometries at 256, atoms at 2,560, and Hessian elements at eight million.
Mass weighting, Eckart projection, and the eigensolver use FP64 for the comparison.
This check uses a fixed checkpoint; it does not establish accuracy after further training.

## Final matched streaming results

All measurements use one H100 and uncached backbone features.
Each precision case starts from the same trained checkpoint and optimizer state as its model's controls.
The context checkpoint has 500,000 training steps.
Timings include complete updates and source reads through four workers with prefetch four.
Full shape warmup stays outside the timer.
Batches cap geometries at 64 and have no Hessian-element cap.
The context model caps atoms at 2,560; extra-Orb models cap atoms at 2,048.

The mixed recipe uses the fused FP16 head and FP16 hidden activations.
It also uses native FP16 matrices and hidden activations in Orb's node and edge MLPs across five message-passing rounds.
Orb's encoder, decoder, attention gates, LayerNorm boundaries, aggregation, and residual additions stay FP32.
The final Hessian projections, geometry, radial functions, charges readout, and loss stay FP32.
Charges can still change slightly because their input features change.

| Model | FP32 ms/update | Mixed ms/update | Throughput gain | Peak allocated GiB, FP32 to mixed |
|---|---:|---:|---:|---:|
| Context head, internal width 1024, all pairs | 109.1 | 84.9 | 28.5% | 65.5 to 50.7 |
| Extra trainable Orb round, backbone graph | 82.6 | 66.9 | 23.4% | 40.7 to 32.1 |
| Extra trainable Orb round, Hessian graph | 119.4 | 95.8 | 24.7% | 65.7 to 51.3 |

The head-only recipe takes 86.7 ms/update on the context model, with a 25.8% throughput gain.
It uses 50.6 GiB and leaves the entire backbone in FP32.
It captures most of the gain; backbone precision adds about 2% more throughput.

All three models pass head-gradient checks and isolated backbone checks for the combined recipe.
Worst aggregate head-gradient errors are 0.97%, 1.23%, and 0.37%, respectively.
The context backbone probe has 2.30% parameter-gradient error and 1.76% coordinate-gradient error.
Every timed optimizer update succeeds, and all frozen parameter hashes remain unchanged.

## Full validation comparison

The context model uses the same 10,000 held-out geometries for all four cases.
The extra-Orb models use the 48-geometry diagnostic and require their own full validation comparison.

| Context policy | Projected Hessian MAE | Change | Projected eigenvalue MAE change | Projected absolute mode cosine change |
|---|---:|---:|---:|---:|
| FP32 initial control | 0.0133268181 | reference | reference | reference |
| Fused FP16 head only | 0.0133268926 | +0.00056% | +0.00444% | -0.00528 percentage points |
| Fused head and message-passing MLPs | 0.0133274319 | +0.00461% | +0.00581% | -0.02169 percentage points |
| FP32 final control | 0.0133267965 | -0.00016% | +0.00026% | -0.01993 percentage points |

The mode cosine variation in the FP32 controls is comparable to the combined recipe's change.
These results compare precision at a fixed checkpoint.
They do not establish equal final accuracy after a new long training run.

Sparse uniform validation now uses HDF5 chunk caching.
The first full-validation attempts used array caching and loaded many unneeded geometries.
Only those benchmark attempts were cancelled and replaced.
Production training and collection jobs remain untouched.

## Stability check

Each recipe also completes 512 consecutive optimizer updates on the same repeated training cohort.
Each processes 14,645 geometries without nonfinite losses or skipped optimizer updates.
Initial mean losses agree within about 0.004%.
Final mean losses differ from the initial FP32 run by -2.22% for head-only and +2.28% for combined precision.
The small held-out diagnostic also differs after these updates.
This cyclic test checks stability and cannot establish convergence on the full training set.
Two additional FP32 runs complete the same test without nonfinite losses or skipped updates.
Their final losses range from 0.06335 to 0.07100, alongside 0.07093 in the first FP32 run.
Training trajectories therefore vary even without a precision change.
These repeats do not establish statistical equivalence between the recipes.

| 512-step trial | Final mean training loss | Projected MAE, 48 geometries | Eigenvalue MAE | Absolute mode cosine |
|---|---:|---:|---:|---:|
| FP32 A | 0.070926 | 0.021026 | 0.186213 | 0.343483 |
| FP32 B | 0.070999 | 0.021388 | 0.216645 | 0.325948 |
| FP32 C | 0.063350 | 0.019910 | 0.164725 | 0.347906 |
| Fused head only | 0.069352 | 0.021820 | 0.200124 | 0.336084 |
| Fused head and message MLPs | 0.072542 | 0.021564 | 0.202246 | 0.370100 |

The candidate is ready for a longer paired training comparison.
The head-only recipe captures most of the measured speed gain with fewer changed operations.

## Using the candidate

The following context enables the combined recipe without changing parameter objects or checkpoint keys:

```python
from hip_orb.mixed_precision import LinearPolicy, linear_precision

head = LinearPolicy("fp16_f32", fuse_silu=True, activation_dtype="fp16")
backbone = LinearPolicy(
    "fp16",
    include=("base.model.gnn_stacks.*._*mlp.mlp.NN-*",),
    activation_dtype="fp16",
)
with linear_precision(model, head=head, backbone=backbone):
    # Run the existing direct-Hessian training loop with gradient scaling.
    ...
```

Use `backbone=LinearPolicy()` for the head-only recipe.
The benchmarks use dynamic gradient scaling with an initial scale of 1024.
Unscale gradients before clipping and retain FP32 optimizer state.
The experimental context is not enabled by the production runner's default configuration.

## Reproduction

Run `python -m hip_orb.benchmark_precision --help` for required input arguments.
Inputs include private train and validation indices, Orb weights, model configuration, and the trained checkpoint.
Use `--case-labels` to select policies and `--streaming` to include source reads.
Keep reports and deployment paths outside version control.

Official operation references:

- [PyTorch 2.8 matrix multiplication](https://docs.pytorch.org/docs/2.8/generated/torch.mm.html)
- [PyTorch 2.8 float32 matrix precision](https://docs.pytorch.org/docs/2.8/generated/torch.set_float32_matmul_precision.html)
- [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/2.8/amp.html)

### Production comparison

`configs/train_2m_context_mlp1024_all_balanced_fp16_lr_1e-3.json` matches the original
balanced context run: 128 maximum geometries, 4,096 maximum atoms, four million
Hessian elements, global batch order, and two million optimizer updates.
It enables `precision_policy: fp16-message-head`. Gradient scaling starts at
1,024. Checkpoints preserve the scale and skipped-update count. Skipped updates
consume a batch but do not advance the training step or learning rate schedule.
Validation uses the same precision policy as training.

The later speed update pads odd linear input widths and skips unused derivatives
of frozen readout inputs. It also adds fused AdamW with explicit trainable-cache
invalidation. Parameter version counters alone do not detect fused updates in
this environment. See [SPEED.md](SPEED.md) for deployment and verification.
