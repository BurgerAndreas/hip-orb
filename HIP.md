# HIP–OrbMol

This branch starts at Orb v0.7.0, commit
`82bab14c091509f2f8f5800ed725dcc596a60953`. Collection uses PyTorch 2.8.0.
The upstream model implementation stays unchanged. HIP lives in `hip_orb`.

## Architecture

Freeze all original parameters, including the energy and charge heads.
Keep the backbone in evaluation mode. Extract detached node and edge features.
Read diagonal blocks from node features. Read off-diagonal blocks from a pair adapter.
For original message edges, the adapter receives the frozen message-passing edge features.
For missing edges, it receives endpoint features, geometry, and an absent-edge flag.
The Hessian graph is fully connected by default. `--hessian-cutoff` changes only that graph.
It never changes the original cutoff graph or the original energy and force heads.

The optional context readout uses `--pair-residual-layers` and `--neighbor-pooling`.
Residual MLP layers refine directed pair features.
Cosine-weighted means pool local pair features into each endpoint atom.
One node update feeds the diagonal head and a further directed pair update.
Tensor gathers and indexed sums process all pairs across the batch.
`--radial-features bessel` adds independent fixed Bessel features with a cosine envelope.
`--radial-cutoff` sets the radial envelope and pooling radius, without removing Hessian pairs.
The context configuration uses width 256, two residual layers, and 16 radial features with a 12 Å radius.
`configs/train_2m_block_rmse_single_context_lr_1e-3.json` selects single-view corrected rotations and Block RMSE.
It uses 2M steps, peak LR 0.001, cosine decay, and 15% linear warm-up.

`--context-rounds 2` adds a second pair-to-node-to-pair round with independent weights.
The second round pools refined first-round pairs and applies residual node and pair updates.
`--mlp-hidden 1024` expands internal readout MLP layers while retaining 256-wide node and pair states.
The default internal width equals `--hidden`; the default context count is one.
`--all-to-all` retains every nonself Hessian pair and pools context over every other atom.
It also learns independent readout Bessel frequencies without a hard radial envelope; a log-distance channel retains long-range distance information.
The 12 Å radial scale initializes basis frequencies but does not truncate these graphs.
`--orb-initialization random` applies the standard PyTorch initialization to every added processor parameter.
`--orb-message-graph backbone` uses the original Orb graph, cutoff, and frozen edge features.
`--orb-message-graph hessian` uses directed all-to-all edges with a fresh trainable radial/angular encoder.
The dense encoder has independent trainable Bessel frequencies, l≤3 spherical harmonics, and a learned projection to Orb's feature width.
Dense messages have no cutoff. Frozen backbone charges and energy/force predictions stay unchanged.
The three `train_2m_*_all_lr_1e-3.json` configurations compare global context pooling with random Orb rounds on either graph.
All three use feature width 256, internal MLP width 1024, and the same Block RMSE schedule.
The Orb variants replace context pooling; their pair adapter and block readouts retain the common residual MLP architecture.
`configs/train_2m_block_rmse_context_2rounds_lr_1e-3.json` tests a second round at width 256.
`configs/train_2m_block_rmse_context_mlp1024_lr_1e-3.json` tests one round with internal width 1024.
Both retain the context baseline's radial features and training settings.
They use 50,000-update sessions with a three-hour safety limit before checkpointing and continuation.

`--extra-orb-round` adds one trainable Orb `AttentionInteractionNetwork` after the frozen backbone.
The extra processor starts from an independent copy of Orb's final processor weights.
It uses the original message graph, polynomial 6 Å cutoff, attention gates, and frozen conditioning features.
Its updated node and edge features feed the Hessian readout.
Charges still come from the original frozen features.
The head optimizer and checkpoints include the extra processor.
`configs/train_2m_block_rmse_single_context_orb_extra_lr_1e-3.json` adds this round to the width-256 context readout.
Extra-round configurations checkpoint after 50,000 updates or three hours, whichever comes first.
This session limit leaves time for checkpoint completion before the four-hour scheduler limit.
The overall 2M-step target has no duration limit.

`--translation-projection` applies the Cartesian translation projector to both Hessian axes.
Block row, column, and global means compute this projection without constructing a dense projector.
Equal-sized molecules share a batched reduction; projections never mix different molecules.
The projection applies after Coulomb addition during training and prediction.
It enforces three translation zero modes to numerical precision and preserves transpose symmetry.
Training targets remain unchanged. This option does not remove rotational modes.
`configs/train_2m_block_rmse_single_context_orb_extra_translation_lr_1e-3.json` changes only this option from the extra-round run.
`configs/train_2m_context_translation_lr_1e-3.json` tests translation projection on the default context baseline without an extra Orb round.

Enforce matrix symmetry during assembly. An off-diagonal block can be nonsymmetric.
Add analytical, fixed-charge Coulomb curvature for every atom pair.
Support the original optional erf damping. Reject periodic systems for now.
Use the checkpoint's Coulomb constant.

Train on `teacher Hessian - fixed-charge Coulomb Hessian`.
The residual includes charge response, repulsion, and short-range curvature.
Do not add the analytical term twice. Original energy and forces remain unchanged.
Exact rotation equivariance is not guaranteed. Augmentation must rerun the frozen backbone.

## Bounded training and checks

Install with `uv sync --extra hip`. Run unit tests with `pytest tests/hip/test_heads.py`.
GPU integration tests load the real checkpoint and compare the original predictions.
Set `HIP_ORB_REAL_TESTS=1` to enable them.

Start with `scripts/hip_preflight.sbatch`. It uses generated molecules and synthetic tensors only.
It cannot start dataset training. Tests cover generated sizes through 350 atoms.
The synthetic optimizer step checks gradients and freeze behavior, not model accuracy.
The source tag retains stale version metadata; this fork reports `0.7.0+hip.0`.
This metadata correction does not change original model operations or parameters.

Run `hip-orb-train --help` for the streaming training options.
Short and long runs use the same runner, loader, optimizer step, validation, and checkpoint code.
Use `scripts/hip_train.sbatch` for both runs. Set `HIP_TRAIN_CONFIG` in `.env` to select configuration values.
`configs/smoke_pilot.json` sets batch size 128, 64 updates, and three validation batches.
It also checks validation every 32 updates. The runner always performs final validation.
`configs/train.json` selects the context head as the default baseline.
It uses width 256, internal MLP width 256, one context round, two residual pair layers, and neighbor pooling.
It uses 16 independent Bessel radial features with a 12 Å envelope and the frozen Orb backbone.
Training uses single-view corrected rotations, Block RMSE, and 2,000,000 optimizer updates without an overall duration limit.
The learning rate rises linearly to 0.001 over 300,000 updates, then follows cosine decay to zero.
The older loss-comparison configurations retain their explicit 1,000,000-update budgets and simpler heads.
The schedule uses the absolute update number and retains its position across job restarts.
Validation runs every 200,000 updates and at the end, using at most 10,000 geometries each time.
The fixed seed selects the same validation geometries for each check.
New training uses balanced packed batches, with a maximum of 64 geometries and 2,560 atoms.
It has no Hessian-element cap. Shard windows preserve source-file locality between updates.
Four loader workers use persistent processes, prefetch four batches each, and cache up to 1 GiB each.
Set `HIP_SLURM_CPUS_PER_TASK=8` in `.env` for this loader configuration.
Validation retains equal-size batches, with a maximum of 256 geometries and an eight-million-element Hessian budget.
Historical comparison configurations retain their recorded batch limits.
The short-session experiments use two-hour jobs and checkpoint after 25,000 updates or 90 minutes.
Set `HIP_SLURM_PARTITION` and optional `HIP_SLURM_TIME_LIMIT` in the private environment for these submissions.
Training continuations inherit the actual Slurm partition if a pending job moves partitions.
Use `scripts/hip_train_steps.sbatch` on clusters with four-hour GPU limits.
The default baseline checkpoints after 50,000 updates or three hours and submits one successor if training remains incomplete.
Other configurations retain their specified session limits.
Only the last job performs final validation. Failed jobs do not submit successors.
Submit from the repository root, with the training environment and selected configuration.
The model, optimizer, epoch position, and learning rate schedule resume from the checkpoint.
Queue time does not affect the step budget. Scheduler job limits still apply.
The runner writes progress after the first update and every 1,000 updates.
`configs/train_mae.json` uses mean absolute Hessian error with the same step budget, data, seed, and validation settings.
The baseline configuration selects Block RMSE. The runner without a configuration retains its MSE default.
`--loss-type mae` selects MAE over all full-matrix elements.
Checkpoints and reports record the loss type. Resuming with a different loss type fails.
`configs/train_block_rmse.json` selects `--loss-type block_rmse` with the same training settings.
This loss averages the Frobenius norm divided by three for each 3-by-3 atom-pair error block.
It includes diagonal and off-diagonal blocks and gives each geometry equal weight.
The loss uses a reshape and tensor norm reductions, with no Python loops or CPU transfers.
The GPU computes the block norms in parallel. Autograd gives a zero gradient for exactly zero blocks.
The loss preserves rotation invariance and grows linearly with error magnitude.
`configs/train_mass_weighted_mse.json` selects `--loss-type mass_weighted_mse` with the same training settings.
It applies inverse square root atomic masses to both axes of the full Hessian error before computing MSE.
Each atomic mass repeats across its three Cartesian coordinates. Masses use ASE standard values in atomic mass units.
Tensor multiplication implements the basis change without constructing mass matrices or using Python loops.
Checkpoints and reports record the mass convention. Validation retains the same physical and vibrational metrics.
Both configurations use the same training and separate validation indexes.
The pilot selects 50-atom shards through configuration. Long training selects 2–350 atoms.
The Hessian element budget reduces batch sizes for larger molecules. Shard tails can produce smaller batches.
The runner reloads the saved checkpoint to check that it can read it.
The runner reads OrbMol HDF5 shards. It records sample indices.
It verifies that every original model parameter and buffer stays unchanged.
Validation metrics include mass-weighted vibrational Hessians, eigenvalues, low-mode subspace overlap, and rotation error.
`first_mode_abs_cosine` reports mean absolute cosine after mass weighting and removal of rigid translation and rotation.
`first_mode_abs_cosine_rms` reports the square root of mean squared cosine (`first_mode_overlap`).
These two averages differ; the RMS value does not equal mean absolute cosine.

The nine `configs/train_2m_*_lr_*.json` configurations use 2,000,000 updates and no overall duration limit.
They compare MAE, block RMSE, and mass-weighted MSE at peak learning rates 0.0003, 0.001, and 0.003.
Warm-up lasts 300,000 updates, followed by cosine decay to zero.
Each run starts with a fresh head and uses the existing 100,000-update checkpoint sessions.
Validation uses 10,000 geometries every 200,000 updates and at the end.
Validation also measures rotation error. It requires separate validation shards.
New runs sample uniform proper rotations using sign-corrected Gaussian QR.
The previous sampler omitted the QR sign correction and restricted orientation coverage.
Each batch shares a rotation, applied to coordinates and target Hessian blocks.
`--translation-scale 1.0` optionally shifts each molecule independently by up to 1 angstrom per Cartesian axis.
Translation leaves the target Hessian unchanged. The default shift is zero.
Reports and checkpoints record the augmentation settings. Completed runs retain their original augmentation.
Augmentation improves coverage but does not enforce exact rotation equivariance in the Orb backbone or matrix head.
The nine 2M-step sweep configurations enable translation shifts of 1 angstrom and rotation consistency with weight 0.1.
Each update uses two independent rotations and averages their reference losses.
The total objective adds 0.1 times the discrepancy between predictions transformed into the same frame.
The discrepancy uses the configured loss, including mass weighting when selected.
Gradients flow through both head predictions. The Orb backbone remains frozen.
Training loss histories record the total objective, not the reference loss alone.
The extra view requires another backbone evaluation and increases training cost.
`python -m hip_orb.check_rotation_training` checks real GPU batches through Slurm before the sweep starts.
The nine `train_2m_mse_ablation_*.json` configurations use MSE and peak LR 0.001.
Eight configurations cross corrected versus legacy rotations, translation on/off, and consistency on/off.
All eight use two reference views, so consistency comparisons retain the same reference supervision and view count.
The ninth configuration uses legacy rotations and one view to reproduce the original training objective.
All runs use fresh heads, seed 7, 2M updates, and the same data order and validation settings.
Validation always uses corrected rotations, independent of each run's training sampler.
Supply `--train-index` and `--validation-index` for the packed collections.
The runner never splits training geometries to create validation data.

Supply scheduler settings through your submission command.
Set cache and data paths in an ignored `.env`.
Set `HIP_TRAIN_INDEX` and `HIP_VALIDATION_INDEX` for the separate collections.
Copy `.env.example` to `.env` and replace its private path and scheduler placeholders.
The example contains no credentials. Keep tokens in `.env` or an exported environment variable.
All HIP job scripts load `.env`. Exported HIP, Hugging Face, and FairChem settings override file values.
Submit training with `bash scripts/submit_job.sh scripts/hip_train_steps.sbatch` from the repository root.
The helper reads account, partition, optional QoS, and log paths from `.env`.
Continuation jobs use the same helper and exported run settings.
Select other loss configurations with `HIP_TRAIN_CONFIG`; give each run a separate `HIP_TRAIN_OUTPUT`.
Override `HIP_SLURM_PARTITION` for CPU-only preparation jobs.

`python -m hip_orb.benchmark` checks latency on molecules with 20, 50, 70, 110, 200, and 350 atoms.
It includes graph construction and frozen backbone inference in HIP timing.
The autograd Hessian reference uses the collection's angular adapter and eight-row batched VJPs.

## Training execution

The runner defaults to `--execution-mode optimized`. Use `eager` for the reference implementation.
Execution settings stay separate from the model architecture and optimizer state.
The optimized path keeps the configured precision, loss, augmentation, and neighbor rules.
It supports existing checkpoints without converting weights.

The largest change removes gradient contention at absent message edges.
The previous gather maps absent edges to one padded zero row.
Its backward pass accumulates many gradients into that row, then discards them.
A Triton kernel skips these absent edges during accumulation.
Valid repeated edge indices still accumulate their gradients.
The implementation supports higher derivatives through a differentiable Torch fallback.
CPU execution, other dtypes, and installations without Triton use Torch.

The training path also returns Hessian batches directly, without splitting and stacking matrices.
Packed coordinates avoid per-molecule ASE construction for supported nonperiodic batches.
The graph builder keeps Orb's radius, neighbor limit, and edge ordering after canonicalization.
Generic and periodic graph checks remain available through the reference path.
Known charge and conditioning indices avoid redundant work and device synchronization.
Frozen backbone features are computed again at every update.

The launcher defaults to `expandable_segments:True` for variable batch shapes.
Set `HIP_CUDA_ALLOC_CONF=""` to use the standard CUDA allocator.
This reduces reserved memory; it does not reduce the model's active tensor memory.

For cluster continuations, create a complete package snapshot:

```bash
python -m hip_orb.runtime_snapshot --source-package hip_orb --output /path/to/new/runtime
python -m hip_orb.runtime_snapshot --output /path/to/new/runtime --verify
```

Set `HIP_TRAIN_RUNTIME_PACKAGE` to that directory in the private environment.
Transfer the complete snapshot, including its manifest. Do not copy a selected module list.
Test that exact snapshot before use. Never overwrite a snapshot used by an allocation.
The launcher verifies its manifest and uses Python's safe import path.
Set `HIP_TRAIN_EXECUTION_MODE="eager"` to restore the reference execution path.

Continuation jobs inherit exported environment values from their parent allocation.
Changing only `.env` cannot replace an inherited runtime setting.
For controlled updates, set `HIP_TRAIN_RUNTIME_POINTER` to a private file such as `.env.runtime`.
Its first line contains the complete snapshot directory.
The launcher reads it on each allocation, so the next continuation selects the verified snapshot.
Update this pointer atomically after all checks pass. Existing allocations retain their imported code.
