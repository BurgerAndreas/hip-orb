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

Enforce matrix symmetry during assembly. An off-diagonal block can be nonsymmetric.
Add analytical, fixed-charge Coulomb curvature for every atom pair.
Support the original optional erf damping. Reject periodic systems for now.
Use the checkpoint's Coulomb constant, not a fitted replacement.

Train on `teacher Hessian - fixed-charge Coulomb Hessian`.
The residual includes charge response, repulsion, and short-range curvature.
Do not add the analytical term twice. Original energy and forces remain unchanged.
Exact rotation equivariance is not guaranteed. Augmentation must rerun the frozen backbone.

## Bounded training and checks

Install with `uv sync --extra hip`. Run unit tests with `pytest tests/hip/test_heads.py`.
GPU integration tests load the real checkpoint and compare the original predictions.
Set `HIP_ORB_REAL_TESTS=1` to enable them.

Run `hip-orb-train --help` for the overfit and pilot commands.
The runner reads completed OrbMol HDF5 shards. It records sample indices, not file locations.
It verifies that every original model parameter and buffer stays unchanged.
Pilot metrics include mass-weighted vibrational Hessians, eigenvalues, and low-mode subspace overlap.
The pilot also measures rotation error. Its geometry split is provisional, not molecule-disjoint.

`scripts/hip_pilot.sbatch` runs tests, a four-molecule overfit, and a 96/32 pilot.
Create `outputs/` before submission. Supply scheduler settings through your submission command.
Set cache and data paths in an ignored `.env`. Never publish runtime logs or local settings.

## Full-training gate

Full 10-million-sample training is intentionally unavailable in the bounded runner.
Require all of these checks before adding or launching it:

- Successful real-molecule overfit and unchanged original predictions.
- A verified molecule-disjoint training/validation split.
- Acceptable held-out curvature and low-mode metrics, including rotation checks.
- Real transition-state P-RFO comparisons using the same energy/force model.
- Measured full-model speed improvement against collection autograd Hessians.
- Measured peak memory across the intended atom-count range, through 350 atoms.

The small P-RFO implementation is a controlled comparison tool, not a production optimizer.
Its synthetic saddle test does not establish molecular transition-state performance.
No pilot result alone can silently enable full training.

After the pilot, `python -m hip_orb.benchmark` checks 20, 50, 70, 110, 200, and 350 atoms.
It includes graph construction and frozen backbone inference in HIP timing.
The reference uses the collection's angular adapter and eight-row batched VJPs.
This measures single-molecule latency, not optimal multi-molecule throughput.
