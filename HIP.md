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
Use the checkpoint's Coulomb constant.

Train on `teacher Hessian - fixed-charge Coulomb Hessian`.
The residual includes charge response, repulsion, and short-range curvature.
Do not add the analytical term twice. Original energy and forces remain unchanged.
Exact rotation equivariance is not guaranteed. Augmentation must rerun the frozen backbone.

## Bounded training and checks

Install with `uv sync --extra hip`. Run unit tests with `pytest tests/hip/test_heads.py`.
GPU integration tests load the real checkpoint and compare the original predictions.
Set `HIP_ORB_REAL_TESTS=1` to enable them.

Run `hip-orb-train --help` for the overfit and pilot commands.
The runner reads OrbMol HDF5 shards. It records sample indices.
It verifies that every original model parameter and buffer stays unchanged.
Validation metrics include mass-weighted vibrational Hessians, eigenvalues, low-mode subspace overlap, and rotation error. 

`scripts/hip_pilot.sbatch` runs tests, a four-molecule overfit, and a 96/32 pilot.
Create `outputs/` before submission. Supply scheduler settings through your submission command.
Set cache and data paths in an ignored `.env`. 

`python -m hip_orb.benchmark` checks the latency on molecules 20, 50, 70, 110, 200, and 350 atoms.
It includes graph construction and frozen backbone inference in HIP timing.
The autograd Hessian reference uses the collection's angular adapter and eight-row batched VJPs.
