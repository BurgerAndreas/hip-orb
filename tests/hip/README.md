# HIP regression tests

Run `python -m pytest -q tests/hip` for the maintained suite.
CUDA tests skip automatically on CPU machines.
Run CUDA tests inside a Slurm allocation on the cluster.

Keep tests that protect these behaviors:

- Checkpoint resume, optimizer counters, schedules, and batch order.
- FP32 mixture forwards and FP32 combined activation gradients.
- Zero, very small, and very large gradients.
- Empty batches, irregular widths, and optional biases.
- Higher derivatives and reference fallbacks.
- Missing edges, shared radial inputs, and dense gradient reductions.
- Rotation transforms and vibration metrics against independent references.

Kernel tests use representative parameter combinations.
They do not repeat every possible combination of the same boundaries.
Every assertion and numerical tolerance stays in place.
The core kernel suite contains 74 cases, reduced from 178.

Performance measurements remain separate from correctness tests.
Run throughput, capacity, and accuracy audits explicitly when execution changes.
Original test matrices and audit results remain in the workspace's cluster evidence.
Archived CUDA graph checks remain under `experiments/training_speed_20261021`.
