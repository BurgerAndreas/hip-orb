# Training runtime

This change combines the cache, launcher, conditioning optimization and requested
checkpoint stop into one installation. Install the `cache` extra when using the
bounded shard cache.

1. Configure paths and allocation options in the [single-node launcher guide](single_node_launcher.md).
2. Use its [bounded cache and node-local locks](shard_cache.md) to share shards across loader workers under one cache lease.
3. Add `--single-owner-broadcast` to enable the [conditioning optimization](single_owner_broadcast.md). It remains opt-in.
4. Set `HIP_ORB_STOP_REQUEST_FILE` before launch. Create that file to checkpoint
   at an update boundary; remove it before explicitly resuming from `head.pt`.
   See [checkpoint and resume semantics](checkpoint_on_request.md).

The launcher defaults to one training process. Multiple ranks require an explicit
entrypoint that synchronizes optimizer updates and isolates rank outputs. The
stop-consensus helper supports multiple ranks, but does not supply that entrypoint.

For a two-GPU smoke test, run `tests/hip/check_training_runtime_gpu.py` with
`torchrun --standalone --nproc_per_node=2`. It checks head losses, gradients,
parameters and Adam states on each GPU, then checks NCCL stop consensus. Run the
cache and checkpoint tests separately with pytest. These are correctness checks;
they do not measure end-to-end training throughput or production resume.

## Two-view distributed continuation

Use `torchrun --standalone --nproc_per_node=2 -m hip_orb.distributed_train`
with the normal runner arguments. This entrypoint supports float32 packed
`block_rmse` with two reference views. It shards each global batch by molecule,
retains full-batch rotation seeds, sums molecule-weighted gradients, clips once,
and advances identical AdamW states. Global continuation limits are 64 molecules
and 1,792 atoms. Rank zero owns the primary output; other ranks use private outputs.
Use `--lr-start-step 2000000` to preserve the source fine-tuning cosine schedule.
On NVIDIA, use `--float32-matmul-precision highest --deterministic` for this
continuation. The pretrained default `high` permits TF32, which produced about
20% relative gradient error between packed and per-molecule execution on L40S.
Full float32 with deterministic kernels reduced the measured relative error to
about 0.0053%; the fused and PyTorch loss paths agreed. The entrypoint configures
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization. Precision and
determinism are recorded in execution metadata. Validation tolerances are unchanged.
The source checkpoint's batch plan fingerprint and cursor must match; do not
reset them when moving shard paths. Real-data two-rank validation is required
before production use.

For real-data validation, replace the entrypoint with
`tests/hip/check_distributed_continuation.py`. On the first batch it compares
a two-molecule single-GPU update against the distributed update, including
loss, gradients, parameters and Adam state, then restores the source state
before the requested continuation. Run a short session and verify both rank
checkpoints have identical tensors, optimizer state, step and cursor.
Periodic checkpoints default to every 1,000 updates (`--checkpoint-every`).
