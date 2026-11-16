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
