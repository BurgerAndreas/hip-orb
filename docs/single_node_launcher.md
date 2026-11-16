# Configurable single-node launcher

**Dependency:** merge the bounded-shard-cache PR first. This branch targets the
base directly and intentionally contains only launcher/configuration changes.
Optional `HIP_ORB_STOP_REQUEST_FILE` also requires the checkpoint-on-request PR.
No production launcher, runtime or checkpoint is modified by this example.

Set `HIP_ORB_RUNTIME`, `HIP_ORB_PYTHON`, `HIP_ORB_OUTPUT`, `HIP_ORB_CONFIG`,
`HIP_ORB_TRAIN_INDEX`, `HIP_ORB_VALIDATION_INDEX`, `HIP_ORB_CACHE_ROOT` and
`HIP_ORB_CHECKSUMS` to your paths. Index shard paths must resolve under the cache.
Set `HIP_ORB_CACHE_BYTES` to choose the disk bound (cache default 64 GiB).
Use `HIP_ORB_CHECKPOINT` and `HIP_ORB_WEIGHTS` for an explicit resume and backbone.
Load site modules/ROCm before submission. Pass account, partition, GPU count,
CPUs, memory, walltime and log destinations as sbatch options; no private paths
or site account names are embedded in reusable code.

```sh
# Example after exporting the required paths above:
export HIP_ORB_WORKERS=4 HIP_ORB_PREFETCH=8
export HIP_ORB_LOADER_CACHE_MODE=chunks HIP_ORB_OPEN_FILES=8
export HIP_ORB_CACHE_BYTES=51539607552
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION --gpus=1 \
  --cpus-per-task=16 --output="$HIP_ORB_OUTPUT/session-%j.log" \
  "$HIP_ORB_RUNTIME/scripts/hip_single_node.sbatch"
```

Ordinary defaults remain two workers, prefetch two, arrays and eight open files.
Four workers/rank, prefetch eight, chunk reads, eight open files and **three GPU
ranks on one MI100 node** were validated deployment settings, not universal
recommendations. The base repository lacks the deployed packed-consistency
multi-GPU update implementation. The launcher defaults to one process and rejects
multiple ranks without `HIP_ORB_DISTRIBUTED_ENTRYPOINT`. To use three ranks,
supply a reviewed entrypoint that synchronizes full optimizer updates and isolates
rank outputs, set `HIP_ORB_NPROC=3`, and request `--gpus=3 --cpus-per-task=48`.
That entrypoint must use the tracked dataset/cache integration; do not combine it
with the old external cache monkey-patch. This launcher does not claim to port
the entire production runtime or make its final checkpoint compatible with base.

The launcher calls one srun task on the compute node, obtains the output writer
lock, and invokes the cache lease wrapper outside torchrun. Its node-local lock
root defaults to job scratch or /tmp, shared by all workers on that host. The
global shared-cache lease excludes other cooperating sessions until training exits.
Multi-node allocations are rejected. This launcher never submits a successor;
resume deliberately with the same config/checkpoint after reviewing the report.
Trailing arguments are passed unchanged to the training command.

The round-two/round-three H100 runtime evidence documents selected mixed-precision
MLP backward implementations. Related `backward_precision`, `scaled_mlp`, and
execution switches already exist in base83e5156. They are not prerequisites for
this launcher/cache change; no runtime snapshot is transplanted or additional
precision change enabled. H100 speed estimates do not establish an MI100 gain.
