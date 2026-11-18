"""Multi-GPU data-parallel continuation for packed molecular training.

Each existing packed batch is split by molecule across ranks. The ranks reduce
the batch-weighted gradients before one synchronized optimizer step, so each
saved step consumes exactly one entry from the original batch plan.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch
import torch.distributed as dist

_completed_updates = 0
_update_log_interval = max(1, int(os.environ.get("HIP_ORB_UPDATE_LOG_INTERVAL", "100")))


def _slice_batch(batch, selected):
    sizes = batch["sizes"].tolist()
    atom_offsets = [0]
    hessian_offsets = [0]
    for size in sizes:
        atom_offsets.append(atom_offsets[-1] + size)
        width = 3 * size
        hessian_offsets.append(hessian_offsets[-1] + width * (width + 1) // 2)

    result = dict(batch)
    result["sizes"] = batch["sizes"][selected]
    for key in ("charge", "spin", "omol_index"):
        result[key] = batch[key][selected]
    for key in ("coords", "atomic_numbers", "rotated_coords"):
        if key not in batch:
            continue
        result[key] = torch.cat([
            batch[key][atom_offsets[i]:atom_offsets[i + 1]] for i in selected
        ], dim=0)
    result["hessian_upper"] = torch.cat([
        batch["hessian_upper"][hessian_offsets[i]:hessian_offsets[i + 1]]
        for i in selected
    ], dim=0)
    return result


def _microbatches(selected, sizes, pair_limit):
    """Keep rank order and bound each group's quadratic activation footprint."""
    group = []
    pair_work = 0
    for molecule in selected:
        cost = sizes[molecule] ** 2
        if group and pair_work + cost > pair_limit:
            yield group
            group, pair_work = [], 0
        group.append(molecule)
        pair_work += cost
    if group:
        yield group


def _distributed_consistency_update(
    model,
    optimizer,
    batch,
    device="cuda",
    seed=7,
    loss_type="block_rmse",
    *,
    scaler=None,
    loss_scale=1.0,
    consistency_weight=0.1,
    consistency_seed=1000010,
    reuse_topology=True,
    fused_loss=True,
):
    if scaler is not None:
        raise ValueError("Distributed MI100 continuation currently requires float32")
    if loss_type != "block_rmse" or model.head.translation_projection:
        raise ValueError("Distributed packed continuation requires block_rmse without translation projection")

    from hip_orb.packed_consistency import _reference_consistency_terms, consistency_terms
    from hip_orb.packed_training import compact_indices, compact_loss, packed_graph, prepare_packed
    from hip_orb.two_view_graph import reuse_graph, topology_reuse_safe

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # Keep saved optimizer/model state unchanged while rematerializing the
    # large trainable pair-head activations during backward on 32 GiB MI100s.
    model.head.checkpoint_pair_activations = True
    model.head._single_owner_broadcast = getattr(model.head, "single_owner_broadcast", False)
    sizes_all = batch["sizes"].tolist()
    total = len(sizes_all)
    if not 0 < total <= 64 or sum(sizes_all) > 1792:
        raise ValueError("Global batch exceeds continuation caps: 64 molecules / 1792 atoms")
    # Greedy largest-first assignment balances quadratic all-pairs work much
    # better than alternating adjacent molecules from the locality-ordered
    # batch. Sort each rank's assignment back into batch order afterwards.
    assignments = [[] for _ in range(world_size)]
    rank_costs = [0.0] * world_size
    for molecule in sorted(range(total), key=lambda i: (sizes_all[i] ** 2, -i), reverse=True):
        target = min(range(world_size), key=lambda r: (rank_costs[r], r))
        assignments[target].append(molecule)
        rank_costs[target] += sizes_all[molecule] ** 2
    selected = sorted(assignments[rank])
    local_count = len(selected)
    optimizer.zero_grad(set_to_none=True)

    local_loss = torch.zeros((), device=device, dtype=torch.float32)
    if local_count:
        # Generate the original full-batch augmentations first so each molecule
        # receives the same deterministic views it would have received on one GPU.
        first_all = prepare_packed(batch, seed)
        second_all = prepare_packed(batch, consistency_seed)
        # The default retains the serial memory footprint. Oversize molecules
        # always run alone. All groups contribute to the same optimizer update.
        pair_limit = getattr(model, "_distributed_microbatch_pair_limit", 0)
        for group in _microbatches(selected, sizes_all, pair_limit):
            first_batch = _slice_batch(first_all, group)
            second_batch = _slice_batch(second_all, group)
            indices = compact_indices(tuple(first_batch["sizes"].tolist()), str(device))
            upper = first_batch["hessian_upper"].to(device, non_blocking=True)
            graph = packed_graph(first_batch, model.adapter, device)
            if reuse_topology and topology_reuse_safe(first_batch, model.adapter, graph):
                second_graph = reuse_graph(graph, second_batch)
            else:
                second_graph = packed_graph(second_batch, model.adapter, device)

            def view(current):
                feature = model.features(current, sizes=first_batch["sizes"])
                feature.packed_layout = indices
                return model.head._batched_forward(feature, as_tensor="pair-blocks")

            first, first_diag = view(graph)
            second, second_diag = view(second_graph)
            r1 = first_batch["rotation"].to(device, non_blocking=True)
            r2 = second_batch["rotation"].to(device, non_blocking=True)
            if fused_loss and first.is_cuda:
                from hip_orb.consistency_kernels import two_view_block_loss
                group_loss = two_view_block_loss(
                    first, first_diag, second, second_diag, upper, r1, r2,
                    indices, consistency_weight,
                )
            else:
                group_loss = 0.5 * (
                    compact_loss(first, first_diag, upper, r1, indices)
                    + compact_loss(second, second_diag, upper, r2, indices)
                )
                group_loss = group_loss + consistency_weight * _reference_consistency_terms(
                    first, first_diag, second, second_diag, r2 @ r1.T, indices
                )
            # Compact loss averages the group's molecules. Recover their share
            # of the global mean before accumulating gradients on each rank.
            group_loss = group_loss * (loss_scale * len(group) / total)
            local_loss = local_loss + group_loss.detach()
            group_loss.backward()

    # The loss is a mean over local molecules; weighted local means sum to the
    # original global-batch mean.  One packed all-reduce synchronizes every head
    # gradient before identical AdamW state advances on both ranks.
    parameters = list(model.head.parameters())
    active = torch.tensor([p.grad is not None for p in parameters], device=device, dtype=torch.int32)
    dist.all_reduce(active, op=dist.ReduceOp.MAX)
    flat = torch.cat([
        (torch.zeros_like(p) if p.grad is None else p.grad).reshape(-1)
        for p in parameters
    ])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
    loss_value = float(local_loss)
    if not math.isfinite(loss_value):
        raise FloatingPointError("Distributed packed consistency loss is not finite")
    offset = 0
    for parameter, has_grad in zip(parameters, active.tolist(), strict=True):
        count = parameter.numel()
        parameter.grad = (flat[offset:offset + count].view_as(parameter)
                          if has_grad else None)
        offset += count
    torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
    optimizer.step()

    global _completed_updates
    _completed_updates += 1
    if rank == 0 and (_completed_updates == 1 or _completed_updates % _update_log_interval == 0):
        print(json.dumps({"event": "mi100_update_complete",
                          "updates_in_process": _completed_updates,
                          "single_owner_broadcast": model.head._single_owner_broadcast,
                          "batch_examples": total,
                          "estimated_rank_pair_cost": rank_costs}), flush=True)
    from hip_orb.mixed_precision import invalidate_trainable_precision_caches
    invalidate_trainable_precision_caches(model)
    return loss_value, total


def distributed_limit_reached(local):
    flag = torch.tensor(int(local), device=f"cuda:{torch.cuda.current_device()}", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise RuntimeError(f"Expected at least two GPU ranks, got WORLD_SIZE={world_size}")
    # Set before CUDA context/handle creation; runner can enable deterministic kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    # Keep nonzero ranks' runner bookkeeping and reports isolated from the
    # authoritative checkpoint written by rank 0.
    if rank != 0:
        for i, value in enumerate(sys.argv[1:-1], start=1):
            if value == "--output":
                sys.argv[i + 1] = os.path.join(sys.argv[i + 1], f".ddp-rank{rank}")
                break
            if value.startswith("--output="):
                sys.argv[i] = value + f".ddp-rank{rank}"
                break

    from hip_orb import runner
    runner.packed_consistency_update = _distributed_consistency_update
    runner.distributed_limit_reached = distributed_limit_reached
    runner.main()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
