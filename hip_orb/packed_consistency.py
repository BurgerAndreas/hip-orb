"""Two supervised rotation views with fused losses and guarded graph reuse."""

import math

import torch

from hip_orb.packed_training import compact_indices, compact_loss, packed_graph, prepare_packed
from hip_orb.two_view_graph import reuse_graph, topology_reuse_safe

__all__ = ["consistency_terms", "packed_consistency_update"]


def _reference_consistency_terms(a, ad, b, bd, relative, indices):
    pair = torch.linalg.vector_norm(b - relative @ a @ relative.T, dim=(-2, -1))
    diagonal = torch.linalg.vector_norm(bd - relative @ ad @ relative.T, dim=(-2, -1))
    return (pair / 3 * indices["pair_weight"]).sum() + (diagonal / 3 * indices["diag_weight"]).sum()


def consistency_terms(a, ad, b, bd, relative, indices):
    if a.is_cuda:
        from hip_orb.consistency_kernels import consistency_terms as fused

        return fused(a, ad, b, bd, relative, indices)
    return _reference_consistency_terms(a, ad, b, bd, relative, indices)


def packed_consistency_update(
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
    if loss_type != "block_rmse" or model.head.translation_projection:
        raise ValueError("Packed rotation consistency requires unprojected block RMSE")
    indices = compact_indices(tuple(batch["sizes"].tolist()), str(device))
    upper = batch["hessian_upper"].to(device, non_blocking=True)
    first_batch = prepare_packed(batch, seed)
    second_batch = prepare_packed(batch, consistency_seed)
    graph = packed_graph(first_batch, model.adapter, device)
    if reuse_topology and topology_reuse_safe(batch, model.adapter, graph):
        second_graph = reuse_graph(graph, second_batch)
    else:
        second_graph = packed_graph(second_batch, model.adapter, device)

    def view(current):
        feature = model.features(current, sizes=batch["sizes"])
        feature.packed_layout = indices
        return model.head._batched_forward(feature, as_tensor="pair-blocks")

    first, first_diag = view(graph)
    second, second_diag = view(second_graph)
    r1 = first_batch["rotation"].to(device, non_blocking=True)
    r2 = second_batch["rotation"].to(device, non_blocking=True)
    if fused_loss and first.is_cuda:
        from hip_orb.consistency_kernels import two_view_block_loss

        loss = two_view_block_loss(
            first, first_diag, second, second_diag, upper, r1, r2, indices, consistency_weight
        )
    else:
        loss = 0.5 * (
            compact_loss(first, first_diag, upper, r1, indices)
            + compact_loss(second, second_diag, upper, r2, indices)
        )
        loss = loss + consistency_weight * _reference_consistency_terms(
            first, first_diag, second, second_diag, r2 @ r1.T, indices
        )
    loss = loss * loss_scale
    loss_value = float(loss.detach())
    if not math.isfinite(loss_value):
        raise FloatingPointError("Packed consistency loss is not finite")
    optimizer.zero_grad(set_to_none=True)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    try:
        torch.nn.utils.clip_grad_norm_(
            model.head.parameters(), 10.0, error_if_nonfinite=scaler is None
        )
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        raise FloatingPointError("Packed consistency gradients are not finite") from error
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()
    from hip_orb.mixed_precision import invalidate_trainable_precision_caches

    invalidate_trainable_precision_caches(model)
    return loss_value, len(batch["sizes"])
