"""Pair-cutoff Hessian inference for HIP-Orb with local context pooling.

Only atom pairs closer than ``cutoff`` (same molecule) get off-diagonal blocks.
Context pooling averages over these pairs only. For an all-to-all head, the
blocks therefore differ from dense inference when a molecule has pairs beyond
the cutoff. If the cutoff is larger than every pair distance, the result is
the same as dense inference.

The analytical fixed-charge Coulomb term is added for the kept pairs only. Its
diagonal is the negative sum of the kept off-diagonal Coulomb blocks, so the
truncated Coulomb Hessian stays translation invariant.

Output: upper-triangle pair indices ``i < j`` (int32), their 3x3 blocks
(row atom ``i``, column atom ``j``), and the N diagonal 3x3 blocks.
``pair_chunk`` (optional) limits the number of pairs in the head at one time.
With chunks, the pair features are computed twice: once for pooling, once
for the readout.
"""

import torch

from hip_orb.coulomb import fixed_charge_blocks
from hip_orb.neighbor_search import RadiusPairs, neighbor_pairs


def _edge_lookup(senders, receivers, total):
    ordered, order = (senders.long() * total + receivers.long()).sort()

    def lookup(a, b):
        if not len(ordered):
            return torch.full_like(a, -1)
        query = a * total + b
        slot = torch.searchsorted(ordered, query).clamp_max(len(ordered) - 1)
        return torch.where(ordered[slot] == query, order[slot], -1)

    return lookup


def _check_head(head):
    if not head.neighbor_pooling:
        raise NotImplementedError("Cutoff inference expects a neighbor-pooling head")
    if len(head.extra_context_rounds):
        raise NotImplementedError("Cutoff inference supports one context round")
    if head.shared_geometry or hasattr(head, "orb_distance_encoder"):
        raise NotImplementedError("Cutoff inference does not support the dense Orb extension")
    if head.translation_projection:
        raise NotImplementedError("Cutoff inference does not support translation projection")
    if (
        getattr(head, "readout_state_mode", "none") != "none"
        or getattr(head, "pair_state_conditioning", "none") != "none"
    ):
        raise NotImplementedError("Cutoff inference does not support state-conditioned readouts")


@torch.no_grad()
def cutoff_hessian(
    model,
    graph,
    cutoff,
    pair_chunk=None,
    *,
    residual_only=False,
    timings=None,
    search="auto",
    pair_future=None,
):
    """Inference wrapper; async CPU work must be submitted before this call."""
    return cutoff_forward(
        model,
        graph,
        cutoff,
        pair_chunk,
        residual_only=residual_only,
        timings=timings,
        search=search,
        pair_future=pair_future,
    )


def cutoff_forward(
    model,
    graph,
    cutoff,
    pair_chunk=None,
    *,
    residual_only=False,
    timings=None,
    search="auto",
    pair_future=None,
    block_sink=None,
    checkpoint_chunks=False,
):
    """Differentiable sparse Hessian, including the frozen backbone forward.

    ``block_sink(i, j, blocks)`` streams inference blocks instead of retaining
    the full output. It is deliberately unavailable with gradients enabled.
    In that mode the return value contains only ``diagonal`` and ``pairs``.
    """
    if block_sink is not None and torch.is_grad_enabled():
        raise ValueError("Block sinks require inference without gradients")
    if pair_chunk is not None and pair_chunk < 1:
        raise ValueError("Pair chunk must be positive")
    head = model.head
    _check_head(head)
    features = model.features(graph)
    nodes, edges = features.nodes, features.edges
    positions = features.positions
    charges = features.charges.reshape(-1)
    total = len(nodes)
    streaming = block_sink is not None and pair_future is None
    if streaming:
        plan = RadiusPairs(positions, features.sizes, cutoff, backend=search)
        i_all = j_all = None
        pairs = 0
    else:
        i_all, j_all = (
            neighbor_pairs(positions, features.sizes, cutoff, backend=search)
            if pair_future is None
            else pair_future.result(positions.device)
        )
        pairs = len(i_all)
    chunk = pair_chunk or 262144

    def batches():
        if streaming:
            yield from plan.chunks(chunk)
        else:
            for start in range(0, pairs, chunk):
                yield i_all[start : start + chunk], j_all[start : start + chunk]

    lookup = _edge_lookup(features.senders, features.receivers, total)
    head.pair_adapter[0]._hip_input_grad_columns = None
    head.pair_adapter[0]._hip_radial_source = None
    fused = (
        getattr(head, "_fused_pair_inputs", False)
        and nodes.is_cuda
        and all(t.is_contiguous() and t.dtype == torch.float32 for t in (nodes, edges, charges))
    )
    padded = None if fused else torch.cat((edges, nodes.new_zeros((1, nodes.shape[-1]))))

    def inputs(a, b, ids, direction, radii):
        if fused:
            from hip_orb.head_fusion import frozen_pair_inputs

            return frozen_pair_inputs(
                nodes, edges, charges, a, b, ids, direction.contiguous(), radii.contiguous()
            )
        present = ids >= 0
        edge = padded[torch.where(present, ids, len(edges))]
        return torch.cat(
            (
                nodes[a],
                nodes[b],
                edge,
                direction,
                radii,
                charges[a, None],
                charges[b, None],
                present[:, None].to(edge.dtype),
            ),
            dim=-1,
        )

    def pair_states(i, j):
        delta = positions[i] - positions[j]
        distance = delta.norm(dim=-1, keepdim=True)
        unit = delta / distance
        radial = head.radial_embedding(distance)
        first = head.pair_residual(head.pair_adapter(inputs(i, j, lookup(i, j), unit, radial)))
        second = head.pair_residual(head.pair_adapter(inputs(j, i, lookup(j, i), -unit, radial)))
        return distance.reshape(-1), first, second

    def evaluate(function, *args):
        if checkpoint_chunks and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def pool_weights(distance):
        if head.all_to_all:
            return torch.ones_like(distance)
        pool_cutoff = head.cutoff if head.cutoff is not None else head.radial_cutoff
        weights = 0.5 * (torch.cos(torch.pi * distance / pool_cutoff) + 1)
        return weights * (distance < pool_cutoff).to(weights.dtype)

    # Pass 1: local context pooling over the kept pairs.
    pooled = None
    count = positions.new_zeros((total, 1))
    cached = None
    for i, j in batches():
        i, j = i.long(), j.long()
        distance, first, second = evaluate(pair_states, i, j)
        w = pool_weights(distance)[:, None]
        if pooled is None:
            pooled = torch.zeros(
                (total, first.shape[-1]),
                device=first.device,
                dtype=torch.promote_types(first.dtype, w.dtype),
            )
        pooled.index_add_(0, i, first * w).index_add_(0, j, second * w)
        count.index_add_(0, i, w).index_add_(0, j, w)
        if streaming:
            pairs += len(i)
        elif pairs <= chunk and not checkpoint_chunks:
            cached = (first, second)
        del first, second
    if pooled is None:
        pooled = positions.new_zeros((total, head.config["hidden"]))
    pooled = pooled / count.clamp_min(1e-8)
    node_inputs = head.node_update(torch.cat((nodes, charges[:, None], pooled), dim=-1))
    del pooled
    diagonal = head.node_head(node_inputs).reshape(-1, 3, 3).float()
    diagonal = (diagonal + diagonal.transpose(-1, -2)) * 0.5

    # Pass 2: pair update and block readout.
    blocks = (
        None
        if block_sink is not None
        else torch.empty((pairs, 3, 3), device=positions.device, dtype=torch.float32)
    )

    def readout(i, j, states):
        if states is None:
            _, first, second = pair_states(i, j)
        else:
            first, second = states
        first = first + head.pair_update(torch.cat((first, node_inputs[i], node_inputs[j]), dim=-1))
        second = second + head.pair_update(
            torch.cat((second, node_inputs[j], node_inputs[i]), dim=-1)
        )
        return (
            head.edge_head(first).reshape(-1, 3, 3)
            + head.edge_head(second).reshape(-1, 3, 3).transpose(-1, -2)
        ).float() * 0.5

    s = 0
    for i, j in batches():
        i, j = i.long(), j.long()
        e = s + len(i)
        if cached is not None:
            part = readout(i, j, cached)
            cached = None
        else:
            part = evaluate(readout, i, j, None)
        if head.cutoff is not None:
            distance = (positions[i] - positions[j]).norm(dim=-1)
            part = torch.where((distance < head.cutoff)[:, None, None], part, 0)
        if not residual_only:
            coulomb, coulomb_diagonal = fixed_charge_blocks(
                positions, charges, i, j, features.constant, features.sigma
            )
            part = part + coulomb
            diagonal = diagonal + coulomb_diagonal
        if block_sink is None:
            blocks[s:e] = part
        else:
            block_sink(i, j, part)
        s = e
    if block_sink is not None:
        return {"diagonal": diagonal, "pairs": pairs}
    return {"i": i_all, "j": j_all, "blocks": blocks, "diagonal": diagonal}


def blocks_to_dense(result, n):
    """Assemble a (3N, 3N) matrix from one molecule's cutoff output (for tests)."""
    i, j = result["i"].long(), result["j"].long()
    h = result["blocks"].new_zeros((n, n, 3, 3))
    h[i, j] = result["blocks"]
    h[j, i] = result["blocks"].transpose(-1, -2)
    ids = torch.arange(n, device=h.device)
    h[ids, ids] = result["diagonal"]
    return h.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
