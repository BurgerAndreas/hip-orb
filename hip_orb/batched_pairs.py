"""Cached within-molecule pair indices and dynamic message-edge lookup."""
from functools import lru_cache

import torch

from hip_orb.block_layout import layout, require


@lru_cache(maxsize=8)
def dense_node_metadata(sizes, device):
    """Store offsets for canonical, sender-major directed molecular edges."""
    node_offsets, edge_offsets, widths = [], [], []
    node_start = edge_start = 0
    for n in sizes:
        node_offsets.extend([node_start] * n)
        edge_offsets.extend([edge_start] * n)
        widths.extend([n] * n)
        node_start += n
        edge_start += n * (n - 1)
    return tuple(torch.tensor(v, device=device, dtype=torch.long)
                 for v in (node_offsets, edge_offsets, widths))


def dense_edge_slots(sizes, senders, receivers):
    """Map valid nonself edges into the canonical dense edge array."""
    origin, edge_origin, width = dense_node_metadata(tuple(sizes), str(senders.device))
    a, b = senders - origin[senders], receivers - origin[senders]
    return edge_origin[senders] + a * (width[senders] - 1) + b - (b > a).long()


def batch_topology(features):
    sizes = [int(n) for n in features.sizes.tolist()]
    if not sizes or min(sizes) < 1 or sum(sizes) != len(features.nodes):
        raise ValueError("Batch sizes do not cover the nodes")
    indices = getattr(features, "packed_layout", None)
    if indices is None:
        indices = layout(tuple(sizes), str(features.nodes.device))
    i, j = indices["i"], indices["j"]
    if features.pairs is not None:
        source, target, forward, reverse = [], [], [], []
        offset = 0
        for n, pair in zip(sizes, features.pairs, strict=True):
            source.append(pair["i"] + offset)
            target.append(pair["j"] + offset)
            forward.append(pair["forward"])
            reverse.append(pair["reverse"])
            offset += n
        provided_i, provided_j = torch.cat(source), torch.cat(target)
        if provided_i.shape != i.shape:
            raise ValueError("Batched blocks require complete pair indices")
        require(((provided_i == i) & (provided_j == j)).all(),
                "Batched blocks require canonical pair indices")
        fwd, rev = torch.cat(forward), torch.cat(reverse)
    elif getattr(features, "dense_ordered", False):
        fwd = dense_edge_slots(sizes, i, j)
        rev = dense_edge_slots(sizes, j, i)
    else:
        node_owner = indices["node_owner"]
        require((node_owner[features.senders] == node_owner[features.receivers]).all(),
                "Message edges cross molecules")
        total = len(features.nodes)
        ordered, order = (features.senders * total + features.receivers).sort()
        if len(ordered) > 1:
            require((ordered[1:] != ordered[:-1]).all(), "Duplicate directed message edges")

        def lookup(a, b):
            if not len(ordered):
                return torch.full_like(a, -1)
            query = a * total + b
            slot = torch.searchsorted(ordered, query).clamp_max(len(ordered) - 1)
            return torch.where(ordered[slot] == query, order[slot], -1)

        fwd, rev = lookup(i, j), lookup(j, i)
    delta = features.positions[i] - features.positions[j]
    distance = delta.norm(dim=-1, keepdim=True)
    require((distance > 0).all(), "Hessian pairs must have nonzero distance")
    return sizes, i, j, fwd, rev, delta, distance
