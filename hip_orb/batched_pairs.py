"""Cached within-molecule pair indices and dynamic message-edge lookup."""
import torch

from hip_orb.block_layout import layout, require


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
