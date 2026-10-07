"""Bounded geometry-free index cache for dense block assembly."""
from functools import lru_cache

import torch


def require(condition, message):
    """Check CUDA scalars without copying them to the CPU."""
    if condition.is_cuda:
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@lru_cache(maxsize=8)
def layout(sizes, device):
    device = torch.device(device)
    source, target, node_owners, dense_forward, dense_reverse, dense_diag = [], [], [], [], [], []
    dense_sizes = []
    atom_offset = dense_offset = 0
    xyz = torch.arange(3, device=device)
    for b, n in enumerate(sizes):
        d = 3*n
        i, j = torch.triu_indices(n, n, offset=1, device=device)
        source.append(i + atom_offset)
        target.append(j + atom_offset)
        node_owners.append(torch.full((n,), b, device=device, dtype=torch.long))
        row = 3*i[:, None, None] + xyz[None, :, None]
        col = 3*j[:, None, None] + xyz[None, None, :]
        dense_forward.append((dense_offset + row*d + col).reshape(-1))
        dense_reverse.append((dense_offset + col*d + row).reshape(-1))
        nodes = torch.arange(n, device=device)
        row = 3*nodes[:, None, None] + xyz[None, :, None]
        col = 3*nodes[:, None, None] + xyz[None, None, :]
        dense_diag.append((dense_offset + row*d + col).reshape(-1))
        dense_sizes.append(d*d)
        atom_offset += n
        dense_offset += dense_sizes[-1]
    i, j = torch.cat(source), torch.cat(target)
    return dict(i=i, j=j, node_owner=torch.cat(node_owners), dense_forward=torch.cat(dense_forward),
                dense_reverse=torch.cat(dense_reverse), dense_diag=torch.cat(dense_diag),
                dense_sizes=dense_sizes)


def assemble(layout, nodes, sizes, blocks, diagonal):
    entries = layout["dense_sizes"]
    flat = nodes.new_zeros(sum(entries))
    flat[layout["dense_forward"]] = blocks.reshape(-1)
    # Reverse indices already transpose the destination coordinates.
    flat[layout["dense_reverse"]] = blocks.reshape(-1)
    flat[layout["dense_diag"]] = diagonal.reshape(-1)
    return [part.reshape(3*n, 3*n) for part, n in zip(flat.split(entries), sizes)]
