"""Unweighted translation projection of Cartesian Hessians."""

import torch


def project_translations(hessian):
    """Compute P H P with block means, without constructing the projector."""
    dimension = hessian.shape[-1]
    if dimension < 3 or dimension % 3 or hessian.shape[-2] != dimension:
        raise ValueError("Translation projection requires square 3N-by-3N Hessians")
    n = dimension // 3
    blocks = hessian.reshape(*hessian.shape[:-2], n, 3, n, 3)
    row_mean = blocks.mean(dim=-2, keepdim=True)
    column_mean = blocks.mean(dim=-4, keepdim=True)
    grand_mean = row_mean.mean(dim=-4, keepdim=True)
    return (blocks - row_mean - column_mean + grand_mean).reshape_as(hessian)


def project_hessian_list(hessians):
    """Batch equal-sized molecules; keep each molecule's projection separate."""
    groups = {}
    for index, hessian in enumerate(hessians):
        groups.setdefault(hessian.shape[-1], []).append(index)
    projected = [None] * len(hessians)
    for indices in groups.values():
        values = project_translations(torch.stack([hessians[i] for i in indices]))
        for index, value in zip(indices, values.unbind(), strict=True):
            projected[index] = value
    return projected
