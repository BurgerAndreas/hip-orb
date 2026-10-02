"""Analytical nonperiodic Hessian, holding predicted charges fixed."""

import math

import torch


def fixed_charge_hessian(positions, charges, constant=14.3996, sigma=None):
    """Return a (3N, 3N) tensor. This excludes all charge-response derivatives."""
    n = len(positions)
    i, j = torch.triu_indices(n, n, offset=1, device=positions.device)
    delta = positions[i] - positions[j]
    r = delta.norm(dim=-1)
    if bool((r <= 0).any()):
        raise ValueError("Coulomb pairs must have nonzero distance")
    if sigma is None:
        first = -r.pow(-2)
        second = 2 * r.pow(-3)
    else:
        if sigma <= 0:
            raise ValueError("Damping sigma must be positive")
        a = 1 / (sigma * math.sqrt(2))
        erf = torch.erf(a * r)
        derivative = 2 * a / math.sqrt(math.pi) * torch.exp(-(a * r).square())
        first = derivative / r - erf / r.square()
        second = -2 * a * a * derivative - 2 * derivative / r.square() + 2 * erf / r.pow(3)
    unit = delta / r[:, None]
    outer = unit[:, :, None] * unit[:, None, :]
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    blocks = (second - first / r)[:, None, None] * outer + (first / r)[:, None, None] * eye
    blocks = blocks * (constant * charges.reshape(-1)[i] * charges.reshape(-1)[j])[:, None, None]
    h = positions.new_zeros((n, n, 3, 3))
    h[i, j] = -blocks
    h[j, i] = -blocks
    h.index_put_((i, i), blocks, accumulate=True)
    h.index_put_((j, j), blocks, accumulate=True)
    return h.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
