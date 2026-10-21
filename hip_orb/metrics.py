"""Hessian and vibrational-space metrics for training validation."""

import torch


def vibrational_basis(positions, masses):
    """Remove mass-weighted translation and rotation by an SVD rank test."""
    root = masses.sqrt()
    center = (positions * masses[:, None]).sum(0) / masses.sum()
    r = positions - center
    eye = torch.eye(3, device=r.device, dtype=r.dtype)
    translations = (root[:, None, None] * eye[None]).reshape(-1, 3)
    rotations = torch.stack(
        [torch.linalg.cross(eye[k].expand_as(r), r) * root[:, None] for k in range(3)], dim=-1
    ).reshape(-1, 3)
    rigid = torch.cat((translations, rotations), dim=1)
    u, s, _ = torch.linalg.svd(rigid, full_matrices=True)
    rank = int((s > s.max() * 1e-6).sum())
    return u[:, rank:]


def hessian_metrics(predicted, target, positions, masses):
    error = predicted - target
    output = {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "relative_rmse": float(
            error.square().mean().sqrt() / target.square().mean().sqrt().clamp_min(1e-12)
        ),
    }
    basis = vibrational_basis(positions, masses)
    if not basis.shape[1]:
        return output
    weights = masses.repeat_interleave(3).rsqrt()

    def reduced(h):
        return basis.T @ (h * weights[:, None] * weights[None, :]) @ basis

    p, t = reduced(predicted), reduced(target)
    pe, pv = torch.linalg.eigh(p)
    te, tv = torch.linalg.eigh(t)
    k = min(3, len(te))
    output.update(
        vibrational_mae=float((p - t).abs().mean()),
        eigenvalue_mae=float((pe - te).abs().mean()),
        first_mode_abs_cosine=float((pv[:, 0] @ tv[:, 0]).abs()),
        first_mode_overlap=float((pv[:, 0] @ tv[:, 0]).square()),
        low_subspace_overlap=float((pv[:, :k].T @ tv[:, :k]).square().sum() / k),
        negative_modes_predicted=int((pe < -1e-6).sum()),
        negative_modes_target=int((te < -1e-6).sum()),
    )
    return output


def rotate_hessians(hessians, rotation):
    """Rotate atom-pair blocks without a dense Kronecker matrix."""
    if hessians.shape[-1] != hessians.shape[-2] or hessians.shape[-1] % 3:
        raise ValueError('Hessians require square dimensions divisible by three')
    if (hessians.is_cuda and hessians.dtype == torch.float32 and hessians.is_contiguous()
            and rotation.shape == (3, 3)
            and rotation.device == hessians.device and rotation.dtype == hessians.dtype
            and not (torch.is_grad_enabled() and (hessians.requires_grad or rotation.requires_grad))):
        from hip_orb.block_rotation import rotate_blocks
        return rotate_blocks(hessians, rotation.contiguous())
    n = hessians.shape[-1] // 3
    blocks = hessians.reshape(*hessians.shape[:-2], n, 3, n, 3).movedim(-3, -2)
    r = rotation[..., None, None, :, :]
    result = r @ blocks @ r.transpose(-1, -2)
    return result.movedim(-2, -3).reshape_as(hessians)


def batched_hessian_metrics(predicted, target, positions, masses):
    """Evaluate equal-size molecules with batched decompositions."""
    if not len(predicted):
        return []
    error = predicted - target
    rmse = error.square().mean((-2, -1)).sqrt()
    basic = torch.stack((error.abs().mean((-2, -1)), rmse,
                         rmse / target.square().mean((-2, -1)).sqrt().clamp_min(1e-12)), -1)
    # Keep each original SVD and projection. Batched projections select different
    # TF32 algorithms and can change nearly degenerate mode statistics.
    bases = [vibrational_basis(x, m) for x, m in zip(positions, masses, strict=True)]
    ranks = [predicted.shape[-1] - basis.shape[-1] for basis in bases]
    rows = [dict(zip(('mae', 'rmse', 'relative_rmse'), values, strict=True))
            for values in basic.cpu().tolist()]
    weights = masses.repeat_interleave(3, dim=-1).rsqrt()
    for rank in sorted(set(ranks)):
        if rank == predicted.shape[-1]:
            continue
        ids = [i for i, value in enumerate(ranks) if value == rank]
        def reduced(h, i):
            basis, inv = bases[i], weights[i]
            return basis.T @ (h * inv[:, None] * inv[None, :]) @ basis
        p = torch.stack([reduced(predicted[i], i) for i in ids])
        t = torch.stack([reduced(target[i], i) for i in ids])
        values, vectors = torch.linalg.eigh(torch.cat((p, t)))
        pe, te = values.chunk(2)
        pv, tv = vectors.chunk(2)
        k = min(3, te.shape[-1])
        cosine = (pv[..., 0] * tv[..., 0]).sum(-1)
        # Preserve the scalar dot-product algorithm for this sensitive statistic.
        cross = torch.stack([a[:, :k].T @ b[:, :k]
                             for a, b in zip(pv, tv, strict=True)])
        overlap = cross.square().sum((-2, -1)) / k
        metrics = torch.stack(((p - t).abs().mean((-2, -1)),
                               (pe - te).abs().mean(-1), cosine.abs(), cosine.square(),
                               overlap, (pe < -1e-6).sum(-1), (te < -1e-6).sum(-1)), -1)
        keys = ('vibrational_mae', 'eigenvalue_mae', 'first_mode_abs_cosine',
                'first_mode_overlap', 'low_subspace_overlap',
                'negative_modes_predicted', 'negative_modes_target')
        for i, values in zip(ids, metrics.cpu().tolist(), strict=True):
            row = dict(zip(keys, values, strict=True))
            for key in keys[-2:]:
                row[key] = int(row[key])
            rows[i].update(row)
    return rows
