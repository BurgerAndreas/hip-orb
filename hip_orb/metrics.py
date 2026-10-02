"""Hessian and vibrational-space metrics for pilot evaluation."""

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
        first_mode_overlap=float((pv[:, 0] @ tv[:, 0]).square()),
        low_subspace_overlap=float((pv[:, :k].T @ tv[:, :k]).square().sum() / k),
        negative_modes_predicted=int((pe < -1e-6).sum()),
        negative_modes_target=int((te < -1e-6).sum()),
    )
    return output
