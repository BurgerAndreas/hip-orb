"""Small partitioned rational-function step for controlled pilot comparisons."""

import torch


def prfo_step(gradient, hessian, trust_radius=0.1, previous_mode=None):
    """Maximize along one mode; minimize in its orthogonal complement."""
    if trust_radius <= 0:
        raise ValueError("Trust radius must be positive")
    values, vectors = torch.linalg.eigh((hessian + hessian.T) * 0.5)
    index = 0 if previous_mode is None else int((vectors.T @ previous_mode).abs().argmax())
    order = torch.cat(
        (
            values.new_tensor([index], dtype=torch.long),
            torch.arange(len(values), device=values.device)[
                torch.arange(len(values), device=values.device) != index
            ],
        )
    )
    values, vectors = values[order], vectors[:, order]
    g = vectors.T @ gradient
    step = torch.zeros_like(g)

    def partition(v, force, uphill):
        if len(v) == 0 or bool(force.norm() < 1e-14):
            return torch.zeros_like(force)
        matrix = v.new_zeros((len(v) + 1, len(v) + 1))
        matrix[:-1, :-1] = torch.diag(v)
        matrix[:-1, -1] = force
        matrix[-1, :-1] = force
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        root = -1 if uphill else 0
        denominator = eigenvectors[-1, root]
        if denominator.abs() < 1e-12:
            return (
                -force / (v - eigenvalues[root]).clamp(max=-1e-8)
                if uphill
                else -force / (v - eigenvalues[root]).clamp(min=1e-8)
            )
        return eigenvectors[:-1, root] / denominator

    step[:1] = partition(values[:1], g[:1], True)
    step[1:] = partition(values[1:], g[1:], False)
    cartesian = vectors @ step
    cartesian *= min(1.0, trust_radius / max(float(cartesian.norm()), 1e-14))
    return cartesian, vectors[:, 0]


def saddle_search(evaluate, initial, max_steps=100, tolerance=1e-3, trust_radius=0.1):
    """Evaluate returns (gradient, Hessian) in the coordinates being optimized."""
    x = initial.clone()
    mode = None
    for iteration in range(max_steps + 1):
        gradient, hessian = evaluate(x)
        if bool(gradient.square().mean().sqrt() < tolerance):
            negative = int((torch.linalg.eigvalsh(hessian) < -1e-6).sum())
            return {"converged": negative == 1, "steps": iteration, "negative_modes": negative}
        if iteration < max_steps:
            step, mode = prfo_step(gradient, hessian, trust_radius, mode)
            x += step
    return {"converged": False, "steps": max_steps}
