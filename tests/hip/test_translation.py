import pytest
import torch

from hip_orb.model import Features, HessianHead
from hip_orb.translation import project_translations, project_hessian_list


@pytest.mark.parametrize("n", [1, 2, 7])
def test_projection_matches_dense_projector_and_preserves_symmetry(n):
    torch.manual_seed(n)
    h = torch.randn(2, 3*n, 3*n, dtype=torch.float64)
    h = (h + h.transpose(-1, -2)) * 0.5
    translations = torch.eye(3, dtype=h.dtype).repeat(n, 1)
    p = torch.eye(3*n, dtype=h.dtype) - translations @ translations.T / n
    got = project_translations(h)
    torch.testing.assert_close(got, p @ h @ p)
    torch.testing.assert_close(got, got.transpose(-1, -2))
    torch.testing.assert_close(got @ translations, torch.zeros(2, 3*n, 3, dtype=h.dtype), atol=1e-12, rtol=0)
    torch.testing.assert_close(project_translations(got), got)


def test_projection_commutes_with_rotation_and_atom_permutation():
    torch.manual_seed(8)
    h = torch.randn(9, 9, dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3, dtype=h.dtype))
    q = torch.kron(torch.eye(3, dtype=h.dtype), rotation.contiguous())
    torch.testing.assert_close(project_translations(q @ h @ q.T), q @ project_translations(h) @ q.T)
    indices = (torch.tensor([2, 0, 1])[:, None]*3 + torch.arange(3)).flatten()
    torch.testing.assert_close(project_translations(h[indices][:, indices]), project_translations(h)[indices][:, indices])


def test_projection_gradients_and_mixed_size_batch_isolation():
    h = torch.randn(2, 6, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(project_translations, (h,))
    samples = [h[0], torch.randn(9, 9, dtype=h.dtype), h[1]]
    for original, projected in zip(samples, project_hessian_list(samples), strict=True):
        torch.testing.assert_close(projected, project_translations(original))


def test_head_applies_projection_after_coulomb_and_supports_backpropagation():
    torch.manual_seed(10)
    features = Features(torch.randn(5, 4), torch.empty(0, 4), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.randn(5, 3), torch.randn(5), torch.tensor([2, 3]))
    plain = HessianHead(4, 8)
    constrained = HessianHead(4, 8, translation_projection=True)
    constrained.load_state_dict(plain.state_dict())
    got = constrained(features)
    for predicted, raw in zip(got, plain(features), strict=True):
        torch.testing.assert_close(predicted, project_translations(raw))
        n = predicted.shape[-1] // 3
        translations = torch.eye(3).repeat(n, 1)
        torch.testing.assert_close(predicted @ translations, torch.zeros(3*n, 3), atol=1e-4, rtol=0)
    sum(h.square().mean() for h in got).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in constrained.parameters())
