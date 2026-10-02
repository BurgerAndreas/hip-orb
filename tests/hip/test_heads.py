from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hip_orb.coulomb import fixed_charge_hessian
from hip_orb.model import Features, HessianHead, HIPOrb
from hip_orb.prfo import saddle_search


def sample(n=3, latent=4):
    return Features(
        torch.randn(n, latent),
        torch.randn(2, latent),
        torch.tensor([0, 1]),
        torch.tensor([1, 0]),
        torch.randn(n, 3),
        torch.randn(n),
        torch.tensor([n]),
    )


@pytest.mark.parametrize("sigma", [None, 0.7])
def test_coulomb_matches_autograd_and_translation(sigma):
    torch.manual_seed(1)
    x = torch.randn(4, 3, dtype=torch.float64)
    q = torch.randn(4, dtype=x.dtype)
    i, j = torch.triu_indices(4, 4, 1)

    def energy(x):
        r = (x[i] - x[j]).norm(dim=-1)
        kernel = 1 / r if sigma is None else torch.erf(r / (sigma * 2**0.5)) / r
        return (14.3996 * q[i] * q[j] * kernel).sum()

    expected = torch.autograd.functional.hessian(energy, x).reshape(12, 12)
    got = fixed_charge_hessian(x, q, sigma=sigma)
    torch.testing.assert_close(got, expected)
    translation = torch.eye(3, dtype=x.dtype).repeat(4, 1)
    torch.testing.assert_close(
        got @ translation, torch.zeros(12, 3, dtype=x.dtype), atol=1e-12, rtol=0
    )


def test_coulomb_rotation():
    f = sample()
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    transform = torch.kron(torch.eye(3), r.contiguous())
    h = fixed_charge_hessian(f.positions, f.charges)
    rotated = fixed_charge_hessian(f.positions @ r.T, f.charges)
    torch.testing.assert_close(rotated, transform @ h @ transform.T, atol=1e-4, rtol=1e-4)


def test_full_graph_symmetry_gradients_and_cutoff():
    torch.manual_seed(2)
    f = sample()
    head = HessianHead(4, 12)
    h = head(f, residual_only=True)[0]
    torch.testing.assert_close(h, h.T)
    # Pair (0,2) has no original message edge but has a learned block.
    assert h[:3, 6:].abs().sum() > 0
    head.cutoff = 1e-6
    truncated = head(f, residual_only=True)[0]
    assert truncated[:3, 6:].abs().sum() == 0
    assert f.edges.shape == (2, 4)
    h.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_batch_and_permutation():
    torch.manual_seed(3)
    f = sample()
    head = HessianHead(4, 12)
    expected = head(f)[0]
    batched = Features(
        torch.cat([f.nodes, f.nodes]),
        torch.cat([f.edges, f.edges]),
        torch.cat([f.senders, f.senders + 3]),
        torch.cat([f.receivers, f.receivers + 3]),
        torch.cat([f.positions, f.positions]),
        torch.cat([f.charges, f.charges]),
        torch.tensor([3, 3]),
    )
    for h in head(batched):
        torch.testing.assert_close(h, expected)
    perm = torch.tensor([2, 0, 1])
    inverse = torch.argsort(perm)
    reordered = Features(
        f.nodes[perm],
        f.edges,
        inverse[f.senders],
        inverse[f.receivers],
        f.positions[perm],
        f.charges[perm],
        f.sizes,
    )
    cart = (perm[:, None] * 3 + torch.arange(3)).reshape(-1)
    torch.testing.assert_close(head(reordered)[0], expected[cart][:, cart])


def test_frozen_wrapper_preserves_energy_force_and_graph():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(4, 4)
            self.model.latent_dim = 4

        def predict(self, graph, split):
            return {"energy": self.model.weight.square().sum(), "forces": graph.forces.clone()}

    base = Base()
    graph = SimpleNamespace(forces=torch.randn(3, 3))
    before = base.predict(graph, False)
    wrapper = HIPOrb(base, None, hidden=12)
    f = sample()
    wrapper.features = lambda graph: f
    wrapper.train()
    result = wrapper.predict(graph)
    assert not base.training and all(not p.requires_grad for p in base.parameters())
    torch.testing.assert_close(result["energy"], before["energy"])
    torch.testing.assert_close(result["forces"], before["forces"])


def test_tiny_overfit_and_prfo():
    torch.manual_seed(4)
    f = sample(n=2)
    teacher = HessianHead(4, 12)
    target = teacher(f)[0].detach()
    student = HessianHead(4, 12)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.01)
    initial = float((student(f)[0] - target).square().mean().detach())
    for _ in range(250):
        optimizer.zero_grad()
        loss = (student(f)[0] - target).square().mean()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < initial * 0.001
    h = torch.diag(torch.tensor([-1.0, 2.0]))
    result = saddle_search(lambda x: (h @ x, h), torch.tensor([0.4, -0.3]))
    assert result["converged"] and result["negative_modes"] == 1
