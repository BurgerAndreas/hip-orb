from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hip_orb.coulomb import fixed_charge_hessian
from hip_orb.model import Features, HessianHead, HIPOrb, backbone_latent_dim


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


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


def test_wrapper_reads_processor_dimension_without_top_level_attribute():
    base = nn.Module()
    base.model = nn.Module()
    base.model.gnn_stacks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    for layer in base.model.gnn_stacks:
        layer.latent_dim = 4
    before = {key: value.clone() for key, value in base.state_dict().items()}
    wrapper = HIPOrb(base, None, hidden=12)
    assert not hasattr(base.model, "latent_dim")
    assert wrapper.head.config["latent_dim"] == 4
    assert all(not parameter.requires_grad for parameter in base.parameters())
    for key, value in before.items():
        torch.testing.assert_close(base.state_dict()[key], value, atol=0, rtol=0)
    assert wrapper.head(sample())[0].shape == (9, 9)


def test_dimension_detection_rejects_missing_or_inconsistent_metadata():
    with pytest.raises(ValueError, match="consistent"):
        backbone_latent_dim(nn.Module())
    backbone = nn.Module()
    backbone.gnn_stacks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
    backbone.gnn_stacks[0].latent_dim = 4
    backbone.gnn_stacks[1].latent_dim = 8
    with pytest.raises(ValueError, match="consistent"):
        backbone_latent_dim(backbone)


@pytest.mark.parametrize("n", [1, 3, 12])
def test_fused_pairs_preserve_outputs_and_gradients(n):
    import copy
    f = sample(n=max(n, 2))
    if n == 1:
        f = Features(f.nodes[:1], f.edges[:0], f.senders[:0], f.receivers[:0],
                     f.positions[:1], f.charges[:1], torch.tensor([1]))
    fused = HessianHead(4, 12)
    original = copy.deepcopy(fused)
    original.fuse_pairs = False
    got, expected = fused(f)[0], original(f)[0]
    torch.testing.assert_close(got, expected)
    got.square().mean().backward()
    expected.square().mean().backward()
    for left, right in zip(fused.parameters(), original.parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad)


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


def test_independent_bessel_radial_features_are_finite_and_cut_off_smoothly():
    features = sample(n=3)
    head = HessianHead(4, 12, radial_features="bessel", radial_cutoff=4.0,
                       radial_bases=8)
    distance = torch.tensor([1.0, 3.999, 4.0, 5.0])
    radial = head.radial_embedding(distance)
    assert radial.shape == (4, 8)
    assert torch.isfinite(radial).all()
    assert radial[2:].count_nonzero() == 0
    h = head(features, residual_only=True)[0]
    assert h.shape == (9, 9) and torch.isfinite(h).all()
    h.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    batched = Features(
        torch.cat((features.nodes, features.nodes)), torch.cat((features.edges, features.edges)),
        torch.cat((features.senders, features.senders + 3)),
        torch.cat((features.receivers, features.receivers + 3)),
        torch.cat((features.positions, features.positions)),
        torch.cat((features.charges, features.charges)), torch.tensor([3, 3]),
    )
    head.zero_grad(set_to_none=True)
    batch_h = head(batched, residual_only=True)
    torch.testing.assert_close(batch_h[0], head(features, residual_only=True)[0])
    torch.testing.assert_close(batch_h[1], head(features, residual_only=True)[0])


@pytest.mark.parametrize("context", [False, True])
def test_batch_and_permutation(context):
    torch.manual_seed(3)
    f = sample()
    head = HessianHead(4, 12, radial_features="bessel" if context else "log",
                       pair_residual_layers=2 if context else 0, neighbor_pooling=context)
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


def test_context_diagonal_uses_neighbors_and_all_heads_receive_gradients():
    from dataclasses import replace
    torch.manual_seed(23)
    features = sample()
    head = HessianHead(4, 12, radial_features="bessel", pair_residual_layers=2,
                       neighbor_pooling=True)
    initial = head(features, residual_only=True)[0]
    positions = features.positions.clone()
    positions[1] += torch.tensor([1.0, 0.0, 0.0])
    changed = head(replace(features, positions=positions), residual_only=True)[0]
    assert not torch.allclose(initial[:3, :3], changed[:3, :3])
    torch.testing.assert_close(initial, initial.T)
    initial.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    singleton = Features(features.nodes[:1], features.edges[:0], features.senders[:0],
                         features.receivers[:0], features.positions[:1], features.charges[:1],
                         torch.tensor([1]))
    assert torch.isfinite(head(singleton, residual_only=True)[0]).all()


@pytest.mark.parametrize("rounds,internal", [(2, 256), (1, 1024)])
def test_context_capacity_variants_preserve_feature_width_and_checkpoint(tmp_path, rounds, internal):
    from dataclasses import replace
    torch.manual_seed(37)
    f = sample()
    head = HessianHead(4, 256, radial_features="bessel", pair_residual_layers=2,
                       neighbor_pooling=True, context_rounds=rounds, mlp_hidden=internal)
    assert head.pair_adapter[0].out_features == internal
    assert head.pair_adapter[-1].out_features == 256
    assert head.pair_residual[0].update[0].out_features == internal
    assert head.pair_update[-1].out_features == 256
    predicted = head(f, residual_only=True)[0]
    torch.testing.assert_close(predicted, predicted.T)
    predicted.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    perm = torch.tensor([2, 0, 1]); inverse = torch.argsort(perm)
    reordered = replace(f, nodes=f.nodes[perm], charges=f.charges[perm], positions=f.positions[perm],
                        senders=inverse[f.senders], receivers=inverse[f.receivers])
    cart = (perm[:, None] * 3 + torch.arange(3)).flatten()
    torch.testing.assert_close(head(reordered, residual_only=True)[0], predicted[cart][:, cart])
    batched = Features(torch.cat((f.nodes, f.nodes)), torch.cat((f.edges, f.edges)),
        torch.cat((f.senders, f.senders + 3)), torch.cat((f.receivers, f.receivers + 3)),
        torch.cat((f.positions, f.positions)), torch.cat((f.charges, f.charges)), torch.tensor([3, 3]))
    for value in head(batched, residual_only=True):
        torch.testing.assert_close(value, predicted)
    checkpoint = tmp_path / "head.pt"
    torch.save({"head": head.state_dict(), "head_config": head.config}, checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    restored = HessianHead(**saved["head_config"])
    restored.load_state_dict(saved["head"])
    torch.testing.assert_close(restored(f, residual_only=True)[0], predicted)


def test_second_context_round_uses_refined_pairs_to_update_diagonals():
    torch.manual_seed(39)
    f = sample()
    one = HessianHead(4, 8, radial_features="bessel", neighbor_pooling=True)
    torch.manual_seed(39)
    two = HessianHead(4, 8, radial_features="bessel", neighbor_pooling=True, context_rounds=2)
    two.load_state_dict(one.state_dict(), strict=False)
    before_one, before_two = one(f, residual_only=True)[0], two(f, residual_only=True)[0]
    with torch.no_grad():
        one.pair_update[-1].bias.add_(1.0)
        two.pair_update[-1].bias.add_(1.0)
    # First-round diagonals precede pair refinement; second-round diagonals use its output.
    torch.testing.assert_close(one(f, residual_only=True)[0][:3, :3], before_one[:3, :3])
    assert not torch.allclose(two(f, residual_only=True)[0][:3, :3], before_two[:3, :3])


@pytest.mark.parametrize("conditioned", [False, True])
def test_extra_orb_round_uses_original_graph_and_trains_only_the_extension(conditioned):
    from orb_models.common.models.gns import AttentionInteractionNetwork
    from orb_models.common.models.nn_util import get_cutoff
    from copy import deepcopy

    class Conditioner(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(4))

        def forward(self, graph):
            return graph.node_features["latent"] * 0 + self.bias, graph.edge_features["latent"] * 0 + self.bias

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.gnn_stacks = nn.ModuleList([AttentionInteractionNetwork(
                4, 2, 8, distance_cutoff=True, attention_gate="sigmoid",
                conditioning=("additive", "additive") if conditioned else "none")])
            self.conditioner = Conditioner() if conditioned else None

        def forward(self, graph):
            cond = self.conditioner(graph) if self.conditioner else (None, None)
            nodes, edges = self.gnn_stacks[0](graph.node_features["latent"], graph.edge_features["latent"],
                graph.senders, graph.receivers, get_cutoff(graph.edge_features["vectors"].norm(dim=-1)),
                cond_nodes=cond[0], cond_edges=cond[1])
            return {"node_features": nodes, "edge_features": edges}

    class Charges(nn.Module):
        def forward(self, nodes, graph):
            return nodes[:, 0]

    base = nn.Module()
    base.model = Backbone()
    base.heads = nn.ModuleDict({"latent_charges": Charges()})
    base.coulomb_module = SimpleNamespace(coulomb_constant=14.3996, direct_coulomb_erf_damping_sigma=None)
    graph = SimpleNamespace(system_features={"pbc": torch.tensor([False])},
        node_features={"positions": torch.randn(3, 3), "latent": torch.randn(3, 4)},
        edge_features={"vectors": torch.tensor([[1., 0, 0], [-1., 0, 0]]), "latent": torch.randn(2, 4)},
        senders=torch.tensor([0, 1]), receivers=torch.tensor([1, 0]), n_node=torch.tensor([3]))
    model = HIPOrb(base, None, hidden=12, radial_features="bessel",
                   pair_residual_layers=2, neighbor_pooling=True, extra_orb_round=True)
    layer = model.head.orb_message_layer
    for original, extra in zip(base.model.gnn_stacks[-1].parameters(), layer.parameters()):
        torch.testing.assert_close(original, extra)
        assert original.data_ptr() != extra.data_ptr()
    before = deepcopy(base.state_dict())
    with torch.no_grad():
        frozen = base.model(graph)
        cond = base.model.conditioner(graph) if conditioned else (None, None)
        expected = layer(frozen["node_features"], frozen["edge_features"], graph.senders, graph.receivers,
                         get_cutoff(graph.edge_features["vectors"].norm(dim=-1)),
                         cond_nodes=cond[0], cond_edges=cond[1])
    captured = []
    hook = layer.register_forward_pre_hook(lambda module, inputs: captured.append(inputs))
    features = model.features(graph)
    hook.remove()
    assert captured[0][2] is graph.senders and captured[0][3] is graph.receivers
    torch.testing.assert_close(features.nodes, expected[0])
    torch.testing.assert_close(features.edges, expected[1])
    torch.testing.assert_close(features.charges, frozen["node_features"][:, 0])
    assert features.nodes.requires_grad and features.edges.requires_grad
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    model.train()
    optimizer.zero_grad()
    model(graph)[0].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert all(p.grad is None for p in base.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(base.model.gnn_stacks[-1].parameters(), layer.parameters()))
    for key, value in before.items():
        torch.testing.assert_close(base.state_dict()[key], value, atol=0, rtol=0)
    restored = HIPOrb(deepcopy(base), None, hidden=12, radial_features="bessel",
                      pair_residual_layers=2, neighbor_pooling=True, extra_orb_round=True)
    restored.head.load_state_dict(model.head.state_dict())
    torch.testing.assert_close(restored(graph)[0], model(graph)[0])


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


def test_tiny_overfit():
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
@pytest.mark.parametrize("n", [1, 2, 50, 70, 110, 200, 350])
def test_generated_sizes_without_dataset(n):
    torch.manual_seed(n)
    features = Features(
        torch.randn(n, 4),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.randn(n, 3),
        torch.randn(n),
        torch.tensor([n]),
    )
    head = HessianHead(4, 12).eval()
    with torch.no_grad():
        h = head(features)[0]
    assert h.shape == (3 * n, 3 * n)
    assert torch.isfinite(h).all()
    torch.testing.assert_close(h, h.T)
    features.positions = features.positions + torch.tensor([2.0, -3.0, 1.0])
    with torch.no_grad():
        shifted = head(features)[0]
    torch.testing.assert_close(shifted, h, atol=2e-3, rtol=2e-4)


def test_residual_addition_has_no_double_counting():
    features = sample()
    head = HessianHead(4, 12)
    for parameter in head.parameters():
        nn.init.zeros_(parameter)
    expected = fixed_charge_hessian(features.positions, features.charges)
    torch.testing.assert_close(head(features)[0], expected)
    assert torch.count_nonzero(head(features, residual_only=True)[0]) == 0


def test_charge_response_is_not_fixed_charge_curvature():
    positions = torch.tensor([[0.0, 0, 0], [1.4, 0.2, 0]], dtype=torch.float64)

    def charges(x):
        distance = (x[0] - x[1]).norm()
        return torch.stack((distance.sin(), -distance.sin()))

    def energy(x):
        q = charges(x)
        return 14.3996 * q[0] * q[1] / (x[0] - x[1]).norm()

    full = torch.autograd.functional.hessian(energy, positions).reshape(6, 6)
    fixed = fixed_charge_hessian(positions, charges(positions).detach())
    assert (full - fixed).abs().max() > 1


def test_head_input_gradcheck():
    torch.manual_seed(11)
    features = sample(n=2)
    features.nodes = features.nodes.double().requires_grad_()
    features.edges = features.edges.double()
    features.positions = features.positions.double()
    features.charges = features.charges.double()
    head = HessianHead(4, 6).double()

    def evaluate(nodes):
        from dataclasses import replace

        return head(replace(features, nodes=nodes), residual_only=True)[0]

    assert torch.autograd.gradcheck(evaluate, (features.nodes,))


def test_pinned_version_metadata():
    from orb_models import __version__

    assert __version__.split("+")[0].lstrip("v") == "0.7.0"
