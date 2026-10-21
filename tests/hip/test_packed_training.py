import copy

import numpy as np
import pytest
import torch

from hip_orb.data import unpack_hessian
from hip_orb.losses import hessian_error_loss
from hip_orb.model import Features, HessianHead
from hip_orb.packed_training import (
    collate_packed,
    packed_blocks,
    packed_indices,
    packed_loss,
    prepare_packed,
)


@pytest.mark.parametrize("sizes", [(1, 2, 5), (3, 3, 3)])
@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("loss_type", ["block_rmse", "mse", "mae"])
def test_packed_predictions_loss_and_gradients(sizes, project, loss_type):
    torch.manual_seed(17)
    total = sum(sizes)
    feature = Features(
        torch.randn(total, 4, dtype=torch.float64),
        torch.randn(0, 4, dtype=torch.float64),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.randn(total, 3, dtype=torch.float64),
        torch.randn(total, dtype=torch.float64),
        torch.tensor(sizes),
    )
    head = HessianHead(4, 8, neighbor_pooling=True, translation_projection=project).double()
    reference = copy.deepcopy(head)
    model = type("Model", (), {"head": head})()
    indices = packed_indices(sizes)
    actual = packed_blocks(model, feature, indices, torch.tensor(sizes))
    feature2 = copy.deepcopy(feature)
    del feature2.packed_layout
    matrices = reference(feature2)
    expected = torch.cat(
        [
            m.reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1, 3, 3)
            for n, m in zip(sizes, matrices, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    targets = [torch.randn(3 * n, 3 * n, dtype=torch.float64) for n in sizes]
    target = torch.cat(
        [
            m.reshape(n, 3, n, 3).permute(0, 2, 1, 3).reshape(-1, 3, 3)
            for n, m in zip(sizes, targets, strict=True)
        ]
    )
    loss = packed_loss(actual, target, indices, loss_type)
    ref_loss = torch.stack(
        [
            hessian_error_loss((p - t)[None], loss_type)
            for p, t in zip(matrices, targets, strict=True)
        ]
    ).mean()
    torch.testing.assert_close(loss, ref_loss, atol=1e-7, rtol=1e-7)
    loss.backward()
    ref_loss.backward()
    for p, q in zip(head.parameters(), reference.parameters(), strict=True):
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-7)


def test_upper_indices_and_rotation_match_equal_size_path():
    samples = []
    for n in (1, 3, 7):
        samples.append(
            {
                "coords": np.random.default_rng(n).normal(size=(n, 3)).astype("float32"),
                "atomic_numbers": np.ones(n, dtype="int16"),
                "hessian_upper": np.arange(3 * n * (3 * n + 1) // 2, dtype="float32"),
                "charge": np.array(0),
                "spin": np.array(1),
                "omol_index": np.array(n),
            }
        )
    batch = prepare_packed(collate_packed(samples))
    blocks = batch["hessian_upper"][batch["indices"]["upper_ids"]].reshape(-1, 3, 3)
    ref = torch.cat(
        [
            unpack_hessian(torch.from_numpy(s["hessian_upper"]), len(s["coords"]))
            .reshape(len(s["coords"]), 3, len(s["coords"]), 3)
            .permute(0, 2, 1, 3)
            .reshape(-1, 3, 3)
            for s in samples
        ]
    )
    torch.testing.assert_close(blocks, ref, atol=0, rtol=0)
    r = batch["rotation"]
    torch.testing.assert_close(r @ r.T, torch.eye(3), atol=1e-6, rtol=1e-6)
    assert torch.linalg.det(r) > 0


@pytest.mark.parametrize(
    "sizes", [(1,), (1, 2, 5), (20, 20), (70, 3, 350, 20), (1, 40, 70, 150, 250, 350)]
)
def test_vectorized_device_indices_match_reference(sizes):
    from hip_orb.packed_training import device_indices

    actual = device_indices(torch.tensor(sizes), "cpu")
    expected = packed_indices(sizes)
    for key in expected:
        if key == "dense_sizes":
            assert expected[key] == actual[key]
        else:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-8, rtol=1e-7)


def test_precomputed_mixed_plan_covers_every_geometry_and_respects_budget():
    from hip_orb.benchmark_capacity import cohort_batches
    from hip_orb.benchmark_packed import coverage, mixed_batches

    groups = [(20, list(range(100))), (70, list(range(100, 160))), (350, list(range(160, 165)))]
    batches = mixed_batches(groups, batch_size=128, limit=4_000_000)
    assert coverage(batches) == coverage(cohort_batches(groups, 128, 4_000_000))
    sizes = {i: n for n, ids in groups for i in ids}
    assert len([i for b in batches for i in b]) == 165
    assert any(len({sizes[i] for i in b}) > 1 for b in batches)
    for b in batches:
        assert len(b) <= 128 and sum(9 * sizes[i] ** 2 for i in b) <= 4_000_000


def test_segmented_projection_is_self_adjoint_and_has_the_correct_backward():
    from hip_orb.packed_training import project_packed

    sizes = torch.tensor([5, 1, 3])
    indices = packed_indices(tuple(sizes.tolist()))
    torch.manual_seed(29)
    x = torch.randn(35, 3, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x)
    px = project_packed(x, indices, sizes)
    py = project_packed(y, indices, sizes)
    torch.testing.assert_close((px * y).sum(), (x * py).sum(), atol=1e-12, rtol=1e-12)
    (px * y).sum().backward()
    torch.testing.assert_close(x.grad, py, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(project_packed(px, indices, sizes), px, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("sizes", [(1,), (1, 2, 7), (3, 5)])
@pytest.mark.parametrize("loss_type", ["block_rmse", "mse", "mae"])
def test_compact_pair_loss_matches_dense_values_and_gradients(sizes, loss_type):
    from hip_orb.packed_training import compact_indices, compact_loss

    torch.manual_seed(19)
    count = sum(sizes)
    feature = Features(torch.randn(count, 4, dtype=torch.float64),
                       torch.empty(0, 4, dtype=torch.float64),
                       torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                       torch.randn(count, 3, dtype=torch.float64),
                       torch.randn(count, dtype=torch.float64) * 0.1, torch.tensor(sizes))
    head = HessianHead(4, 8, radial_features="bessel", all_to_all=True,
                       neighbor_pooling=True, pair_residual_layers=1).double()
    reference = copy.deepcopy(head)
    dense_indices = packed_indices(sizes)
    compact = compact_indices(sizes, "cpu")
    upper = torch.randn(sum((3*n)*(3*n+1)//2 for n in sizes), dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))[0]
    dense = packed_blocks(type("Model", (), {"head": reference})(), feature,
                          dense_indices, torch.tensor(sizes))
    target = rotation @ upper[dense_indices["upper_ids"]].reshape(-1, 3, 3) @ rotation.T
    expected = packed_loss(dense, target, dense_indices, loss_type)
    feature.packed_layout = compact
    pairs, diagonal = head._batched_forward(feature, as_tensor="pair-blocks")
    got = compact_loss(pairs, diagonal, upper, rotation, compact, loss_type)
    torch.testing.assert_close(got, expected, atol=1e-7, rtol=1e-7)
    ga = torch.autograd.grad(got, tuple(head.parameters()), allow_unused=True)
    gb = torch.autograd.grad(expected, tuple(reference.parameters()), allow_unused=True)
    for a, b in zip(ga, gb, strict=True):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_nonfinite_forward_stops_before_backward_or_optimizer(monkeypatch, bad_value):
    from types import SimpleNamespace

    from hip_orb import packed_training as pt

    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter])
    head = SimpleNamespace(translation_projection=False,
                           _batched_forward=lambda *args, **kwargs: (None, None))
    model = SimpleNamespace(head=head, adapter=None, _compact_pair_loss=True,
                            features=lambda *args, **kwargs: SimpleNamespace())
    batch = {'sizes': torch.tensor([1]), 'hessian_upper': torch.zeros(6),
             'rotation': torch.eye(3)}
    monkeypatch.setattr(pt, 'prepare_packed', lambda value, seed: value)
    monkeypatch.setattr(pt, 'packed_graph', lambda *args: None)
    monkeypatch.setattr(pt, 'compact_indices', lambda *args: {})
    monkeypatch.setattr(pt, 'compact_loss', lambda *args: parameter.sum() * bad_value)
    monkeypatch.setattr(optimizer, 'step', lambda *a, **k: pytest.fail('Optimizer must not run'))
    with pytest.raises(FloatingPointError, match='not finite'):
        pt.packed_update(model, optimizer, batch, device='cpu')
    assert parameter.grad is None
    torch.testing.assert_close(parameter.detach(), torch.ones(1))
    assert not optimizer.state


def test_mass_weighted_packed_and_compact_match_dense_values_and_gradients():
    from hip_orb.packed_training import compact_indices, compact_loss
    sizes = (2, 5, 3)
    masses = torch.tensor([1., 12., 16., 1., 14., 12., 32., 1., 1., 16.], dtype=torch.float64)
    rng = torch.Generator().manual_seed(83)
    matrices = [torch.randn(3*n, 3*n, dtype=torch.float64, generator=rng) for n in sizes]
    leaves = [m.requires_grad_() for m in matrices]
    matrices = [(m + m.T).div(2) for m in leaves]
    targets = [torch.randn_like(m) for m in matrices]
    targets = [(m + m.T)/2 for m in targets]
    upper = torch.cat([m[torch.triu_indices(len(m), len(m)).unbind()] for m in targets])
    blocks = torch.cat([m.reshape(n,3,n,3).permute(0,2,1,3).reshape(-1,3,3)
                        for n,m in zip(sizes,matrices)])
    target_blocks = torch.cat([m.reshape(n,3,n,3).permute(0,2,1,3).reshape(-1,3,3)
                               for n,m in zip(sizes,targets)])
    indices = packed_indices(sizes)
    full = packed_loss(blocks, target_blocks, indices, "mass_weighted_block_rmse", masses)
    pieces, offset = [], 0
    for n,p,t in zip(sizes,matrices,targets):
        pieces.append(hessian_error_loss((p-t)[None], "mass_weighted_block_rmse", masses[offset:offset+n][None]))
        offset += n
    reference = torch.stack(pieces).mean()
    ci = compact_indices(sizes,"cpu")
    offsets = torch.tensor([0, sizes[0]**2, sizes[0]**2+sizes[1]**2])
    node_offsets = torch.tensor([0, sizes[0], sizes[0]+sizes[1]])
    pair_owner = torch.repeat_interleave(torch.arange(3),torch.tensor([n*(n-1)//2 for n in sizes]))
    n = torch.tensor(sizes)[pair_owner]
    local_i = ci["i"]-node_offsets[pair_owner]
    local_j = ci["j"]-node_offsets[pair_owner]
    off = blocks[offsets[pair_owner]+local_i*n+local_j]
    node_owner = ci["node_owner"]
    local = torch.arange(sum(sizes))-node_offsets[node_owner]
    diag = blocks[offsets[node_owner]+local*(torch.tensor(sizes)[node_owner]+1)]
    compact = compact_loss(off,diag,upper,torch.eye(3,dtype=torch.float64),ci,"mass_weighted_block_rmse",masses)
    torch.testing.assert_close(full, reference, atol=1e-7,rtol=1e-7)
    torch.testing.assert_close(compact, reference, atol=1e-7,rtol=1e-7)
    refgrads = torch.autograd.grad(reference,leaves,retain_graph=True)
    for loss in (full,compact):
        for a,b in zip(torch.autograd.grad(loss,leaves,retain_graph=True),refgrads):
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)

@pytest.mark.parametrize('context', [False, True])
@pytest.mark.parametrize('radial', ['bessel', 'log_gaussian'])
def test_dense_extension_compact_loss_preserves_encoder_gradients(context, radial):
    from test_orb_extension import base_and_graph
    from hip_orb.model import HIPOrb
    from hip_orb.packed_training import compact_indices, compact_loss
    torch.manual_seed(101)
    base, graph = base_and_graph()
    model = HIPOrb(base, None, hidden=8, mlp_hidden=16, all_to_all=True,
                   radial_features=radial, radial_bases=4, pair_residual_layers=1,
                   neighbor_pooling=context, extra_orb_round=True,
                   orb_initialization='random', orb_message_graph='hessian',
                   orb_combine_backbone_edges=True)
    reference = copy.deepcopy(model)
    model._dense_indexing = True
    sizes = (3, 2)
    full_indices = packed_indices(sizes)
    ci = compact_indices(sizes, 'cpu')
    upper = torch.randn(sum((3*n)*(3*n+1)//2 for n in sizes))
    rotation = torch.linalg.qr(torch.randn(3,3))[0]
    feature = reference.features(graph, sizes=graph.n_node)
    blocks = packed_blocks(reference, feature, full_indices, graph.n_node)
    target = rotation @ upper[full_indices['upper_ids']].reshape(-1,3,3) @ rotation.T
    expected = packed_loss(blocks, target, full_indices)
    feature = model.features(graph, sizes=graph.n_node)
    feature.packed_layout = ci
    pairs, diagonal = model.head._batched_forward(feature, as_tensor='pair-blocks')
    actual = compact_loss(pairs, diagonal, upper, rotation, ci)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    ga = torch.autograd.grad(actual, tuple(model.head.parameters()))
    gb = torch.autograd.grad(expected, tuple(reference.head.parameters()))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=2e-6,rtol=2e-5)
