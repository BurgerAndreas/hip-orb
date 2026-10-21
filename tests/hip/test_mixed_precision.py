import copy
from collections import OrderedDict

import pytest
import torch

from hip_orb.mixed_precision import LinearPolicy, linear_precision, quantize_fp8


class Small(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.model = torch.nn.Sequential(
            torch.nn.Linear(17, 32), torch.nn.SiLU(), torch.nn.Linear(32, 17)
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(17, 32), torch.nn.SiLU(), torch.nn.Linear(32, 9)
        )


def test_bf16_policy_keeps_output_layer_and_parameters_fp32():
    torch.manual_seed(47)
    model = Small()
    reference = copy.deepcopy(model)
    x = torch.randn(7, 17, requires_grad=True)
    state_keys = list(model.state_dict())
    with linear_precision(model, head=LinearPolicy("bf16")) as stats:
        y = model.head(x)
        expected = reference.head[2](
            reference.head[1](
                torch.nn.functional.linear(
                    x.bfloat16(),
                    reference.head[0].weight.bfloat16(),
                    reference.head[0].bias.bfloat16(),
                ).float()
            )
        )
        torch.testing.assert_close(y, expected, atol=0, rtol=0)
        y.square().sum().backward()
        expected.square().sum().backward()
        for p, q in zip(model.head.parameters(), reference.head.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
            assert p.dtype == p.grad.dtype == torch.float32
        assert stats["head_layers"] == ["head.0"]
        assert stats["calls"]["bf16"] == 1
    assert list(model.state_dict()) == state_keys
    torch.testing.assert_close(model.head(x), reference.head(x), atol=0, rtol=0)


def test_policy_restores_after_error_and_frozen_weight_cache_refreshes():
    model = Small()
    before = model.base.model[0].forward
    x = torch.randn(4, 17)
    model.base.requires_grad_(False)
    with (
        pytest.raises(RuntimeError, match="stop"),
        linear_precision(model, backbone=LinearPolicy("bf16")),
    ):
        first = model.base.model(x)
        with torch.no_grad():
            model.base.model[0].weight.add_(1)
        assert not torch.equal(model.base.model(x), first)
        raise RuntimeError("stop")
    assert model.base.model[0].forward == before


def test_fp8_scaling_zero_and_small_values():
    for magnitude in (0, 1e-20, 1e-5, 1, 1e12):
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0]) * magnitude
        q, s = quantize_fp8(x)
        assert torch.isfinite(q.float()).all() and s > 0
        torch.testing.assert_close(q.float() * s, x, atol=magnitude * 0.05, rtol=0.05)


def test_square_selection_preserves_input_and_output_projections():
    model = Small()
    model.head = torch.nn.Sequential(
        torch.nn.Linear(17, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 512),
        torch.nn.SiLU(),
        torch.nn.Linear(512, 9),
    )
    with linear_precision(model, head=LinearPolicy("bf16", selection="square")) as stats:
        model.head(torch.randn(4, 17)).sum().backward()
        assert stats["head_layers"] == ["head.2"]
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in model.head.parameters()
        )


def test_row_scaling_retains_large_and_small_rows_without_overflow():
    from hip_orb.precision_scaled import row_quantize

    x = torch.randn(17, 23) * torch.logspace(-12, 12, 17)[:, None]
    for dtype in (torch.float16, torch.float8_e4m3fn):
        q, s = row_quantize(x, dtype, padded=True, fused=False)
        assert q.shape == (32, 32) and s.shape == (32, 1)
        assert torch.isfinite(q.float()).all() and torch.isfinite(s).all()
        reconstructed = q.float() * s
        relative = (reconstructed[:17, :23] - x).norm(dim=1) / x.norm(dim=1)
        assert (relative < (0.001 if dtype == torch.float16 else 0.05)).all()
        assert not reconstructed[17:].any() and not reconstructed[:, 23:].any()


def test_named_selection_changes_only_the_selected_layer():
    model = Small()
    with linear_precision(model, backbone=LinearPolicy("bf16", include=("base.model.0",))) as stats:
        model.base.model(torch.randn(4, 17))
        assert stats["backbone_layers"] == ["base.model.0"]


def test_fusion_restores_activation_after_error():
    model = Small()
    before = model.head[1].forward
    with (
        pytest.raises(RuntimeError, match="stop"),
        linear_precision(model, head=LinearPolicy("fp16_f32", fuse_silu=True)),
    ):
        assert model.head[1].forward != before
        raise RuntimeError("stop")
    assert model.head[1].forward == before


def test_orb_hidden_activations_return_to_fp32_before_normalization():
    model = Small()
    model.base.model = torch.nn.Sequential(
        OrderedDict(
            mlp=torch.nn.Sequential(
                OrderedDict(
                    [
                        ("NN-0", torch.nn.Linear(17, 32)),
                        ("A-0", torch.nn.SiLU()),
                        ("NN-1", torch.nn.Linear(32, 32)),
                        ("A-1", torch.nn.SiLU()),
                        ("NN-2", torch.nn.Linear(32, 17)),
                    ]
                )
            ),
            norm=torch.nn.LayerNorm(17),
        )
    )
    dtypes = {}
    hooks = [
        layer.register_forward_hook(
            lambda module, args, out, key=name: dtypes.update({key: out.dtype})
        )
        for name, layer in model.base.model.mlp.named_children()
    ]
    with linear_precision(
        model,
        backbone=LinearPolicy("fp16", include=("base.model.mlp.NN-*",), activation_dtype="fp16"),
    ):
        y = model.base.model(torch.randn(4, 17))
        y.square().sum().backward()
    assert dtypes["NN-0"] == dtypes["NN-1"] == torch.float16
    assert dtypes["NN-2"] == y.dtype == torch.float32
    for hook in hooks:
        hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GEMMs need CUDA")
def test_half_cache_refreshes_without_reusing_autograd_graph():
    from hip_orb.mixed_precision import half_linear_float_output
    from hip_orb.precision_kernels import fused_half_silu

    for operation in (half_linear_float_output, fused_half_silu):
        w = torch.randn(32, 17, device="cuda", requires_grad=True)
        cache = {}
        for _ in range(2):
            x = torch.randn(7, 17, device="cuda", requires_grad=True)
            y = operation(x, w, cache=cache)
            ref = operation(x, w)
            torch.testing.assert_close(y, ref, atol=0, rtol=0)
            grad = torch.randn_like(y)
            actual = torch.autograd.grad(y, (x, w), grad)
            expected = torch.autograd.grad(ref, (x, w), grad)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with torch.no_grad():
                w.add_(0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GEMMs need CUDA")
def test_empty_precision_linears_keep_zero_parameter_gradients():
    from hip_orb.mixed_precision import fp8_linear, half_linear_float_output
    from hip_orb.precision_kernels import fused_half_silu
    from hip_orb.precision_scaled import scaled_linear

    for operation in (fp8_linear, half_linear_float_output, scaled_linear, fused_half_silu):
        x = torch.empty(0, 17, device="cuda", requires_grad=True)
        w = torch.randn(9, 17, device="cuda", requires_grad=True)
        b = torch.randn(9, device="cuda", requires_grad=True)
        y = operation(x, w, b)
        assert y.shape == (0, 9)
        y.sum().backward()
        assert x.grad is not None and x.grad.numel() == 0
        assert w.grad is not None and not w.grad.any()
        assert b.grad is not None and not b.grad.any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GEMMs need CUDA")
def test_scaled_native_forward_and_backward():
    from hip_orb.precision_audit import scaled_operator_audit

    assert all(row["passed"] for row in scaled_operator_audit())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused SiLU needs CUDA")
def test_fused_native_forward_and_backward():
    from hip_orb.precision_audit import fused_operator_audit

    assert all(row["passed"] for row in fused_operator_audit())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native FP8 needs CUDA")
def test_native_fp8_forward_and_surrogate_backward():
    from hip_orb.precision_audit import operator_audit

    assert all(r["passed"] for r in operator_audit())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native GEMMs need CUDA")
def test_native_gemms_reject_unsupported_higher_derivatives():
    from hip_orb.mixed_precision import fp8_linear, half_linear_float_output
    from hip_orb.precision_kernels import fused_half_silu
    from hip_orb.precision_scaled import scaled_linear

    for operation in (fp8_linear, half_linear_float_output, scaled_linear, fused_half_silu):
        x = torch.randn(17, 23, device="cuda", requires_grad=True)
        w = torch.randn(9, 23, device="cuda", requires_grad=True)
        with pytest.raises(RuntimeError, match="first-order gradients only"):
            torch.autograd.grad(operation(x, w).sum(), x, create_graph=True)


def test_aligned_weight_cache_zero_padding_and_refresh():
    from hip_orb.mixed_precision import cached_half_weight

    weight = torch.nn.Parameter(torch.randn(32, 791))
    cache = {}
    first = cached_half_weight(weight, torch.float16, cache, alignment=16)
    assert first.shape == (32, 800)
    assert not first[:, 791:].any()
    assert cached_half_weight(weight, torch.float16, cache, alignment=16) is first
    with torch.no_grad():
        weight.add_(1)
    second = cached_half_weight(weight, torch.float16, cache, alignment=16)
    torch.testing.assert_close(second[:, :791], weight.half(), atol=0, rtol=0)
    assert second is not first and not second[:, 791:].any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA half-input FP32-output GEMM")
@pytest.mark.parametrize("fused", [False, True])
def test_aligned_half_gemm_values_and_gradients(fused):
    from hip_orb.mixed_precision import half_linear_float_output
    from hip_orb.precision_kernels import fused_half_silu

    torch.manual_seed(92)
    x = torch.randn(2, 17, 791, device="cuda", requires_grad=True)
    w = (torch.randn(64, 791, device="cuda") * 0.03).requires_grad_()
    b = torch.randn(64, device="cuda", requires_grad=True)
    operation = fused_half_silu if fused else half_linear_float_output
    kwargs = {} if fused else {"dtype": torch.float16}
    expected = operation(x, w, b, **kwargs)
    got = operation(x, w, b, alignment=16, **kwargs)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=2e-5)
    gradient = torch.randn_like(got)
    ga = torch.autograd.grad(got, (x, w, b), gradient)
    gb = torch.autograd.grad(expected, (x, w, b), gradient)
    for a, c in zip(ga, gb, strict=True):
        assert a.shape == c.shape
        assert (a - c).norm() / c.norm().clamp_min(1e-12) < 0.001


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused half GEMM")
def test_restricted_input_backward_preserves_trainable_slice_and_parameter_gradients():
    from hip_orb.precision_kernels import fused_half_silu

    torch.manual_seed(83)
    frozen = torch.randn(71, 768, device="cuda")
    radial = torch.randn(71, 17, device="cuda", requires_grad=True)
    x = torch.cat((frozen, radial), dim=1)
    w = torch.randn(64, 785, device="cuda", requires_grad=True) * 0.02
    b = torch.randn(64, device="cuda", requires_grad=True)
    reference = fused_half_silu(x, w, b, alignment=16)
    got = fused_half_silu(x, w, b, alignment=16, input_grad_columns=(768, 785))
    torch.testing.assert_close(got, reference, atol=0, rtol=0)
    probe = torch.randn_like(got)
    ga = torch.autograd.grad(got, (radial, w, b), probe, retain_graph=True)
    gb = torch.autograd.grad(reference, (radial, w, b), probe)
    for a, c in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, c, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused AdamW")
def test_fused_optimizer_refreshes_cached_half_weights():
    from hip_orb.mixed_precision import cached_half_weight

    weight = torch.nn.Parameter(torch.randn(32, 791, device="cuda"))
    cache = {}
    first = cached_half_weight(weight, torch.float16, cache, alignment=16)
    optimizer = torch.optim.AdamW([weight], lr=0.1, fused=True)
    weight.grad = torch.ones_like(weight)
    optimizer.step()
    from hip_orb.mixed_precision import invalidate_trainable_precision_caches

    model = torch.nn.Module()
    model._hip_precision_trainable_caches = [cache]
    invalidate_trainable_precision_caches(model)
    second = cached_half_weight(weight, torch.float16, cache, alignment=16)
    assert second is not first
    torch.testing.assert_close(second[:, :791], weight.half(), atol=0, rtol=0)
    assert not second[:, 791:].any()


def test_versionless_updates_invalidate_trainable_casts_and_restore_scope():
    from hip_orb.mixed_precision import invalidate_trainable_precision_caches

    model = Small()
    x = torch.randn(4, 17)
    with linear_precision(model, head=LinearPolicy("bf16")):
        before = model.head(x)
        version = model.head[0].weight._version
        model.head[0].weight.data.add_(1)
        assert model.head[0].weight._version == version
        invalidate_trainable_precision_caches(model)
        after = model.head(x)
        assert not torch.equal(before, after)
    assert not hasattr(model, "_hip_precision_trainable_caches")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused head kernels need CUDA")
@pytest.mark.parametrize("shape", [(7, 17, 9), (71, 791, 64), (513, 1024, 256)])
@pytest.mark.parametrize("half_output", [False, True])
def test_head_fusion_forward_and_backward(shape, half_output):
    from hip_orb.precision_kernels import fused_half_silu

    torch.manual_seed(159)
    m, k, n = shape
    x = torch.randn(m, k, device="cuda", requires_grad=True)
    w = (torch.randn(n, k, device="cuda") / k**0.5).requires_grad_()
    b = torch.randn(n, device="cuda", requires_grad=True)
    reference = fused_half_silu(x, w, b, alignment=16, half_output=half_output)
    probe = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, (x, w, b), probe)
    actual = fused_half_silu(x, w, b, alignment=16, half_output=half_output,
                             fused_input_cast=True)
    error = (actual.float() - reference.float()).norm() / reference.float().norm()
    assert error < 0.001
    got = torch.autograd.grad(actual, (x, w, b), probe)
    for a, e in zip(got, expected, strict=True):
        assert (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-12) < 0.002


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused head kernels need CUDA")
def test_fused_cast_padding_with_strided_inputs():
    from hip_orb.head_fusion import cast_half_padded

    for dtype in (torch.float32, torch.float16):
        x = torch.randn(19, 34, device="cuda", dtype=dtype)[:, ::2]
        got = cast_half_padded(x, 32)
        torch.testing.assert_close(got[:, :17], x.half(), atol=0, rtol=0)
        assert not got[:, 17:].any()
        assert cast_half_padded(x, 17).shape == x.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Fused pair inputs need CUDA")
def test_fused_pair_inputs_with_missing_edges_and_radial_gradients():
    from hip_orb.head_fusion import frozen_pair_inputs

    torch.manual_seed(190)
    nodes = torch.randn(5, 16, device="cuda")
    edges = torch.randn(3, 16, device="cuda")
    charges = torch.randn(5, device="cuda")
    a = torch.tensor([0, 2, 4, 1], device="cuda")
    b = torch.tensor([1, 4, 0, 3], device="cuda")
    ids = torch.tensor([0, -1, 2, -1], device="cuda")
    direction = torch.randn(4, 3, device="cuda")
    radii = torch.randn(4, 17, device="cuda", requires_grad=True)
    edge = torch.cat((edges, nodes.new_zeros(1, 16)))[torch.where(ids < 0, 3, ids)]
    expected = torch.cat((nodes[a], nodes[b], edge, direction, radii,
                          charges[a, None], charges[b, None], (ids >= 0)[:, None]), dim=1)
    got = frozen_pair_inputs(nodes, edges, charges, a, b, ids, direction, radii)
    torch.testing.assert_close(got, expected, atol=0, rtol=0)
    probe = torch.randn_like(got)
    actual_grad = torch.autograd.grad(got, radii, probe)[0]
    expected_grad = torch.autograd.grad(expected, radii, probe)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    with pytest.raises(ValueError, match="frozen"):
        frozen_pair_inputs(nodes.requires_grad_(), edges, charges, a, b, ids, direction, radii)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Direct radial backward needs CUDA")
@pytest.mark.parametrize("rows", [0, 71])
def test_direct_radial_backward_avoids_full_input_gradient(rows):
    from hip_orb.precision_kernels import fused_half_silu

    torch.manual_seed(291)
    frozen = torch.randn(rows, 768, device="cuda")
    radial = torch.randn(rows, 17, device="cuda", requires_grad=True)
    x = torch.cat((frozen, radial), dim=1)
    w = (torch.randn(64, 785, device="cuda") * 0.02).requires_grad_()
    b = torch.randn(64, device="cuda", requires_grad=True)
    expected = fused_half_silu(x, w, b, alignment=16, input_grad_columns=(768, 785))
    actual = fused_half_silu(x.detach(), w, b, alignment=16, input_grad_columns=(768, 785),
                             radial_source=radial, fused_input_cast=True)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    probe = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (radial, w, b), probe)
    ge = torch.autograd.grad(expected, (radial, w, b), probe)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Head fusion guards need CUDA")
@pytest.mark.parametrize("trainable_features", [False, True])
def test_head_input_fusion_preserves_full_head_and_feature_gradients(trainable_features):
    from hip_orb.model import Features, HessianHead

    torch.manual_seed(391)
    model = torch.nn.Module()
    model.head = HessianHead(4, 8, radial_features="bessel", all_to_all=True,
                             radial_bases=3, mlp_hidden=16, neighbor_pooling=True).cuda()
    feature = Features(
        torch.randn(5, 4, device="cuda", requires_grad=trainable_features),
        torch.randn(4, 4, device="cuda", requires_grad=trainable_features),
        torch.tensor([0, 1, 2, 3], device="cuda"),
        torch.tensor([1, 0, 3, 2], device="cuda"),
        torch.randn(5, 3, device="cuda", requires_grad=trainable_features),
        torch.randn(5, device="cuda", requires_grad=trainable_features),
        torch.tensor([2, 3], device="cuda"))
    model.head._radial_only_backward = True
    params = tuple(model.head.parameters())
    if trainable_features:
        params += (feature.nodes, feature.edges, feature.positions, feature.charges)
    policy = LinearPolicy("fp16_f32", fuse_silu=True, activation_dtype="fp16", input_alignment=16)
    with linear_precision(model, head=policy):
        expected = torch.cat([v.flatten() for v in model.head(feature)])
        probe = torch.randn_like(expected)
        reference_grad = torch.autograd.grad(expected, params, probe)
        model.head._fused_pair_inputs = True
        got = torch.cat([v.flatten() for v in model.head(feature)])
        actual_grad = torch.autograd.grad(got, params, probe)
        torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)
        for actual, reference in zip(actual_grad, reference_grad, strict=True):
            torch.testing.assert_close(actual, reference, atol=2e-4, rtol=0.002)
        assert model.head.pair_adapter[0]._hip_radial_source is None
        if trainable_features:
            assert model.head.pair_adapter[0]._hip_input_grad_columns is None


def test_targeted_fp32_boundary_accepts_half_input_and_preserves_gradients():
    model = Small()
    reference = copy.deepcopy(model)
    x = torch.randn(7, 17, dtype=torch.float16, requires_grad=True)
    original = model.head[0].forward
    with linear_precision(model, head=LinearPolicy('bf16'),
                          overrides={'head.0': LinearPolicy('fp32')}) as stats:
        actual = model.head(x)
        expected = reference.head(x.float())
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        expected.sum().backward()
        for p, q in zip(model.head.parameters(), reference.head.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
        assert stats['overrides'] == {'head.0': 'fp32'}
    assert model.head[0].forward == original


@pytest.mark.parametrize('overrides,match', [
    ({'head.missing': LinearPolicy('fp32')}, 'Unmatched'),
    ({'head.*': LinearPolicy('fp32'), 'head.0': LinearPolicy('bf16')}, 'Overlapping'),
])
def test_precision_override_errors_restore_all_forwards(overrides, match):
    model = Small()
    originals = {name: module.forward for name, module in model.named_modules()}
    with (pytest.raises(ValueError, match=match),
          linear_precision(model, head=LinearPolicy('bf16'), overrides=overrides)):
        pass
    assert all(module.forward == originals[name] for name, module in model.named_modules())
    assert not hasattr(model, '_hip_precision_trainable_caches')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA half-input FP32-output GEMM')
@pytest.mark.parametrize('safe_policy', [LinearPolicy('fp32'), LinearPolicy('fp16_scaled')])
def test_targeted_policy_handles_half_activation_overflow_and_backward(safe_policy):
    model = Small().cuda()
    with torch.no_grad():
        model.head[0].weight.fill_(64)
        model.head[0].bias.zero_()
        model.head[2].weight.fill_(1 / 32)
        model.head[2].bias.zero_()
    reference = copy.deepcopy(model)
    x = torch.full((7, 17), 128.0, device='cuda', requires_grad=True)
    policy = LinearPolicy('fp16_f32', fuse_silu=True, activation_dtype='fp16')
    with torch.no_grad(), linear_precision(model, head=policy):
        assert not torch.isfinite(model.head(x)).all()
    with linear_precision(model, head=policy, overrides={'head.0': safe_policy}):
        actual = model.head(x)
        expected = reference.head(x)
        torch.testing.assert_close(actual, expected, atol=0, rtol=1e-3)
        actual.sum().backward()
        expected.sum().backward()
        for p, q in zip(model.head.parameters(), reference.head.parameters(), strict=True):
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=1e-3)


def test_context_precision_boundary_must_cover_the_next_matrix_product():
    model = Small()
    model.head = torch.nn.Module()
    model.head.node_update = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False), torch.nn.SiLU(),
        torch.nn.Linear(4, 4, bias=False), torch.nn.SiLU(),
        torch.nn.Linear(4, 4, bias=False))
    with torch.no_grad():
        model.head.node_update[0].weight.fill_(1)
        model.head.node_update[2].weight.fill_(3.25)
        model.head.node_update[4].weight.fill_(16)
    reference = copy.deepcopy(model)
    x = torch.full((7, 4), 1277.0, requires_grad=True)
    # FP32 output at the failing layer alone still overflows the next FP16 input cast.
    with linear_precision(model, head=LinearPolicy('fp16'),
                          overrides={'head.node_update.2': LinearPolicy('fp32')}):
        assert not torch.isfinite(model.head.node_update(x)).all()
    with linear_precision(model, head=LinearPolicy('fp16'),
                          overrides={'head.node_update.*': LinearPolicy('fp32')}):
        actual = model.head.node_update(x)
        expected = reference.head.node_update(x)
        assert actual.max() > 3_000_000
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        expected.sum().backward()
        for p, q in zip(model.head.parameters(), reference.head.parameters(), strict=True):
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
