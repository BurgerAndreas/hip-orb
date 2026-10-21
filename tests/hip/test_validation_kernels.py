import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("n,b,contiguous", [(1, 2, True), (16, 64, False), (350, 2, True)])
def test_block_rotation_fp32_against_double(n, b, contiguous):
    from hip_orb.metrics import rotate_hessians

    h = torch.randn(b, 3 * n, 3 * n, device="cuda")
    r = torch.linalg.qr(torch.randn(3, 3, device="cuda")).Q
    if contiguous:
        r = r.contiguous()
    transform = torch.kron(
        torch.eye(n, device="cuda", dtype=torch.float64), r.double().contiguous()
    )
    reference = transform @ h.double() @ transform.T
    actual = rotate_hessians(h, r)
    torch.testing.assert_close(actual.double(), reference, rtol=2e-5, atol=2e-6)


def test_batched_metrics_keep_native_float32_projections():
    from hip_orb.metrics import batched_hessian_metrics, hessian_metrics

    torch.manual_seed(62)
    old = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for n in [2, 4, 16, 40]:
            p = torch.randn(4, 3 * n, 3 * n, device="cuda")
            p = p + p.transpose(-1, -2)
            t = torch.randn_like(p)
            t = t + t.transpose(-1, -2)
            x = torch.randn(4, n, 3, device="cuda")
            m = torch.rand(4, n, device="cuda") + 1
            actual = batched_hessian_metrics(p, t, x, m)
            for i, row in enumerate(actual):
                expected = hessian_metrics(p[i], t[i], x[i], m[i])
                for key in row:
                    assert row[key] == pytest.approx(expected[key], rel=1e-5, abs=1e-5), key
    finally:
        torch.set_float32_matmul_precision(old)
