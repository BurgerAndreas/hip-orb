"""Independent numerical checks for native low-precision matrix operations."""

import torch

from hip_orb.mixed_precision import fp8_linear, half_linear_float_output, pad_matrix, quantize_fp8


def comparison(actual, reference):
    a, r = actual.detach().double().reshape(-1), reference.detach().double().reshape(-1)
    difference = a - r
    na, nr = a.norm(), r.norm()
    return {
        "finite": bool(torch.isfinite(a).all()),
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(difference.norm() / nr.clamp_min(1e-30)),
        "cosine": float((a @ r) / (na * nr).clamp_min(1e-30))
        if nr > 0
        else (1.0 if na == 0 else 0.0),
    }


def scaled_operator_audit(device="cuda"):
    from hip_orb.precision_scaled import row_quantize, scaled_linear

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    records = []

    def row_oracle(a, bt):
        qa, sa = row_quantize(a, torch.float16, fused=False)
        qb, sb = row_quantize(bt, torch.float16, fused=False)
        return (qa.float() @ qb.float().T) * sa * sb.T

    try:
        for mode in ("fp16_scaled", "fp8_row", "bf16_compensated"):
            for m, k, n in ((17, 23, 9), (32, 256, 1024), (8193, 7, 9)):
                torch.manual_seed(911)
                x = (
                    torch.randn(m, k, device=device)
                    * torch.logspace(-6, 6, m, device=device)[:, None]
                ).requires_grad_()
                w = torch.randn(n, k, device=device, requires_grad=True)
                b = torch.randn(n, device=device, requires_grad=True)
                g = torch.randn(m, n, device=device) * 1e-3
                y = scaled_linear(x, w, b, mode=mode)
                (y * g).sum().backward()
                if mode == "bf16_compensated":
                    expected_y = x.detach() @ w.detach().T + b.detach()
                    expected_dx = g @ w.detach()
                    expected_dw = g.T @ x.detach()
                else:
                    dtype = torch.float16 if mode == "fp16_scaled" else torch.float8_e4m3fn
                    qa, sa = row_quantize(x.detach(), dtype, fused=False)
                    qw, sw = row_quantize(w.detach(), dtype, fused=False)
                    xd, wd = qa.float() * sa, qw.float() * sw
                    expected_y = (qa.float() @ qw.float().T) * sa * sw.T + b.detach()
                    expected_dx = row_oracle(g, wd.T)
                    expected_dw = row_oracle(g.T, xd.T)
                row = {
                    "mode": mode,
                    "shape": (m, k, n),
                    "forward": comparison(y, expected_y),
                    "input_gradient": comparison(x.grad, expected_dx),
                    "weight_gradient": comparison(w.grad, expected_dw),
                    "bias_gradient": comparison(b.grad, g.sum(0)),
                }
                row["passed"] = all(
                    row[key]["finite"] and row[key]["relative_l2"] < 3e-4
                    for key in ("forward", "input_gradient", "weight_gradient", "bias_gradient")
                )
                records.append(row)
                if not row["passed"]:
                    raise AssertionError(f"Scaled operator failed: {row}")
        return records
    finally:
        torch.set_float32_matmul_precision(previous)


def fused_operator_audit(device="cuda"):
    from hip_orb.precision_kernels import fused_half_silu

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    records = []
    try:
        for shape in ((17, 23, 9), (32, 256, 1024)):
            for half_output in (False, True):
                for use_bias in (False, True):
                    torch.manual_seed(411)
                    m, k, n = shape
                    x = torch.randn(m, k, device=device, requires_grad=True)
                    w = torch.randn(n, k, device=device, requires_grad=True)
                    b = torch.randn(n, device=device, requires_grad=True) if use_bias else None
                    g = torch.randn(m, n, device=device) * 1e-3
                    y = fused_half_silu(x, w, b, half_output=half_output)
                    (y * g).sum().backward()
                    xd, wd = x.detach().half().float(), w.detach().half().float()
                    pre = xd @ wd.T
                    if b is not None:
                        pre += b.detach()
                    sigmoid = pre.sigmoid()
                    expected_y = pre * sigmoid
                    upstream = g
                    if half_output:
                        expected_y = expected_y.half().float()
                        upstream = g.half().float()
                    dz = upstream * sigmoid * (1 + pre * (1 - sigmoid))
                    dh = dz.half().float()
                    row = {
                        "shape": shape,
                        "half_output": half_output,
                        "bias": use_bias,
                        "forward": comparison(y, expected_y),
                        "input_gradient": comparison(x.grad, dh @ wd),
                        "weight_gradient": comparison(w.grad, dh.T @ xd),
                    }
                    if b is not None:
                        row["bias_gradient"] = comparison(b.grad, dz.sum(0))
                    row["passed"] = all(
                        v["finite"] and v["relative_l2"] < 1e-3
                        for v in row.values()
                        if isinstance(v, dict)
                    )
                    records.append(row)
                    if not row["passed"]:
                        raise AssertionError(f"Fused operator failed: {row}")
        return records
    finally:
        torch.set_float32_matmul_precision(previous)


def operator_audit(device="cuda"):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    records = []
    try:
        for shape in ((17, 23, 9), (32, 256, 1024), (64, 1024, 256)):
            for dtype in (torch.bfloat16, torch.float16):
                torch.manual_seed(711)
                m, k, n = shape
                x = torch.randn(m, k, device=device, requires_grad=True)
                w = torch.randn(n, k, device=device, requires_grad=True)
                b = torch.randn(n, device=device, requires_grad=True)
                g = torch.randn(m, n, device=device) * 1e-3
                y = half_linear_float_output(x, w, b, dtype=dtype)
                (y * g).sum().backward()
                xd, wd, gd = (v.detach().to(dtype).float() for v in (x, w, g))
                row = {
                    "shape": shape,
                    "backward": str(dtype) + "_fp32_output",
                    "forward": comparison(y, xd @ wd.T + b.detach()),
                    "input_gradient": comparison(x.grad, gd @ wd),
                    "weight_gradient": comparison(w.grad, gd.T @ xd),
                    "bias_gradient": comparison(b.grad, g.sum(0)),
                }
                row["passed"] = all(
                    row[key]["finite"] and row[key]["relative_l2"] < 3e-4
                    for key in ("forward", "input_gradient", "weight_gradient", "bias_gradient")
                )
                records.append(row)
                if not row["passed"]:
                    raise AssertionError(f"Half-input backward implementation failed: {row}")
        for shape in ((17, 23, 9), (32, 256, 1024), (64, 1024, 256)):
            for backward in ("bf16", "fp8_e4m3", "fp8_e5m2"):
                torch.manual_seed(711)
                m, k, n = shape
                x = torch.randn(m, k, device=device, requires_grad=True)
                w = torch.randn(n, k, device=device, requires_grad=True)
                b = torch.randn(n, device=device, requires_grad=True)
                g = torch.randn(m, n, device=device) * 1e-3
                y = fp8_linear(x, w, b, backward=backward)
                (y * g).sum().backward()
                mp, kp, npad = ((v + 15) // 16 * 16 for v in (m, k, n))
                qx, sx = quantize_fp8(pad_matrix(x.detach(), mp, kp))
                qw, sw = quantize_fp8(pad_matrix(w.detach(), npad, kp))
                xd, wd = qx.float() * sx, qw.float() * sw
                expected_y = (xd @ wd.T)[:m, :n] + b.detach()
                gp = pad_matrix(g, mp, npad)
                if backward == "bf16":
                    expected_dx = (gp.bfloat16() @ wd.bfloat16())[:m, :k].float()
                    expected_dw = (gp.bfloat16().T @ xd.bfloat16())[:n, :k].float()
                else:
                    dtype = torch.float8_e5m2 if backward == "fp8_e5m2" else torch.float8_e4m3fn
                    qg, sg = quantize_fp8(gp, dtype)
                    gd = qg.float() * sg
                    expected_dx = (gd @ wd)[:m, :k]
                    expected_dw = (gd.T @ xd)[:n, :k]
                row = {
                    "shape": shape,
                    "backward": backward,
                    "forward": comparison(y, expected_y),
                    "input_gradient": comparison(x.grad, expected_dx),
                    "weight_gradient": comparison(w.grad, expected_dw),
                    "bias_gradient": comparison(b.grad, g.sum(0)),
                }
                row["passed"] = all(
                    row[key]["finite"] and row[key]["relative_l2"] < 3e-4
                    for key in ("forward", "input_gradient", "weight_gradient", "bias_gradient")
                )
                records.append(row)
                if not row["passed"]:
                    raise AssertionError(f"FP8 backward implementation failed: {row}")
        # Zero inputs and zero upstream gradients must remain finite.
        x = torch.zeros(17, 23, device=device, requires_grad=True)
        w = torch.zeros(9, 23, device=device, requires_grad=True)
        fp8_linear(x, w, backward="fp8_e4m3").sum().backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all()
        assert not x.grad.any() and not w.grad.any()
        return records
    finally:
        torch.set_float32_matmul_precision(previous)
