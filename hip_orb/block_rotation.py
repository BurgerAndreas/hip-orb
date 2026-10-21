"""FP32 rotation of independent 3 by 3 Hessian blocks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rotate(H, R, OUT, N: tl.constexpr, COUNT, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair = index % (N * N)
    base = (index // (N * N)) * (9 * N * N) + (pair // N) * 9 * N + (pair % N) * 3
    r00 = tl.load(R)
    r01 = tl.load(R + 1)
    r02 = tl.load(R + 2)
    r10 = tl.load(R + 3)
    r11 = tl.load(R + 4)
    r12 = tl.load(R + 5)
    r20 = tl.load(R + 6)
    r21 = tl.load(R + 7)
    r22 = tl.load(R + 8)
    a = tl.load(H + base, index < COUNT, 0)
    b = tl.load(H + base + 1, index < COUNT, 0)
    c = tl.load(H + base + 2, index < COUNT, 0)
    d = tl.load(H + base + 3 * N, index < COUNT, 0)
    e = tl.load(H + base + 3 * N + 1, index < COUNT, 0)
    f = tl.load(H + base + 3 * N + 2, index < COUNT, 0)
    g = tl.load(H + base + 6 * N, index < COUNT, 0)
    h = tl.load(H + base + 6 * N + 1, index < COUNT, 0)
    i = tl.load(H + base + 6 * N + 2, index < COUNT, 0)
    x0 = r00 * a + r01 * d + r02 * g
    x1 = r00 * b + r01 * e + r02 * h
    x2 = r00 * c + r01 * f + r02 * i
    y0 = r10 * a + r11 * d + r12 * g
    y1 = r10 * b + r11 * e + r12 * h
    y2 = r10 * c + r11 * f + r12 * i
    z0 = r20 * a + r21 * d + r22 * g
    z1 = r20 * b + r21 * e + r22 * h
    z2 = r20 * c + r21 * f + r22 * i
    tl.store(OUT + base, x0 * r00 + x1 * r01 + x2 * r02, index < COUNT)
    tl.store(OUT + base + 1, x0 * r10 + x1 * r11 + x2 * r12, index < COUNT)
    tl.store(OUT + base + 2, x0 * r20 + x1 * r21 + x2 * r22, index < COUNT)
    tl.store(OUT + base + 3 * N, y0 * r00 + y1 * r01 + y2 * r02, index < COUNT)
    tl.store(OUT + base + 3 * N + 1, y0 * r10 + y1 * r11 + y2 * r12, index < COUNT)
    tl.store(OUT + base + 3 * N + 2, y0 * r20 + y1 * r21 + y2 * r22, index < COUNT)
    tl.store(OUT + base + 6 * N, z0 * r00 + z1 * r01 + z2 * r02, index < COUNT)
    tl.store(OUT + base + 6 * N + 1, z0 * r10 + z1 * r11 + z2 * r12, index < COUNT)
    tl.store(OUT + base + 6 * N + 2, z0 * r20 + z1 * r21 + z2 * r22, index < COUNT)


def rotate_blocks(hessians, rotation):
    out = torch.empty_like(hessians)
    n = hessians.shape[-1] // 3
    count = hessians.numel() // 9
    if count:
        _rotate[(triton.cdiv(count, 128),)](hessians, rotation, out, n, count, 128)
    return out
