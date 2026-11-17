"""Bound temporary MLP activations in the frozen Orb backbone."""

from contextlib import contextmanager

import torch


@contextmanager
def bounded_backbone_mlps(backbone, rows=65536):
    """Evaluate independent MLP rows in chunks without changing state keys.

    Only the encoder/interaction pointwise MLPs are affected. Attention and
    graph reductions still see the complete graph. This is an inference-only
    setting for the frozen backbone, including when training the Hessian head.
    GEMM row shapes can change floating-point rounding.
    """
    if rows < 1:
        raise ValueError("Backbone chunk rows must be positive")
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError("Backbone chunking requires frozen parameters")
    changed = []
    try:
        for name, module in backbone.named_modules():
            if name.rsplit(".", 1)[-1] not in ("_edge_fn", "_node_fn", "_edge_mlp", "_node_mlp"):
                continue
            original = module.forward
            own_forward = module.__dict__.get("forward")

            def forward(x, original=original, module=module):
                if torch.is_grad_enabled() or module.training:
                    raise ValueError("Chunked backbone MLPs require eval mode without gradients")
                if len(x) <= rows:
                    return original(x)
                first = original(x[:rows])
                output = first.new_empty((len(x), *first.shape[1:]))
                output[:rows] = first
                for start in range(rows, len(x), rows):
                    output[start : start + rows] = original(x[start : start + rows])
                return output

            changed.append((module, own_forward))
            module.forward = forward
        yield
    finally:
        for module, original in changed:
            if original is None:
                del module.forward
            else:
                module.forward = original
