"""Training execution settings, separate from architecture and checkpoint weights."""

from contextlib import ExitStack
from importlib.util import find_spec
from types import MethodType

import torch

EXECUTION_MODES = ("eager", "optimized")


def edge_values(edges, ids):
    """Avoid a shared absent-edge gradient row on supported CUDA installations."""
    if edges.is_cuda and edges.dtype == torch.float32 and find_spec("triton") is not None:
        from hip_orb.fused_gather import masked_edge_gather

        return masked_edge_gather(edges, ids)
    padded = torch.cat((edges, edges.new_zeros((1, edges.shape[-1]))))
    return padded[torch.where(ids >= 0, ids, len(edges))]


def joined_pair_inputs(nodes, edges, charges, i, j, forward, reverse, delta, distance):
    a, b = torch.cat((i, j)), torch.cat((j, i))
    ids = torch.cat((forward, reverse))
    present = ids >= 0
    edge = edge_values(edges, ids)
    unit = delta / distance
    radial = torch.log1p(distance)
    return torch.cat(
        (
            nodes[a],
            nodes[b],
            edge,
            torch.cat((unit, -unit)),
            torch.cat((radial, radial)),
            charges[a, None],
            charges[b, None],
            present[:, None].to(edge.dtype),
        ),
        dim=-1,
    )


def known_charge_forward(self, node_features, batch):
    """Retain Orb's charge centering and use its existing node owners."""
    from orb_models.common.models import segment_ops

    charges = self.mlp(node_features)
    if self.enforce_total_charge:
        owners = batch.node_batch_index
        means = segment_ops.scatter_mean(
            charges, owners, batch.n_node, dim=0, dim_size=len(batch.n_node)
        )
        charges = charges - means[owners]
        if batch.system_features is not None and "total_charge" in batch.system_features:
            total = batch.system_features["total_charge"].to(dtype=charges.dtype)
            shifts = total / batch.n_node.to(dtype=charges.dtype)
            charges = charges + shifts[owners, None]
    return charges * self.charge_scale


def configure_execution(
    model,
    mode="optimized",
    *,
    fused_norm=True,
    aligned_inputs=True,
    speed_features=True,
    scaled_gradients=True,
    compact_activations=True,
    backward_fp16=True,
    backward_input_fp32=None,
    shared_projection=True,
    fused_reductions=True,
):
    """Select execution without changing parameters, graph rules, or checkpoints.

    The optimized path keeps parameters and accumulated gradients in float32.
    Large MLP matrix products use locally scaled FP16 operands. Mixture heads
    keep native FP32 forward operations and use FP16 only in backward products.
    Selected frozen backbone MLPs use FP16. Norms and losses keep float32.
    Reconfiguration restores the original methods before applying a new mode.
    """
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Execution mode must be one of {EXECUTION_MODES}")
    previous = getattr(model, "_execution_context", None)
    if previous is not None:
        previous.close()
        del model._execution_context
    head = model.head
    parameter = next(head.parameters())
    cuda = parameter.is_cuda
    cuda_float32 = cuda and parameter.dtype == torch.float32
    if not hasattr(model, "_execution_originals"):
        model._execution_originals = {
            "head_forward": head.forward,
            "head_tensor": head.forward_tensor,
            "pair_inputs": head.pair_inputs,
            "rms_norms": [
                (m, m.forward) for m in model.modules() if isinstance(m, torch.nn.RMSNorm)
            ],
            "aligned_linears": [
                (m, m.forward)
                for m in head.modules()
                if isinstance(m, torch.nn.Linear) and m.in_features % 16 and m.out_features >= 128
            ],
        }
        conditioner = getattr(model.base.model, "conditioner", None)
        if conditioner is not None:
            model._execution_originals["conditioner"] = (conditioner, conditioner.forward)
        # Torch ModuleDict has no dict.get method.
        charges = (
            model.base.heads["latent_charges"] if "latent_charges" in model.base.heads else None  # noqa: SIM401
        )
        if charges is not None:
            model._execution_originals["charges"] = (charges, charges.forward)
    originals = model._execution_originals
    head.forward, head.forward_tensor = originals["head_forward"], originals["head_tensor"]
    head.pair_inputs = originals["pair_inputs"]
    for module, forward in originals["rms_norms"]:
        module.forward = forward
    for module, forward in originals["aligned_linears"]:
        module.forward = forward
    for name in ("conditioner", "charges"):
        if name in originals:
            module, forward = originals[name]
            module.forward = forward
    for name in ("_edge_gather", "_tensor_training"):
        if hasattr(head, name):
            delattr(head, name)
    head._join_radial_pairs = head._fused_trainable_pairs = False
    head._join_context_pairs = False
    head._project_geometry_first = False
    if hasattr(head, "orb_feature_adapter"):
        head.orb_feature_adapter._edge_checkpoint_limit = 0
    model._dense_message_kernels = False
    model._tensor_graph = model._async_metadata = False
    model._dense_indexing = mode != "eager"
    if mode != "eager":
        if cuda_float32 and aligned_inputs:
            from hip_orb.aligned_linear import aligned_linear

            def aligned(self, x):
                return aligned_linear(x, self.weight, self.bias)

            for module, _ in originals["aligned_linears"]:
                module.forward = MethodType(aligned, module)
        if cuda_float32 and fused_norm and find_spec("triton") is not None:
            from hip_orb.fused_norm import rms_norm

            def normalized(self, x):
                return rms_norm(x, self.normalized_shape, self.weight, self.eps)

            for module, _ in originals["rms_norms"]:
                module.forward = MethodType(normalized, module)
        head._tensor_training = True
        head._edge_gather = edge_values
        head.pair_inputs = joined_pair_inputs
        model._tensor_graph = cuda_float32
        model._async_metadata = True
        if "conditioner" in originals:
            from hip_orb.compilation import shape_known_conditioning

            module, _ = originals["conditioner"]
            module.forward = MethodType(shape_known_conditioning, module)
        if "charges" in originals:
            from orb_models.forcefield.models.forcefield_heads import LatentChargeHead

            module, _ = originals["charges"]
            if isinstance(module, LatentChargeHead):
                module.forward = MethodType(known_charge_forward, module)
    fast = bool(
        mode != "eager"
        and cuda_float32
        and speed_features
        and find_spec("triton") is not None
        and not hasattr(head, "backbone_mixture")
    )
    mixture_backward = bool(
        mode != "eager"
        and cuda_float32
        and speed_features
        and backward_fp16
        and hasattr(head, "backbone_mixture")
        and find_spec("triton") is not None
    )
    mixture_weight_only = bool(
        mixture_backward
        and (head.shared_geometry if backward_input_fp32 is None else backward_input_fp32)
    )
    project_shared = bool(
        mode != "eager"
        and cuda_float32
        and speed_features
        and shared_projection
        and head.shared_geometry
    )
    head._project_geometry_first = project_shared
    frozen_half = False
    if fast:
        from hip_orb.fused_linear_silu import fused_linear_silu_layers
        from hip_orb.mixed_precision import LinearPolicy, linear_precision
        from hip_orb.scaled_mlp import scaled_mlp_layers

        stack = ExitStack()
        try:
            if scaled_gradients and compact_activations:
                stack.enter_context(scaled_mlp_layers(model, fused_reductions=fused_reductions))
            stack.enter_context(
                fused_linear_silu_layers(
                    model, scaled=scaled_gradients, compact=compact_activations
                )
            )
            # Only detached frozen features use half precision. Trainable
            # gradients retain FP32, including very small Hessian gradients.
            if not any(p.requires_grad for p in model.base.model.parameters()):
                changed = stack.enter_context(
                    linear_precision(
                        model,
                        backbone=LinearPolicy(
                            "fp16",
                            include=("base.model.gnn_stacks.*._*mlp.mlp.NN-*",),
                            activation_dtype="fp16",
                        ),
                    )
                )
                frozen_half = bool(changed["backbone_layers"])
        except BaseException:
            stack.close()
            raise
        model._execution_context = stack
        head._join_radial_pairs = head._fused_trainable_pairs = True
        model._dense_message_kernels = True
        if hasattr(head, "orb_feature_adapter"):
            head.orb_feature_adapter._edge_checkpoint_limit = (
                650_000 if compact_activations else 300_000
            )
    if mixture_backward:
        from hip_orb.backward_precision import backward_mlp_precision

        stack = ExitStack()
        try:
            stack.enter_context(
                backward_mlp_precision(
                    model, fused_reductions=fused_reductions, weight_only=mixture_weight_only
                )
            )
        except BaseException:
            stack.close()
            raise
        model._execution_context = stack
    precision = (
        f"float32-{torch.get_float32_matmul_precision()}"
        if parameter.dtype == torch.float32
        else str(parameter.dtype).removeprefix("torch.")
    )
    result = {
        "mode": mode,
        "backbone_precision": precision,
        "head_precision": precision,
        "tensor_training": mode != "eager",
        "tensor_molecular_graph": bool(model._tensor_graph),
        "masked_edge_kernel": bool(
            mode != "eager" and cuda_float32 and find_spec("triton") is not None
        ),
        "known_metadata": mode != "eager",
    }
    result["fused_rms_norm"] = bool(
        mode != "eager" and cuda_float32 and fused_norm and find_spec("triton") is not None
    )
    result["dense_indexing"] = model._dense_indexing
    result["aligned_fp32_inputs"] = bool(mode != "eager" and cuda_float32 and aligned_inputs)
    result.update(
        fused_fp32_silu=fast,
        fused_dense_pair_inputs=fast,
        frozen_backbone_mlp_fp16=frozen_half,
        edge_checkpoint_limit=(650_000 if compact_activations else 300_000) if fast else 0,
        scaled_fp16_gradients=bool((fast and scaled_gradients) or mixture_backward),
        compact_saved_activations=bool(fast and compact_activations),
        compact_mlp_kernels=bool(fast and compact_activations and scaled_gradients),
        fp16_min_rows=16384 if fast and scaled_gradients else None,
        dense_message_kernels=fast,
        precision_guard=("fp32-mixture-forward" if hasattr(head, "backbone_mixture") else None),
    )
    if frozen_half:
        result["backbone_precision"] = precision + "+fp16-message-mlps"
    if fast and scaled_gradients:
        result["head_precision"] = precision + "+locally-scaled-fp16-mlps"
    result.update(
        fp32_forward_scaled_backward=mixture_backward,
        fp32_backward_activations=mixture_weight_only,
        fused_gradient_reductions=bool(
            ((fast and scaled_gradients and compact_activations) or mixture_backward)
            and fused_reductions
        ),
        project_shared_geometry_first=project_shared,
    )
    if mixture_backward:
        result["head_precision"] = precision + (
            "+locally-scaled-fp16-weight-gradients"
            if mixture_weight_only
            else "+locally-scaled-fp16-backward"
        )
        result["fp16_min_rows"] = 16384
    model.execution_config = result
    return result
