"""Training execution settings, separate from architecture and checkpoint weights."""

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
    return torch.cat((nodes[a], nodes[b], edge, torch.cat((unit, -unit)),
                      torch.cat((radial, radial)), charges[a, None], charges[b, None],
                      present[:, None].to(edge.dtype)), dim=-1)


def known_charge_forward(self, node_features, batch):
    """Retain Orb's charge centering and use its existing node owners."""
    from orb_models.common.models import segment_ops
    charges = self.mlp(node_features)
    if self.enforce_total_charge:
        owners = batch.node_batch_index
        means = segment_ops.scatter_mean(charges, owners, batch.n_node, dim=0,
                                          dim_size=len(batch.n_node))
        charges = charges - means[owners]
        if batch.system_features is not None and "total_charge" in batch.system_features:
            total = batch.system_features["total_charge"].to(dtype=charges.dtype)
            shifts = total / batch.n_node.to(dtype=charges.dtype)
            charges = charges + shifts[owners, None]
    return charges * self.charge_scale


def configure_execution(model, mode="optimized"):
    """Select execution without changing parameters, graph rules, or checkpoints.

    The optimized path retains the configured float32 precision throughout
    the backbone and readout.
    Reconfiguration restores the original methods before applying a new mode.
    """
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Execution mode must be one of {EXECUTION_MODES}")
    head = model.head
    parameter = next(head.parameters())
    cuda = parameter.is_cuda
    cuda_float32 = cuda and parameter.dtype == torch.float32
    if not hasattr(model, "_execution_originals"):
        model._execution_originals = {"head_forward": head.forward,
                                      "head_tensor": head.forward_tensor,
                                      "pair_inputs": head.pair_inputs}
        conditioner = getattr(model.base.model, "conditioner", None)
        if conditioner is not None:
            model._execution_originals["conditioner"] = (conditioner, conditioner.forward)
        # Torch ModuleDict has no dict.get method.
        charges = model.base.heads["latent_charges"] if "latent_charges" in model.base.heads else None  # noqa: SIM401
        if charges is not None:
            model._execution_originals["charges"] = (charges, charges.forward)
    originals = model._execution_originals
    head.forward, head.forward_tensor = originals["head_forward"], originals["head_tensor"]
    head.pair_inputs = originals["pair_inputs"]
    for name in ("conditioner", "charges"):
        if name in originals:
            module, forward = originals[name]
            module.forward = forward
    for name in ("_edge_gather", "_tensor_training"):
        if hasattr(head, name):
            delattr(head, name)
    model._tensor_graph = model._async_metadata = False
    if mode != "eager":
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
    precision = (f"float32-{torch.get_float32_matmul_precision()}" if parameter.dtype == torch.float32
                 else str(parameter.dtype).removeprefix("torch."))
    result = {"mode": mode, "backbone_precision": precision,
              "head_precision": precision,
              "tensor_training": mode != "eager", "tensor_molecular_graph": bool(model._tensor_graph),
              "masked_edge_kernel": bool(mode != "eager" and cuda_float32 and find_spec("triton") is not None),
              "known_metadata": mode != "eager"}
    model.execution_config = result
    return result
