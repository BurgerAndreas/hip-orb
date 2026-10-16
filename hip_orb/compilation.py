"""Reversible compiler candidates; keep state dictionaries and frozen weights intact."""
from contextlib import contextmanager
from types import MethodType

import torch

from hip_orb.model import directed_pair_inputs


def shape_known_conditioning(self, batch):
    """Preserve conditioning math; give repeat_interleave its known output size."""
    assert (batch.system_features is not None and "total_charge" in batch.system_features
            and "spin_multiplicity" in batch.system_features)
    combined = torch.cat((self.charge_embedding(batch.system_features["total_charge"]),
                          self.spin_embedding(batch.system_features["spin_multiplicity"])), dim=-1)
    nodes = combined.repeat_interleave(batch.n_node, dim=0,
                output_size=batch.node_features["positions"].shape[0]) if self.emits_node_embs else None
    edges = combined.repeat_interleave(batch.n_edge, dim=0,
                output_size=batch.senders.shape[0]) if self.emits_edge_embs else None
    return nodes, edges


@contextmanager
def compiler_candidate(model, *, backbone=False, pair_inputs=False, shaped_conditioning=False,
                       triton_pairs=False):
    previous_backbone = model.base.model._compiled_call_impl
    previous_inputs = model.head.pair_inputs
    conditioner = getattr(model.base.model, "conditioner", None)
    previous_conditioner = conditioner.forward if conditioner is not None else None
    try:
        if (backbone or shaped_conditioning) and conditioner is not None:
            conditioner.forward = MethodType(shape_known_conditioning, conditioner)
        if backbone:
            model.base.model.compile(dynamic=True, fullgraph=True,
                                     options={"force_same_precision": True})
        if pair_inputs:
            model.head.pair_inputs = torch.compile(directed_pair_inputs, dynamic=True, fullgraph=True)
        if triton_pairs:
            from hip_orb.fused_pair_inputs import fused_directed_pair_inputs
            model.head.pair_inputs = fused_directed_pair_inputs
        yield
    finally:
        model.base.model._compiled_call_impl = previous_backbone
        model.head.pair_inputs = previous_inputs
        if conditioner is not None:
            conditioner.forward = previous_conditioner
