"""Independent Orb processors and geometric encoders for a new message graph."""

from copy import deepcopy
from functools import lru_cache

import torch
from torch import nn

from orb_models.common.models.angular import SphericalHarmonics
from orb_models.common.models.rbf import BesselBasis
from orb_models.common.models.nn_util import mlp_and_layer_norm
from hip_orb.radial import make_radial


def make_orb_processor(template, initialization):
    processor = deepcopy(template)
    if initialization == "random":
        for module in processor.modules():
            if list(module.parameters(recurse=False)):
                reset = getattr(module, "reset_parameters", None)
                if reset is None:
                    raise ValueError(f"No standard initialization for {type(module).__name__}")
                reset()
    elif initialization != "pretrained":
        raise ValueError("Orb initialization must be pretrained or random")
    return processor.requires_grad_(True)


@lru_cache(maxsize=8)
def dense_message_topology(sizes, device):
    """Cache directed, nonself edges in molecule and sender order."""
    widths = torch.tensor(sizes, device=device, dtype=torch.long)
    counts = widths * (widths - 1)
    total_edges = sum(n * (n - 1) for n in sizes)
    owners = torch.arange(len(sizes), device=device).repeat_interleave(counts, output_size=total_edges)
    edge_offsets = torch.nn.functional.pad(counts.cumsum(0), (1, 0))
    node_offsets = torch.nn.functional.pad(widths.cumsum(0), (1, 0))
    t = torch.arange(total_edges, device=device) - edge_offsets[owners]
    n = widths[owners]
    a, b = t // (n - 1), t % (n - 1)
    b = b + (b >= a).long()
    return a + node_offsets[owners], b + node_offsets[owners]


class DenseOrbEdgeEncoder(nn.Module):
    """Fresh radial/angular encoding, with no hard distance cutoff."""

    def __init__(self, latent_dim, radial_scale=12.0, radial_bases=16,
                 radial_features="bessel", radial_log_range=4.0):
        super().__init__()
        self.radial = make_radial(radial_features, radial_bases, radial_scale, radial_log_range)
        self.angular = SphericalHarmonics(lmax=3, normalize=True, normalization="component")
        self.projection = mlp_and_layer_norm(radial_bases * 16 + 1, latent_dim, 1024,
                                             2, activation="silu", mlp_norm="rms_norm")

    def forward(self, vectors):
        distance = vectors.norm(dim=-1)
        radial = self.radial(distance)
        angular = self.angular(vectors)
        combined = (radial[:, :, None] * angular[:, None, :]).flatten(1)
        return self.projection(torch.cat((combined, torch.log1p(distance[:, None])), dim=-1))


class BackboneEdgeResidual(nn.Module):
    """Add normalized pretrained features only on matching directed dense edges."""

    def __init__(self, latent_dim):
        super().__init__()
        self.norm = nn.RMSNorm(latent_dim)
        self.projection = nn.Linear(latent_dim, latent_dim, bias=False)

    def forward(self, geometric, backbone_edges, senders, receivers, dense_senders,
                dense_receivers, total_nodes, *, dense_sizes=None):
        if dense_sizes is not None:
            from hip_orb.batched_pairs import dense_edge_slots
            slots = dense_edge_slots(dense_sizes, senders, receivers)
            residual = self.projection(self.norm(backbone_edges))
            return geometric.index_add(0, slots, residual)
        # Sorting supports any edge ordering. searchsorted avoids an N-total squared map.
        dense_keys = dense_senders * total_nodes + dense_receivers
        keys, order = dense_keys.sort()
        old_keys = senders * total_nodes + receivers
        slots = torch.searchsorted(keys, old_keys)
        from hip_orb.block_layout import require
        require((slots < len(keys)).all(), "Backbone edges must exist in the dense graph")
        require((keys[slots] == old_keys).all(), "Backbone edges must match directed dense edges")
        residual = self.projection(self.norm(backbone_edges))
        return geometric.index_add(0, order[slots], residual)


class OrbFeatureAdapters(nn.Module):
    """Learn Hessian-specific element and geometry residuals before Orb processing."""

    def __init__(self, latent_dim, mode, mlp_hidden=1024, atom_dim=64,
                 radial_scale=12.0, radial_bases=16, radial_features="bessel", radial_log_range=4.0):
        super().__init__()
        from hip_orb.model import mlp
        self.mode = mode
        if mode in ("atom", "both"):
            self.atom_embedding = nn.Embedding(119, atom_dim)
            self.node_norm = nn.RMSNorm(latent_dim)
            self.node_update = mlp(latent_dim + atom_dim + 1, mlp_hidden, latent_dim)
            self._small_output(self.node_update[-1])
        if mode in ("edge", "both"):
            self.edge_atom_embedding = nn.Embedding(119, atom_dim)
            self.edge_norm = nn.RMSNorm(latent_dim)
            self.radial = make_radial(radial_features, radial_bases, radial_scale, radial_log_range)
            self.angular = SphericalHarmonics(lmax=3, normalize=True, normalization="component")
            self.edge_update = mlp(latent_dim + 2 * atom_dim + radial_bases * 16 + 1,
                                   mlp_hidden, latent_dim)
            self._small_output(self.edge_update[-1])

    @staticmethod
    def _small_output(layer):
        nn.init.normal_(layer.weight, std=1e-3 / layer.in_features**0.5)
        nn.init.zeros_(layer.bias)

    def edge_residual(self, edges, numbers, positions, senders, receivers):
        vectors = positions[receivers] - positions[senders]
        distances = vectors.norm(dim=-1)
        geometry = (self.radial(distances)[:, :, None] * self.angular(vectors)[:, None, :]).flatten(1)
        atoms = self.edge_atom_embedding(numbers.long())
        inputs = torch.cat((self.edge_norm(edges), atoms[senders], atoms[receivers],
                            geometry, torch.log1p(distances[:, None])), dim=-1)
        return self.edge_update(inputs)

    def forward(self, nodes, edges, numbers, charges, positions, senders, receivers):
        if hasattr(self, "atom_embedding"):
            inputs = torch.cat((self.node_norm(nodes), self.atom_embedding(numbers.long()),
                                charges[:, None]), dim=-1)
            nodes = nodes + self.node_update(inputs)
        if hasattr(self, "edge_update"):
            if (self.training and torch.is_grad_enabled()
                    and len(edges) > getattr(self, "_edge_checkpoint_limit", 0)):
                from torch.utils.checkpoint import checkpoint
                residual = checkpoint(self.edge_residual, edges, numbers, positions, senders,
                                      receivers, use_reentrant=False)
            else:
                residual = self.edge_residual(edges, numbers, positions, senders, receivers)
            edges = edges + residual
        return nodes, edges


class FrozenLayerMixture(nn.Module):
    """Mix normalized frozen layers with separate trainable node and edge weights."""

    def __init__(self, latent_dim, layers):
        super().__init__()
        self.node_norms = nn.ModuleList(nn.RMSNorm(latent_dim) for _ in range(layers))
        self.edge_norms = nn.ModuleList(nn.RMSNorm(latent_dim) for _ in range(layers))
        logits = torch.full((layers,), -2.0)
        logits[-1] = 2.0
        self.node_logits = nn.Parameter(logits.clone())
        self.edge_logits = nn.Parameter(logits.clone())

    def forward(self, captured):
        node_weights, edge_weights = self.node_logits.softmax(0), self.edge_logits.softmax(0)
        nodes = sum(w * norm(pair[0].detach()) for w, norm, pair in zip(node_weights, self.node_norms, captured, strict=True))
        edges = sum(w * norm(pair[1].detach()) for w, norm, pair in zip(edge_weights, self.edge_norms, captured, strict=True))
        return nodes, edges
