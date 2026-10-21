"""Frozen backbone, independent Hessian graph, and trainable block readouts."""

from dataclasses import dataclass
from collections import OrderedDict
from copy import copy

import torch
from torch import nn
from orb_models.common.models.rbf import BesselBasis

from hip_orb.coulomb import fixed_charge_hessian, fixed_charge_blocks
from hip_orb.batched_pairs import batch_topology
from hip_orb.block_layout import layout, assemble
from hip_orb.translation import project_hessian_list
from hip_orb.translation import project_translations
from hip_orb.radial import make_radial


@dataclass
class Features:
    nodes: torch.Tensor
    edges: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    positions: torch.Tensor
    charges: torch.Tensor
    sizes: torch.Tensor
    constant: float | torch.Tensor = 14.3996
    sigma: float | None = None
    pairs: list[dict[str, torch.Tensor]] | None = None
    geometry: torch.Tensor | None = None
    dense_ordered: bool = False


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


class ResidualPairLayer(nn.Module):
    def __init__(self, hidden, mlp_hidden=None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.update = mlp(hidden, mlp_hidden or hidden, hidden)

    def forward(self, features):
        return features + self.update(self.norm(features))


def pool_directed_pairs(first, second, i, j, weights, count):
    pooled = torch.zeros((len(count), first.shape[-1]), device=first.device,
                         dtype=torch.promote_types(first.dtype, weights.dtype))
    pooled = pooled.index_add(0, i, first * weights).index_add(0, j, second * weights)
    return pooled / count.clamp_min(1e-8)


class ContextRound(nn.Module):
    """Refine latent node and pair states after a further neighbor aggregation."""

    def __init__(self, hidden, mlp_hidden):
        super().__init__()
        self.node_update = mlp(2 * hidden, mlp_hidden, hidden)
        self.pair_update = mlp(3 * hidden, mlp_hidden, hidden)

    def forward(self, nodes, pooled, first, second, i, j):
        nodes = nodes + self.node_update(torch.cat((nodes, pooled), dim=-1))
        first = first + self.pair_update(torch.cat((first, nodes[i], nodes[j]), dim=-1))
        second = second + self.pair_update(torch.cat((second, nodes[j], nodes[i]), dim=-1))
        return nodes, first, second


def directed_pair_inputs(nodes, edges, charges, i, j, forward, reverse, delta, distance):
    """Pure tensor region suitable for gather/concatenation compiler fusion."""
    a, b = torch.cat((i, j)), torch.cat((j, i))
    ids = torch.cat((forward, reverse))
    present = ids >= 0
    padded = torch.cat((edges, nodes.new_zeros((1, nodes.shape[-1]))))
    edge = padded[torch.where(present, ids, edges.shape[0])]
    unit = delta / distance
    radial = torch.log1p(distance)
    return torch.cat((nodes[a], nodes[b], edge, torch.cat((unit, -unit)),
                      torch.cat((radial, radial)), charges[a, None], charges[b, None],
                      present[:, None].to(edge.dtype)), dim=-1)


def backbone_latent_dim(backbone):
    """Read Orb's actual message-passing width without changing the backbone."""
    dimensions = []
    direct = getattr(backbone, "latent_dim", None)
    if direct is not None:
        dimensions.append(int(direct))
    # Orb 0.7 MoleculeGNS stores this on each processor, not on itself.
    for processor in getattr(backbone, "gnn_stacks", ()):
        dimension = getattr(processor, "latent_dim", None)
        if dimension is None:
            raise ValueError("Orb processor does not expose its feature dimension")
        dimensions.append(int(dimension))
    if not dimensions or min(dimensions) < 1 or len(set(dimensions)) != 1:
        raise ValueError("Cannot establish one consistent Orb node and edge feature dimension")
    return dimensions[0]


class HessianHead(nn.Module):
    """Read diagonal blocks from nodes and off-diagonal blocks from pair features.

    Existing message edges supply frozen latent edge features. A trainable pair
    adapter also uses endpoint features and geometry, including missing edges.
    No Hessian edges enter the frozen message-passing graph.
    """

    def __init__(self, latent_dim=256, hidden=128, cutoff=None,
                 radial_features="log", radial_cutoff=12.0, radial_bases=16,
                 pair_residual_layers=0, neighbor_pooling=False, translation_projection=False,
                 context_rounds=1, mlp_hidden=None, all_to_all=False,
                 radial_log_range=4.0, shared_geometry=False, checkpoint_context=False):
        super().__init__()
        if cutoff is not None and cutoff <= 0:
            raise ValueError("Hessian cutoff must be positive or None")
        if all_to_all and cutoff is not None:
            raise ValueError("All-to-all Hessian graphs cannot have a cutoff")
        if radial_features not in ("log", "bessel", "log_gaussian"):
            raise ValueError("Radial features must be log, bessel, or log_gaussian")
        if radial_features == "log_gaussian" and not all_to_all:
            raise ValueError("Log Gaussian features require an all-to-all Hessian graph")
        if shared_geometry and radial_features == "log":
            raise ValueError("Shared geometry requires an expanded radial basis")
        if radial_cutoff <= 0 or radial_bases < 1:
            raise ValueError("Radial cutoff and basis count must be positive")
        mlp_hidden = hidden if mlp_hidden is None else mlp_hidden
        if hidden < 1 or mlp_hidden < 1:
            raise ValueError("Feature and MLP widths must be positive")
        if context_rounds < 1 or (context_rounds > 1 and not neighbor_pooling):
            raise ValueError("Multiple context rounds require neighbor pooling; round count must be positive")
        self.radial_features_type = radial_features
        self.radial_cutoff = float(radial_cutoff)
        self.radial_bases = int(radial_bases)
        self.all_to_all = all_to_all
        self.shared_geometry = shared_geometry
        self.checkpoint_context = checkpoint_context
        self.radial = (make_radial(radial_features, self.radial_bases, self.radial_cutoff,
                                  radial_log_range, trainable=all_to_all)
                       if radial_features != "log" and not shared_geometry else None)
        radial_dim = self.radial_bases + int(all_to_all) if radial_features != "log" else 1
        self.config = {"latent_dim": latent_dim, "hidden": hidden, "cutoff": cutoff,
                       "radial_features": radial_features, "radial_cutoff": self.radial_cutoff,
                       "radial_bases": self.radial_bases}
        if radial_features == "log_gaussian":
            self.config["radial_log_range"] = radial_log_range
        if shared_geometry:
            self.geometry_projection = nn.Linear(latent_dim, radial_dim)
            self.config["shared_geometry"] = True
        if all_to_all:
            self.config["all_to_all"] = True
        if context_rounds != 1:
            self.config["context_rounds"] = context_rounds
        if mlp_hidden != hidden:
            self.config["mlp_hidden"] = mlp_hidden
        self.cutoff = cutoff
        self.translation_projection = translation_projection
        if translation_projection:
            self.config["translation_projection"] = True
        self.fuse_pairs = radial_features == "log"
        self.batch_pairs = True
        self.cache_pairs = True
        self._pair_indices = OrderedDict()
        self.pair_inputs = directed_pair_inputs
        self.node_head = mlp(latent_dim + 1, mlp_hidden, 9)
        # Two nodes, one directed edge, unit vector, radial features, two charges, edge flag.
        self.pair_adapter = mlp(3 * latent_dim + 6 + radial_dim, mlp_hidden, hidden)
        self.edge_head = nn.Linear(hidden, 9)
        if pair_residual_layers < 0:
            raise ValueError("Pair residual layer count must be nonnegative")
        self.pair_residual = nn.Sequential(
            *(ResidualPairLayer(hidden, mlp_hidden) for _ in range(pair_residual_layers)))
        self.neighbor_pooling = neighbor_pooling
        if pair_residual_layers or neighbor_pooling:
            self.config.update(pair_residual_layers=pair_residual_layers,
                               neighbor_pooling=neighbor_pooling)
        if neighbor_pooling:
            self.node_update = mlp(latent_dim + 1 + hidden, mlp_hidden, hidden)
            self.node_head = mlp(hidden, mlp_hidden, 9)
            self.pair_update = mlp(3 * hidden, mlp_hidden, hidden)
        self.extra_context_rounds = nn.ModuleList(
            ContextRound(hidden, mlp_hidden) for _ in range(context_rounds - 1))

    def forward(self, features: Features, *, residual_only=False, as_tensor=False):
        if as_tensor:
            return self.forward_tensor(features, residual_only=residual_only)
        if self.batch_pairs:
            hessians = self._batched_forward(features, residual_only=residual_only)
        else:
            if self.neighbor_pooling or len(self.pair_residual):
                raise ValueError("Context heads require batched pair evaluation")
            hessians = self._loop_forward(features, residual_only=residual_only)
        return project_hessian_list(hessians) if self.translation_projection else hessians

    def forward_tensor(self, features: Features, *, residual_only=False):
        """Return an equal-size batch without splitting and restacking matrices."""
        hessians = self._batched_forward(features, residual_only=residual_only, as_tensor=True)
        return project_translations(hessians) if self.translation_projection else hessians

    def radial_embedding(self, distance):
        if self.shared_geometry:
            raise ValueError("Shared geometry must use the cached directed encoder output")
        if self.radial_features_type == "log":
            return torch.log1p(distance.reshape(-1, 1))
        # Use Orb's fixed Bessel form with a separate radius for the Hessian graph.
        distance = distance.reshape(-1)
        if self.all_to_all:
            return torch.cat((self.radial(distance), torch.log1p(distance[:, None])), dim=-1)
        envelope = 0.5 * (torch.cos(torch.pi * distance / self.radial_cutoff) + 1.0)
        envelope = envelope * (distance < self.radial_cutoff).to(distance.dtype)
        return self.radial(distance) * envelope[:, None]

    def _batched_forward(self, features, *, residual_only=False, as_tensor=False):
        sizes, i, j, forward, reverse, delta, distance = batch_topology(features)
        charges = features.charges.reshape(-1)
        self.pair_adapter[0]._hip_input_grad_columns = None
        self.pair_adapter[0]._hip_radial_source = None
        if (getattr(self, "_radial_only_backward", False) and self.all_to_all
                and self.radial_features_type == "bessel"
                and not any(t.requires_grad for t in (features.nodes, features.edges,
                                                     features.positions, features.charges))):
            start = 3 * features.nodes.shape[-1] + 3
            self.pair_adapter[0]._hip_input_grad_columns = (start, start + self.radial_bases + 1)

        def inputs(a, b, ids, direction, radii):
            if (getattr(self, "_fused_pair_inputs", False) and features.nodes.is_cuda
                    and not any(t.requires_grad for t in (features.nodes, features.edges, charges, direction))
                    and all(t.is_contiguous() and t.dtype == torch.float32
                            for t in (features.nodes, features.edges, charges, radii))):
                from hip_orb.head_fusion import frozen_pair_inputs
                if self.pair_adapter[0]._hip_input_grad_columns is not None:
                    self.pair_adapter[0]._hip_radial_source = radii
                return frozen_pair_inputs(features.nodes, features.edges, charges,
                                          a, b, ids, direction.contiguous(), radii)
            present = ids >= 0
            if hasattr(self, "_edge_gather"):
                edge = self._edge_gather(features.edges, ids)
            else:
                padded = torch.cat((features.edges, features.nodes.new_zeros((1, features.nodes.shape[-1]))))
                edge = padded[torch.where(present, ids, len(features.edges))]
            node_a, node_b = features.nodes[a], features.nodes[b]
            return torch.cat((node_a, node_b, edge, direction, radii,
                              charges[a, None], charges[b, None], present[:, None].to(edge.dtype)), dim=-1)

        if self.fuse_pairs:
            joined = self.pair_inputs(features.nodes, features.edges, charges, i, j,
                                      forward, reverse, delta, distance)
            directed = self.pair_residual(self.pair_adapter(joined))
            first, second = directed.chunk(2) if len(i) else (directed, directed)
        else:
            unit = delta / distance
            if self.shared_geometry:
                if features.geometry is None:
                    raise ValueError("Shared geometry features are missing")
                if hasattr(self, "_edge_gather"):
                    gather = self._edge_gather
                else:
                    gather = lambda values, ids: values[ids]
                if getattr(self, "_project_geometry_first", False) and features.dense_ordered:
                    projected = self.geometry_projection(features.geometry)
                    first_radial = gather(projected, forward)
                    second_radial = gather(projected, reverse)
                else:
                    first_radial = self.geometry_projection(gather(features.geometry, forward))
                    second_radial = self.geometry_projection(gather(features.geometry, reverse))
            else:
                first_radial = second_radial = self.radial_embedding(distance)
            if getattr(self, "_join_radial_pairs", False):
                if getattr(self, "_fused_trainable_pairs", False) and features.dense_ordered:
                    from hip_orb.dense_pair_inputs import dense_pair_inputs
                    joined = dense_pair_inputs(features.nodes, features.edges, charges, i, j,
                                               forward, reverse, unit.contiguous(), first_radial,
                                               second_radial, sizes)
                else:
                    joined = torch.cat((inputs(i, j, forward, unit, first_radial),
                                        inputs(j, i, reverse, -unit, second_radial)))
                directed = self.pair_residual(self.pair_adapter(joined))
                first, second = directed.split(len(i)) if len(i) else (directed, directed)
            else:
                first = self.pair_residual(self.pair_adapter(inputs(i, j, forward, unit, first_radial)))
                second = self.pair_residual(self.pair_adapter(inputs(j, i, reverse, -unit, second_radial)))
        self.pair_adapter[0]._hip_radial_source = None
        node_inputs = torch.cat((features.nodes, charges[:, None]), dim=-1)
        if self.neighbor_pooling:
            def context_mlp(module, value):
                if self.checkpoint_context and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    return checkpoint(module, value, use_reentrant=False)
                return module(value)
            # Local weighted means prevent the context scale from growing with molecule size.
            pool_cutoff = self.cutoff if self.cutoff is not None else self.radial_cutoff
            weights = 0.5 * (torch.cos(torch.pi * distance / pool_cutoff) + 1)
            weights = weights * (distance < pool_cutoff).to(weights.dtype)
            if self.all_to_all:
                weights = torch.ones_like(distance)
            count = weights.new_zeros((len(features.nodes), 1))
            count = count.index_add(0, i, weights).index_add(0, j, weights)
            def pool(first, second):
                return pool_directed_pairs(first, second, i, j, weights, count)
            pooled = pool(first, second)
            node_inputs = context_mlp(self.node_update, torch.cat((node_inputs, pooled), dim=-1))
            if getattr(self, "_join_context_pairs", getattr(self, "_join_radial_pairs", False)):
                joined = torch.cat((torch.cat((first, node_inputs[i], node_inputs[j]), dim=-1),
                                    torch.cat((second, node_inputs[j], node_inputs[i]), dim=-1)))
                updated = context_mlp(self.pair_update, joined)
                a, b = updated.split(len(i)) if len(i) else (updated, updated)
                first, second = first + a, second + b
            else:
                first = first + context_mlp(self.pair_update, torch.cat((first, node_inputs[i], node_inputs[j]), dim=-1))
                second = second + context_mlp(self.pair_update, torch.cat((second, node_inputs[j], node_inputs[i]), dim=-1))
            for context in self.extra_context_rounds:
                pooled = pool(first, second)
                node_inputs, first, second = context(node_inputs, pooled, first, second, i, j)
        first_blocks = self.edge_head(first).reshape(-1, 3, 3)
        second_blocks = self.edge_head(second).reshape(-1, 3, 3)
        blocks = (first_blocks + second_blocks.transpose(-1, -2)) * 0.5
        diagonal = self.node_head(node_inputs).reshape(-1, 3, 3)
        diagonal = (diagonal + diagonal.transpose(-1, -2)) * 0.5
        if self.cutoff is not None:
            blocks = torch.where((distance < self.cutoff)[:, :, None], blocks, 0)
        if not residual_only:
            coulomb, coulomb_diagonal = fixed_charge_blocks(features.positions, charges, i, j,
                                                            features.constant, features.sigma)
            blocks = blocks + coulomb
            diagonal = diagonal + coulomb_diagonal
        if as_tensor == "pair-blocks":
            return blocks, diagonal
        indices = getattr(features, "packed_layout", None)
        if indices is None:
            indices = layout(tuple(sizes), str(features.nodes.device))
        return assemble(indices, features.nodes, sizes, blocks, diagonal, as_tensor=as_tensor)

    def _loop_forward(self, features: Features, *, residual_only=False):
        results = []
        offset = 0
        for molecule, size in enumerate(features.sizes.tolist()):
            n = int(size)
            nodes = features.nodes[offset : offset + n]
            positions = features.positions[offset : offset + n]
            charges = features.charges[offset : offset + n].reshape(-1)
            # Keep a directed map: off-diagonal blocks need not be symmetric.
            pair = features.pairs[molecule] if features.pairs is not None else None
            mapping = None
            if pair is None:
                mapping = torch.full((n, n), -1, device=nodes.device, dtype=torch.long)
                mask = (features.senders >= offset) & (features.senders < offset + n)
                if bool(((features.receivers[mask] < offset) | (features.receivers[mask] >= offset + n)).any()):
                    raise ValueError("Message edges cross molecules")
                ids = torch.arange(len(features.edges), device=nodes.device)[mask]
                mapping[features.senders[mask] - offset, features.receivers[mask] - offset] = ids
                key = (n, str(nodes.device))
                if not self.cache_pairs:
                    i, j = torch.triu_indices(n, n, offset=1, device=nodes.device)
                else:
                    if key not in self._pair_indices:
                        self._pair_indices[key] = torch.triu_indices(n, n, offset=1, device=nodes.device)
                        if len(self._pair_indices) > 8:
                            self._pair_indices.popitem(last=False)
                    self._pair_indices.move_to_end(key)
                    i, j = self._pair_indices[key]
            if pair is not None:
                i, j = pair["i"], pair["j"]
                forward_ids, reverse_ids = pair["forward"], pair["reverse"]
            distance = (positions[i] - positions[j]).norm(dim=-1)
            if bool((distance <= 0).any()):
                raise ValueError("Hessian pairs must have nonzero distance")
            if self.cutoff is not None:
                keep = distance < self.cutoff
                i, j = i[keep], j[keep]
                if pair is not None:
                    forward_ids, reverse_ids = forward_ids[keep], reverse_ids[keep]

            def read_pair(a, b, edge_ids=None, mapping=mapping, nodes=nodes, positions=positions, charges=charges):
                edge_ids = mapping[a, b] if edge_ids is None else edge_ids
                present = edge_ids >= 0
                edge = nodes.new_zeros((len(a), nodes.shape[-1]))
                edge[present] = features.edges[edge_ids[present]]
                delta = positions[a] - positions[b]
                r = delta.norm(dim=-1, keepdim=True)
                inputs = torch.cat(
                    (
                        nodes[a],
                        nodes[b],
                        edge,
                        delta / r,
                        self.radial_embedding(r.reshape(-1)),
                        charges[a, None],
                        charges[b, None],
                        present[:, None].to(nodes.dtype),
                    ),
                    dim=-1,
                )
                return self.edge_head(self.pair_adapter(inputs)).reshape(-1, 3, 3)

            if self.fuse_pairs:
                directed = read_pair(torch.cat((i, j)), torch.cat((j, i)),
                                     torch.cat((forward_ids, reverse_ids)) if pair is not None else None)
                forward, reverse = directed.split(len(i)) if len(i) else (directed, directed)
                blocks = (forward + reverse.transpose(-1, -2)) * 0.5
            else:
                blocks = (read_pair(i, j, forward_ids if pair is not None else None) +
                          read_pair(j, i, reverse_ids if pair is not None else None).transpose(-1, -2)) * 0.5
            diagonal = self.node_head(torch.cat((nodes, charges[:, None]), dim=-1)).reshape(n, 3, 3)
            diagonal = (diagonal + diagonal.transpose(-1, -2)) * 0.5
            h = nodes.new_zeros((n, n, 3, 3))
            h[i, j] = blocks
            h[j, i] = blocks.transpose(-1, -2)
            ids = torch.arange(n, device=nodes.device)
            h[ids, ids] = diagonal
            h = h.permute(0, 2, 1, 3).reshape(3 * n, 3 * n)
            if not residual_only:
                h = h + fixed_charge_hessian(positions, charges, features.constant, features.sigma)
            results.append(h)
            offset += n
        if offset != len(features.nodes):
            raise ValueError("Batch sizes do not cover the nodes")
        return results


class HIPOrb(nn.Module):
    """Freeze every existing parameter and preserve the original E/F interface."""

    def __init__(self, base, adapter, hidden=128, cutoff=None,
                 radial_features="log", radial_cutoff=12.0, radial_bases=16,
                 pair_residual_layers=0, neighbor_pooling=False, extra_orb_round=False,
                 translation_projection=False, context_rounds=1, mlp_hidden=None,
                 all_to_all=False, orb_initialization="pretrained", orb_message_graph="backbone",
                 orb_rounds=1, orb_combine_backbone_edges=False, orb_feature_adapters="none",
                 radial_log_range=4.0, shared_geometry=False, backbone_layer_mix=0,
                 checkpoint_context=False):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapter = adapter
        self.head = HessianHead(backbone_latent_dim(base.model), hidden, cutoff,
                                radial_features, radial_cutoff, radial_bases,
                                pair_residual_layers, neighbor_pooling, translation_projection,
                                context_rounds, mlp_hidden, all_to_all,
                                radial_log_range, shared_geometry, checkpoint_context)
        self.head.to(next(base.parameters()))
        if orb_rounds < 1 or (not extra_orb_round and orb_rounds != 1):
            raise ValueError("Orb rounds require an enabled extension and a positive count")
        if orb_combine_backbone_edges and (not extra_orb_round or orb_message_graph != "hessian"):
            raise ValueError("Combining backbone edges requires the dense Orb extension")
        if orb_feature_adapters not in ("none", "atom", "edge", "both"):
            raise ValueError("Unknown Orb feature adapter mode")
        if orb_feature_adapters != "none" and not extra_orb_round:
            raise ValueError("Feature adapters require an extra Orb round")
        if shared_geometry and (not extra_orb_round or orb_message_graph != "hessian" or orb_feature_adapters != "none"):
            raise ValueError("Shared geometry requires the dense Orb extension without feature adapters")
        if backbone_layer_mix and not extra_orb_round:
            raise ValueError("A backbone layer mixture requires an extra Orb round")
        if extra_orb_round:
            from orb_models.common.models.gns import AttentionInteractionNetwork
            stacks = getattr(base.model, "gnn_stacks", None)
            if not stacks or not isinstance(stacks[-1], AttentionInteractionNetwork):
                raise ValueError("An extra Orb round requires an Orb attention processor")
            if backbone_layer_mix < 0 or backbone_layer_mix > len(stacks):
                raise ValueError("Invalid number of frozen backbone layers to mix")
            if backbone_layer_mix:
                if orb_message_graph == "hessian" and not orb_combine_backbone_edges:
                    raise ValueError("A dense layer mixture must retain backbone edge features")
                from hip_orb.orb_extension import FrozenLayerMixture
                self.head.backbone_mixture = FrozenLayerMixture(self.head.config["latent_dim"], backbone_layer_mix)
                self.head.config["backbone_layer_mix"] = backbone_layer_mix
            # Register with the head so its optimizer and checkpoints include this layer.
            from hip_orb.orb_extension import make_orb_processor, DenseOrbEdgeEncoder
            if orb_message_graph not in ("backbone", "hessian"):
                raise ValueError("Orb message graph must be backbone or hessian")
            if orb_message_graph == "hessian" and not all_to_all:
                raise ValueError("The dense Orb extension requires the all-to-all Hessian graph")
            self.head.orb_message_layer = make_orb_processor(stacks[-1], orb_initialization)
            if orb_rounds > 1:
                self.head.extra_orb_layers = nn.ModuleList(
                    make_orb_processor(stacks[-1], orb_initialization) for _ in range(orb_rounds - 1))
                self.head.config["orb_rounds"] = orb_rounds
            self.head.config.update(extra_orb_round=True,
                                    extra_orb_initialization=("copy_final_processor" if orb_initialization == "pretrained" else "random"))
            if orb_message_graph != "backbone":
                self.head.config["orb_message_graph"] = orb_message_graph
                expanded_basis = "log_gaussian" if radial_features == "log_gaussian" else "bessel"
                encoder = DenseOrbEdgeEncoder(self.head.config["latent_dim"], radial_cutoff, radial_bases,
                                              expanded_basis, radial_log_range)
                if shared_geometry:
                    self.head.shared_geometry_encoder = encoder
                    self.head.orb_distance_encoder = nn.Linear(self.head.config["latent_dim"], self.head.config["latent_dim"])
                else:
                    self.head.orb_distance_encoder = encoder
                if orb_combine_backbone_edges:
                    from hip_orb.orb_extension import BackboneEdgeResidual
                    self.head.orb_backbone_edge_residual = BackboneEdgeResidual(self.head.config["latent_dim"])
                    self.head.config["orb_combine_backbone_edges"] = True
            if orb_feature_adapters != "none":
                from hip_orb.orb_extension import OrbFeatureAdapters
                self.head.orb_feature_adapter = OrbFeatureAdapters(
                    self.head.config["latent_dim"], orb_feature_adapters, mlp_hidden or hidden,
                    radial_scale=radial_cutoff, radial_bases=radial_bases,
                    radial_features="log_gaussian" if radial_features == "log_gaussian" else "bessel",
                    radial_log_range=radial_log_range)
                self.head.config["orb_feature_adapters"] = orb_feature_adapters
            self.head.to(next(base.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def features(self, graph, *, sizes=None, pairs=None):
        if not getattr(graph, "_hip_known_molecular", False) and bool(graph.system_features["pbc"].any()):
            raise NotImplementedError("HIP analytical Coulomb currently supports molecules only")
        senders, receivers = graph.senders, graph.receivers
        dense_graph = None
        if hasattr(self.head, "orb_distance_encoder"):
            from hip_orb.orb_extension import dense_message_topology
            shape = graph.n_node if sizes is None else sizes
            if hasattr(graph, "_hip_dense_topology"):
                senders, receivers = graph._hip_dense_topology
            else:
                senders, receivers = dense_message_topology(tuple(shape.tolist()), str(graph.senders.device))
            dense_graph = copy(graph)
            dense_graph.senders, dense_graph.receivers = senders, receivers
            dense_graph.n_edge = graph.n_node * (graph.n_node - 1)
            dense_graph.edge_features = {"vectors": graph.node_features["positions"][receivers] - graph.node_features["positions"][senders]}
        captured, hooks = [], []
        if hasattr(self.head, "backbone_mixture"):
            count = self.head.config["backbone_layer_mix"]
            hooks = [layer.register_forward_hook(lambda module, args, result: captured.append(result))
                     for layer in self.base.model.gnn_stacks[-count:]]
        try:
            with torch.no_grad():
                out = self.base.model(graph)
        finally:
            for hook in hooks:
                hook.remove()
        with torch.no_grad():
            charges = self.base.heads["latent_charges"](out["node_features"], graph).reshape(-1)
            cond_nodes, cond_edges = None, None
            if hasattr(self.head, "orb_message_layer"):
                from orb_models.common.models.nn_util import get_cutoff
                message_cutoff = (graph.node_features["positions"].new_ones((len(senders), 1))
                                  if dense_graph is not None else get_cutoff(graph.edge_features["vectors"].norm(dim=-1)))
                conditioner = getattr(self.base.model, "conditioner", None)
                if conditioner is not None:
                    cond_nodes, cond_edges = conditioner(dense_graph if dense_graph is not None else graph)
        nodes, edges = out["node_features"].detach(), out["edge_features"].detach()
        if hasattr(self.head, "backbone_mixture"):
            nodes, edges = self.head.backbone_mixture(captured)
        geometry = None
        if dense_graph is not None:
            from hip_orb.block_layout import require
            vectors = dense_graph.edge_features["vectors"].detach()
            require((vectors.norm(dim=-1) > 0).all(), "Dense Orb edges require distinct positions")
            if self.head.shared_geometry:
                geometry = self.head.shared_geometry_encoder(vectors)
                geometric = self.head.orb_distance_encoder(geometry)
            else:
                geometric = self.head.orb_distance_encoder(vectors)
            edges = (self.head.orb_backbone_edge_residual(
                geometric, edges, graph.senders, graph.receivers, senders, receivers, len(nodes),
                dense_sizes=tuple(shape.tolist()) if getattr(self, "_dense_indexing", False) else None)
                if hasattr(self.head, "orb_backbone_edge_residual") else geometric)
            # CPU-prepared lookup IDs refer to the original Orb edges, not this new graph.
            pairs = None
        if hasattr(self.head, "orb_message_layer"):
            if hasattr(self.head, "orb_feature_adapter"):
                nodes, edges = self.head.orb_feature_adapter(
                    nodes, edges, graph.node_features["atomic_numbers"], charges,
                    graph.node_features["positions"].detach(), senders, receivers)
            if getattr(self, "_dense_message_kernels", False) and dense_graph is not None:
                from hip_orb.dense_interaction import dense_interaction
                nodes, edges = dense_interaction(
                    self.head.orb_message_layer, nodes, edges, senders, receivers, message_cutoff,
                    cond_nodes=cond_nodes, cond_edges=cond_edges, sizes=tuple(shape.tolist()))
            else:
                nodes, edges = self.head.orb_message_layer(
                    nodes, edges, senders, receivers, message_cutoff,
                    cond_nodes=cond_nodes, cond_edges=cond_edges)
            for processor in getattr(self.head, "extra_orb_layers", ()):
                nodes, edges = processor(nodes, edges, senders, receivers, message_cutoff,
                                         cond_nodes=cond_nodes, cond_edges=cond_edges)
        dimension = self.head.config["latent_dim"]
        if out["node_features"].shape[-1] != dimension or out["edge_features"].shape[-1] != dimension:
            raise ValueError("Orb output feature dimensions differ from the Hessian head")
        coulomb = self.base.coulomb_module
        return Features(
            nodes,
            edges,
            senders,
            receivers,
            graph.node_features["positions"].detach(),
            charges.detach(),
            graph.n_node if sizes is None else sizes,
            coulomb.coulomb_constant if getattr(self, "_async_metadata", False) else float(coulomb.coulomb_constant),
            coulomb.direct_coulomb_erf_damping_sigma,
            pairs,
            geometry,
            dense_graph is not None and getattr(self, "_dense_indexing", False),
        )

    def forward(self, graph):
        return self.head(self.features(graph))

    def predict(self, graph, *, energy_forces=True):
        hessians = self(graph)
        # Delegate E/F, without replacing heads or changing the force graph.
        result = self.base.predict(graph, split=False) if energy_forces else {}
        return result | {"hessian": hessians}

    @classmethod
    def pretrained(cls, device="cuda", precision="float32-high", weights_path=None, **kwargs):
        from orb_models import __version__
        from orb_models.forcefield.pretrained import orbmol_v2

        if __version__.split("+")[0].lstrip("v") != "0.7.0":
            raise RuntimeError("HIP collection compatibility requires Orb 0.7.0")
        options = {"device": device, "precision": precision, "compile": False}
        if weights_path is not None:
            options["weights_path"] = str(weights_path)
        base, adapter = orbmol_v2(**options)
        base.disable_stress()
        return cls(base, adapter, **kwargs)
