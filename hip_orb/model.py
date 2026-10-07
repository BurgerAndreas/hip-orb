"""Frozen backbone, independent Hessian graph, and trainable block readouts."""

from dataclasses import dataclass
from collections import OrderedDict

import torch
from torch import nn

from hip_orb.coulomb import fixed_charge_hessian, fixed_charge_blocks
from hip_orb.batched_pairs import batch_topology
from hip_orb.block_layout import layout, assemble


@dataclass
class Features:
    nodes: torch.Tensor
    edges: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    positions: torch.Tensor
    charges: torch.Tensor
    sizes: torch.Tensor
    constant: float = 14.3996
    sigma: float | None = None
    pairs: list[dict[str, torch.Tensor]] | None = None


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


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

    def __init__(self, latent_dim=256, hidden=128, cutoff=None):
        super().__init__()
        if cutoff is not None and cutoff <= 0:
            raise ValueError("Hessian cutoff must be positive or None")
        self.config = {"latent_dim": latent_dim, "hidden": hidden, "cutoff": cutoff}
        self.cutoff = cutoff
        self.fuse_pairs = True
        self.batch_pairs = True
        self.cache_pairs = True
        self._pair_indices = OrderedDict()
        self.node_head = mlp(latent_dim + 1, hidden, 9)
        # Two nodes, one directed edge, unit vector, distance, two charges, edge flag.
        self.pair_adapter = mlp(3 * latent_dim + 7, hidden, hidden)
        self.edge_head = nn.Linear(hidden, 9)

    def forward(self, features: Features, *, residual_only=False):
        if self.batch_pairs:
            return self._batched_forward(features, residual_only=residual_only)
        return self._loop_forward(features, residual_only=residual_only)

    def _batched_forward(self, features, *, residual_only=False):
        sizes, i, j, forward, reverse, delta, distance = batch_topology(features)
        charges = features.charges.reshape(-1)
        unit = delta / distance
        radial = torch.log1p(distance)

        def inputs(a, b, ids, direction, radii):
            present = ids >= 0
            padded = torch.cat((features.edges, features.nodes.new_zeros((1, features.nodes.shape[-1]))))
            edge = padded[torch.where(present, ids, len(features.edges))]
            return torch.cat((features.nodes[a], features.nodes[b], edge, direction, radii,
                              charges[a, None], charges[b, None], present[:, None].to(edge.dtype)), dim=-1)

        if self.fuse_pairs:
            joined = inputs(torch.cat((i, j)), torch.cat((j, i)), torch.cat((forward, reverse)),
                            torch.cat((unit, -unit)), torch.cat((radial, radial)))
            directed = self.edge_head(self.pair_adapter(joined))
            first_blocks, second_blocks = directed.reshape(-1, 3, 3).chunk(2) if len(i) else (
                directed.reshape(-1, 3, 3), directed.reshape(-1, 3, 3))
        else:
            first_blocks = self.edge_head(self.pair_adapter(inputs(i, j, forward, unit, radial))).reshape(-1, 3, 3)
            second_blocks = self.edge_head(self.pair_adapter(inputs(j, i, reverse, -unit, radial))).reshape(-1, 3, 3)
        blocks = (first_blocks + second_blocks.transpose(-1, -2)) * 0.5
        diagonal = self.node_head(torch.cat((features.nodes, charges[:, None]), dim=-1)).reshape(-1, 3, 3)
        diagonal = (diagonal + diagonal.transpose(-1, -2)) * 0.5
        if self.cutoff is not None:
            blocks = torch.where((distance < self.cutoff)[:, :, None], blocks, 0)
        if not residual_only:
            coulomb, coulomb_diagonal = fixed_charge_blocks(features.positions, charges, i, j,
                                                            features.constant, features.sigma)
            blocks = blocks + coulomb
            diagonal = diagonal + coulomb_diagonal
        indices = layout(tuple(sizes), str(features.nodes.device))
        return assemble(indices, features.nodes, sizes, blocks, diagonal)

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
                        torch.log1p(r),
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

    def __init__(self, base, adapter, hidden=128, cutoff=None):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.adapter = adapter
        self.head = HessianHead(backbone_latent_dim(base.model), hidden, cutoff)
        self.head.to(next(base.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def features(self, graph, *, sizes=None, pairs=None):
        if bool(graph.system_features["pbc"].any()):
            raise NotImplementedError("HIP analytical Coulomb currently supports molecules only")
        out = self.base.model(graph)
        dimension = self.head.config["latent_dim"]
        if out["node_features"].shape[-1] != dimension or out["edge_features"].shape[-1] != dimension:
            raise ValueError("Orb output feature dimensions differ from the Hessian head")
        charges = self.base.heads["latent_charges"](out["node_features"], graph).reshape(-1)
        coulomb = self.base.coulomb_module
        return Features(
            out["node_features"].detach(),
            out["edge_features"].detach(),
            graph.senders,
            graph.receivers,
            graph.node_features["positions"].detach(),
            charges.detach(),
            graph.n_node if sizes is None else sizes,
            float(coulomb.coulomb_constant),
            coulomb.direct_coulomb_erf_damping_sigma,
            pairs,
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
