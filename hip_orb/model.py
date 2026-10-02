"""Frozen backbone, independent Hessian graph, and trainable block readouts."""

from dataclasses import dataclass

import torch
from torch import nn

from hip_orb.coulomb import fixed_charge_hessian


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


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, outputs),
    )


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
        self.node_head = mlp(latent_dim + 1, hidden, 9)
        # Two nodes, one directed edge, unit vector, distance, two charges, edge flag.
        self.pair_adapter = mlp(3 * latent_dim + 7, hidden, hidden)
        self.edge_head = nn.Linear(hidden, 9)

    def forward(self, features: Features, *, residual_only=False):
        results = []
        offset = 0
        for size in features.sizes.tolist():
            n = int(size)
            nodes = features.nodes[offset : offset + n]
            positions = features.positions[offset : offset + n]
            charges = features.charges[offset : offset + n].reshape(-1)
            # Keep a directed map: off-diagonal blocks need not be symmetric.
            mapping = torch.full((n, n), -1, device=nodes.device, dtype=torch.long)
            mask = (features.senders >= offset) & (features.senders < offset + n)
            if bool(
                (
                    (features.receivers[mask] < offset) | (features.receivers[mask] >= offset + n)
                ).any()
            ):
                raise ValueError("Message edges cross molecules")
            ids = torch.arange(len(features.edges), device=nodes.device)[mask]
            mapping[features.senders[mask] - offset, features.receivers[mask] - offset] = ids
            i, j = torch.triu_indices(n, n, offset=1, device=nodes.device)
            distance = (positions[i] - positions[j]).norm(dim=-1)
            if bool((distance <= 0).any()):
                raise ValueError("Hessian pairs must have nonzero distance")
            if self.cutoff is not None:
                keep = distance < self.cutoff
                i, j = i[keep], j[keep]

            def read_pair(a, b, mapping=mapping, nodes=nodes, positions=positions, charges=charges):
                edge_ids = mapping[a, b]
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

            blocks = (read_pair(i, j) + read_pair(j, i).transpose(-1, -2)) * 0.5
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
        self.head = HessianHead(base.model.latent_dim, hidden, cutoff)
        self.head.to(next(base.parameters()))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def features(self, graph):
        if bool(graph.system_features["pbc"].any()):
            raise NotImplementedError("HIP analytical Coulomb currently supports molecules only")
        out = self.base.model(graph)
        charges = self.base.heads["latent_charges"](out["node_features"], graph).reshape(-1)
        coulomb = self.base.coulomb_module
        return Features(
            out["node_features"].detach(),
            out["edge_features"].detach(),
            graph.senders,
            graph.receivers,
            graph.node_features["positions"].detach(),
            charges.detach(),
            graph.n_node,
            float(coulomb.coulomb_constant),
            coulomb.direct_coulomb_erf_damping_sigma,
        )

    def forward(self, graph):
        return self.head(self.features(graph))

    def predict(self, graph, *, energy_forces=True):
        hessians = self(graph)
        # Delegate E/F, without replacing heads or changing the force graph.
        result = self.base.predict(graph, split=False) if energy_forces else {}
        return result | {"hessian": hessians}

    @classmethod
    def pretrained(cls, device="cuda", precision="float32-high", **kwargs):
        from orb_models import __version__
        from orb_models.forcefield.pretrained import orbmol_v2

        if __version__.lstrip("v") != "0.7.0":
            raise RuntimeError("HIP collection compatibility requires Orb 0.7.0")
        base, adapter = orbmol_v2(device=device, precision=precision, compile=False)
        base.disable_stress()
        return cls(base, adapter, **kwargs)
