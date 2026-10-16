"""Independent Orb processors and geometric encoders for a new message graph."""

from copy import deepcopy
from functools import lru_cache

import torch
from torch import nn

from orb_models.common.models.angular import SphericalHarmonics
from orb_models.common.models.rbf import BesselBasis
from orb_models.common.models.nn_util import mlp_and_layer_norm


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
    senders, receivers = [], []
    offset = 0
    for n in sizes:
        ids = torch.arange(n, device=device)
        a = ids[:, None].expand(n, n).reshape(-1)
        b = ids[None, :].expand(n, n).reshape(-1)
        keep = a != b
        senders.append(a[keep] + offset)
        receivers.append(b[keep] + offset)
        offset += n
    return torch.cat(senders), torch.cat(receivers)


class DenseOrbEdgeEncoder(nn.Module):
    """Fresh radial/angular encoding, with no hard distance cutoff."""

    def __init__(self, latent_dim, radial_scale=12.0, radial_bases=16):
        super().__init__()
        self.radial = BesselBasis(radial_scale, radial_bases, trainable=True)
        self.angular = SphericalHarmonics(lmax=3, normalize=True, normalization="component")
        self.projection = mlp_and_layer_norm(radial_bases * 16 + 1, latent_dim, 1024,
                                             2, activation="silu", mlp_norm="rms_norm")

    def forward(self, vectors):
        distance = vectors.norm(dim=-1)
        radial = self.radial(distance)
        angular = self.angular(vectors)
        combined = (radial[:, :, None] * angular[:, None, :]).flatten(1)
        return self.projection(torch.cat((combined, torch.log1p(distance[:, None])), dim=-1))
