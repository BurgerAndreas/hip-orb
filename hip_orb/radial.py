"""Nonperiodic molecular distance features without a finite support boundary."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from orb_models.common.models.rbf import BesselBasis


class LogGaussianBasis(nn.Module):
    """Gaussian log-distance features, a constant, and a smooth inverse distance."""

    def __init__(self, num_bases=16, log_range=4.0):
        super().__init__()
        if num_bases < 4 or not math.isfinite(log_range) or log_range <= 0:
            raise ValueError("Log Gaussian basis requires at least four channels and a positive log range")
        self.num_bases = num_bases
        count = num_bases - 2
        self.centers = nn.Parameter(torch.linspace(0.0, log_range, count))
        width = log_range / (count - 1)
        self.raw_widths = nn.Parameter(torch.full((count,), math.log(math.expm1(width))))

    def forward(self, distances):
        # Distances are measured in Angstrom; the reference length is one Angstrom.
        x = torch.log1p(distances.reshape(-1, 1))
        widths = F.softplus(self.raw_widths) + 1e-4
        gaussian = torch.exp(-0.5 * ((x - self.centers) / widths).square())
        return torch.cat((gaussian, torch.ones_like(x), torch.reciprocal(1 + distances.reshape(-1, 1))), dim=-1)


def make_radial(kind, num_bases, radial_scale=12.0, log_range=4.0, *, trainable=True):
    if kind == "log_gaussian":
        return LogGaussianBasis(num_bases, log_range)
    if kind == "bessel":
        return BesselBasis(radial_scale, num_bases, trainable=trainable)
    raise ValueError("Unknown expanded radial basis")
