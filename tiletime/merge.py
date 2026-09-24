"""Stitch processed items back into one video using a TilePlan."""

import logging

import torch
import torch.nn.functional as F

from .plan import PLAN_VERSION, TilePlan

log = logging.getLogger("KM-TileTime")

BLENDS = ("feather", "hard cut")
CURVES = ("linear", "smoothstep")
OUTPUT_SIZES = ("scaled", "source")
EPS = 1e-8


def _ramp_up(pos, a, b, blend, curve, width_pct):
    """0 -> 1 ramp centred in the overlap [a, b)."""
    mid = (a + b) / 2.0
    width = 0.0 if blend == "hard cut" else (b - a) * width_pct / 100.0
    if width <= 0:
        return (pos >= mid).double()
    t = ((pos - (mid - width / 2.0)) / width).clamp(0.0, 1.0)
    if curve == "smoothstep":
        t = t * t * (3.0 - 2.0 * t)
    return t


def axis_weights(spans, length, blend="feather", curve="smoothstep", width_pct=100):
    """1D weights for pieces along one axis.

    spans: (start, end) per piece, sorted by start, together covering [0, length).
    Returns float32 [len(spans), length], zero outside each piece. Sides without a
    neighbour keep weight 1.
    """
    pos = torch.arange(length, dtype=torch.float64) + 0.5  # pixel centres
    out = torch.zeros(len(spans), length, dtype=torch.float64)
    for i, (s, e) in enumerate(spans):
        w = torch.zeros(length, dtype=torch.float64)
        w[s:e] = 1.0
        if i > 0:
            w *= _ramp_up(pos, s, spans[i - 1][1], blend, curve, width_pct)
        if i < len(spans) - 1:
            w *= 1.0 - _ramp_up(pos, spans[i + 1][0], e, blend, curve, width_pct)
        out[i] = w
    return out.float()
