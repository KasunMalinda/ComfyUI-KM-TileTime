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


def _resize(piece, h, w):
    """Resize [F, H, W, C] to [F, h, w, C]: area when shrinking, bilinear otherwise."""
    x = piece.permute(0, 3, 1, 2)
    if h <= x.shape[2] and w <= x.shape[3]:
        x = F.interpolate(x, size=(h, w), mode="area")
    else:
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1)


def _check_items(items, plan):
    if not isinstance(plan, TilePlan):
        raise ValueError("tile_plan is missing: connect the tile_plan output of KM Tile & Time Split")
    if plan.version != PLAN_VERSION:
        raise ValueError(
            f"tile_plan version {plan.version} is not supported (expected {PLAN_VERSION}): "
            "re-run KM Tile & Time Split"
        )
    if len(items) != plan.n_items:
        raise ValueError(
            f"expected {plan.n_items} items ({plan.n_tiles} tiles x {plan.n_chunks} chunks), got {len(items)}"
        )
    for i, it in enumerate(items):
        if it.ndim != 4:
            raise ValueError(
                f"item {i} must be an IMAGE batch [frames, height, width, channels], got shape {tuple(it.shape)}"
            )
    _, h0, w0, c0 = items[0].shape
    for i, it in enumerate(items):
        f, h, w, c = it.shape
        if (h, w) != (h0, w0):
            raise ValueError(
                f"item {i} is {w}x{h} but item 0 is {w0}x{h0}: "
                "every processed tile must come back at the same size"
            )
        if c != c0:
            raise ValueError(
                f"item {i} has {c} channels but item 0 has {c0}: keep the channel count the same for every item"
            )
        if f < plan.chunk_len:
            raise ValueError(
                f"item {i} has {f} frames, expected {plan.chunk_len}: the processing branch dropped frames"
            )
        if f > plan.chunk_len:
            log.warning(
                "KM Tile & Time Merge: item %d has %d frames, expected %d; extra frames trimmed",
                i, f, plan.chunk_len,
            )
    return h0, w0, c0


def merge_items(items, plan, blend="feather", curve="smoothstep", width_pct=100, output_size="scaled"):
    """Blend processed items into one [n_frames, H', W', C] float32 tensor on CPU."""
    if blend not in BLENDS or curve not in CURVES or output_size not in OUTPUT_SIZES:
        raise ValueError(
            f"invalid merge settings: blend={blend!r}, curve={curve!r}, output_size={output_size!r}"
        )
    h0, w0, c0 = _check_items(items, plan)
    if output_size == "source":
        sy = sx = 1.0
    else:
        sy, sx = h0 / plan.tile_h, w0 / plan.tile_w

    y_spans = [(round(y * sy), round((y + plan.tile_h) * sy)) for y in plan.ys]
    x_spans = [(round(x * sx), round((x + plan.tile_w) * sx)) for x in plan.xs]
    t_spans = [(t, t + plan.chunk_len) for t in plan.chunk_starts]
    canvas_h, canvas_w = round(plan.padded_h * sy), round(plan.padded_w * sx)
    canvas_t = plan.n_frames + plan.pad_t

    wy = axis_weights(y_spans, canvas_h, blend, curve, width_pct)
    wx = axis_weights(x_spans, canvas_w, blend, curve, width_pct)
    wt = axis_weights(t_spans, canvas_t, blend, curve, width_pct)

    out = torch.zeros(canvas_t, canvas_h, canvas_w, c0, dtype=torch.float32)
    for i, item in enumerate(items):
        r, c, k = plan.item_coords(i)
        (y0, y1), (x0, x1), (t0, t1) = y_spans[r], x_spans[c], t_spans[k]
        piece = item[: plan.chunk_len].to(device="cpu", dtype=torch.float32)
        if tuple(piece.shape[1:3]) != (y1 - y0, x1 - x0):
            piece = _resize(piece, y1 - y0, x1 - x0)
        w = (
            wt[k, t0:t1].view(-1, 1, 1, 1)
            * wy[r, y0:y1].view(1, -1, 1, 1)
            * wx[c, x0:x1].view(1, 1, -1, 1)
        )
        out[t0:t1, y0:y1, x0:x1] += piece * w

    # Separable normalization: the sum over all pieces of wt*wy*wx equals the
    # product of the per-axis sums, so dividing axis by axis needs no second
    # full-size buffer.
    out.div_(wt.sum(0).clamp_min(EPS).view(-1, 1, 1, 1))
    out.div_(wy.sum(0).clamp_min(EPS).view(1, -1, 1, 1))
    out.div_(wx.sum(0).clamp_min(EPS).view(1, 1, -1, 1))

    out_h, out_w = round(plan.src_h * sy), round(plan.src_w * sx)
    return out[: plan.n_frames, :out_h, :out_w].contiguous()
