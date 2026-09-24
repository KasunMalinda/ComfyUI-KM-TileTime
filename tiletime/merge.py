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


def _scaled_pos(pos, returned, tile):
    """round(pos * returned / tile), rounding halves up, in exact integer math."""
    return (2 * pos * returned + tile) // (2 * tile)


def _place(starts, returned, tile, canvas):
    """Canvas placement of unresampled pieces along one axis.

    Each piece of size `returned` starts at _scaled_pos(start). Returns
    (spans, crops, fills): spans[i] = (s, e) on the canvas, crops[i] = pixels
    dropped from the piece's end where it runs past the canvas, fills[i] =
    pixels of edge replication added at its end to close a gap before the next
    piece or the canvas end. With exact rounding and non-negative overlap
    neighbours always touch or overlap, so crops and fills stay 0; they are a
    guard only.
    """
    spans, crops, fills = [], [], []
    for i, start in enumerate(starts):
        s = _scaled_pos(start, returned, tile)
        e = min(s + returned, canvas)
        nxt = _scaled_pos(starts[i + 1], returned, tile) if i + 1 < len(starts) else canvas
        fill = max(0, nxt - e)
        spans.append((s, e + fill))
        crops.append(s + returned - e)
        fills.append(fill)
    return spans, crops, fills


def _fit(piece, dim, crop, fill):
    """Drop `crop` pixels from the end of `dim`, then replicate the edge `fill` times."""
    if crop:
        piece = piece.narrow(dim, 0, piece.shape[dim] - crop)
    if fill:
        edge = piece.narrow(dim, piece.shape[dim] - 1, 1)
        sizes = [-1] * piece.ndim
        sizes[dim] = fill
        piece = torch.cat([piece, edge.expand(*sizes)], dim=dim)
    return piece


def merge_items(items, plan, blend="feather", curve="smoothstep", width_pct=100, output_size="scaled"):
    """Blend processed items into one [n_frames, H', W', C] float32 tensor on CPU.

    "scaled" places every returned piece unresampled at its scaled position, so
    tile pixels reach the output unchanged apart from blending in overlaps.
    "source" resizes each piece back to the planned tile size first.
    """
    if blend not in BLENDS or curve not in CURVES or output_size not in OUTPUT_SIZES:
        raise ValueError(
            f"invalid merge settings: blend={blend!r}, curve={curve!r}, output_size={output_size!r}"
        )
    h0, w0, c0 = _check_items(items, plan)
    ph, pw = (plan.tile_h, plan.tile_w) if output_size == "source" else (h0, w0)

    canvas_h = _scaled_pos(plan.padded_h, ph, plan.tile_h)
    canvas_w = _scaled_pos(plan.padded_w, pw, plan.tile_w)
    canvas_t = plan.n_frames + plan.pad_t
    y_spans, y_crops, y_fills = _place(plan.ys, ph, plan.tile_h, canvas_h)
    x_spans, x_crops, x_fills = _place(plan.xs, pw, plan.tile_w, canvas_w)
    t_spans = [(t, t + plan.chunk_len) for t in plan.chunk_starts]
    if any(y_fills) or any(x_fills):
        log.warning("KM Tile & Time Merge: filled a gap between tiles by edge replication")

    wy = axis_weights(y_spans, canvas_h, blend, curve, width_pct)
    wx = axis_weights(x_spans, canvas_w, blend, curve, width_pct)
    wt = axis_weights(t_spans, canvas_t, blend, curve, width_pct)

    out = torch.zeros(canvas_t, canvas_h, canvas_w, c0, dtype=torch.float32)
    for i, item in enumerate(items):
        r, c, k = plan.item_coords(i)
        (y0, y1), (x0, x1), (t0, t1) = y_spans[r], x_spans[c], t_spans[k]
        piece = item[: plan.chunk_len].to(device="cpu", dtype=torch.float32)
        if tuple(piece.shape[1:3]) != (ph, pw):
            piece = _resize(piece, ph, pw)  # "source" mode only
        piece = _fit(piece, 1, y_crops[r], y_fills[r])
        piece = _fit(piece, 2, x_crops[c], x_fills[c])
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

    out_h = _scaled_pos(plan.src_h, ph, plan.tile_h)
    out_w = _scaled_pos(plan.src_w, pw, plan.tile_w)
    return out[: plan.n_frames, :out_h, :out_w].contiguous()
