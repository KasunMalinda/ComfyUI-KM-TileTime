"""Stitch processed items back into one video using a TilePlan."""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .color import gain_offset, lowpass_stats, solve_neighbours
from .plan import PLAN_VERSION, TilePlan

log = logging.getLogger("KM-TileTime")

BLENDS = ("feather", "hard cut")
CURVES = ("linear", "smoothstep")
OUTPUT_SIZES = ("scaled", "source")
COLOR_MATCHES = ("off", "neighbours", "source")
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


@dataclass(frozen=True)
class Layout:
    """Where every piece lands on the output canvas (shared by merging and the overlay)."""

    piece_h: int
    piece_w: int
    canvas_t: int
    canvas_h: int
    canvas_w: int
    out_h: int
    out_w: int
    t_spans: list
    y_spans: list
    y_crops: list
    y_fills: list
    x_spans: list
    x_crops: list
    x_fills: list


def output_layout(plan, piece_h, piece_w):
    """Layout for pieces of size piece_h x piece_w (the returned size, or the tile size in "source" mode)."""
    canvas_h = _scaled_pos(plan.padded_h, piece_h, plan.tile_h)
    canvas_w = _scaled_pos(plan.padded_w, piece_w, plan.tile_w)
    y_spans, y_crops, y_fills = _place(plan.ys, piece_h, plan.tile_h, canvas_h)
    x_spans, x_crops, x_fills = _place(plan.xs, piece_w, plan.tile_w, canvas_w)
    return Layout(
        piece_h=piece_h,
        piece_w=piece_w,
        canvas_t=plan.n_frames + plan.pad_t,
        canvas_h=canvas_h,
        canvas_w=canvas_w,
        out_h=_scaled_pos(plan.src_h, piece_h, plan.tile_h),
        out_w=_scaled_pos(plan.src_w, piece_w, plan.tile_w),
        t_spans=[(t, t + plan.chunk_len) for t in plan.chunk_starts],
        y_spans=y_spans,
        y_crops=y_crops,
        y_fills=y_fills,
        x_spans=x_spans,
        x_crops=x_crops,
        x_fills=x_fills,
    )


def piece_size(items, plan, output_size):
    """(piece_h, piece_w): the returned item size, or the planned tile size in "source" mode."""
    if output_size == "source":
        return plan.tile_h, plan.tile_w
    return int(items[0].shape[1]), int(items[0].shape[2])


def _prepare(item, plan, lay, r, c):
    """One item as it lands on the canvas: trimmed frames, float32 CPU, resized ("source"), fitted."""
    piece = item[: plan.chunk_len].to(device="cpu", dtype=torch.float32)
    if tuple(piece.shape[1:3]) != (lay.piece_h, lay.piece_w):
        piece = _resize(piece, lay.piece_h, lay.piece_w)  # "source" mode only
    piece = _fit(piece, 1, lay.y_crops[r], lay.y_fills[r])
    return _fit(piece, 2, lay.x_crops[c], lay.x_fills[c])


def _shared(a, b):
    """Intersection of two (start, end) spans."""
    return max(a[0], b[0]), min(a[1], b[1])


def _neighbour_pairs(items, plan, lay):
    """Low-pass statistics of every region two neighbouring items share.

    Neighbours are the tile to the right and below in the same chunk, and the
    same tile in the next chunk. Returns (p, q, mean_p, std_p, mean_q, std_q).
    """
    pairs = []
    for i in range(plan.n_items):
        r, c, k = plan.item_coords(i)
        for r2, c2, k2 in ((r, c + 1, k), (r + 1, c, k), (r, c, k + 1)):
            if r2 >= plan.rows or c2 >= plan.cols or k2 >= plan.n_chunks:
                continue
            j = (r2 * plan.cols + c2) * plan.n_chunks + k2
            ts = _shared(lay.t_spans[k], lay.t_spans[k2])
            ys = _shared(lay.y_spans[r], lay.y_spans[r2])
            xs = _shared(lay.x_spans[c], lay.x_spans[c2])
            if ts[1] <= ts[0] or ys[1] <= ys[0] or xs[1] <= xs[0]:
                continue
            stats = []
            for n, (rr, cc, kk) in ((i, (r, c, k)), (j, (r2, c2, k2))):
                t0, y0, x0 = lay.t_spans[kk][0], lay.y_spans[rr][0], lay.x_spans[cc][0]
                piece = _prepare(items[n], plan, lay, rr, cc)
                region = piece[ts[0] - t0:ts[1] - t0, ys[0] - y0:ys[1] - y0, xs[0] - x0:xs[1] - x0]
                stats.append(lowpass_stats(region))
            (mp, sp), (mq, sq) = stats
            pairs.append((i, j, mp, sp, mq, sq))
    return pairs


def _color_corrections(items, plan, lay, color_match):
    """Per-item (gain, offset) float32 [k] tensors, or None when no correction applies.

    Only the first k channels are corrected: k = min(3, item channels), and in
    "source" mode also at most the source channel count. Channels from k on
    (alpha, for example) are left alone.
    """
    if color_match == "off":
        return None
    k = min(3, int(items[0].shape[-1]))
    if color_match == "source":
        if len(plan.src_stats) != plan.n_items:
            raise ValueError(
                "color_match 'source' needs a tile_plan from KM Tile & Time Split v2: re-run Split"
            )
        k = min(k, len(plan.src_stats[0][0]))
        corrections = []
        for i, item in enumerate(items):
            r, c, _ = plan.item_coords(i)
            mean, std = lowpass_stats(_prepare(item, plan, lay, r, c))
            ref_mean, ref_std = (torch.tensor(v[:k], dtype=torch.float64) for v in plan.src_stats[i])
            corrections.append(gain_offset(mean[:k], std[:k], ref_mean, ref_std))
    else:
        pairs = [
            (p, q, mp[:k], sp[:k], mq[:k], sq[:k])
            for p, q, mp, sp, mq, sq in _neighbour_pairs(items, plan, lay)
        ]
        a, b = solve_neighbours(plan.n_items, pairs)
        if a is None:
            return None
        corrections = list(zip(a, b))
    gains = torch.stack([g for g, _ in corrections])
    offsets = torch.stack([o for _, o in corrections])
    log.info(
        "KM Tile & Time Merge: color_match %s, largest gain change %.3f, largest offset %.3f",
        color_match, float((gains - 1).abs().max()), float(offsets.abs().max()),
    )
    return [(g.float(), o.float()) for g, o in corrections]


def merge_items(items, plan, blend="feather", curve="smoothstep", width_pct=100, output_size="scaled",
                color_match="off"):
    """Blend processed items into one [n_frames, H', W', C] float32 tensor on CPU.

    "scaled" places every returned piece unresampled at its scaled position, so
    tile pixels reach the output unchanged apart from blending in overlaps.
    "source" resizes each piece back to the planned tile size first.
    color_match "neighbours" or "source" applies a per-item gain and offset
    before blending (see color.py).
    """
    if (blend not in BLENDS or curve not in CURVES or output_size not in OUTPUT_SIZES
            or color_match not in COLOR_MATCHES):
        raise ValueError(
            f"invalid merge settings: blend={blend!r}, curve={curve!r}, "
            f"output_size={output_size!r}, color_match={color_match!r}"
        )
    _, _, c0 = _check_items(items, plan)
    lay = output_layout(plan, *piece_size(items, plan, output_size))
    if any(lay.y_fills) or any(lay.x_fills):
        log.warning("KM Tile & Time Merge: filled a gap between tiles by edge replication")
    corrections = _color_corrections(items, plan, lay, color_match)

    wy = axis_weights(lay.y_spans, lay.canvas_h, blend, curve, width_pct)
    wx = axis_weights(lay.x_spans, lay.canvas_w, blend, curve, width_pct)
    wt = axis_weights(lay.t_spans, lay.canvas_t, blend, curve, width_pct)

    out = torch.zeros(lay.canvas_t, lay.canvas_h, lay.canvas_w, c0, dtype=torch.float32)
    for i, item in enumerate(items):
        r, c, k = plan.item_coords(i)
        (y0, y1), (x0, x1), (t0, t1) = lay.y_spans[r], lay.x_spans[c], lay.t_spans[k]
        piece = _prepare(item, plan, lay, r, c)
        if corrections is not None:
            # A new tensor: _prepare can return a view of the caller's item.
            gain, offset = corrections[i]
            n_ch = gain.shape[0]
            piece = torch.cat([piece[..., :n_ch] * gain + offset, piece[..., n_ch:]], dim=-1)
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

    return out[: plan.n_frames, : lay.out_h, : lay.out_w].contiguous()
