"""Layout math for KM Tile & Time: frame rules, tile grid, chunks and TilePlan.

Pure Python (no torch, no ComfyUI imports) so the preview route and the tests
can use it directly.
"""

import math
from dataclasses import dataclass, fields

PLAN_VERSION = 2

# name -> (step, offset, min_valid). A frame count n is valid when
# n >= min_valid and (n - offset) % step == 0.
FRAME_RULES = {
    "none": (1, 0, 1),
    "4n+1": (4, 1, 1),
    "8n+1": (8, 1, 1),
    "17n": (17, 0, 17),
}

# Old frame rule names, kept so workflows saved before the 4k+1/8k+1/17k ->
# 4n+1/8n+1/17n rename still load.
LEGACY_FRAME_RULES = {"4k+1": "4n+1", "8k+1": "8n+1", "17k": "17n"}


def canonical_frame_rule(name):
    """Map a legacy frame_rule name to its current name; other names unchanged."""
    return LEGACY_FRAME_RULES.get(name, name)


TILE_MODES = ("grid", "target size")
OVERLAP_MODES = ("percent", "pixels")
MULTIPLES = (1, 8, 16, 32, 64)
MAX_OVERLAP_PERCENT = 49.0
MAX_OVERLAP_FRACTION = MAX_OVERLAP_PERCENT / 100.0


# ---------------------------------------------------------------------------
# frame rules
# ---------------------------------------------------------------------------

def _rule(name):
    try:
        return FRAME_RULES[name]
    except KeyError:
        raise ValueError(
            f"unknown frame_rule {name!r}, expected one of: {', '.join(FRAME_RULES)}"
        ) from None


def is_valid_frames(n, rule):
    step, offset, min_valid = _rule(rule)
    return n >= min_valid and (n - offset) % step == 0


def snap_frames_down(n, rule):
    """Largest valid count <= n (never below the rule's minimum)."""
    step, offset, min_valid = _rule(rule)
    if n <= min_valid:
        return min_valid
    return n - (n - offset) % step


def snap_frames_up(n, rule):
    """Smallest valid count >= n."""
    step, offset, min_valid = _rule(rule)
    if n <= min_valid:
        return min_valid
    return n + (-(n - offset)) % step


def snap_chunk_frames(chunk_frames, rule, notes):
    """Apply the frame rule to chunk_frames (0 = chunking off stays 0)."""
    if chunk_frames <= 0 or is_valid_frames(chunk_frames, rule):
        return chunk_frames
    snapped = snap_frames_down(chunk_frames, rule)
    notes.append(f"chunk_frames {chunk_frames} -> {snapped} ({rule})")
    return snapped


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

def round_up(x, m):
    return int(math.ceil(x / m - 1e-9)) * m


def round_down(x, m):
    return int(math.floor(x / m + 1e-9)) * m


@dataclass(frozen=True)
class SplitParams:
    tile_mode: str = "grid"
    rows: int = 2
    cols: int = 2
    target_width: int = 1280
    target_height: int = 720
    overlap_mode: str = "percent"
    overlap: float = 10.0
    multiple_of: int = 16
    chunk_frames: int = 0
    chunk_overlap: int = 8
    frame_rule: str = "none"

    def __post_init__(self):
        canonical = canonical_frame_rule(self.frame_rule)
        if canonical != self.frame_rule:
            object.__setattr__(self, "frame_rule", canonical)
        if self.tile_mode not in TILE_MODES:
            raise ValueError(f"tile_mode must be one of {TILE_MODES}, got {self.tile_mode!r}")
        if self.overlap_mode not in OVERLAP_MODES:
            raise ValueError(f"overlap_mode must be one of {OVERLAP_MODES}, got {self.overlap_mode!r}")
        if self.multiple_of not in MULTIPLES:
            raise ValueError(f"multiple_of must be one of {MULTIPLES}, got {self.multiple_of!r}")
        _rule(self.frame_rule)
        for name in ("rows", "cols", "target_width", "target_height"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("overlap", "chunk_frames", "chunk_overlap"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, got {getattr(self, name)}")
        maxima = {
            "rows": 16, "cols": 16, "target_width": 8192, "target_height": 8192,
            "chunk_frames": 10000, "chunk_overlap": 1000, "overlap": 4096,
        }
        for name, limit in maxima.items():
            if getattr(self, name) > limit:
                raise ValueError(f"{name} must be at most {limit}, got {getattr(self, name)}")

    @classmethod
    def from_dict(cls, values):
        """Build from widget values. Unknown keys (such as preset) are ignored."""
        casts = {"overlap": float, "tile_mode": str, "overlap_mode": str, "frame_rule": str}
        kwargs = {}
        for f in fields(cls):
            if f.name in values:
                cast = casts.get(f.name, int)
                try:
                    kwargs[f.name] = cast(values[f.name])
                except (TypeError, ValueError):
                    raise ValueError(f"{f.name} has an invalid value: {values[f.name]!r}") from None
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# spatial axis layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisLayout:
    tile: int
    starts: tuple
    pad: int


def even_starts(size, tile, count):
    """Evenly spaced starts: first at 0, last ending exactly at size."""
    if count == 1:
        return (0,)
    span = size - tile
    return tuple(int(round(i * span / (count - 1))) for i in range(count))


def _finish_axis(size, tile, count, m, axis, notes):
    if count > 1 and tile >= size:
        notes.append(f"{axis}: one tile covers the source, using 1 tile instead of {count}")
        count, tile = 1, round_up(size, m)
    if count == 1:
        return AxisLayout(tile, (0,), tile - size)
    return AxisLayout(tile, even_starts(size, tile, count), 0)


def grid_axis(size, count, overlap_mode, overlap, m, axis, notes):
    if count == 1:
        return _finish_axis(size, round_up(size, m), 1, m, axis, notes)
    # With even spacing the overlap o between neighbours satisfies
    # count * tile - (count - 1) * o = size. Solve for the tile that gives the
    # requested overlap, then round up (the overlap only grows).
    k = count - 1
    if overlap_mode == "percent":
        if overlap > MAX_OVERLAP_PERCENT:
            notes.append(f"overlap capped at {MAX_OVERLAP_PERCENT:g}%")
        p = min(overlap, MAX_OVERLAP_PERCENT) / 100.0
        tile = size / (count - k * p)
    else:
        # o <= 0.49 * tile  <=>  o <= 0.49 * size / (count - 0.49 * k)
        limit = MAX_OVERLAP_FRACTION * size / (count - MAX_OVERLAP_FRACTION * k)
        o = float(overlap)
        if o > limit:
            notes.append(f"{axis} overlap capped at {int(limit)} px")
            o = limit
        tile = (size + k * o) / count
    return _finish_axis(size, round_up(tile, m), count, m, axis, notes)


def target_axis(size, target, overlap_mode, overlap, m, axis, notes):
    tile = max(m, round_down(target, m))
    if size <= tile:
        return _finish_axis(size, round_up(size, m), 1, m, axis, notes)
    if overlap_mode == "percent":
        if overlap > MAX_OVERLAP_PERCENT:
            notes.append(f"overlap capped at {MAX_OVERLAP_PERCENT:g}%")
        o = tile * min(overlap, MAX_OVERLAP_PERCENT) / 100.0
    else:
        o = float(overlap)
        if o > MAX_OVERLAP_FRACTION * tile:
            o = MAX_OVERLAP_FRACTION * tile
            notes.append(f"{axis} overlap capped at {int(o)} px")
    count = int(math.ceil((size - o) / (tile - o) - 1e-9))
    return _finish_axis(size, tile, count, m, axis, notes)


# ---------------------------------------------------------------------------
# temporal chunks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkLayout:
    length: int
    starts: tuple
    pad: int


def chunk_axis(n_frames, chunk_frames, chunk_overlap, rule, notes):
    length = snap_chunk_frames(chunk_frames, rule, notes)
    if length == 0 or length >= n_frames:
        total = snap_frames_up(n_frames, rule)
        if total != n_frames:
            notes.append(f"clip padded {n_frames} -> {total} frames ({rule}), removed again by Merge")
        return ChunkLayout(total, (0,), total - n_frames)
    o = min(chunk_overlap, length - 1)
    count = int(math.ceil((n_frames - o) / (length - o)))
    return ChunkLayout(length, even_starts(n_frames, length, count), 0)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TilePlan:
    n_frames: int
    src_h: int
    src_w: int
    channels: int
    pad_t: int
    pad_h: int
    pad_w: int
    tile_h: int
    tile_w: int
    ys: tuple
    xs: tuple
    chunk_len: int
    chunk_starts: tuple
    frame_rule: str = "none"
    notes: tuple = ()
    src_stats: tuple = ()  # per item ((means...), (stds...)) of the source region, set by Split
    order: str = "tile-major"
    version: int = PLAN_VERSION

    @property
    def rows(self):
        return len(self.ys)

    @property
    def cols(self):
        return len(self.xs)

    @property
    def n_tiles(self):
        return self.rows * self.cols

    @property
    def n_chunks(self):
        return len(self.chunk_starts)

    @property
    def n_items(self):
        return self.n_tiles * self.n_chunks

    @property
    def padded_h(self):
        return self.src_h + self.pad_h

    @property
    def padded_w(self):
        return self.src_w + self.pad_w

    def item_coords(self, index):
        """Item index -> (row, col, chunk). Order is tile-major, tiles row by row."""
        tile, chunk = divmod(index, self.n_chunks)
        row, col = divmod(tile, self.cols)
        return row, col, chunk


def make_plan(n_frames, height, width, channels, params):
    if min(n_frames, height, width, channels) < 1:
        raise ValueError(
            f"source must have at least 1 frame, pixel and channel, got "
            f"{n_frames} frames, {width}x{height}, {channels} channels"
        )
    notes = []
    m = params.multiple_of
    if params.tile_mode == "grid":
        ay = grid_axis(height, params.rows, params.overlap_mode, params.overlap, m, "height", notes)
        ax = grid_axis(width, params.cols, params.overlap_mode, params.overlap, m, "width", notes)
    else:
        ay = target_axis(height, params.target_height, params.overlap_mode, params.overlap, m, "height", notes)
        ax = target_axis(width, params.target_width, params.overlap_mode, params.overlap, m, "width", notes)
    ch = chunk_axis(n_frames, params.chunk_frames, params.chunk_overlap, params.frame_rule, notes)
    if ay.pad or ax.pad:
        notes.append(f"source padded to {width + ax.pad}x{height + ay.pad} (edge pixels), removed again by Merge")
    return TilePlan(
        n_frames=n_frames, src_h=height, src_w=width, channels=channels,
        pad_t=ch.pad, pad_h=ay.pad, pad_w=ax.pad,
        tile_h=ay.tile, tile_w=ax.tile, ys=ay.starts, xs=ax.starts,
        chunk_len=ch.length, chunk_starts=ch.starts,
        frame_rule=params.frame_rule,
        notes=tuple(dict.fromkeys(notes)),  # drop duplicates, keep order
    )


# ---------------------------------------------------------------------------
# description (Split's info output and the node's info panel)
# ---------------------------------------------------------------------------

def _overlaps(starts, length):
    return [starts[i] + length - starts[i + 1] for i in range(len(starts) - 1)]


def _span_text(values):
    if not values:
        return "0"
    lo, hi = min(values), max(values)
    return f"{lo}" if lo == hi else f"{lo}-{hi}"


def _plural(n, word):
    return word if n == 1 else word + "s"


def describe(plan):
    ox = _span_text(_overlaps(plan.xs, plan.tile_w))
    oy = _span_text(_overlaps(plan.ys, plan.tile_h))
    lines = [
        f"Source {plan.src_w}x{plan.src_h}, {plan.n_frames}f, {plan.channels}ch",
        f"Grid {plan.cols} {_plural(plan.cols, 'col')} x {plan.rows} {_plural(plan.rows, 'row')} "
        f"= {plan.n_tiles} {_plural(plan.n_tiles, 'tile')}, "
        f"{plan.tile_w}x{plan.tile_h} each (overlap {ox}/{oy} px)",
    ]
    if plan.n_chunks == 1:
        lines.append(f"Chunks 1 x {plan.chunk_len}f (whole clip), frame rule {plan.frame_rule}")
    else:
        ot = _span_text(_overlaps(plan.chunk_starts, plan.chunk_len))
        lines.append(
            f"Chunks {plan.n_chunks} x {plan.chunk_len}f (overlap {ot}), frame rule {plan.frame_rule}"
        )
    lines.append(f"Items {plan.n_items} through the branch")
    lines.append(
        f"Output {plan.src_w}x{plan.src_h} at 1x (x2 model: {plan.src_w * 2}x{plan.src_h * 2})"
    )
    if plan.notes:
        lines.append("Adjustments: " + "; ".join(plan.notes))
    return "\n".join(lines)
