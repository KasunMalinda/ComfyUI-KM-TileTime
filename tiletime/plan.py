"""Layout math for KM Tile & Time: frame rules, tile grid, chunks and TilePlan.

Pure Python (no torch, no ComfyUI imports) so the preview route and the tests
can use it directly.
"""

import math
from dataclasses import dataclass, fields

PLAN_VERSION = 1

# name -> (step, offset, min_valid). A frame count n is valid when
# n >= min_valid and (n - offset) % step == 0.
FRAME_RULES = {
    "none": (1, 0, 1),
    "4k+1": (4, 1, 1),
    "8k+1": (8, 1, 1),
    "17k": (17, 0, 17),
}

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
