"""Info panel text for the Split node, built from plan.py so it always
matches what a run produces."""

from .plan import SplitParams, describe, make_plan, round_down, snap_chunk_frames


def _parse_shape(shape):
    try:
        n, h, w, c = (int(v) for v in shape)
    except (TypeError, ValueError):
        raise ValueError(f"shape must be [frames, height, width, channels], got {shape!r}") from None
    return n, h, w, c


def _unmeasured_text(params):
    notes = []
    lines = ["Source not measured: press Measure to read the source"]
    if params.tile_mode == "target size":
        m = params.multiple_of
        tw = max(m, round_down(params.target_width, m))
        th = max(m, round_down(params.target_height, m))
        lines.append(f"Tiles up to {tw}x{th} (fewer or smaller if the source is smaller)")
    else:
        lines.append(f"Grid {params.cols} cols x {params.rows} rows = {params.rows * params.cols} tiles")
    length = snap_chunk_frames(params.chunk_frames, params.frame_rule, notes)
    if length:
        lines.append(f"Chunks of {length}f (overlap {params.chunk_overlap}), frame rule {params.frame_rule}")
    else:
        lines.append(f"Chunks off (whole clip), frame rule {params.frame_rule}")
    if notes:
        lines.append("Adjustments: " + "; ".join(notes))
    return "\n".join(lines)


def preview_text(shape, params):
    """shape: [frames, height, width, channels] or None. params: widget values dict."""
    split_params = SplitParams.from_dict(params)
    if shape is None:
        return _unmeasured_text(split_params)
    return describe(make_plan(*_parse_shape(shape), split_params))
