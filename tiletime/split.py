"""Cut a video tensor into the items described by a TilePlan."""

import torch


def pad_source(images, plan):
    """Replicate the last frame, bottom row and right column up to the padded size."""
    if not (plan.pad_t or plan.pad_h or plan.pad_w):
        return images
    n, h, w, c = images.shape
    out = images.new_empty(n + plan.pad_t, h + plan.pad_h, w + plan.pad_w, c)
    out[:n, :h, :w] = images
    if plan.pad_w:
        out[:n, :h, w:] = images[:, :, -1:].expand(-1, -1, plan.pad_w, -1)
    if plan.pad_h:
        out[:n, h:, :] = out[:n, h - 1:h, :].expand(-1, plan.pad_h, -1, -1)
    if plan.pad_t:
        out[n:, :, :] = out[n - 1:n, :, :].expand(plan.pad_t, -1, -1, -1)
    return out


def split_items(images, plan):
    """Return the item list in tile-major order. Items are views when no padding is needed."""
    expected = (plan.n_frames, plan.src_h, plan.src_w, plan.channels)
    if tuple(images.shape) != expected:
        raise ValueError(f"images shape {tuple(images.shape)} does not match the plan {expected}")
    src = pad_source(images, plan)
    items = []
    for y in plan.ys:
        for x in plan.xs:
            for t in plan.chunk_starts:
                items.append(src[t:t + plan.chunk_len, y:y + plan.tile_h, x:x + plan.tile_w, :])
    return items
