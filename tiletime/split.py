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


def normalize_mask(mask, n_frames, height, width):
    """Bring a MASK to [n_frames, height, width]; a single frame is broadcast without copying."""
    m = mask.unsqueeze(0) if mask.ndim == 2 else mask
    if m.ndim != 3:
        raise ValueError(f"mask must be [frames, height, width], got shape {tuple(mask.shape)}")
    if tuple(m.shape[1:]) != (height, width):
        raise ValueError(
            f"mask is {m.shape[2]}x{m.shape[1]} but images are {width}x{height}: "
            "the mask must match the image size"
        )
    if m.shape[0] == 1 and n_frames > 1:
        return m.expand(n_frames, height, width)
    if m.shape[0] != n_frames:
        raise ValueError(
            f"mask has {m.shape[0]} frames but images have {n_frames}: "
            f"use a mask with {n_frames} frames or a single frame"
        )
    return m


def split_mask_items(mask, plan):
    """Cut a [N, H, W] mask exactly like the images (same tiles, chunks and padding)."""
    src = pad_source(mask.unsqueeze(-1), plan)[..., 0]
    items = []
    for y in plan.ys:
        for x in plan.xs:
            for t in plan.chunk_starts:
                items.append(src[t:t + plan.chunk_len, y:y + plan.tile_h, x:x + plan.tile_w])
    return items


def ones_mask_items(plan):
    """All-ones masks for every item, as views of a single value (no per-item memory)."""
    one = torch.ones(1, 1, 1).expand(plan.chunk_len, plan.tile_h, plan.tile_w)
    return [one] * plan.n_items
