"""Debug overlay for KM Tile & Time Merge: tile outlines, numbers, overlap bands, chunk labels.

Drawn on a copy of the merged result. Lines and tints use torch; text uses
Pillow, which ships with ComfyUI.
"""

import colorsys
import random

import torch
from PIL import Image, ImageDraw, ImageFont

TINT = 0.2  # overlap bands are pulled this far toward white
LABEL_BOX_ALPHA = 0.6


def overlay_connected(prompt, unique_id, slot=1):
    """True when any node in the API prompt takes output `slot` of node `unique_id`."""
    target = str(unique_id)
    for node in (prompt or {}).values():
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == target and value[1] == slot:
                return True
    return False


def placeholder():
    """1x1 black IMAGE returned when nothing uses the overlay output."""
    return torch.zeros(1, 1, 1, 3)


def tile_colors(n, seed=42):
    """n distinct RGB colors in 0..1: even hues, alternating saturation and brightness, shuffled."""
    colors = []
    for i in range(n):
        hue = (i / n) % 1.0
        saturation = 0.9 + (i % 2) * 0.1
        value = 0.8 + ((i // 2) % 2) * 0.2
        colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
    random.Random(seed).shuffle(colors)
    return colors


def sizes_for_width(width):
    """(line_px, font_px) scaled between 512 and 1920 px of output width."""
    f = min(1.0, max(0.0, (width - 512) / (1920 - 512)))
    return int(round(2 + 4 * f)), int(round(14 + 34 * f))


def frame_labels(t_spans, n_frames):
    """Label per output frame: 'chunk k' or 'chunk k-(k+1) blend' (1-based)."""
    labels = []
    for f in range(n_frames):
        ks = [k + 1 for k, (s, e) in enumerate(t_spans) if s <= f < e]
        if len(ks) >= 2:
            labels.append(f"chunk {ks[0]}-{ks[-1]} blend")
        else:
            labels.append(f"chunk {ks[0] if ks else 1}")
    return labels


def _coverage(spans, length):
    cov = torch.zeros(length, dtype=torch.int32)
    for s, e in spans:
        cov[max(0, s):min(length, e)] += 1
    return cov


def _font(px):
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # Pillow older than 10.1 has no sized default font
        return ImageFont.load_default()


def _text_alpha(text, px):
    """Rendered text as a float alpha mask [h, w] in 0..1."""
    font = _font(px)
    left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    img = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    ImageDraw.Draw(img).text((-left, -top), text, fill=255, font=font)
    data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    return data.view(img.height, img.width).float() / 255.0


def _blend_region(out, frames, y, x, alpha, color, channels):
    """out[frames, y:y+h, x:x+w, :channels] = mix(out, color, alpha), clipped to the frame."""
    h = min(alpha.shape[0], out.shape[1] - y)
    w = min(alpha.shape[1], out.shape[2] - x)
    if h <= 0 or w <= 0:
        return
    a = alpha[:h, :w].unsqueeze(-1)
    region = out[frames, y:y + h, x:x + w, :channels]
    out[frames, y:y + h, x:x + w, :channels] = region * (1 - a) + color[:channels] * a


def draw_overlay(image, plan, lay):
    """Overlay copy of a merged [T, H, W, C] image, using Merge's output layout."""
    out = image.clone()
    frames, height, width, chans = out.shape
    ch = min(chans, 3)
    line, font_px = sizes_for_width(width)

    band = (_coverage(lay.y_spans, height) >= 2)[:, None] | (_coverage(lay.x_spans, width) >= 2)[None, :]
    for f in range(frames):  # one frame at a time keeps the indexing temporaries small
        fr = out[f]
        fr[band, :ch] = fr[band, :ch] * (1 - TINT) + TINT

    colors = tile_colors(plan.n_tiles)
    for r, (y0, y1) in enumerate(lay.y_spans):
        for c, (x0, x1) in enumerate(lay.x_spans):
            y0c, y1c, x0c, x1c = min(y0, height), min(y1, height), min(x0, width), min(x1, width)
            if y1c <= y0c or x1c <= x0c:
                continue
            color = torch.tensor(colors[r * plan.cols + c], dtype=torch.float32)
            for ya, yb, xa, xb in (
                (y0c, min(y0c + line, y1c), x0c, x1c),
                (max(y1c - line, y0c), y1c, x0c, x1c),
                (y0c, y1c, x0c, min(x0c + line, x1c)),
                (y0c, y1c, max(x1c - line, x0c), x1c),
            ):
                out[:, ya:yb, xa:xb, :ch] = color[:ch]
            alpha = _text_alpha(str(r * plan.cols + c + 1), font_px)
            _blend_region(out, slice(None), y0c + 2 * line, x0c + 2 * line, alpha, color, ch)

    white = torch.ones(3)
    black = torch.zeros(3)
    labels = frame_labels(lay.t_spans, frames)
    for text in dict.fromkeys(labels):
        idx = [f for f, t in enumerate(labels) if t == text]
        alpha = _text_alpha(text, font_px)
        pad = max(2, font_px // 4)
        box = torch.full((alpha.shape[0] + 2 * pad, alpha.shape[1] + 2 * pad), LABEL_BOX_ALPHA)
        top = max(0, height - box.shape[0])  # bottom-left: tile 1's number owns the top-left corner
        _blend_region(out, idx, top, 0, box, black, ch)
        _blend_region(out, idx, top + pad, pad, alpha, white, ch)
    return out
