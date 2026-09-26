"""Per-item color correction for KM Tile & Time Merge (torch only).

Every item gets one gain `a` and one offset `b` per channel (x' = a * x + b).
Statistics are taken from a low-pass copy (area-downscaled, a few sampled
frames) so they measure tone and color cast, not the detail a model adds.
"""

import torch
import torch.nn.functional as F

LOWPASS_SIDE = 64
MAX_FRAMES = 8
GAIN_MIN = 0.8
GAIN_MAX = 1.25
STD_EPS = 1e-4
NEIGHBOUR_LAMBDA = 1e-3


def lowpass_stats(x, max_side=LOWPASS_SIDE, max_frames=MAX_FRAMES):
    """Per-channel (mean, std) of [F, H, W, C] after frame sampling and area downscale.

    Returns two float64 tensors of shape [C] on CPU.
    """
    frames = x.shape[0]
    if frames > max_frames:
        idx = torch.linspace(0, frames - 1, max_frames).round().long()
        x = x[idx.to(x.device)]
    x = x.to(device="cpu", dtype=torch.float32)
    h, w = x.shape[1], x.shape[2]
    scale = max_side / max(h, w)
    if scale < 1:
        size = (max(1, round(h * scale)), max(1, round(w * scale)))
        x = F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="area").permute(0, 2, 3, 1)
    flat = x.reshape(-1, x.shape[-1]).double()
    return flat.mean(0), flat.std(0, unbiased=False)


def item_stats(items):
    """Source statistics for the plan: one ((means...), (stds...)) pair per item."""
    out = []
    for item in items:
        mean, std = lowpass_stats(item)
        out.append((tuple(mean.tolist()), tuple(std.tolist())))
    return tuple(out)


def gain_offset(item_mean, item_std, ref_mean, ref_std):
    """Gain and offset that move (item_mean, item_std) to (ref_mean, ref_std), gain clamped."""
    a = (ref_std / item_std.clamp_min(STD_EPS)).clamp(GAIN_MIN, GAIN_MAX)
    b = ref_mean - a * item_mean
    return a, b


def solve_neighbours(n_items, pairs, lam=NEIGHBOUR_LAMBDA):
    """Per-item gains and offsets that make overlapping items agree.

    pairs: (p, q, mean_p, std_p, mean_q, std_q) with float64 [C] statistics of
    the region items p and q share. Solves, per channel, the regularised least
    squares problems
        sum (g_p + log s_p - g_q - log s_q)^2 + lam * P * sum g^2   (a = exp(g))
        sum (a_p m_p + b_p - a_q m_q - b_q)^2 + lam * P * sum b^2
    through their normal equations (a graph Laplacian plus a ridge term).
    Returns (a, b) as float64 [n_items, C], or (None, None) when there are no pairs.
    """
    if not pairs:
        return None, None
    channels = pairs[0][2].shape[0]
    reg = lam * len(pairs)
    lap = torch.zeros(n_items, n_items, dtype=torch.float64)
    for p, q, *_ in pairs:
        lap[p, p] += 1
        lap[q, q] += 1
        lap[p, q] -= 1
        lap[q, p] -= 1
    system = lap + reg * torch.eye(n_items, dtype=torch.float64)

    def solve(targets):
        rhs = torch.zeros(n_items, channels, dtype=torch.float64)
        for (p, q, *_), target in zip(pairs, targets):
            rhs[p] += target
            rhs[q] -= target
        return torch.linalg.solve(system, rhs)

    # g_p - g_q = log s_q - log s_p
    g = solve([torch.log(sq.clamp_min(STD_EPS)) - torch.log(sp.clamp_min(STD_EPS))
               for _, _, _, sp, _, sq in pairs])
    a = torch.exp(g).clamp(GAIN_MIN, GAIN_MAX)
    # b_p - b_q = a_q m_q - a_p m_p
    b = solve([a[q] * mq - a[p] * mp for p, q, mp, _, mq, _ in pairs])
    return a, b
