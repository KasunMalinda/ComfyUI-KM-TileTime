"""Shared test clips and helpers."""

import torch

from tiletime.plan import make_plan


def smooth_video(n, h, w, c=3):
    """Low-frequency test clip: resampling it changes values only slightly."""
    t = torch.linspace(0, 1, n).view(n, 1, 1, 1)
    y = torch.linspace(0, 1, h).view(1, h, 1, 1)
    x = torch.linspace(0, 1, w).view(1, 1, w, 1)
    ch = torch.arange(c).view(1, 1, 1, c)
    return 0.5 + 0.25 * torch.sin(3 * x + 2 * y + t + ch) * torch.cos(2 * x - y + 0.5 * ch)


def random_video(n, h, w, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, h, w, c, generator=g)


def split_and_merge(video, params, **merge_kwargs):
    from tiletime.merge import merge_items
    from tiletime.split import split_items

    plan = make_plan(*video.shape, params)
    items = split_items(video, plan)
    return plan, items, merge_items(items, plan, **merge_kwargs)
