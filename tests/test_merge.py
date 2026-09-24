import logging
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from helpers import random_video, smooth_video, split_and_merge
from tiletime.merge import merge_items
from tiletime.plan import SplitParams, make_plan
from tiletime.split import split_items


ROUND_TRIP_CASES = [
    SplitParams(multiple_of=1),
    SplitParams(rows=3, cols=4, overlap_mode="pixels", overlap=6, multiple_of=8),
    SplitParams(rows=1, cols=1, multiple_of=16),  # padded
    SplitParams(tile_mode="target size", target_width=48, target_height=32, multiple_of=16, chunk_frames=9, chunk_overlap=3),
    SplitParams(tile_mode="target size", target_width=64, target_height=64, chunk_frames=13, frame_rule="4k+1"),
    SplitParams(rows=2, cols=2, chunk_frames=0, frame_rule="17k"),  # temporal padding
    SplitParams(rows=16, cols=16, multiple_of=1),
]


@pytest.mark.parametrize("params", ROUND_TRIP_CASES)
@pytest.mark.parametrize("blend,curve,width", [
    ("feather", "smoothstep", 100), ("feather", "linear", 30), ("hard cut", "linear", 100),
])
def test_identity_round_trip(params, blend, curve, width):
    video = random_video(23, 70, 110)
    _, _, out = split_and_merge(video, params, blend=blend, curve=curve, width_pct=width)
    assert out.shape == video.shape
    assert torch.allclose(out, video, atol=1e-5)


def test_identity_single_image_rgba():
    video = random_video(1, 33, 47, c=4)
    _, _, out = split_and_merge(video, SplitParams(rows=2, cols=2, frame_rule="8k+1"))
    assert torch.allclose(out, video, atol=1e-5)


def test_2x_nearest_scaled_and_source():
    video = random_video(20, 48, 80)
    params = SplitParams(rows=2, cols=3, multiple_of=8, chunk_frames=9, chunk_overlap=2)
    plan = make_plan(*video.shape, params)
    up = [i.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2) for i in split_items(video, plan)]
    scaled = merge_items(up, plan, output_size="scaled")
    assert torch.allclose(scaled, video.repeat_interleave(2, 1).repeat_interleave(2, 2), atol=1e-5)
    source = merge_items(up, plan, output_size="source")
    assert torch.allclose(source, video, atol=1e-5)


def test_non_integer_scale():
    video = smooth_video(5, 64, 96)
    plan = make_plan(*video.shape, SplitParams(rows=2, cols=2, multiple_of=16))
    items = split_items(video, plan)
    th, tw = round(plan.tile_h * 1.5), round(plan.tile_w * 1.5)
    resized = [F.interpolate(i.permute(0, 3, 1, 2), size=(th, tw), mode="bilinear", align_corners=False)
               .permute(0, 2, 3, 1) for i in items]
    out = merge_items(resized, plan)
    ref = F.interpolate(video.permute(0, 3, 1, 2), size=(96, 144), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    assert out.shape == ref.shape
    assert (out - ref).abs().max() < 0.02


# --- frame counts and errors ---------------------------------------------------

def _plan_and_items(n=20, chunk=9):
    video = random_video(n, 32, 32)
    plan = make_plan(*video.shape, SplitParams(multiple_of=1, chunk_frames=chunk, chunk_overlap=2))
    return video, plan, split_items(video, plan)


def test_extra_frames_trimmed_with_warning(caplog):
    video, plan, items = _plan_and_items()
    longer = [torch.cat([i, i[-1:]]) for i in items]
    with caplog.at_level(logging.WARNING, logger="KM-TileTime"):
        out = merge_items(longer, plan)
    assert torch.allclose(out, video, atol=1e-5)
    assert "extra frames trimmed" in caplog.text


def test_fewer_frames_error():
    _, plan, items = _plan_and_items()
    items[5] = items[5][:-1]
    with pytest.raises(ValueError, match="item 5 has 8 frames, expected 9"):
        merge_items(items, plan)


def test_item_count_error():
    _, plan, items = _plan_and_items()
    with pytest.raises(ValueError, match=rf"expected {plan.n_items} items \(4 tiles x {plan.n_chunks} chunks\), got {plan.n_items - 1}"):
        merge_items(items[:-1], plan)


def test_inconsistent_size_error():
    _, plan, items = _plan_and_items()
    items[3] = items[3][:, :-1]
    with pytest.raises(ValueError, match="every processed tile must come back at the same size"):
        merge_items(items, plan)


def test_channel_error():
    _, plan, items = _plan_and_items()
    items[2] = items[2][..., :2]
    with pytest.raises(ValueError, match="item 2 has 2 channels but item 0 has 3"):
        merge_items(items, plan)


def test_missing_plan_and_bad_version():
    _, plan, items = _plan_and_items()
    with pytest.raises(ValueError, match="tile_plan is missing"):
        merge_items(items, None)
    with pytest.raises(ValueError, match="version 99 is not supported"):
        merge_items(items, replace(plan, version=99))
