import pytest
import torch

from helpers import random_video
from tiletime.plan import SplitParams, make_plan
from tiletime.split import split_items


def split(video, params):
    plan = make_plan(*video.shape, params)
    return plan, split_items(video, plan)


def test_items_have_identical_shape_and_tile_major_order():
    video = random_video(200, 90, 130)
    plan, items = split(video, SplitParams(rows=2, cols=3, multiple_of=1, chunk_frames=81))
    assert len(items) == plan.n_items == 18
    assert {tuple(i.shape) for i in items} == {(81, plan.tile_h, plan.tile_w, 3)}
    # item 4 = tile (0, 1), chunk 1
    y, x, t = plan.ys[0], plan.xs[1], plan.chunk_starts[1]
    assert torch.equal(items[4], video[t:t + 81, y:y + plan.tile_h, x:x + plan.tile_w])


def test_items_are_views_without_padding():
    video = random_video(10, 64, 64)
    _, items = split(video, SplitParams(multiple_of=1))
    assert all(i.untyped_storage().data_ptr() == video.untyped_storage().data_ptr() for i in items)


def test_padding_replicates_edges():
    video = random_video(70, 50, 40)
    params = SplitParams(rows=1, cols=1, multiple_of=16, chunk_frames=81, frame_rule="4n+1")
    _, (item,) = split(video, params)
    assert item.shape == (73, 64, 48, 3)
    assert torch.equal(item[72], item[69]) and torch.equal(item[69, :50, :40], video[69])  # repeated last frame
    assert torch.equal(item[:70, :50, 47], video[:, :, 39])  # replicated right column
    assert torch.equal(item[:70, 63, :40], video[:, 49, :])  # replicated bottom row


def test_split_rejects_mismatched_images():
    plan = make_plan(5, 10, 10, 3, SplitParams())
    with pytest.raises(ValueError, match="does not match the plan"):
        split_items(torch.zeros(5, 10, 11, 3), plan)
