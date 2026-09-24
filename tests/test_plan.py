import pytest

from tiletime.plan import SplitParams, chunk_axis, describe, is_valid_frames, make_plan


# --- chunks ------------------------------------------------------------------

def test_chunks_200_frames_wan():
    ch = chunk_axis(200, 81, 8, "4k+1", [])
    assert (ch.length, ch.starts, ch.pad) == (81, (0, 60, 119), 0)


def test_chunk_frames_snapped_down_and_reported():
    notes = []
    ch = chunk_axis(300, 80, 8, "4k+1", notes)
    assert ch.length == 77 and notes == ["chunk_frames 80 -> 77 (4k+1)"]


def test_one_chunk_padded_to_rule():
    notes = []
    ch = chunk_axis(70, 81, 8, "4k+1", notes)
    assert (ch.length, ch.starts, ch.pad) == (73, (0,), 3)
    assert "clip padded 70 -> 73" in notes[0]


def test_chunking_off_without_rule_is_whole_clip():
    ch = chunk_axis(70, 0, 8, "none", [])
    assert (ch.length, ch.starts, ch.pad) == (70, (0,), 0)


def test_chunk_overlap_larger_than_chunk_is_clamped():
    ch = chunk_axis(100, 10, 50, "none", [])
    assert ch.length == 10 and ch.starts[0] == 0 and ch.starts[-1] == 90


# --- plan --------------------------------------------------------------------

@pytest.mark.parametrize("params", [
    SplitParams(),
    SplitParams(rows=16, cols=16, multiple_of=8),
    SplitParams(tile_mode="target size", target_width=640, target_height=360, overlap_mode="pixels", overlap=48),
    SplitParams(tile_mode="target size", chunk_frames=33, chunk_overlap=4, frame_rule="8k+1"),
])
@pytest.mark.parametrize("shape", [(1, 64, 64, 3), (50, 1080, 1920, 3), (245, 2880, 4096, 4), (7, 100, 37, 3)])
def test_plan_covers_source(params, shape):
    n, h, w, c = shape
    plan = make_plan(n, h, w, c, params)
    assert plan.xs[0] == 0 and plan.xs[-1] + plan.tile_w == w + plan.pad_w
    assert plan.ys[0] == 0 and plan.ys[-1] + plan.tile_h == h + plan.pad_h
    assert plan.chunk_starts[0] == 0
    assert plan.chunk_starts[-1] + plan.chunk_len == n + plan.pad_t
    assert plan.tile_w % params.multiple_of == 0 and plan.tile_h % params.multiple_of == 0
    for starts, tile in ((plan.xs, plan.tile_w), (plan.ys, plan.tile_h), (plan.chunk_starts, plan.chunk_len)):
        assert all(starts[i + 1] - starts[i] <= tile for i in range(len(starts) - 1)), "gap between pieces"
    assert is_valid_frames(plan.chunk_len, params.frame_rule)


def test_item_coords_tile_major():
    plan = make_plan(200, 100, 100, 3, SplitParams(rows=2, cols=3, multiple_of=1, chunk_frames=81))
    assert plan.n_chunks == 3 and plan.n_items == 18
    assert plan.item_coords(0) == (0, 0, 0)
    assert plan.item_coords(2) == (0, 0, 2)
    assert plan.item_coords(3) == (0, 1, 0)
    assert plan.item_coords(9) == (1, 0, 0)


def test_make_plan_rejects_empty_source():
    with pytest.raises(ValueError, match="at least 1 frame"):
        make_plan(0, 10, 10, 3, SplitParams())


@pytest.mark.parametrize("field,value", [
    ("rows", 17),
    ("cols", 17),
    ("target_width", 8193),
    ("target_height", 8193),
    ("chunk_frames", 10001),
    ("chunk_overlap", 1001),
    ("overlap", 4097),
])
def test_split_params_rejects_values_above_widget_maxima(field, value):
    with pytest.raises(ValueError, match=field):
        SplitParams(**{field: value})


def test_describe_lists_layout_and_adjustments():
    params = SplitParams(tile_mode="target size", target_width=1280, target_height=720,
                         chunk_frames=80, chunk_overlap=8, frame_rule="4k+1")
    text = describe(make_plan(200, 1080, 1920, 3, params))
    assert text.splitlines() == [
        "Source 1920x1080, 200f, 3ch",
        "Grid 2 cols x 2 rows = 4 tiles, 1280x720 each (overlap 640/360 px)",
        "Chunks 3 x 77f (overlap 15-16), frame rule 4k+1",
        "Items 12 through the branch",
        "Output 1920x1080 at 1x (x2 model: 3840x2160)",
        "Adjustments: chunk_frames 80 -> 77 (4k+1)",
    ]
