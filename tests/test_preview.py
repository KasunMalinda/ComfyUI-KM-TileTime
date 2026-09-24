import pytest

from tiletime.plan import SplitParams, describe, make_plan
from tiletime.preview import preview_text


def test_preview_matches_split_info():
    widgets = {"tile_mode": "grid", "rows": 3, "cols": 3, "multiple_of": "16", "chunk_frames": 81, "frame_rule": "4k+1"}
    expected = describe(make_plan(245, 2880, 4096, 3, SplitParams.from_dict(widgets)))
    assert preview_text([245, 2880, 4096, 3], widgets) == expected


def test_preview_unmeasured_target_mode():
    text = preview_text(None, {"tile_mode": "target size", "target_width": 1290, "target_height": 720,
                               "multiple_of": "16", "chunk_frames": 80, "frame_rule": "4k+1"})
    assert text.splitlines() == [
        "Source not measured: press Measure to read the source",
        "Tiles up to 1280x720 (fewer or smaller if the source is smaller)",
        "Chunks of 77f (overlap 8), frame rule 4k+1",
        "Adjustments: chunk_frames 80 -> 77 (4k+1)",
    ]


def test_preview_unmeasured_grid_chunking_off():
    text = preview_text(None, {"rows": 2, "cols": 3})
    assert text.splitlines() == [
        "Source not measured: press Measure to read the source",
        "Grid 3 cols x 2 rows = 6 tiles",
        "Chunks off (whole clip), frame rule none",
    ]


@pytest.mark.parametrize("shape", [[1, 2, 3], "abc", [1, "x", 3, 4]])
def test_preview_rejects_bad_shape(shape):
    with pytest.raises(ValueError, match="shape must be"):
        preview_text(shape, {})
