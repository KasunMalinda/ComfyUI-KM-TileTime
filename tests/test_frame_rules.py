import pytest

from tiletime.plan import (
    LEGACY_FRAME_RULES,
    SplitParams,
    canonical_frame_rule,
    is_valid_frames,
    snap_chunk_frames,
    snap_frames_down,
    snap_frames_up,
)


# --- frame rules -----------------------------------------------------------

@pytest.mark.parametrize("rule,valid,invalid", [
    ("none", [1, 2, 80], [0]),
    ("4n+1", [1, 5, 77, 81], [2, 4, 80]),
    ("8n+1", [1, 9, 121], [8, 120]),
    ("17n", [17, 34, 136], [1, 16, 18, 135]),
])
def test_is_valid_frames(rule, valid, invalid):
    assert all(is_valid_frames(n, rule) for n in valid)
    assert not any(is_valid_frames(n, rule) for n in invalid)


@pytest.mark.parametrize("rule,n,down,up", [
    ("4n+1", 80, 77, 81),
    ("4n+1", 70, 69, 73),
    ("8n+1", 120, 113, 121),
    ("17n", 140, 136, 153),
    ("17n", 5, 17, 17),
    ("none", 70, 70, 70),
])
def test_snap(rule, n, down, up):
    assert snap_frames_down(n, rule) == down
    assert snap_frames_up(n, rule) == up


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="unknown frame_rule"):
        is_valid_frames(5, "3k")


def test_snap_chunk_frames_reports_change():
    notes = []
    assert snap_chunk_frames(80, "4n+1", notes) == 77 and notes == ["chunk_frames 80 -> 77 (4n+1)"]
    assert snap_chunk_frames(0, "4n+1", notes) == 0 and len(notes) == 1


# --- params ------------------------------------------------------------------

def test_params_from_dict_casts_combo_strings_and_ignores_unknown_keys():
    p = SplitParams.from_dict({"multiple_of": "32", "overlap": 12, "preset": "x", "rows": 3})
    assert p.multiple_of == 32 and p.overlap == 12.0 and p.rows == 3


@pytest.mark.parametrize("bad", [
    {"tile_mode": "diagonal"},
    {"overlap_mode": "inches"},
    {"multiple_of": 12},
    {"frame_rule": "2k"},
    {"rows": 0},
    {"chunk_frames": -1},
    {"rows": "abc"},
])
def test_params_reject_bad_values(bad):
    with pytest.raises(ValueError):
        SplitParams.from_dict(bad)


# --- legacy frame rule names ------------------------------------------------

@pytest.mark.parametrize("legacy,current", LEGACY_FRAME_RULES.items())
def test_canonical_frame_rule_maps_legacy_names(legacy, current):
    assert canonical_frame_rule(legacy) == current


@pytest.mark.parametrize("name", ["none", "4n+1", "8n+1", "17n", "unknown"])
def test_canonical_frame_rule_leaves_other_names_unchanged(name):
    assert canonical_frame_rule(name) == name


@pytest.mark.parametrize("legacy,current", LEGACY_FRAME_RULES.items())
def test_split_params_from_dict_migrates_legacy_frame_rule(legacy, current):
    assert SplitParams.from_dict({"frame_rule": legacy}).frame_rule == current


def test_split_params_migrates_legacy_frame_rule_on_direct_construction():
    p = SplitParams(frame_rule="4k+1")
    assert p.frame_rule == "4n+1"


def test_node_split_runs_with_a_legacy_frame_rule_value():
    from helpers import random_video
    from tiletime.plan import make_plan
    from tiletime.split import split_items

    params = SplitParams.from_dict({"rows": 1, "cols": 1, "multiple_of": 16, "chunk_frames": 81, "frame_rule": "4k+1"})
    video = random_video(70, 64, 64)
    plan = make_plan(*video.shape, params)
    items = split_items(video, plan)
    assert plan.frame_rule == "4n+1"
    assert len(items) == plan.n_items
