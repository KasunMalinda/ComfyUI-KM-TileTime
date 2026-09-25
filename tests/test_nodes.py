import pytest
import torch

from helpers import random_video
from tiletime.nodes import KMTileTimeMerge, KMTileTimeSplit
from tiletime.plan import LEGACY_FRAME_RULES
from tiletime.presets import CUSTOM, preset_labels


def test_nodes_round_trip_and_ui_payload():
    video = random_video(30, 40, 60)
    split = KMTileTimeSplit().split(
        video, preset="custom", tile_mode="grid", rows=2, cols=2, target_width=1280, target_height=720,
        overlap_mode="percent", overlap=10.0, multiple_of="8", chunk_frames=17, chunk_overlap=4, frame_rule="8n+1",
    )
    items, plan, info = split["result"]
    assert split["ui"] == {"km_source": [[30, 40, 60, 3]], "km_info": [info]}
    (out,) = KMTileTimeMerge().merge(items, [plan], ["feather"], ["smoothstep"], [100], ["scaled"])
    assert torch.allclose(out, video, atol=1e-5)


def test_split_input_types_publish_presets():
    spec = KMTileTimeSplit.INPUT_TYPES()["required"]
    labels, opts = spec["preset"]
    assert labels[0] == "custom" and set(opts["km_presets"]) == set(labels[1:])
    assert KMTileTimeSplit.OUTPUT_NODE is True


# --- VALIDATE_INPUTS (API-format prompts skip the built-in combo check) ----

def test_validate_inputs_accepts_current_values():
    assert KMTileTimeSplit.VALIDATE_INPUTS(preset=CUSTOM, frame_rule="4n+1") is True


@pytest.mark.parametrize("legacy", LEGACY_FRAME_RULES)
def test_validate_inputs_accepts_legacy_frame_rules(legacy):
    assert KMTileTimeSplit.VALIDATE_INPUTS(preset=CUSTOM, frame_rule=legacy) is True


@pytest.mark.parametrize("legacy,current", LEGACY_FRAME_RULES.items())
def test_validate_inputs_accepts_old_preset_labels(legacy, current):
    old_label = next(label for label in preset_labels() if f", {current}," in label).replace(
        f", {current},", f", {legacy},"
    )
    assert KMTileTimeSplit.VALIDATE_INPUTS(preset=old_label, frame_rule="none") is True


def test_validate_inputs_rejects_unknown_frame_rule():
    result = KMTileTimeSplit.VALIDATE_INPUTS(preset=CUSTOM, frame_rule="3k")
    assert isinstance(result, str) and "3k" in result


def test_validate_inputs_rejects_unknown_preset():
    result = KMTileTimeSplit.VALIDATE_INPUTS(preset="not a real preset", frame_rule="none")
    assert isinstance(result, str) and "not a real preset" in result
