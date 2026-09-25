import pytest

from tiletime.plan import LEGACY_FRAME_RULES, SplitParams, is_valid_frames
from tiletime.presets import CUSTOM, PRESETS, canonical_preset_label, preset_labels, preset_widget_values


def test_labels_show_values():
    assert preset_labels() == [
        "custom",
        "Wan 2.1/2.2 VACE (16, 4n+1, 1280x720, 81/8)",
        "Wan 2.2 TI2V-5B (32, 4n+1, 1280x704, 121/8)",
        "LTX-2 / 2.3 / 2.5 (32, 8n+1, 1280x704, 121/16)",
        "MiniMax H3 (32, 17n, 1280x704, 136/17)",
    ]


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.name)
def test_preset_values_satisfy_their_own_rules(preset):
    assert preset.width % preset.multiple_of == 0
    assert preset.height % preset.multiple_of == 0
    assert is_valid_frames(preset.chunk_frames, preset.frame_rule)
    assert preset.chunk_overlap < preset.chunk_frames
    SplitParams.from_dict(preset.widget_values())  # accepted by the node


def test_widget_values_keyed_by_label():
    values = preset_widget_values()
    assert CUSTOM not in values
    assert values["MiniMax H3 (32, 17n, 1280x704, 136/17)"] == {
        "tile_mode": "target size",
        "target_width": 1280,
        "target_height": 704,
        "multiple_of": "32",
        "chunk_frames": 136,
        "chunk_overlap": 17,
        "frame_rule": "17n",
    }


# --- legacy preset labels ---------------------------------------------------

@pytest.mark.parametrize("legacy,current", LEGACY_FRAME_RULES.items())
def test_canonical_preset_label_maps_legacy_rule_token(legacy, current):
    old_label = f"Wan 2.1/2.2 VACE (16, {legacy}, 1280x720, 81/8)"
    new_label = f"Wan 2.1/2.2 VACE (16, {current}, 1280x720, 81/8)"
    assert canonical_preset_label(old_label) == new_label


@pytest.mark.parametrize("label", preset_labels() + [CUSTOM])
def test_canonical_preset_label_leaves_current_labels_and_custom_alone(label):
    assert canonical_preset_label(label) == label
