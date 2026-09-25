import torch

from helpers import random_video
from tiletime.nodes import KMTileTimeMerge, KMTileTimeSplit


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
