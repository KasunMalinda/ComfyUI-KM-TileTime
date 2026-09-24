"""Model presets for KM Tile & Time Split.

One table is the single source of truth: it builds the dropdown labels and is
sent to the browser through the preset combo's input options, where
web/js/tiletime_split.js uses it to fill the widgets.
"""

from dataclasses import dataclass

CUSTOM = "custom"


@dataclass(frozen=True)
class Preset:
    name: str
    multiple_of: int
    frame_rule: str
    width: int
    height: int
    chunk_frames: int
    chunk_overlap: int

    @property
    def label(self):
        return (
            f"{self.name} ({self.multiple_of}, {self.frame_rule}, "
            f"{self.width}x{self.height}, {self.chunk_frames}/{self.chunk_overlap})"
        )

    def widget_values(self):
        """Widget name -> value, in the form the Split widgets hold them."""
        return {
            "tile_mode": "target size",
            "target_width": self.width,
            "target_height": self.height,
            "multiple_of": str(self.multiple_of),
            "chunk_frames": self.chunk_frames,
            "chunk_overlap": self.chunk_overlap,
            "frame_rule": self.frame_rule,
        }


# Locally run video models only. Sources: model cards and ComfyUI's
# nodes_wan.py / nodes_lt.py for Wan and LTX; Comfyui-MMH3-UltimateUpscale
# (nodes/nodes.py) for MiniMax H3. Wan and LTX chunk overlaps are chosen
# defaults, not model requirements.
PRESETS = (
    Preset("Wan 2.1/2.2 VACE", 16, "4k+1", 1280, 720, 81, 8),
    Preset("Wan 2.2 TI2V-5B", 32, "4k+1", 1280, 704, 121, 8),
    Preset("LTX-2 / 2.3 / 2.5", 32, "8k+1", 1280, 704, 121, 16),
    Preset("MiniMax H3", 32, "17k", 1280, 704, 136, 17),
)


def preset_labels():
    return [CUSTOM] + [p.label for p in PRESETS]


def preset_widget_values():
    return {p.label: p.widget_values() for p in PRESETS}
