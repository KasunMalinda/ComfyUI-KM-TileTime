"""ComfyUI node classes for KM Tile & Time (widgets, list flags, UI payload)."""

from dataclasses import replace

from .color import item_stats
from .merge import BLENDS, COLOR_MATCHES, CURVES, OUTPUT_SIZES, merge_items, output_layout, piece_size
from .overlay import draw_overlay, overlay_connected, placeholder
from .plan import (
    FRAME_RULES,
    MULTIPLES,
    OVERLAP_MODES,
    TILE_MODES,
    SplitParams,
    canonical_frame_rule,
    describe,
    make_plan,
)
from .presets import CUSTOM, canonical_preset_label, preset_labels, preset_widget_values
from .split import normalize_mask, ones_mask_items, split_items, split_mask_items

CATEGORY = "KM/TileTime"


class KMTileTimeSplit:
    """Cut a video into overlapping tiles and frame chunks, output as one list."""

    CATEGORY = CATEGORY
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE", "MASK", "KM_TILE_PLAN", "STRING")
    RETURN_NAMES = ("image_tiles", "mask_tiles", "tile_plan", "info")
    OUTPUT_IS_LIST = (True, True, False, False)
    # Output node so the info panel's Measure button can target it with a
    # partial run (only Split and the nodes feeding it execute).
    OUTPUT_NODE = True
    SEARCH_ALIASES = ["tile", "tiles", "split", "chunk", "video tile", "tiled upscale", "km"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "preset": (preset_labels(), {
                    "default": CUSTOM,
                    "km_presets": preset_widget_values(),
                    "tooltip": "Fills the widgets below for a model. Values stay editable.",
                }),
                "tile_mode": (list(TILE_MODES), {"default": "grid"}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16}),
                "target_width": ("INT", {"default": 1280, "min": 64, "max": 8192}),
                "target_height": ("INT", {"default": 720, "min": 64, "max": 8192}),
                "overlap_mode": (list(OVERLAP_MODES), {"default": "percent"}),
                "overlap": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 4096.0, "step": 1.0}),
                "multiple_of": ([str(m) for m in MULTIPLES], {"default": "16"}),
                "chunk_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10000,
                    "tooltip": "Frames per chunk. 0 turns temporal chunking off.",
                }),
                "chunk_overlap": ("INT", {"default": 8, "min": 0, "max": 1000}),
                "frame_rule": (list(FRAME_RULES), {"default": "none"}),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Cut with the same plan as images. Without a mask, mask_tiles are all white.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, preset, frame_rule):
        # Naming these two inputs makes ComfyUI skip its own combo check for
        # them and call this instead, so workflows saved with the old frame
        # rule names (4k+1, 8k+1, 17k) still validate.
        if canonical_frame_rule(frame_rule) not in FRAME_RULES:
            return f"frame_rule has an invalid value: {frame_rule!r}, expected one of: {', '.join(FRAME_RULES)}"
        if canonical_preset_label(preset) not in preset_labels():
            return f"preset has an invalid value: {preset!r}, expected one of: {', '.join(preset_labels())}"
        return True

    def split(self, images, preset, mask=None, **widgets):
        params = SplitParams.from_dict(widgets)
        n, h, w, c = (int(v) for v in images.shape)
        plan = make_plan(n, h, w, c, params)
        info = describe(plan)
        items = split_items(images, plan)
        if mask is None:
            mask_items = ones_mask_items(plan)
        else:
            mask_items = split_mask_items(normalize_mask(mask, n, h, w), plan)
        plan = replace(plan, src_stats=item_stats(items))
        return {
            "ui": {"km_source": [[n, h, w, c]], "km_info": [info]},
            "result": (items, mask_items, plan, info),
        }


class KMTileTimeMerge:
    """Blend processed tiles and chunks back into one video."""

    CATEGORY = CATEGORY
    FUNCTION = "merge"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "overlay")
    INPUT_IS_LIST = True
    SEARCH_ALIASES = ["merge tiles", "stitch", "untile", "km"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_plan": ("KM_TILE_PLAN",),
                "blend": (list(BLENDS), {"default": "feather"}),
                "feather_curve": (list(CURVES), {"default": "smoothstep"}),
                "feather_width": ("INT", {
                    "default": 100, "min": 0, "max": 100,
                    "tooltip": "Percent of each overlap covered by the blend ramp.",
                }),
                "output_size": (list(OUTPUT_SIZES), {"default": "scaled"}),
                "color_match": (list(COLOR_MATCHES), {
                    "default": "off",
                    "tooltip": "neighbours: make tiles agree in their overlaps (keeps the model's look). "
                               "source: return each tile to the source colors (upscaling).",
                }),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    def merge(self, tiles, tile_plan, blend, feather_curve, feather_width, output_size, color_match,
              prompt=None, unique_id=None):
        plan = tile_plan[0] if tile_plan else None
        image = merge_items(
            tiles, plan, blend[0], feather_curve[0], feather_width[0], output_size[0], color_match[0]
        )
        # The overlay is drawn only when something uses it, so it costs nothing otherwise
        # and can never end up on the main output.
        if overlay_connected(prompt[0] if prompt else None, unique_id[0] if unique_id else None):
            lay = output_layout(plan, *piece_size(tiles, plan, output_size[0]))
            overlay = draw_overlay(image, plan, lay)
        else:
            overlay = placeholder()
        return (image, overlay)


NODE_CLASS_MAPPINGS = {
    "KMTileTimeSplit": KMTileTimeSplit,
    "KMTileTimeMerge": KMTileTimeMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KMTileTimeSplit": "KM Tile & Time Split",
    "KMTileTimeMerge": "KM Tile & Time Merge",
}
