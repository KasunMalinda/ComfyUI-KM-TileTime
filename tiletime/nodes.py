"""ComfyUI node classes for KM Tile & Time (widgets, list flags, UI payload)."""

from .merge import BLENDS, CURVES, OUTPUT_SIZES, merge_items
from .plan import FRAME_RULES, MULTIPLES, OVERLAP_MODES, TILE_MODES, SplitParams, describe, make_plan
from .presets import CUSTOM, preset_labels, preset_widget_values
from .split import split_items

CATEGORY = "KM/TileTime"


class KMTileTimeSplit:
    """Cut a video into overlapping tiles and frame chunks, output as one list."""

    CATEGORY = CATEGORY
    FUNCTION = "split"
    RETURN_TYPES = ("IMAGE", "KM_TILE_PLAN", "STRING")
    RETURN_NAMES = ("tiles", "tile_plan", "info")
    OUTPUT_IS_LIST = (True, False, False)
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
            }
        }

    def split(self, images, preset, **widgets):
        params = SplitParams.from_dict(widgets)
        n, h, w, c = (int(v) for v in images.shape)
        plan = make_plan(n, h, w, c, params)
        info = describe(plan)
        items = split_items(images, plan)
        return {
            "ui": {"km_source": [[n, h, w, c]], "km_info": [info]},
            "result": (items, plan, info),
        }


class KMTileTimeMerge:
    """Blend processed tiles and chunks back into one video."""

    CATEGORY = CATEGORY
    FUNCTION = "merge"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
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
            }
        }

    def merge(self, tiles, tile_plan, blend, feather_curve, feather_width, output_size):
        plan = tile_plan[0] if tile_plan else None
        image = merge_items(tiles, plan, blend[0], feather_curve[0], feather_width[0], output_size[0])
        return (image,)


NODE_CLASS_MAPPINGS = {
    "KMTileTimeSplit": KMTileTimeSplit,
    "KMTileTimeMerge": KMTileTimeMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KMTileTimeSplit": "KM Tile & Time Split",
    "KMTileTimeMerge": "KM Tile & Time Merge",
}
