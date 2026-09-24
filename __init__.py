import logging

try:
    from .tiletime.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except (ImportError, ValueError):
    # ImportError: attempted relative import with no known parent package (pytest)
    # ValueError: attempted relative import beyond top-level package
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .tiletime import routes  # noqa: F401  registers /km_tiletime/* on the ComfyUI server
except Exception as e:  # the nodes still work without the info panel routes
    logging.getLogger("KM-TileTime").warning("KM Tile & Time: info panel routes not registered: %s", e)

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
