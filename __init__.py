import logging

WEB_DIRECTORY = "./web/js"

# ComfyUI imports this folder as a package. pytest also imports this file, because
# the repo root holds __init__.py, but with no parent package; the tests import
# tiletime directly, so the node registration below is skipped there.
if __package__:
    from .tiletime.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:
        from .tiletime import routes  # noqa: F401  registers /km_tiletime/* on the ComfyUI server
    except Exception as e:  # the nodes still work without the info panel routes
        logging.getLogger("KM-TileTime").warning("KM Tile & Time: info panel routes not registered: %s", e)

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
