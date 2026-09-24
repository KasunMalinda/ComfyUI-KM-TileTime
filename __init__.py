import logging

# Only set up the ComfyUI integration when imported as a package (not during pytest)
if __package__:
    from .tiletime.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    try:
        from .tiletime import routes  # noqa: F401  registers /km_tiletime/* on the ComfyUI server
    except Exception as e:  # the nodes still work without the info panel routes
        logging.getLogger("KM-TileTime").warning("KM Tile & Time: info panel routes not registered: %s", e)

    WEB_DIRECTORY = "./web/js"

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
else:
    # When imported directly (e.g., by pytest), provide minimal exports
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = "./web/js"
    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
