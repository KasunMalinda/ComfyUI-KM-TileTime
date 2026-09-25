"""HTTP routes for the Split node's info panel (registered on ComfyUI's server)."""

import inspect

from aiohttp import web
from server import PromptServer

from .preview import preview_text

routes = PromptServer.instance.routes


@routes.post("/km_tiletime/preview")
async def km_tiletime_preview(request):
    try:
        data = await request.json()
        text = preview_text(data.get("shape"), data.get("params") or {})
    except (ValueError, TypeError, AttributeError) as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"text": text})


@routes.get("/km_tiletime/caps")
async def km_tiletime_caps(request):
    """Tell the panel whether this ComfyUI can run only part of a workflow."""
    try:
        import execution

        partial = "partial_execution_list" in inspect.signature(execution.validate_prompt).parameters
    except Exception:
        partial = False
    return web.json_response({"partial": partial})
