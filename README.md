# ComfyUI-KM-TileTime

Two ComfyUI nodes that cut a video into overlapping **tiles** and overlapping **frame chunks**, run every piece through one processing branch, and stitch the results back seamlessly. Use it to fit video models' resolution and frame limits (for example a 4096x2880 plate through Wan or LTX) and to keep VRAM low, since only one piece is on the GPU at a time.

## Nodes

**KM Tile & Time Split** (`KM/TileTime`)

- Inputs `images` (IMAGE, a video or a single image) and optional `mask` (MASK, same size as the images, one frame per image frame or a single frame).
- Outputs, in order:
  - `image_tiles`: one list; ComfyUI runs the branch once per piece.
  - `mask_tiles`: the mask cut exactly like the images, so `mask_tiles[i]` lines up with `image_tiles[i]` (for example for a VACE mask). All white when no mask is connected.
  - `tile_plan`: connect to Merge.
  - `info`: text.
- `preset`: fills the widgets for a model. The label shows its values: size multiple, frame rule, tile size, chunk frames / chunk overlap. Editing a filled value switches it to `custom`.
- `tile_mode`: `grid` (rows x cols) or `target size` (the largest tile the model takes; the tile count follows the source).
- `overlap_mode` / `overlap`: percent of the tile or pixels, capped below half a tile.
- `multiple_of`: tile width and height are rounded to this.
- `chunk_frames` (0 = off) / `chunk_overlap`: temporal chunks. Every chunk has the same length; the last one moves back in time rather than being padded.
- `frame_rule`: `4n+1` (Wan), `8n+1` (LTX), `17n` (MiniMax H3). Chunk length is rounded down to a valid count; a clip that fits in one chunk is padded with its last frame and Merge removes the padding.
- **Info panel** at the bottom shows source size, grid, tile size, chunks, item count and output size, and updates as you change widgets. Press **Measure source** to read the real input: it runs only the nodes that feed Split, from any loader.

**KM Tile & Time Merge** (`KM/TileTime`)

- Inputs `tiles` (the processed list) and `tile_plan`.
- Outputs `images` (the merged video) and `overlay` (see below).
- `blend`: `feather` (`linear` or `smoothstep`, `feather_width` percent of each overlap) or `hard cut`.
- `output_size`: `scaled` keeps the model's scale (for example 2x) or `source` returns the original size.
- `color_match`: removes per-tile color drift before blending.
  - `off` (default): no correction.
  - `neighbours`: makes tiles agree in their overlaps while keeping the model's overall look. Use it for restyles and anything that changes color on purpose.
  - `source`: returns each tile to the source's colors and contrast, keeping the detail the model added. Use it for upscaling and denoising.
- Any returned tile size works (2x, 1.5x...), as long as every piece comes back at the same size. In `scaled` mode tiles are placed without resampling, so their pixels reach the output unchanged apart from blending (and color match, when on).
- `overlay`: a debug view of the merged video with each tile outlined in its own color, tile numbers, lightened overlap bands and a chunk label per frame. It is only drawn when something is connected to it, including when you connect it after a previous run (the UI tracks the connection), and it never touches the main `images` output.

## Presets

| Preset | Multiple | Frame rule | Tile | Chunk / overlap |
|---|---|---|---|---|
| Wan 2.1/2.2 VACE | 16 | 4n+1 | 1280x720 | 81 / 8 |
| Wan 2.2 TI2V-5B | 32 | 4n+1 | 1280x704 | 121 / 8 |
| LTX-2 / 2.3 / 2.5 | 32 | 8n+1 | 1280x704 | 121 / 16 |
| MiniMax H3 | 32 | 17n | 1280x704 | 136 / 17 |

## Typical workflow

Load video -> **KM Tile & Time Split** -> your per-piece branch (upscale, VACE restyle, color...) -> **KM Tile & Time Merge** -> save. Use core **Get Image Size** on the `image_tiles` output if the branch needs the tile size or frame count.

## Notes

- Everything stays in system RAM as tensors: a long 4K clip needs a lot of RAM.
- Tiles, chunks and item order: tiles are numbered row by row from the top-left, and items are tile-major (all chunks of tile 0, then tile 1...).
- Workflows saved with version 1 of Split need their Split to Merge links reconnected: Split's outputs gained `mask_tiles` in second place.

## License

MIT
