# ComfyUI-KM-TileTime

Two ComfyUI nodes that cut a video into overlapping **tiles** and overlapping **frame chunks**, run every piece through one processing branch, and stitch the results back seamlessly. Use it to fit video models' resolution and frame limits (for example a 4096x2880 plate through Wan or LTX) and to keep VRAM low, since only one piece is on the GPU at a time.

## Nodes

**KM Tile & Time Split** (`KM/TileTime`)

- Input `images` (IMAGE, a video or a single image).
- Outputs `tiles` (one list: ComfyUI runs the branch once per piece), `tile_plan` (connect to Merge) and `info` (text).
- `preset`: fills the widgets for a model. The label shows its values: size multiple, frame rule, tile size, chunk frames / chunk overlap. Editing a filled value switches it to `custom`.
- `tile_mode`: `grid` (rows x cols) or `target size` (the largest tile the model takes; the tile count follows the source).
- `overlap_mode` / `overlap`: percent of the tile or pixels, capped below half a tile.
- `multiple_of`: tile width and height are rounded to this.
- `chunk_frames` (0 = off) / `chunk_overlap`: temporal chunks. Every chunk has the same length; the last one moves back in time rather than being padded.
- `frame_rule`: `4k+1` (Wan), `8k+1` (LTX), `17k` (MiniMax H3). Chunk length is rounded down to a valid count; a clip that fits in one chunk is padded with its last frame and Merge removes the padding.
- **Info panel** at the bottom shows source size, grid, tile size, chunks, item count and output size, and updates as you change widgets. Press **Measure source** to read the real input: it runs only the nodes that feed Split, from any loader.

**KM Tile & Time Merge** (`KM/TileTime`)

- Inputs `tiles` (the processed list) and `tile_plan`.
- `blend`: `feather` (`linear` or `smoothstep`, `feather_width` percent of each overlap) or `hard cut`.
- `output_size`: `scaled` keeps the model's scale (for example 2x) or `source` returns the original size.
- Any returned tile size works (2x, 1.5x...), as long as every piece comes back at the same size. In `scaled` mode tiles are placed without resampling, so their pixels reach the output unchanged apart from blending in the overlaps.

## Presets

| Preset | Multiple | Frame rule | Tile | Chunk / overlap |
|---|---|---|---|---|
| Wan 2.1/2.2 VACE | 16 | 4k+1 | 1280x720 | 81 / 8 |
| Wan 2.2 TI2V-5B | 32 | 4k+1 | 1280x704 | 121 / 8 |
| LTX-2 / 2.3 / 2.5 | 32 | 8k+1 | 1280x704 | 121 / 16 |
| MiniMax H3 | 32 | 17k | 1280x704 | 136 / 17 |

## Typical workflow

Load video -> **KM Tile & Time Split** -> your per-piece branch (upscale, VACE restyle, color...) -> **KM Tile & Time Merge** -> save. Use core **Get Image Size** on the `tiles` output if the branch needs the tile size or frame count.

## Notes

- Everything stays in system RAM as tensors: a long 4K clip needs a lot of RAM.
- Tiles, chunks and item order: tiles are numbered row by row from the top-left, and items are tile-major (all chunks of tile 0, then tile 1...).

## License

MIT
