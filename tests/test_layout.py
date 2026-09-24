from tiletime.plan import even_starts, grid_axis, target_axis


# --- spatial axes --------------------------------------------------------------

def test_even_starts_first_at_zero_last_at_end():
    assert even_starts(4096, 1472, 3) == (0, 1312, 2624)
    assert even_starts(100, 100, 1) == (0,)


def test_grid_solves_tile_for_requested_overlap():
    notes = []
    ax = grid_axis(4096, 3, "percent", 10, 16, "width", notes)
    assert ax.tile == 1472 and ax.starts == (0, 1312, 2624) and ax.pad == 0
    overlap = ax.starts[0] + ax.tile - ax.starts[1]
    assert overlap >= 0.10 * ax.tile  # at least the requested 10% of the tile
    assert notes == []


def test_grid_single_tile_pads_to_multiple():
    ax = grid_axis(1000, 1, "percent", 10, 16, "width", [])
    assert (ax.tile, ax.starts, ax.pad) == (1008, (0,), 8)


def test_grid_percent_capped():
    notes = []
    grid_axis(1000, 2, "percent", 80, 1, "width", notes)
    assert notes == ["overlap capped at 49%"]


def test_grid_pixel_overlap_capped_below_half_tile():
    notes = []
    ax = grid_axis(1000, 2, "pixels", 900, 1, "width", notes)
    overlap = 2 * ax.tile - 1000
    assert overlap < ax.tile / 2
    assert notes == ["width overlap capped at 324 px"]


def test_target_rounds_down_to_multiple():
    ax = target_axis(4000, 1290, "percent", 10, 16, "width", [])
    assert ax.tile == 1280


def test_target_small_axis_shrinks_instead_of_padding_to_target():
    ax = target_axis(1000, 1280, "percent", 10, 16, "width", [])
    assert (ax.tile, ax.starts, ax.pad) == (1008, (0,), 8)


def test_target_count_keeps_overlap_at_least_requested():
    ax = target_axis(1920, 1280, "percent", 10, 16, "width", [])
    assert ax.tile == 1280 and ax.starts == (0, 640)
    ax = target_axis(4096, 1280, "pixels", 128, 16, "width", [])
    overlaps = [ax.starts[i] + ax.tile - ax.starts[i + 1] for i in range(len(ax.starts) - 1)]
    assert min(overlaps) >= 128 and ax.starts[-1] + ax.tile == 4096


def test_tile_larger_than_source_collapses_to_one():
    notes = []
    ax = grid_axis(10, 2, "percent", 10, 16, "width", notes)
    assert (ax.tile, ax.starts, ax.pad) == (16, (0,), 6)
    assert "one tile covers the source" in notes[0]
