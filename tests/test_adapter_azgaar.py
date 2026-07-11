"""Azgaar FMG → terrain-bundle adapter tests (offline, synthetic exports)."""

import numpy as np

from toposlicr.adapters.azgaar import azgaar_to_bundle
from toposlicr.bundle import load_bundle


def _square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[
        [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _cells(heights):
    """A 2x2 grid of unit cells over [0,2]x[0,2] with the given heights.

    Cell layout (x, y with y up)::
        (0,1) (1,1)   ← heights[2], heights[3]
        (0,0) (1,0)   ← heights[0], heights[1]
    """
    boxes = [(0, 0, 1, 1), (1, 0, 2, 1), (0, 1, 1, 2), (1, 1, 2, 2)]
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": i, "height": h},
         "geometry": _square(*b)}
        for i, (b, h) in enumerate(zip(boxes, heights, strict=True))]}


def test_heightmap_reflects_cell_heights(tmp_path):
    # bottom-left low (water), top-right high land.
    cells = _cells([5, 40, 40, 90])
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60)
    b = load_bundle(out)
    elev = b.elevation()
    rows, cols = elev.shape
    # top-right quadrant (row 0.. , col high) is the height-90 cell; it must be
    # higher than the bottom-left (height-5) quadrant.
    top_right = elev[: rows // 2, cols // 2:].mean()
    bottom_left = elev[rows // 2:, : cols // 2].mean()
    assert top_right > bottom_left


def test_water_mask_marks_low_cells(tmp_path):
    cells = _cells([5, 40, 40, 90])  # one cell (height 5) below threshold 20
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60, water_threshold=20)
    b = load_bundle(out)
    assert b.water is not None
    assert b.water.any()
    # Roughly a quarter of the grid is water (the one low cell).
    frac = b.water.mean()
    assert 0.1 < frac < 0.4


def test_water_threshold_respected(tmp_path):
    cells = _cells([5, 40, 40, 90])
    b_low = load_bundle(azgaar_to_bundle(cells, tmp_path / "a", cells_across=60,
                                         water_threshold=10))
    b_high = load_bundle(azgaar_to_bundle(cells, tmp_path / "b", cells_across=60,
                                          water_threshold=50))
    low_water = 0 if b_low.water is None else int(b_low.water.sum())
    high_water = 0 if b_high.water is None else int(b_high.water.sum())
    assert high_water > low_water  # higher threshold floods more cells


def test_burg_coordinate_consistency(tmp_path):
    # Burg placed inside the top-right (height 90) cell at world (1.5, 1.5).
    cells = _cells([5, 40, 40, 90])
    burgs = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Highcastle", "population": 30},
         "geometry": {"type": "Point", "coordinates": [1.5, 1.5]}}]}
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60, burgs=burgs)
    feats = list(load_bundle(out).features())
    burg = next(f for f in feats if f.name == "Highcastle")
    # Feature x,y share the cells' coordinate frame → land inside that cell.
    assert 1.0 <= burg.geometry.x <= 2.0
    assert 1.0 <= burg.geometry.y <= 2.0
    assert burg.tags.get("icon") in {"city", "tower"}


def test_burgs_from_csv_rows(tmp_path):
    cells = _cells([30, 40, 50, 90])
    burgs = [{"Burg": "Rowton", "Longitude": 0.5, "Latitude": 0.5, "Population": 5}]
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=40, burgs=burgs)
    feats = list(load_bundle(out).features())
    assert any(f.name == "Rowton" for f in feats)


def test_rivers_become_named_points(tmp_path):
    cells = _cells([30, 40, 50, 90])
    rivers = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Silverflow"},
         "geometry": {"type": "LineString", "coordinates": [[0.2, 0.2], [1.8, 1.8]]}}]}
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=40, rivers_geojson=rivers)
    feats = list(load_bundle(out).features())
    river = next(f for f in feats if f.name == "Silverflow")
    # midpoint of the diagonal line
    assert abs(river.geometry.x - 1.0) < 0.2 and abs(river.geometry.y - 1.0) < 0.2


def test_bundle_runs_through_core(tmp_path):
    from toposlicr.config import parse_config
    from toposlicr.layers import build_layer_model

    cells = _cells([5, 40, 60, 95])
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=120)
    bundle = load_bundle(out)
    cfg = parse_config({
        "region": {"bundle": str(out)},
        "physical": {"model_width_mm": 300, "normalize_layers": 6},
    })
    model = build_layer_model(bundle.to_dem(), cfg)
    assert model.flat is True
    assert model.layer_count >= 2
    for k in range(1, model.layer_count):
        assert model.footprint(k).area <= model.footprint(k - 1).area + 1e-6


def test_meta_maps_pixels_back_to_height(tmp_path):
    cells = _cells([10, 30, 60, 90])
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=80)
    b = load_bundle(out)
    lo, hi = float(b.elevation().min()), float(b.elevation().max())
    # heights span ~10..90 (rasterised, minus water-fill zeros at edges)
    assert hi <= 90 + 1e-3
    assert np.isfinite(lo) and np.isfinite(hi) and hi > lo


def test_burg_with_bad_coordinates_is_skipped_not_crashed(tmp_path):
    cells = _cells([5, 40, 60, 95])
    burgs = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Bad"},
         "geometry": {"type": "Point", "coordinates": [1.0]}},        # short coords
        {"type": "Feature", "properties": {"name": "Good", "population": 5},
         "geometry": {"type": "Point", "coordinates": [1.5, 1.5]}},
    ]}
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60, burgs=burgs)
    names = {f.name for f in load_bundle(out).features()}
    assert "Good" in names and "Bad" not in names


def test_all_water_export_not_normalized_to_zero(tmp_path):
    # every cell below the water threshold → water.png must be all-set, not lost
    cells = _cells([5, 8, 10, 15])
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60, water_threshold=20)
    b = load_bundle(out)
    assert b.water is not None and b.water.any()


def test_geometryless_feature_is_skipped(tmp_path):
    cells = _cells([5, 40, 60, 95])
    cells["features"].append({"type": "Feature", "properties": {"height": 50}})  # no geometry
    out = azgaar_to_bundle(cells, tmp_path / "w", cells_across=60)  # must not crash
    assert load_bundle(out).shape[1] >= 1
