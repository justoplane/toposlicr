"""Regression tests locking in the 15 fixes from the adversarial code review.

Each test maps to one confirmed finding. Everything runs offline (synthetic DEM
or hand-built geometry); no network.
"""

import numpy as np
import pytest
from affine import Affine
from shapely.geometry import MultiPolygon, Point, Polygon

from toposlicr.config import ConfigError, parse_config
from toposlicr.contour import band_polygon, smooth_dem
from toposlicr.coupon import offset_series
from toposlicr.dem.base import DemRaster
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.features.overpass import OverpassError, check_overpass_error
from toposlicr.features.schema import FeatureCollection
from toposlicr.geo import BBox
from toposlicr.labels import (
    LabelReport,
    PlacedLabel,
    read_label_overrides,
    text_to_polygons,
    write_labels_file,
)
from toposlicr.layers import build_layer_model
from toposlicr.nest import add_part_id_score, nest_parts
from toposlicr.parts import OP_ENGRAVE, Part, build_parts_from_model
from toposlicr.scale import _layer_count, solve_scale
from toposlicr.symbology import _assign_lake_layer, build_symbology

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _cfg(**over):
    phys = {"model_width_mm": 300, "ply_thickness_mm": 3.0}
    phys_over = over.pop("physical", {})
    # Only default to interval_m when the caller didn't pin a scale driver.
    if not any(k in phys_over for k in ("exaggeration", "interval_m", "layer_count")):
        phys["interval_m"] = 250
    phys.update(phys_over)
    data = {
        "region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                   "dem": "synthetic"},
        "physical": phys,
        "machine": {"kerf_mm": 0.15},
        "materials": {"default": "birch_3mm", "water": "blue_acrylic_3mm"},
        "symbology": {"lakes": {"mode": over.pop("lake_mode", "score"),
                                "min_area_km2": 0.0}},
    }
    for k, v in over.items():
        data[k] = v
    return parse_config(data)


def _model(cfg):
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    return build_layer_model(dem, cfg, BBOX)


def _square(cx, cy, half):
    return Polygon([(cx - half, cy - half), (cx + half, cy - half),
                    (cx + half, cy + half), (cx - half, cy + half)])


# --- 1: smooth_dem preserves NaN --------------------------------------------

def test_smooth_dem_preserves_nan():
    arr = np.full((8, 8), 100.0)
    arr[0, :] = np.nan
    arr[-1, :] = np.nan
    arr[:, 0] = np.nan
    arr[:, -1] = np.nan
    # add interior relief so smoothing is meaningful
    arr[3:5, 3:5] = 140.0
    out = smooth_dem(arr, sigma_px=1.5)
    assert np.isnan(out).any()
    assert np.isnan(out[0, 0])          # originally-NaN border stays NaN
    assert np.isnan(out[0, 4])
    assert np.isfinite(out[4, 4])       # interior stays finite
    # sigma <= 0 returns NaN untouched
    out0 = smooth_dem(arr, sigma_px=0.0)
    assert np.isnan(out0[0, 0])
    assert out0[4, 4] == pytest.approx(arr[4, 4])


# --- 2: band_polygon excludes the nodata border -----------------------------

def test_band_polygon_excludes_nodata_border():
    n = 14
    data = np.full((n, n), np.nan)
    # interior 8x8 patch (rows/cols 3..10) with a gentle ramp above 100
    for r in range(3, 11):
        for c in range(3, 11):
            data[r, c] = 100.0 + (r - 3) + (c - 3)
    px = 30.0
    transform = Affine.translation(0.0, n * px) * Affine.scale(px, -px)
    dem = DemRaster(data=data.astype("float32"), transform=transform,
                    crs="EPSG:32611", nodata=None)
    z = smooth_dem(dem.data, 1.5)
    xs, ys = dem.pixel_coords()
    band = band_polygon(dem, 99.0, xs, ys, z)   # threshold below interior min
    full_area = (n * px) ** 2
    assert not band.is_empty
    # The band tracks the interior patch, not the full grid rectangle.
    assert band.area < 0.6 * full_area
    interior_area = (8 * px) ** 2
    assert band.area < interior_area * 1.3


# --- 3: nest tolerates / drops empty-outline parts --------------------------

def test_nest_handles_empty_outline_parts():
    cfg = _cfg()
    empty = Part(part_id="L09-P0", kind="ply", material="birch_3mm",
                 outline=MultiPolygon())
    add_part_id_score(empty, cfg)                 # must not raise
    real = Part(part_id="L00-P0", kind="ply", material="birch_3mm",
                outline=MultiPolygon([_square(50, 50, 20)]))
    boards = nest_parts([empty, real], cfg)
    placed = [p for b in boards for p in b.parts]
    assert len(placed) == 1
    assert placed[0].part_id == "L00-P0"


# --- 4: acrylic skips (does not delete) a ply part a lake fully covers -------

def test_acrylic_full_cover_skips_and_leaves_no_empty_part():
    cfg = _cfg(lake_mode="inset")
    model = _model(cfg)
    sym = build_symbology(model, FeatureCollection(), cfg)
    parts = build_parts_from_model(model, sym, cfg)
    k = 2
    ply2 = next(p for p in parts if p.kind == "ply" and p.layer_index == k)
    lake = ply2.outline.buffer(2.0)               # fully covers the layer-2 piece
    sym.per_layer[k].lake_polys.append(lake)

    from toposlicr.acrylic import apply_acrylic
    out_parts, warnings = apply_acrylic(parts, model, sym, cfg)
    assert any("fully covers" in w for w in warnings)
    assert all(not p.outline.is_empty for p in out_parts if p.kind == "ply")


# --- 5: acrylic re-clips a ply part's ops to the new (holed) outline ---------

def test_acrylic_reclips_ops_out_of_lake():
    cfg = _cfg(lake_mode="inset")
    model = _model(cfg)
    sym = build_symbology(model, FeatureCollection(), cfg)
    parts = build_parts_from_model(model, sym, cfg)
    k = 1
    ply = next(p for p in parts if p.kind == "ply" and p.layer_index == k)
    rep = model.footprint(k).representative_point()
    lake = _square(rep.x, rep.y, 6.0)             # small interior hole
    ply.ops[OP_ENGRAVE] = lake.buffer(3.0)        # straddles the lake boundary
    sym.per_layer[k].lake_polys.append(lake)

    from toposlicr.acrylic import apply_acrylic
    apply_acrylic(parts, model, sym, cfg)
    assert OP_ENGRAVE in ply.ops                  # op survives (extends past lake)
    assert ply.ops[OP_ENGRAVE].intersection(lake).area < 1e-6  # clipped out of hole


# --- 6: label file round-trips the bbox center (no drift) -------------------

def test_label_file_records_bbox_center_not_centroid(tmp_path):
    glyphs = text_to_polygons("Pgy", cap_height_mm=4.0)     # has descenders
    minx, miny, maxx, maxy = glyphs.bounds
    bbox_cx, bbox_cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    centroid = glyphs.centroid
    # The glyph's centroid must differ from its bbox center for the test to bite.
    assert abs(bbox_cx - centroid.x) > 0.05

    placed = PlacedLabel(text="Pgy", layer_index=3, geometry=glyphs,
                         anchor=Point(0, 0), placed=True)
    path = tmp_path / "labels.json"
    write_labels_file(LabelReport(placed=[placed]), path)
    overrides = read_label_overrides(path)
    layer, cx, cy = overrides["Pgy"]
    assert layer == 3
    assert abs(cx - bbox_cx) <= 0.011             # matches bbox center (rounded 2dp)
    assert abs(cy - bbox_cy) <= 0.011


# --- 7: layer-count formula & honoring a pinned count -----------------------

def test_layer_count_formula_matches_builder():
    assert _layer_count(0, 2000, 200) == 10       # even division: summit degenerate
    assert _layer_count(0, 2050, 200) == 11       # remainder: extra band


def test_pinned_layer_count_is_honored_and_reported():
    res = solve_scale(model_width_mm=500, real_width_m=40_000, ply_thickness_mm=3.0,
                      layer_count=10, base_elev_m=0, max_elev_m=2000)
    assert res.interval_m == pytest.approx(200.0)   # span/N, not snapped
    assert res.layer_count == 10


def test_built_model_layer_count_near_pin():
    cfg = _cfg(physical={"layer_count": 8})
    model = _model(cfg)
    assert abs(model.layer_count - 8) <= 1


# --- 8: config coerces quoted numerics, rejects garbage ---------------------

def test_config_coerces_string_numerics():
    cfg = _cfg(symbology={"peaks": {"min_prominence_m": "150"},
                          "lakes": {"min_area_km2": "0.05"}})
    assert cfg.symbology.peaks.min_prominence_m == 150.0
    assert isinstance(cfg.symbology.peaks.min_prominence_m, float)
    assert cfg.symbology.lakes.min_area_km2 == 0.05
    assert isinstance(cfg.symbology.lakes.min_area_km2, float)


def test_config_rejects_uncoercible_numeric():
    with pytest.raises(ConfigError):
        _cfg(symbology={"peaks": {"min_prominence_m": "abc"}})


# --- 9: Overpass error payloads are detected --------------------------------

def test_check_overpass_error_detects_remark():
    with pytest.raises(OverpassError):
        check_overpass_error({"remark": "runtime error: Query timed out",
                              "elements": []})


def test_check_overpass_error_passes_normal_payload():
    check_overpass_error({"elements": [{"type": "node", "id": 1}]})
    # A remark alongside real results is not a fatal error.
    check_overpass_error({"remark": "some note", "elements": [{"type": "node"}]})


# --- 10: lake with no visible-band overlap falls back to footprint -----------

def test_assign_lake_layer_footprint_fallback():
    cfg = _cfg()
    model = _model(cfg)
    assert model.layer_count >= 3
    rep = model.footprint(2).representative_point()
    lake = _square(rep.x, rep.y, 2.0)
    empty_bands = {k: Polygon() for k in range(model.layer_count)}  # no overlap
    k = _assign_lake_layer(lake, empty_bands, model)
    assert k >= 1                                  # not spuriously 0
    assert model.footprint(k).contains(lake.centroid)


# --- 11: coupon offset series does not overshoot stop -----------------------

def test_offset_series_no_overshoot():
    vals = offset_series(-0.05, 0.15, 0.03)
    assert all(v <= 0.15 + 1e-9 for v in vals)
    assert max(vals) <= 0.15 + 1e-9
    default = offset_series()
    assert default[0] == pytest.approx(-0.05)
    assert default[-1] == pytest.approx(0.15)


# --- 12: glyph cache honors cap_height_mm -----------------------------------

def test_glyph_cache_honors_cap_height():
    def h(cap):
        g = text_to_polygons("AV", cap_height_mm=cap)
        minx, miny, maxx, maxy = g.bounds
        return maxy - miny
    h2, h6 = h(2.0), h(6.0)
    assert h2 == pytest.approx(2.0, rel=0.3)
    ratio = h6 / h2
    assert abs(ratio - 3.0) / 3.0 < 0.05           # same text, different size
