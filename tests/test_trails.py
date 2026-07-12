"""Trail symbology — OSM parsing, selection, dashing, and per-band scoring."""

import json

import numpy as np
import pytest
from shapely.geometry import LineString

from toposlicr.config import SymbologyConfig, parse_config
from toposlicr.features.overpass import parse_overpass_json
from toposlicr.features.schema import FeatureType
from toposlicr.features.select import auto_choices
from toposlicr.symbology import dash_line


def _payload():
    return {"elements": [
        {"type": "way", "id": 1, "tags": {"highway": "path", "name": "Meadow Loop"},
         "geometry": [{"lat": 36.55, "lon": -118.31}, {"lat": 36.57, "lon": -118.30}]},
        {"type": "way", "id": 2, "tags": {"highway": "footway"},
         "geometry": [{"lat": 36.55, "lon": -118.29}, {"lat": 36.58, "lon": -118.29}]},
        {"type": "way", "id": 3, "tags": {"highway": "track"},  # forest road — excluded
         "geometry": [{"lat": 36.55, "lon": -118.28}, {"lat": 36.58, "lon": -118.28}]},
        {"type": "relation", "id": 9,
         "tags": {"route": "hiking", "name": "Sierra High Route", "network": "nwn"},
         "members": [{"type": "way", "ref": 10, "role": "",
                      "geometry": [{"lat": 36.55, "lon": -118.32},
                                   {"lat": 36.59, "lon": -118.28}]}]},
    ]}


# --- parsing -------------------------------------------------------------

def test_parses_paths_and_route_relations():
    coll = parse_overpass_json(_payload())
    trails = coll.of_type(FeatureType.TRAIL)
    names = {t.name for t in trails}
    assert "Meadow Loop" in names            # named path
    assert "Sierra High Route" in names      # named route relation
    assert len(trails) == 3                  # 2 paths + route; track excluded (not a trail type)
    route = next(t for t in trails if t.name == "Sierra High Route")
    assert route.tags.get("network") == "nwn"
    assert route.importance == 3             # nwn rank


def test_track_highway_is_not_a_trail():
    coll = parse_overpass_json(_payload())
    # highway=track (way id 3) must not become a trail (hiking-only scope).
    assert all(t.tags.get("highway") != "track" for t in coll.of_type(FeatureType.TRAIL))


# --- selection -----------------------------------------------------------

def test_selection_defaults_keep_all_trails():
    coll = parse_overpass_json(_payload())
    chosen = {c.feature.name or "unnamed": c.include for c in auto_choices(coll, SymbologyConfig())}
    assert chosen["Meadow Loop"] and chosen["Sierra High Route"] and chosen["unnamed"]


def test_include_unnamed_false_drops_unnamed_paths():
    coll = parse_overpass_json(_payload())
    sym = SymbologyConfig()
    sym.trails.include_unnamed = False
    chosen = {c.feature.name or "unnamed": c.include for c in auto_choices(coll, sym)}
    assert not chosen["unnamed"]
    assert chosen["Meadow Loop"] and chosen["Sierra High Route"]


def test_min_network_filters_routes_by_grade():
    coll = parse_overpass_json(_payload())
    sym = SymbologyConfig()
    sym.trails.min_network = "iwn"           # only international routes
    inc = {c.feature.name: c.include for c in auto_choices(coll, sym)
           if c.feature.tags.get("network")}
    assert inc["Sierra High Route"] is False  # nwn < iwn


def test_trails_include_false_drops_everything():
    coll = parse_overpass_json(_payload())
    sym = SymbologyConfig()
    sym.trails.include = False
    assert not any(c.include for c in auto_choices(coll, sym)
                   if c.feature.feature_type is FeatureType.TRAIL)


def test_named_path_with_nonwalking_network_is_kept():
    """min_network grades routes only — a named way with a cycle network stays."""
    from shapely.geometry import LineString

    from toposlicr.features.schema import GeoFeature
    from toposlicr.features.select import _auto_include
    f = GeoFeature(FeatureType.TRAIL, LineString([(0, 0), (1, 1)]),
                   name="Riverside Path", importance=1,
                   tags={"highway": "footway", "network": "lcn"})
    assert _auto_include(f, SymbologyConfig()) is True


def test_curved_text_not_upside_down_on_vertical_trail():
    """A descending vertical segment must not render inverted (orient upward)."""
    from toposlicr.labels import text_along_path
    up = text_along_path("Ridge", LineString([(0, 0), (0, 100)]), cap_height_mm=4.0)
    down = text_along_path("Ridge", LineString([(0, 100), (0, 0)]), cap_height_mm=4.0)
    assert up is not None and down is not None
    # Both orient upward → identical placement (down is reversed to match up).
    assert up.equals_exact(down, tolerance=1e-6) or up.bounds == pytest.approx(down.bounds)


# --- dashing -------------------------------------------------------------

def test_dash_line_segments_a_straight_line():
    d = dash_line(LineString([(0, 0), (20, 0)]), dash_mm=2.5, gap_mm=1.5)
    assert d.geom_type == "MultiLineString"
    assert len(d.geoms) == 5                  # period 4mm over 20mm → 5 dashes
    assert d.length == pytest.approx(12.5, abs=0.1)   # 5 × 2.5


def test_dash_line_follows_a_curve():
    curve = LineString([(0, 0), (5, 5), (10, 0), (15, 5)])
    d = dash_line(curve, dash_mm=1.0, gap_mm=1.0)
    assert not d.is_empty
    assert d.length < curve.length            # gaps removed
    # dashes stay on the curve (bbox within the curve's bbox)
    assert d.bounds[0] >= curve.bounds[0] - 1e-6
    assert d.bounds[2] <= curve.bounds[2] + 1e-6


# --- symbology + fictional bundle ---------------------------------------

def _trail_bundle(root, n=140):
    from toposlicr.bundle import BundleMeta, write_gray16
    bdir = root / "trails.terrainbundle"
    bdir.mkdir(parents=True, exist_ok=True)
    ys, xs = np.mgrid[0:n, 0:n] / n
    z = np.exp(-(((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / (2 * 0.06))) + 0.1
    write_gray16(bdir / "heightmap.png", z)
    (bdir / "meta.json").write_text(json.dumps(
        BundleMeta(units_per_pixel=50.0, world_origin=(0.0, n * 50.0)).to_dict()))
    (bdir / "trails.geojson").write_text(json.dumps({
        "type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {"name": "Summit Trail"},
            "geometry": {"type": "LineString",
                         "coordinates": [[1000, 1000], [3500, 4000], [6000, 6000]]}}]}))
    return bdir


def test_bundle_trails_scored_per_band_with_name(tmp_path):
    from toposlicr.bundle import load_bundle
    from toposlicr.layers import build_layer_model
    from toposlicr.pipeline import _bundle_symbology

    bdir = _trail_bundle(tmp_path)
    cfg = parse_config({"region": {"bundle": str(bdir)},
                        "physical": {"model_width_mm": 300, "normalize_layers": 6}})
    bundle = load_bundle(bdir)
    assert bundle.trails() and bundle.trails()[0][1] == "Summit Trail"
    model = build_layer_model(bundle.to_dem(), cfg)
    sym = _bundle_symbology(model, bundle, cfg, [])
    dashed = [k for k, s in sym.per_layer.items() if s.trails and not s.trails.is_empty]
    named = [k for k, s in sym.per_layer.items()
             if s.trail_labels and not s.trail_labels.is_empty]
    assert dashed                          # trail scored on the bands it crosses
    assert len(named) == 1                 # name placed once, on its most-exposed layer
