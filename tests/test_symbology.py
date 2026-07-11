import pytest
from shapely.geometry import LineString, Point, Polygon

from toposlicr.config import parse_config
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.features.schema import FeatureCollection, FeatureType, GeoFeature
from toposlicr.geo import BBox
from toposlicr.labels import (
    LabelRequest,
    place_labels,
    text_to_polygons,
)
from toposlicr.layers import build_layer_model
from toposlicr.symbology import (
    LabelReport,
    SymbologyResult,
    build_symbology,
    hidden_zone,
    visible_band,
)

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _model(**phys):
    data = {
        "region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                   "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, **phys},
    }
    cfg = parse_config(data)
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    return build_layer_model(dem, cfg, BBOX), cfg


def _count_interiors(geom):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return sum(len(p.interiors) for p in polys if p.geom_type == "Polygon")


# --- text_to_polygons -------------------------------------------------------

def test_text_to_polygons_nonempty_and_sized():
    geom = text_to_polygons("AV", cap_height_mm=4.0)
    assert not geom.is_empty
    assert geom.area > 0
    minx, miny, maxx, maxy = geom.bounds
    height = maxy - miny
    # "AV" are capitals with no descenders: height ≈ cap height (allow tolerance).
    assert height == pytest.approx(4.0, rel=0.3)


def test_text_to_polygons_empty_for_blank():
    assert text_to_polygons("", cap_height_mm=4.0).is_empty
    assert text_to_polygons("   ", cap_height_mm=4.0).is_empty


def test_text_to_polygons_has_counter_holes():
    # A glyph with an enclosed counter must yield at least one interior ring.
    geom = text_to_polygons("oe", cap_height_mm=6.0)
    assert _count_interiors(geom) >= 1


# --- visible band / hidden zone ---------------------------------------------

def test_visible_band_is_footprint_minus_above():
    model, _ = _model(exaggeration=1.5)
    for k in range(model.layer_count):
        vb = visible_band(model, k)
        here = model.footprint(k)
        above = model.footprint(k + 1)
        assert vb.within(here.buffer(1e-6))
        expected = here if above.is_empty else here.difference(above)
        assert vb.area == pytest.approx(expected.area, rel=1e-6)


def test_hidden_zone_is_within_layer_above():
    model, _ = _model(exaggeration=1.5)
    for k in range(model.layer_count - 1):
        hz = hidden_zone(model, k)
        if hz.is_empty:
            continue
        assert hz.within(model.footprint(k + 1).buffer(1e-6))
        assert hz.within(model.footprint(k).buffer(1e-6))


def test_top_layer_visible_band_is_whole_footprint():
    model, _ = _model(exaggeration=1.5)
    top = model.layer_count - 1
    vb = visible_band(model, top)
    assert vb.area == pytest.approx(model.footprint(top).area, rel=1e-9)


# --- place_labels -----------------------------------------------------------

def test_place_labels_places_within_band_without_overlap():
    band = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    reqs = [
        LabelRequest("Alpha", Point(40, 40), layer_index=0, cap_height_mm=5.0),
        LabelRequest("Beta", Point(150, 150), layer_index=0, cap_height_mm=5.0),
    ]
    placed = place_labels(reqs, {0: band})
    assert all(p.placed for p in placed)
    for p in placed:
        assert p.geometry.within(band)
    assert not placed[0].geometry.intersects(placed[1].geometry)


def test_place_labels_fails_when_too_large_for_band():
    band = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    req = LabelRequest("Enormous", Point(100, 100), layer_index=0, cap_height_mm=500.0)
    placed = place_labels([req], {0: band})
    assert placed[0].placed is False
    assert placed[0].geometry.is_empty


def test_place_labels_cross_layer_fallback_to_lower_band():
    tiny = Polygon([(99, 99), (101, 99), (101, 101), (99, 101)])   # 2 mm, no room
    big = Polygon([(0, 0), (200, 0), (200, 200), (0, 200)])
    req = LabelRequest("Summit", Point(100, 100), layer_index=2, is_point=True,
                       cap_height_mm=4.0)
    placed = place_labels([req], {2: tiny, 1: big})
    assert placed[0].placed
    assert placed[0].layer_index < 2      # fell back to a lower, larger band
    assert placed[0].geometry.within(big)


# --- build_symbology end-to-end ---------------------------------------------

def test_build_symbology_covers_all_layers_and_places_peak():
    model, cfg = _model(exaggeration=1.5)
    features = FeatureCollection(epsg=4326)
    features.add(GeoFeature(
        FeatureType.PEAK, Point(-118.27, 36.57), name="TestPeak",
        elevation=2400.0, importance=2400.0, osm_id="node/1"))
    features.add(GeoFeature(
        FeatureType.RIVER, LineString([(-118.30, 36.57), (-118.24, 36.56)]),
        name="TestCreek", importance=5, osm_id="way/2"))

    result = build_symbology(model, features, cfg)
    assert isinstance(result, SymbologyResult)
    assert isinstance(result.labels, LabelReport)
    # A per-layer entry for every layer.
    assert set(result.per_layer) == set(range(model.layer_count))
    for k, ls in result.per_layer.items():
        assert ls.layer_index == k
        assert ls.visible_band.area == pytest.approx(visible_band(model, k).area, rel=1e-6)
    # The peak produced a label request (placed or flagged, but present).
    texts = [p.text for p in result.labels.placed]
    assert any("TestPeak" in t for t in texts)
    # label_elevation default appends the elevation.
    assert any("2,400 m" in t for t in texts)


def test_build_symbology_river_clipped_to_some_visible_band():
    model, cfg = _model(exaggeration=1.5)
    features = FeatureCollection(epsg=4326)
    features.add(GeoFeature(
        FeatureType.RIVER, LineString([(-118.32, 36.57), (-118.22, 36.57)]),
        name="LongCreek", importance=5, osm_id="way/9"))
    result = build_symbology(model, features, cfg)
    total_river_len = sum(
        ls.rivers.length for ls in result.per_layer.values() if not ls.rivers.is_empty
    )
    assert total_river_len > 0
