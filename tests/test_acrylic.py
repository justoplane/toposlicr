import pytest
from shapely.geometry import MultiPolygon, Polygon

from toposlicr.acrylic import INTERFERENCE_MM, apply_acrylic, press_fit_offset
from toposlicr.config import parse_config
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.features.schema import FeatureCollection, FeatureType, GeoFeature
from toposlicr.geo import BBox
from toposlicr.layers import build_layer_model
from toposlicr.parts import Part, build_parts_from_model, water_material
from toposlicr.symbology import build_symbology

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _cfg(mode="inset", **machine):
    data = {
        "region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                   "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, "interval_m": 250},
        "machine": {"kerf_mm": 0.15, **machine},
        "materials": {"default": "birch_3mm", "water": "blue_acrylic_3mm"},
        "symbology": {"lakes": {"mode": mode, "min_area_km2": 0.0}},
    }
    return parse_config(data)


def _square(cx, cy, half):
    return Polygon([(cx - half, cy - half), (cx + half, cy - half),
                    (cx + half, cy + half), (cx - half, cy + half)])


def _interiors(geom):
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    return sum(len(p.interiors) for p in polys if p.geom_type == "Polygon")


# --- press_fit_offset -------------------------------------------------------

def test_press_fit_offset_matches_formula():
    cfg = _cfg(kerf_mm=0.15)
    # kerf/2 + kerf/2 + interference = kerf + interference
    assert press_fit_offset(cfg) == pytest.approx(0.15 + INTERFERENCE_MM)
    assert press_fit_offset(cfg) > 0


def test_press_fit_offset_scales_with_kerf():
    assert press_fit_offset(_cfg(kerf_mm=0.30)) > press_fit_offset(_cfg(kerf_mm=0.10))


# --- apply_acrylic on hand-built parts --------------------------------------

def _fake_symbology(lakes_by_layer):
    """Build a minimal SymbologyResult-like object exposing per_layer.lake_polys."""
    from toposlicr.labels import LabelReport
    from toposlicr.symbology import LayerSymbology, SymbologyResult
    per_layer = {}
    for k, polys in lakes_by_layer.items():
        per_layer[k] = LayerSymbology(
            layer_index=k, visible_band=MultiPolygon(),
            rivers=MultiPolygon(), lake_outlines=MultiPolygon(), lake_polys=polys)
    return SymbologyResult(per_layer=per_layer, labels=LabelReport())


class _FakeModel:
    """Just enough LayerModel surface for apply_acrylic (footprint of below)."""

    def __init__(self, footprints):
        self._f = footprints

    def footprint(self, k):
        return self._f.get(k, MultiPolygon())


def test_inset_cuts_hole_and_emits_acrylic():
    ply = Part(part_id="L02-P0", kind="ply", material="birch_3mm",
               outline=MultiPolygon([_square(50, 50, 40)]), layer_index=2)
    lake = _square(50, 50, 10)
    model = _FakeModel({1: MultiPolygon([_square(50, 50, 45)])})  # solid shelf below
    sym = _fake_symbology({2: [lake]})
    cfg = _cfg(mode="inset")

    out, warnings = apply_acrylic([ply], model, sym, cfg)

    # The ply part gained a hole.
    assert _interiors(ply.outline) == 1
    assert ply.area == pytest.approx(80 * 80 - 20 * 20, rel=0.02)
    # Exactly one acrylic part, larger than the lake, correct material/kind/id.
    acrylics = [p for p in out if p.kind == "acrylic"]
    assert len(acrylics) == 1
    ac = acrylics[0]
    assert ac.material == water_material(cfg.materials)
    assert ac.part_id == "W02-P0"
    assert ac.area > lake.area
    assert not warnings


def test_score_mode_is_noop():
    ply = Part(part_id="L02-P0", kind="ply", material="birch_3mm",
               outline=MultiPolygon([_square(50, 50, 40)]), layer_index=2)
    sym = _fake_symbology({2: [_square(50, 50, 10)]})
    out, warnings = apply_acrylic([ply], _FakeModel({}), sym, _cfg(mode="score"))
    assert out == [ply]
    assert warnings == []
    assert _interiors(ply.outline) == 0


def test_base_layer_lake_warns_and_skips():
    ply = Part(part_id="L00-P0", kind="ply", material="birch_3mm",
               outline=MultiPolygon([_square(50, 50, 40)]), layer_index=0)
    sym = _fake_symbology({0: [_square(50, 50, 10)]})
    out, warnings = apply_acrylic([ply], _FakeModel({}), sym, _cfg(mode="inset"))
    assert not [p for p in out if p.kind == "acrylic"]
    assert any("base layer" in w for w in warnings)
    assert _interiors(ply.outline) == 0


def test_lake_spanning_two_ply_parts_cut_from_both():
    left = Part(part_id="L03-P0", kind="ply", material="birch_3mm",
                outline=MultiPolygon([_square(30, 50, 20)]), layer_index=3)
    right = Part(part_id="L03-P1", kind="ply", material="birch_3mm",
                 outline=MultiPolygon([_square(70, 50, 20)]), layer_index=3)
    lake = _square(50, 50, 15)  # straddles both squares (x 35..65 overlaps both)
    model = _FakeModel({2: MultiPolygon([_square(50, 50, 60)])})
    sym = _fake_symbology({3: [lake]})
    out, _ = apply_acrylic([left, right], model, sym, _cfg(mode="inset"))
    # Both ply parts lost area to the lake.
    assert left.area < 40 * 40
    assert right.area < 40 * 40
    assert len([p for p in out if p.kind == "acrylic"]) == 1


# --- end-to-end on a synthetic model ---------------------------------------

def _lake_feature(lon, lat, d=0.01):
    poly = Polygon([(lon - d, lat - d), (lon + d, lat - d),
                    (lon + d, lat + d), (lon - d, lat + d)])
    return GeoFeature(FeatureType.LAKE, poly, name="Test Lake", importance=1.0,
                      osm_id="way/1")


def test_end_to_end_synthetic_produces_acrylic():
    cfg = _cfg(mode="inset")
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    model = build_layer_model(dem, cfg, BBOX)
    # A lake offset from the summit so it lands on a mid (non-base) layer.
    feats = FeatureCollection([_lake_feature(-118.30, 36.58)])
    sym = build_symbology(model, feats, cfg)
    parts = build_parts_from_model(model, sym, cfg)

    out, warnings = apply_acrylic(parts, model, sym, cfg)
    acrylics = [p for p in out if p.kind == "acrylic"]
    assert len(acrylics) >= 1
    assert all(p.material == water_material(cfg.materials) for p in acrylics)
    # The happy-path lake is not on the base layer, so no shelf warning fires.
    assert not any("shelf" in w for w in warnings)
