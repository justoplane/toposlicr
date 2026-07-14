import xml.etree.ElementTree as ET

import pytest
from shapely.geometry import MultiPolygon, Polygon

from toposlicr.config import parse_config
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.geo import BBox
from toposlicr.layers import build_layer_model
from toposlicr.render import (
    registration_score,
    render_composite_preview,
    render_layer_svgs,
)
from toposlicr.svg import SvgDocument, polygon_to_path_d

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


def test_layer_model_is_nested_and_scaled():
    model, _ = _model(exaggeration=1.5)
    assert model.layer_count >= 3
    assert model.model_width_mm == pytest.approx(300)
    # Each higher layer is contained in the one below and smaller.
    for k in range(1, model.layer_count):
        below = model.footprint(k - 1)
        here = model.footprint(k)
        assert here.area <= below.area + 1e-6
        assert here.within(below.buffer(0.01))
    # No spurious nesting warnings for clean synthetic terrain.
    assert not any("protruded" in w for w in model.warnings)


def test_layer_geometry_within_model_bounds():
    model, _ = _model(exaggeration=1.5)
    for layer in model.layers:
        minx, miny, maxx, maxy = layer.geometry.bounds
        assert minx >= -0.5 and miny >= -0.5
        assert maxx <= model.model_width_mm + 0.5
        assert maxy <= model.model_height_mm + 0.5


def test_registration_score_is_boundary_of_layer_above():
    model, _ = _model(exaggeration=1.5)
    reg = registration_score(model, 0)
    assert not reg.is_empty
    # The score lies on layer 1's boundary and inside layer 0.
    assert reg.length == pytest.approx(model.footprint(1).boundary.length, rel=0.05)


def test_render_layer_svgs_are_valid_xml(tmp_path):
    model, cfg = _model(exaggeration=1.5)
    paths = render_layer_svgs(model, cfg, tmp_path)
    assert len(paths) == model.layer_count
    root = ET.parse(paths[0]).getroot()
    assert root.tag.endswith("svg")
    assert root.get("width") == f"{300:g}mm" or root.get("width").endswith("mm")
    # Base layer carries a cut plus a registration score.
    ops = {el.get("data-op") for el in root.iter() if el.tag.endswith("path")}
    assert "cut" in ops
    assert "score_registration" in ops


def test_composite_preview_written(tmp_path):
    model, cfg = _model(exaggeration=1.5)
    p = render_composite_preview(model, cfg, tmp_path / "preview.svg")
    root = ET.parse(p).getroot()
    assert root.tag.endswith("svg")


def test_svg_units_are_mm_with_matching_viewbox():
    doc = SvgDocument(200, 100)
    doc.add_cut(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), "#000000")
    svg = doc.to_svg()
    assert 'width="200mm"' in svg
    assert 'height="100mm"' in svg
    assert 'viewBox="0 0 200 100"' in svg


def test_svg_y_axis_is_flipped():
    # A point at model y=0 (SW corner) maps to SVG y=height.
    d = polygon_to_path_d(Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), height_mm=100)
    assert "0,100" in d  # (x=0, model y=0) -> svg (0, 100)


def test_svg_engrave_is_filled_no_stroke():
    doc = SvgDocument(50, 50)
    doc.add_engrave(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), "#333333")
    svg = doc.to_svg()
    assert 'fill="#333333"' in svg
    assert "stroke=\"none\"" not in svg  # engrave path omits stroke entirely
    assert 'data-op="engrave"' in svg


def test_multipolygon_path_has_multiple_subpaths():
    mp = MultiPolygon([
        Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
    ])
    d = polygon_to_path_d(mp, height_mm=20)
    assert d.count("M") == 2
    assert d.count("Z") == 2


def test_layer_seams_finds_shared_edge_between_split_parts():
    from types import SimpleNamespace

    from shapely.geometry import MultiPolygon, Polygon

    from toposlicr.render import layer_seams
    # Two ply parts of layer 0 sharing the edge x=50 (the seam).
    left = SimpleNamespace(kind="ply", layer_index=0,
                           outline=MultiPolygon([Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])]))
    right = SimpleNamespace(kind="ply", layer_index=0,
                            outline=MultiPolygon([Polygon([(50, 0), (100, 0), (100, 100), (50, 100)])]))
    seams = layer_seams([left, right], 0)
    assert seams is not None and seams.length == pytest.approx(100, abs=0.5)
    # A single unsplit part has no seams.
    assert layer_seams([left], 0) is None


def test_per_layer_svg_marks_panelization_seams(tmp_path):
    from toposlicr.parts import build_parts_from_model
    from toposlicr.panelize import panelize_parts
    from toposlicr.render import render_layer_svgs

    # A wide model on a tiny bed forces the base layers to be split.
    data = {"region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                       "dem": "synthetic"},
            "physical": {"model_width_mm": 400, "layer_count": 6},
            "machine": {"bed_mm": [120, 120]}}
    cfg = parse_config(data)
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    model = build_layer_model(dem, cfg, BBOX)
    parts = panelize_parts(build_parts_from_model(model, None, cfg), model, cfg)
    paths = render_layer_svgs(model, cfg, tmp_path, parts=parts)
    # At least one (split) layer carries the distinct seam color/op.
    assert any('data-op="seam"' in p.read_text() for p in paths)
    assert any(cfg.machine.colors["seam"] in p.read_text() for p in paths)
