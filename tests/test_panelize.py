import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon

from toposlicr.config import parse_config
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.geo import BBox
from toposlicr.layers import build_layer_model
from toposlicr.panelize import panelize_parts
from toposlicr.parts import OP_SCORE_HYDRO, Part, build_parts_from_model

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _rect_part(w, h, pid="L00-P0", **kw):
    outline = MultiPolygon([Polygon([(0, 0), (w, 0), (w, h), (0, h)])])
    return Part(part_id=pid, kind="ply", material="birch_3mm",
                outline=outline, layer_index=0, **kw)


def _cfg(bed, **phys):
    data = {
        "region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                   "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, **phys},
        "machine": {"bed_mm": list(bed), "margin_mm": 0.0},
    }
    return parse_config(data)


def _fits(part, bed):
    minx, miny, maxx, maxy = part.outline.bounds
    return (maxx - minx) <= bed[0] + 1e-6 and (maxy - miny) <= bed[1] + 1e-6


def test_part_smaller_than_bed_passes_through():
    cfg = _cfg((495, 279))
    part = _rect_part(100, 100)
    model, _ = _synthetic_model(cfg)
    out = panelize_parts([part], model, cfg)
    assert len(out) == 1
    assert out[0] is part


def test_oversized_part_splits_and_all_fit():
    bed = (200, 200)
    cfg = _cfg(bed)
    model, _ = _synthetic_model(cfg)
    part = _rect_part(500, 150)  # too wide
    out = panelize_parts([part], model, cfg)
    assert len(out) >= 2
    for p in out:
        assert _fits(p, cfg.machine.usable_bed_mm)
    # Unique ids, parent recorded.
    ids = [p.part_id for p in out]
    assert len(ids) == len(set(ids))
    assert all(p.parent_id == "L00-P0" for p in out)


def test_split_preserves_area():
    bed = (200, 200)
    cfg = _cfg(bed)
    model, _ = _synthetic_model(cfg)
    part = _rect_part(500, 150)
    out = panelize_parts([part], model, cfg)
    total = sum(p.area for p in out)
    assert total == pytest.approx(500 * 150, rel=1e-6)
    # Pieces should not overlap (guillotine): union area == sum of areas.
    union = out[0].outline
    for p in out[1:]:
        union = union.union(p.outline)
    assert union.area == pytest.approx(total, rel=1e-6)


def test_ops_distributed_to_subparts():
    bed = (200, 200)
    cfg = _cfg(bed)
    model, _ = _synthetic_model(cfg)
    part = _rect_part(500, 150)
    # A horizontal score line spanning the full width at mid-height.
    part.add_op(OP_SCORE_HYDRO, LineString([(0, 75), (500, 75)]))
    out = panelize_parts([part], model, cfg)
    total_len = sum(
        p.ops[OP_SCORE_HYDRO].length for p in out if OP_SCORE_HYDRO in p.ops
    )
    assert total_len == pytest.approx(500, rel=1e-3)
    # Each sub-part only carries the portion within it.
    for p in out:
        if OP_SCORE_HYDRO in p.ops:
            assert p.ops[OP_SCORE_HYDRO].within(p.outline.buffer(1e-6))


def _synthetic_model(cfg):
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    return build_layer_model(dem, cfg, BBOX), cfg


def test_realistic_layers_all_fit_tiny_bed():
    # Tiny bed forces real layers to split.
    cfg = _cfg((120, 120), interval_m=250)
    model, _ = _synthetic_model(cfg)
    parts = build_parts_from_model(model, None, cfg)
    assert any(not _fits(p, cfg.machine.usable_bed_mm) for p in parts)  # some oversized
    out = panelize_parts(parts, model, cfg)
    for p in out:
        assert _fits(p, cfg.machine.usable_bed_mm)
    # Every output part is non-empty.
    assert all(p.area > 0 for p in out)


def test_puzzle_joint_still_produces_fitting_parts():
    cfg = _cfg((200, 200))
    cfg.panelization.seam_joint = "puzzle"
    model, _ = _synthetic_model(cfg)
    part = _rect_part(500, 150)
    out = panelize_parts([part], model, cfg)
    assert len(out) >= 2
    for p in out:
        assert _fits(p, cfg.machine.usable_bed_mm)
