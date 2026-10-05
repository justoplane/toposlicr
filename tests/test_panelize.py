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


# --- seam placement: exposed length is the objective ------------------------

from types import SimpleNamespace  # noqa: E402

from toposlicr.panelize import (  # noqa: E402
    PIECE_PENALTY_MM,
    STAGGER_MM,
    fits_bed,
    seam_visibility,
)


def _fake_model(footprints):
    """A LayerModel stand-in: footprint(k) → polygon (empty beyond the list)."""
    def footprint(k):
        if 0 <= k < len(footprints):
            return footprints[k]
        return MultiPolygon()
    return SimpleNamespace(footprint=footprint, layer_count=len(footprints))


def _rect(x0, y0, x1, y1):
    return MultiPolygon([Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])])


def _seam_xs(parts):
    """x positions of vertical seams between sibling pieces."""
    xs = set()
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            shared = parts[i].outline.boundary.intersection(parts[j].outline.boundary)
            if not shared.is_empty and shared.length > 1:
                xs.add(round(shared.centroid.x, 1))
    return sorted(xs)


def test_seams_go_through_the_hidden_corridor():
    """500×150 slab, 200-wide bed: two cuts are needed. The layer above covers
    two vertical strips; both seams must land inside them (minus the margin)."""
    cfg = _cfg((200, 200))
    slab = _rect(0, 0, 500, 150)
    above = _rect(150, 0, 200, 150).union(_rect(300, 0, 350, 150))
    model = _fake_model([slab, MultiPolygon([g for g in above.geoms])])
    out = panelize_parts([_rect_part(500, 150)], model, cfg)
    assert len(out) == 3
    xs = _seam_xs(out)
    assert len(xs) == 2
    m = cfg.panelization.seam_margin_mm
    assert 150 + m <= xs[0] <= 200 - m and 300 + m <= xs[1] <= 350 - m
    total, exposed = seam_visibility(out, model, 0)
    assert total == pytest.approx(300, abs=1) and exposed == pytest.approx(0, abs=1e-6)


def test_extra_piece_when_it_hides_the_seam():
    """One exposed cut would do (bed 300 wide); two hidden cuts cost 2 pieces'
    penalty but no exposed seam, so the splitter takes three pieces."""
    cfg = _cfg((300, 200))
    slab = _rect(0, 0, 500, 150)
    above = _rect(120, 0, 160, 150).union(_rect(340, 0, 380, 150))
    model = _fake_model([slab, MultiPolygon([g for g in above.geoms])])
    out = panelize_parts([_rect_part(500, 150)], model, cfg)
    assert len(out) == 3
    _, exposed = seam_visibility(out, model, 0)
    assert exposed == pytest.approx(0, abs=1e-6)
    assert 2 * PIECE_PENALTY_MM < 150 + 25          # the trade the test relies on
    for p in out:
        assert fits_bed(p.outline, cfg.machine.usable_bed_mm)


def test_no_corridor_uses_one_cut():
    """Nothing to hide behind → the cheapest split is the single cut."""
    cfg = _cfg((300, 200))
    model = _fake_model([_rect(0, 0, 500, 150)])
    out = panelize_parts([_rect_part(500, 150)], model, cfg)
    assert len(out) == 2
    total, exposed = seam_visibility(out, model, 0)
    assert total == pytest.approx(150, abs=1) and exposed == pytest.approx(150, abs=1)


def test_adjacent_layer_seams_stagger():
    """Identical slabs on layers 0 and 1 with no corridor: layer 1's seam must
    not sit on top of layer 0's (brickwork)."""
    cfg = _cfg((300, 200))
    slab = _rect(0, 0, 500, 150)
    model = _fake_model([slab, slab, MultiPolygon()])
    p0 = _rect_part(500, 150, pid="L00-P0")
    p1 = Part(part_id="L01-P0", kind="ply", material="birch_3mm",
              outline=p0.outline, layer_index=1)
    out = panelize_parts([p0, p1], model, cfg)
    xs0 = _seam_xs([p for p in out if p.layer_index == 0])
    xs1 = _seam_xs([p for p in out if p.layer_index == 1])
    assert len(xs0) == 1 and len(xs1) == 1
    assert abs(xs0[0] - xs1[0]) >= STAGGER_MM - 1e-6


def test_islands_split_independently():
    """Two islands, one oversized: only that one is cut, the other stays whole."""
    cfg = _cfg((200, 200))
    big = Polygon([(0, 0), (350, 0), (350, 100), (0, 100)])
    small = Polygon([(400, 0), (450, 0), (450, 50), (400, 50)])
    part = Part(part_id="L00-P0", kind="ply", material="birch_3mm",
                outline=MultiPolygon([big, small]), layer_index=0)
    model = _fake_model([MultiPolygon([big, small])])
    out = panelize_parts([part], model, cfg)
    assert len(out) == 3
    assert any(p.outline.equals(MultiPolygon([small])) for p in out)


def test_fit_counts_rotation():
    """A 100×400 piece only fits the 495×279 bed sideways: not split."""
    cfg = _cfg((495, 279))
    model = _fake_model([_rect(0, 0, 100, 400)])
    tall = Part(part_id="L00-P0", kind="ply", material="birch_3mm",
                outline=_rect(0, 0, 100, 400), layer_index=0)
    assert fits_bed(tall.outline, cfg.machine.usable_bed_mm)
    out = panelize_parts([tall], model, cfg)
    assert len(out) == 1 and out[0] is tall


def test_bent_seam_follows_a_z_shaped_corridor():
    """The hidden zone is a Z: a strip on the lower-left, a strip on the
    upper-right, joined by a thin horizontal band. No straight cut can hide,
    but one Z seam can — and it must still yield two bed-fitting pieces."""
    cfg = _cfg((350, 200))
    slab = _rect(0, 0, 500, 150)
    above = (Polygon([(150, 0), (190, 0), (190, 80), (150, 80)])
             .union(Polygon([(310, 70), (350, 70), (350, 150), (310, 150)]))
             .union(Polygon([(150, 70), (350, 70), (350, 80), (150, 80)])))
    model = _fake_model([slab, MultiPolygon([above]) if above.geom_type == "Polygon"
                         else above])
    out = panelize_parts([_rect_part(500, 150)], model, cfg)
    assert len(out) == 2
    total, exposed = seam_visibility(out, model, 0)
    assert total > 150                      # longer than a straight cut: it jogs
    assert exposed == pytest.approx(0, abs=1e-6)
    for p in out:
        assert fits_bed(p.outline, cfg.machine.usable_bed_mm)
    # The seam has a horizontal jog: some shared boundary runs along y ≈ 75.
    shared = out[0].outline.boundary.intersection(out[1].outline.boundary)
    ys = [round(c[1]) for seg in getattr(shared, "geoms", [shared]) for c in seg.coords]
    assert any(70 <= y <= 80 for y in ys)
