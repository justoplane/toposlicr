import pytest
from shapely.geometry import MultiPolygon, Polygon

from toposlicr.config import parse_config
from toposlicr.nest import nest_parts, validate_boards
from toposlicr.parts import OP_CUT, OP_SCORE_IDS, Part, Placement


def _cfg(bed=(200.0, 200.0), margin=8.0, kerf=0.15):
    data = {
        "region": {"bbox": [-118.35, 36.52, -118.20, 36.62], "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0},
        "machine": {"bed_mm": list(bed), "margin_mm": margin, "kerf_mm": kerf},
    }
    return parse_config(data)


def _square(side, x0=0.0, y0=0.0):
    return MultiPolygon([Polygon([
        (x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])])


def _part(pid, side, material="birch_3mm", kind="ply"):
    return Part(part_id=pid, kind=kind, material=material, outline=_square(side),
                layer_index=0)


def test_nests_small_parts_onto_board():
    cfg = _cfg()
    parts = [_part(f"L00-P{i}", 30.0) for i in range(5)]
    boards = nest_parts(parts, cfg)
    assert boards
    # Every part got a placement.
    all_parts = [p for b in boards for p in b.parts]
    assert len(all_parts) == 5
    assert all(p.placement is not None for p in all_parts)
    # validate_boards must not raise on a good layout.
    validate_boards(boards, cfg)


def test_placed_parts_do_not_overlap_and_stay_in_bounds():
    cfg = _cfg()
    parts = [_part(f"L00-P{i}", 40.0) for i in range(6)]
    boards = nest_parts(parts, cfg)
    for board in boards:
        cuts = [p.placed_geometries()[OP_CUT] for p in board.parts]
        for i in range(len(cuts)):
            minx, miny, maxx, maxy = cuts[i].bounds
            assert minx >= -1e-6 and miny >= -1e-6
            assert maxx <= board.width_mm + 1e-6
            assert maxy <= board.height_mm + 1e-6
            for j in range(i + 1, len(cuts)):
                assert cuts[i].intersection(cuts[j]).area < 1e-3


def test_two_materials_nest_into_separate_board_sequences():
    cfg = _cfg()
    parts = ([_part(f"L00-P{i}", 30.0, material="birch_3mm") for i in range(3)]
             + [_part(f"W00-P{i}", 30.0, material="blue_acrylic_3mm", kind="acrylic")
                for i in range(2)])
    boards = nest_parts(parts, cfg)
    materials = {b.material for b in boards}
    assert materials == {"birch_3mm", "blue_acrylic_3mm"}
    # Each material's boards index from 0.
    for material in materials:
        idxs = sorted(b.index for b in boards if b.material == material)
        assert idxs[0] == 0


def test_multiple_boards_when_group_overflows_one_bed():
    # Tiny bed so a handful of parts need more than one board.
    cfg = _cfg(bed=(120.0, 120.0))
    parts = [_part(f"L00-P{i}", 45.0) for i in range(8)]
    boards = nest_parts(parts, cfg)
    assert len(boards) >= 2
    placed = [p.part_id for b in boards for p in b.parts]
    assert len(placed) == 8
    assert len(set(placed)) == 8  # each placed exactly once


def test_part_larger_than_bed_raises():
    cfg = _cfg(bed=(120.0, 120.0))  # usable 104×104
    parts = [_part("L00-P0", 150.0)]
    with pytest.raises(ValueError, match="exceeds the usable bed"):
        nest_parts(parts, cfg)


def test_part_id_score_added_to_each_part():
    cfg = _cfg()
    parts = [_part(f"L00-P{i}", 30.0) for i in range(3)]
    boards = nest_parts(parts, cfg)
    for board in boards:
        for part in board.parts:
            assert OP_SCORE_IDS in part.ops
            assert not part.ops[OP_SCORE_IDS].is_empty


def test_validate_boards_detects_overlap():
    from toposlicr.parts import Board

    cfg = _cfg()
    a = _part("L00-P0", 40.0)
    b = _part("L00-P1", 40.0)
    # Place both at the same spot → overlap.
    a.placement = Placement(board_index=0, tx=0.0, ty=0.0)
    b.placement = Placement(board_index=0, tx=10.0, ty=10.0)
    board = Board(index=0, material="birch_3mm", width_mm=184.0, height_mm=184.0,
                  parts=[a, b])
    with pytest.raises(ValueError, match="overlap"):
        validate_boards([board], cfg)


def test_validate_boards_detects_out_of_bounds():
    from toposlicr.parts import Board

    cfg = _cfg()
    p = _part("L00-P0", 40.0)
    p.placement = Placement(board_index=0, tx=500.0, ty=0.0)  # pushed off the bed
    board = Board(index=0, material="birch_3mm", width_mm=184.0, height_mm=184.0,
                  parts=[p])
    with pytest.raises(ValueError, match="outside board"):
        validate_boards([board], cfg)


def test_nest_rotates_a_part_that_only_fits_sideways():
    cfg = parse_config({
        "region": {"bbox": [-118.35, 36.52, -118.20, 36.62], "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, "exaggeration": 1.5},
        "machine": {"bed_mm": [495, 279], "margin_mm": 0.0},
    })
    tall = Part(part_id="L00-P0", kind="ply", material="birch_3mm",
                outline=MultiPolygon([Polygon([(0, 0), (100, 0), (100, 400), (0, 400)])]),
                layer_index=0)
    boards = nest_parts([tall], cfg)
    assert len(boards) == 1
    placed = boards[0].parts[0]
    assert placed.placement.rotation_deg == 90.0
    minx, miny, maxx, maxy = placed.placed_geometries()["cut"].bounds
    assert maxx - minx == pytest.approx(400) and maxy - miny == pytest.approx(100)
    assert minx >= -1e-6 and miny >= -1e-6 and maxx <= 495 + 1e-6 and maxy <= 279 + 1e-6
    validate_boards(boards, cfg)      # no overlap / out-of-bed
