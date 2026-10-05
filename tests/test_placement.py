"""Open-space label placement: sized, rotated, curved text in the nearest room."""

import math

import pytest
from shapely.affinity import rotate
from shapely.geometry import LineString, Point, box

from toposlicr.labels import LabelRequest, place_labels
from toposlicr.placement import _upright


def _ribbon(angle_deg, length=200, width=12, center=(100, 100)):
    """A straight band `width` mm across, rotated to `angle_deg`."""
    cx, cy = center
    r = box(cx - length / 2, cy - width / 2, cx + length / 2, cy + width / 2)
    return rotate(r, angle_deg, origin=(cx, cy))


def _text_angle(geom):
    """Orientation of a placed label from its minimum rotated rectangle."""
    from shapely import minimum_rotated_rectangle
    pts = list(minimum_rotated_rectangle(geom).exterior.coords)[:4]
    (x0, y0), (x1, y1) = max(zip(pts, pts[1:] + pts[:1], strict=True),
                             key=lambda e: math.dist(*e))
    return _upright(math.degrees(math.atan2(y1 - y0, x1 - x0)))


def test_thin_diagonal_ribbon_gets_rotated_text():
    band = _ribbon(35)
    req = LabelRequest("Ridgeline", Point(100, 100), layer_index=0, cap_height_mm=4.0)
    placed = place_labels([req], {0: band})
    assert placed[0].placed
    assert placed[0].geometry.within(band.buffer(1e-6))
    assert abs(placed[0].rotation_deg - 35) < 12
    assert abs(_text_angle(placed[0].geometry) - 35) < 12
    assert placed[0].cap_height_mm == pytest.approx(4.0)     # no shrink needed


def test_open_plateau_keeps_text_horizontal_near_the_anchor():
    band = box(0, 0, 300, 300)
    req = LabelRequest("Big Meadow", Point(150, 150), layer_index=0, cap_height_mm=4.0)
    placed = place_labels([req], {0: band})
    p = placed[0]
    assert p.placed and p.rotation_deg == 0.0 and not p.curved
    minx, miny, maxx, maxy = p.geometry.bounds
    assert math.dist(((minx + maxx) / 2, (miny + maxy) / 2), (150, 150)) < 1.0
    assert p.leader is None


def test_cramped_band_uses_the_size_ladder_but_not_below_minimum():
    # 6 mm wide: 4 mm caps don't fit (glyph bbox ~5.3 mm with descenders + gap),
    # 70% does — and a 2.5 mm floor stays respected.
    band = _ribbon(0, width=6.2)
    req = LabelRequest("Narrow Bench", Point(100, 100), layer_index=0, cap_height_mm=4.0,
                       min_cap_height_mm=2.5)
    placed = place_labels([req], {0: band})
    assert placed[0].placed
    assert 2.5 <= placed[0].cap_height_mm < 4.0
    # Raise the floor above what fits → unplaced rather than illegible.
    req2 = LabelRequest("Narrow Bench", Point(100, 100), layer_index=0, cap_height_mm=4.0,
                        min_cap_height_mm=3.9)
    assert place_labels([req2], {0: band})[0].placed is False


def test_short_text_is_used_when_the_full_label_has_no_room():
    band = _ribbon(0, length=40, width=14)          # room for a name, not name+elevation
    req = LabelRequest("Mount Test 4,000 m", Point(100, 100), layer_index=0,
                       cap_height_mm=4.0, short_text="Mount Test")
    placed = place_labels([req], {0: band})
    assert placed[0].placed
    assert placed[0].rendered_text == "Mount Test"
    assert placed[0].text == "Mount Test 4,000 m"      # key stays the full label


def test_summit_falls_back_to_a_lower_band_with_a_leader():
    tiny = Point(100, 100).buffer(3)
    big = box(0, 0, 200, 200).difference(Point(100, 100).buffer(20))
    req = LabelRequest("Summit", Point(100, 100), layer_index=2, is_point=True,
                       cap_height_mm=4.0)
    placed = place_labels([req], {2: tiny, 1: big})
    p = placed[0]
    assert p.placed and p.layer_index == 1
    assert p.geometry.within(big.buffer(1e-6))
    # The leader is scored only where it lies on the label's own exposed band.
    assert p.leader is not None and p.leader.within(big.buffer(1e-6))


def test_labels_avoid_obstacles_and_each_other():
    band = box(0, 0, 200, 60)
    river = LineString([(0, 30), (200, 30)])
    reqs = [LabelRequest("Alpha", Point(100, 30), layer_index=0, cap_height_mm=4.0),
            LabelRequest("Beta", Point(100, 30), layer_index=0, cap_height_mm=4.0)]
    placed = place_labels(reqs, {0: band}, obstacles={0: river})
    assert all(p.placed for p in placed)
    for p in placed:
        assert not p.geometry.intersects(river.buffer(1.0))
    assert not placed[0].geometry.intersects(placed[1].geometry)


def test_long_name_curves_along_a_bending_ribbon():
    # A 12 mm wide ribbon following a 90° arc of radius 60: no straight run is
    # long enough for the name, so it must flow along the curve.
    arc = LineString([(100 + 60 * math.cos(t), 100 + 60 * math.sin(t))
                      for t in [math.radians(a) for a in range(0, 91, 3)]])
    band = arc.buffer(6, cap_style="flat")
    req = LabelRequest("Long Winding Shoulder Name", Point(100 + 60 * math.cos(0.8),
                                                          100 + 60 * math.sin(0.8)),
                       layer_index=0, cap_height_mm=3.0, min_cap_height_mm=3.0)
    placed = place_labels([req], {0: band}, curved=True)
    assert placed[0].placed and placed[0].curved
    assert placed[0].geometry.within(band.buffer(1e-6))
    # Without curving there is no straight room at this size.
    assert place_labels([req], {0: band}, curved=False)[0].placed is False


def test_priority_places_important_features_first():
    band = _ribbon(0, length=40, width=14)           # room for exactly one label
    low = LabelRequest("Minor", Point(100, 100), layer_index=0, cap_height_mm=4.0,
                       priority=0.1, min_cap_height_mm=4.0)
    high = LabelRequest("Major", Point(100, 100), layer_index=0, cap_height_mm=4.0,
                        priority=0.9, min_cap_height_mm=4.0)
    placed = {p.text: p for p in place_labels([low, high], {0: band},
                                              search_radius_mm=10)}
    assert placed["Major"].placed and not placed["Minor"].placed


def test_forced_override_is_honored_verbatim():
    band = box(0, 0, 200, 200)
    req = LabelRequest("Moved", Point(20, 20), layer_index=0, cap_height_mm=4.0,
                       forced_center=(150, 150))
    p = place_labels([req], {0: band})[0]
    minx, miny, maxx, maxy = p.geometry.bounds
    assert ((minx + maxx) / 2, (miny + maxy) / 2) == pytest.approx((150, 150), abs=0.01)


def test_upright_folding():
    assert _upright(0) == 0 and _upright(45) == 45 and _upright(-45) == -45
    assert _upright(135) == -45 and _upright(-135) == 45 and _upright(180) == 0
    assert _upright(90) == 90 and _upright(-90) == 90


def test_lake_in_a_bowl_may_step_up_to_the_terrace_above():
    """Home band is a thin ring (no room); the layer above has a wide band."""
    ring = Point(100, 100).buffer(30).difference(Point(100, 100).buffer(26))
    above = box(0, 0, 200, 200).difference(Point(100, 100).buffer(34))
    req = LabelRequest("Bowl Lake", Point(100, 100), layer_index=1, is_point=True,
                       cap_height_mm=4.0, allow_higher=True)
    p = place_labels([req], {1: ring, 2: above})[0]
    assert p.placed and p.layer_index == 2 and p.leader is not None
    # Without permission it stays unplaced (nothing below, ring too thin).
    req2 = LabelRequest("Bowl Lake", Point(100, 100), layer_index=1, is_point=True,
                        cap_height_mm=4.0)
    assert place_labels([req2], {1: ring, 2: above})[0].placed is False
