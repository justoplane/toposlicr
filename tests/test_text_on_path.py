"""Curved text-on-path for trail names (labels.text_along_path)."""

import pytest
from shapely.geometry import LineString, MultiPolygon

from toposlicr.labels import text_along_path, trail_name_geometry

CAP = 4.0

# A long, gentle upward curve (~205 mm) — smooth enough to letter along.
GENTLE_ARC = LineString([(0, 0), (50, 8), (100, 20), (150, 36), (200, 56)])


def _parts(geom):
    return list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]


def test_curves_text_along_a_gentle_arc():
    out = text_along_path("Summit Trail", GENTLE_ARC, cap_height_mm=CAP)
    assert out is not None and not out.is_empty
    # The name distributes along the path, not stacked at one point.
    assert len(_parts(out)) > 3
    minx, miny, maxx, maxy = out.bounds
    assert (maxx - minx) > 4 * CAP                      # spread out horizontally
    # Sits close to the path (offset just above it).
    assert GENTLE_ARC.distance(out.centroid) < 3 * CAP


def test_text_advances_along_the_line():
    out = text_along_path("Summit Trail", GENTLE_ARC, cap_height_mm=CAP)
    assert out is not None
    minx, _, maxx, _ = out.bounds
    # Centred text on a ~200 mm-wide arc lands around its middle and spans a chunk.
    cx = (minx + maxx) / 2.0
    assert 70 < cx < 130
    assert (maxx - minx) > 20                           # letters run along the path


def test_short_segment_falls_back():
    # A long name cannot fit a 5 mm line → None (caller uses horizontal label).
    out = text_along_path("A Very Long Trail Name", LineString([(0, 0), (5, 0)]),
                          cap_height_mm=CAP)
    assert out is None


def test_sharp_switchback_falls_back():
    zig = LineString([(0, 0), (40, 5), (0, 10), (40, 15), (0, 20)])  # ~166° turns
    assert text_along_path("Long Loop Trail", zig, cap_height_mm=CAP) is None


def test_straight_line_text_is_horizontal_and_above():
    out = text_along_path("Trail", LineString([(0, 0), (100, 0)]), cap_height_mm=CAP)
    assert out is not None and not out.is_empty
    minx, miny, maxx, maxy = out.bounds
    assert miny > 0                                     # offset above the line (y=0)
    assert (maxy - miny) < 2 * CAP                      # one upright line of text


def test_empty_and_degenerate_inputs():
    assert text_along_path("", GENTLE_ARC, cap_height_mm=CAP) is None
    assert text_along_path("Trail", LineString(), cap_height_mm=CAP) is None


def test_reads_left_to_right_when_line_runs_leftward():
    # A right-to-left line is flipped internally so text still reads L→R.
    ltr = text_along_path("Trail", LineString([(0, 0), (100, 0)]), cap_height_mm=CAP)
    rtl = text_along_path("Trail", LineString([(100, 0), (0, 0)]), cap_height_mm=CAP)
    assert ltr is not None and rtl is not None
    # Both span the same horizontal extent (direction normalized).
    assert ltr.bounds[0] == pytest.approx(rtl.bounds[0], abs=1.0)
    assert ltr.bounds[2] == pytest.approx(rtl.bounds[2], abs=1.0)


def test_trail_name_geometry_uses_curve_then_fallback():
    curved = trail_name_geometry("Ridge Path", GENTLE_ARC, cap_height_mm=CAP,
                                 curved=True)
    horizontal = trail_name_geometry("Ridge Path", GENTLE_ARC, cap_height_mm=CAP,
                                     curved=False)
    assert not curved.is_empty and not horizontal.is_empty
    # Curved follows the arc's rise, so its vertical extent differs from the flat one.
    assert curved.bounds != horizontal.bounds
    assert (curved.bounds[3] - curved.bounds[1]) > (horizontal.bounds[3]
                                                    - horizontal.bounds[1])


def test_multilinestring_uses_longest_component():
    from shapely.geometry import MultiLineString
    mls = MultiLineString([LineString([(0, 0), (3, 0)]),      # too short
                           [(0, 20), (60, 20), (120, 20)]])   # long, straight
    out = text_along_path("Trail", mls, cap_height_mm=CAP)
    assert out is not None
    # Placed along the long (y≈20) component, not the short one.
    assert out.centroid.y > 15
