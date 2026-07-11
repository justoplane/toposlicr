import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from toposlicr.contour import (
    band_polygon,
    chaikin,
    clean_geometry,
    smooth_dem,
)
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.geo import BBox

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _utm_field():
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    utm = dem.to_crs(f"EPSG:{BBOX.utm_epsg()}")
    z = smooth_dem(utm.data, 2.0)
    xs, ys = utm.pixel_coords()
    return utm, xs, ys, z


def test_smooth_dem_reduces_variance():
    dem = SyntheticDemProvider(hills=3).fetch(BBOX, 30)
    raw_var = np.var(dem.data)
    sm = smooth_dem(dem.data, 3.0)
    assert np.var(sm) <= raw_var


def test_band_polygons_are_valid_and_nested():
    utm, xs, ys, z = _utm_field()
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    g_low = band_polygon(utm, lo + 0.2 * (hi - lo), xs, ys, z)
    g_high = band_polygon(utm, lo + 0.6 * (hi - lo), xs, ys, z)
    assert g_low.is_valid and g_high.is_valid
    assert g_low.area > g_high.area
    # Higher band sits inside the lower one (small buffer absorbs edge noise).
    assert g_high.within(g_low.buffer(1.0))


def test_band_above_max_is_empty():
    utm, xs, ys, z = _utm_field()
    assert band_polygon(utm, float(np.nanmax(z)) + 100, xs, ys, z).is_empty


def test_chaikin_smooths_and_increases_vertices():
    # Many-vertex ring: Chaikin rounds corners with little area change. (On a
    # sharp 4-vertex square the corner-cutting is intentionally more aggressive.)
    ring = [(0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (5, 10), (0, 10), (0, 5)]
    square = Polygon(ring)
    out = chaikin(square, 1)
    assert out.area == pytest.approx(square.area, rel=0.1)
    n_out = len(out.geoms[0].exterior.coords)
    assert n_out > len(ring)


def test_clean_geometry_drops_slivers():
    big = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    sliver = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])  # 1 mm²
    geom = MultiPolygon([big, sliver])
    out = clean_geometry(geom, simplify_tol_mm=0.2, min_area_mm2=16, min_hole_mm2=16)
    assert isinstance(out, MultiPolygon)
    assert len(out.geoms) == 1
    assert out.geoms[0].area == pytest.approx(2500, rel=0.01)


def test_clean_geometry_drops_pinholes():
    outer = [(0, 0), (50, 0), (50, 50), (0, 50)]
    pinhole = [(10, 10), (11, 10), (11, 11), (10, 11)]  # 1 mm² hole
    poly = Polygon(outer, [pinhole])
    out = clean_geometry(poly, simplify_tol_mm=0.0, min_area_mm2=16, min_hole_mm2=16)
    kept = out.geoms[0]
    assert len(kept.interiors) == 0
