"""Projection + cut-frame tests: the map must come out north-up and rectangular.

A lon/lat box is not axis-aligned in a UTM zone: grid north only matches true
north on the zone's central meridian, and elsewhere the grid is rotated by the
convergence angle (~0.8° for the Sierra example). Projecting the DEM into UTM
therefore produced a visibly tilted base slab. The layer model now projects
into a transverse Mercator centered on the box and clips every band to the
rectangle inscribed in the projected box.
"""

import math

import numpy as np
import pytest
from affine import Affine
from shapely import minimum_rotated_rectangle
from shapely.geometry import Point, Polygon, box

from toposlicr.config import parse_config
from toposlicr.dem.base import DemRaster
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.geo import BBox, projected_frame, reproject_geom
from toposlicr.layers import build_layer_model

# ~1.3° west of UTM zone 11's central meridian → ~0.8° grid convergence.
BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def _model(bbox: BBox = BBOX, **phys):
    cfg = parse_config({
        "region": {"bbox": [bbox.west, bbox.south, bbox.east, bbox.north],
                   "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0,
                     "exaggeration": 1.5, **phys},
    })
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(bbox, 30)
    return build_layer_model(dem, cfg, bbox)


def _dense_box(bbox: BBox, n: int = 50) -> Polygon:
    """The lon/lat box as a densified ring, so projection can curve its edges."""
    lons = np.linspace(bbox.west, bbox.east, n)
    lats = np.linspace(bbox.south, bbox.north, n)
    ring = ([(x, bbox.south) for x in lons] + [(bbox.east, y) for y in lats]
            + [(x, bbox.north) for x in lons[::-1]] + [(bbox.west, y) for y in lats[::-1]])
    return Polygon(ring)


def _tilt_deg(geom) -> float:
    """Angle between the geometry's minimum rotated rectangle and the axes."""
    pts = list(minimum_rotated_rectangle(geom).exterior.coords)[:4]
    (x0, y0), (x1, y1) = max(zip(pts, pts[1:] + pts[:1], strict=True),
                             key=lambda e: math.dist(*e))
    ang = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 90
    return min(ang, 90 - ang)


def test_utm_tilts_the_box_but_the_local_crs_does_not():
    dense = _dense_box(BBOX)
    assert _tilt_deg(reproject_geom(dense, 4326, BBOX.utm_epsg())) > 0.5   # the bug
    assert _tilt_deg(reproject_geom(dense, 4326, BBOX.local_crs())) < 0.02


def test_local_crs_is_centered_and_north_up():
    crs = BBOX.local_crs()
    center = reproject_geom(Point(BBOX.mean_lon, BBOX.mean_lat), 4326, crs)
    north = reproject_geom(Point(BBOX.mean_lon, BBOX.north), 4326, crs)
    assert abs(center.x) < 1e-6 and abs(center.y) < 1e-6
    assert abs(north.x) < 1e-3 and north.y > 0


def test_reproject_geom_accepts_epsg_ints_and_crs_strings():
    pt = Point(BBOX.mean_lon, BBOX.mean_lat)
    via_int = reproject_geom(pt, 4326, BBOX.utm_epsg())
    via_str = reproject_geom(pt, "EPSG:4326", f"EPSG:{BBOX.utm_epsg()}")
    assert via_int.equals_exact(via_str, 1e-6)
    assert reproject_geom(pt, 4326, "EPSG:4326") is pt


def test_projected_frame_matches_real_extent():
    x0, y0, x1, y1 = projected_frame(BBOX, BBOX.local_crs())
    width_m, height_m = BBOX.extent_m()
    assert (x1 - x0) == pytest.approx(width_m, rel=0.005)
    assert (y1 - y0) == pytest.approx(height_m, rel=0.005)
    # The frame lies inside the projected box (edges trimmed inward, never out).
    projected = reproject_geom(_dense_box(BBOX), 4326, BBOX.local_crs())
    assert box(x0, y0, x1, y1).within(projected.buffer(0.01))


def test_base_slab_is_the_full_frame_and_not_tilted():
    model = _model()
    w, h = model.model_width_mm, model.model_height_mm
    base = model.footprint(0)
    assert _tilt_deg(base) < 0.05
    assert base.bounds == pytest.approx((0.0, 0.0, w, h), abs=0.05)
    assert base.symmetric_difference(box(0, 0, w, h)).area < 1e-3 * w * h
    # Corners are sharp: smoothing must not chamfer the frame.
    for x in (0.15, w - 0.15):
        for y in (0.15, h - 0.15):
            assert base.contains(Point(x, y))
    assert model.crs.startswith("+proj=tmerc")
    assert model.utm_epsg == BBOX.utm_epsg()


def test_model_aspect_follows_real_extent():
    model = _model()
    width_m, height_m = BBOX.extent_m()
    expected_h = model.model_width_mm * height_m / width_m
    assert model.model_height_mm == pytest.approx(expected_h, rel=0.005)


def test_features_land_at_the_frame_center():
    from toposlicr.symbology import _to_model

    model = _model()
    pt = _to_model(model, Point(BBOX.mean_lon, BBOX.mean_lat))
    assert pt.x == pytest.approx(model.model_width_mm / 2, abs=0.5)
    assert pt.y == pytest.approx(model.model_height_mm / 2, abs=0.5)


def test_projected_dem_input_is_handled():
    """A DEM that already arrives projected (e.g. terrarium's Web Mercator)."""
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    merc = dem.to_crs("EPSG:3857")
    cfg = parse_config({
        "region": {"bbox": [BBOX.west, BBOX.south, BBOX.east, BBOX.north],
                   "dem": "synthetic"},
        "physical": {"model_width_mm": 300, "ply_thickness_mm": 3.0, "exaggeration": 1.5},
    })
    model = build_layer_model(merc, cfg, BBOX)
    base = model.footprint(0)
    assert _tilt_deg(base) < 0.05
    assert base.area == pytest.approx(model.model_width_mm * model.model_height_mm,
                                      rel=1e-3)


def test_pad_edges_replicates_border_and_shifts_origin():
    data = np.arange(12, dtype="float32").reshape(3, 4)
    dem = DemRaster(data=data, transform=Affine.translation(100, 200) * Affine.scale(10, -10),
                    crs="EPSG:32611")
    padded = dem.pad_edges(1)
    assert padded.shape == (5, 6)
    assert padded.transform * (1, 1) == dem.transform * (0, 0)
    assert padded.data[0, 0] == data[0, 0] and padded.data[-1, -1] == data[-1, -1]
    assert np.array_equal(padded.data[1:-1, 1:-1], data)
    assert dem.pad_edges(0) is dem
