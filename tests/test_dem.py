import numpy as np
import pytest

from toposlicr.dem.base import DemRaster
from toposlicr.dem.cache import make_provider
from toposlicr.dem.synthetic import SyntheticDemProvider
from toposlicr.geo import BBox

BBOX = BBox.from_list([-118.35, 36.52, -118.20, 36.62])


def test_synthetic_dem_shape_and_range():
    dem = SyntheticDemProvider(base_elev_m=500, relief_m=2000, hills=1).fetch(BBOX, 30)
    assert dem.crs == "EPSG:4326"
    assert dem.data.ndim == 2
    lo, hi = dem.elevation_range()
    assert lo >= 500 - 1
    assert hi == pytest.approx(2500, abs=1)


def test_synthetic_is_deterministic():
    a = SyntheticDemProvider(seed=7).fetch(BBOX, 40)
    b = SyntheticDemProvider(seed=7).fetch(BBOX, 40)
    assert np.array_equal(a.data, b.data)


def test_reproject_to_utm_gives_meter_resolution():
    dem = SyntheticDemProvider().fetch(BBOX, 30)
    utm = dem.to_crs(f"EPSG:{BBOX.utm_epsg()}")
    assert utm.crs.endswith(str(BBOX.utm_epsg()))
    rx, ry = utm.resolution
    # Reprojected pixels should be on the order of tens of meters, not degrees.
    assert 5 < rx < 100
    assert 5 < ry < 100


def test_geotiff_roundtrip(tmp_path):
    dem = SyntheticDemProvider().fetch(BBOX, 40)
    p = dem.save_geotiff(tmp_path / "d.tif")
    back = DemRaster.from_geotiff(p)
    assert back.shape == dem.shape
    assert np.allclose(back.data, dem.data, atol=0.5)


def test_make_provider_selects_by_pref():
    assert make_provider("synthetic", BBOX).name == "synthetic"
    assert make_provider("terrarium", BBOX).name == "terrarium"
    # No key + auto → keyless terrarium fallback.
    assert make_provider("auto", BBOX, api_key=None).name == "terrarium"
    # Explicit OT dataset.
    assert make_provider("COP30", BBOX).name == "opentopography"


def test_masked_excludes_nodata():
    data = np.array([[1.0, 2.0], [-9999.0, 4.0]], dtype="float32")
    from affine import Affine
    dem = DemRaster(data=data, transform=Affine.identity(), crs="EPSG:4326",
                    nodata=-9999.0)
    m = dem.masked()
    assert m.count() == 3
    assert float(m.min()) == 1.0
    assert float(m.max()) == 4.0
