"""AWS Terrain Tiles provider — global, keyless.

Fetches Mapzen/AWS "terrarium" RGB-encoded elevation PNG tiles from the public
``elevation-tiles-prod`` S3 bucket, decodes them to meters, mosaics the tiles
covering the bbox and returns a DEM in Web Mercator (EPSG:3857). No API key is
required, which makes this the zero-setup default network source.

Terrarium encoding: ``elevation_m = R*256 + G + B/256 - 32768``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import requests
from affine import Affine

from ..geo import BBox
from .base import DemProvider, DemRaster

_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_TILE_PX = 256
_MERC_MAX = 20037508.342789244  # half the Web Mercator world extent (m)


def _mercator_resolution(lat_deg: float, zoom: int) -> float:
    """Ground resolution (m/px) of a web-mercator tile at a latitude and zoom."""
    return 156543.03392804097 * math.cos(math.radians(lat_deg)) / (2 ** zoom)


def _zoom_for_resolution(bbox: BBox, resolution_m: float) -> int:
    for z in range(0, 16):
        if _mercator_resolution(bbox.mean_lat, z) <= resolution_m:
            return z
    return 15


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(x, n - 1)), max(0, min(y, n - 1))


class TerrariumDemProvider(DemProvider):
    """Global keyless DEM from AWS terrain tiles."""

    name = "terrarium"

    def __init__(self, session: requests.Session | None = None, timeout: float = 30.0):
        self._session = session or requests.Session()
        self._timeout = timeout

    def _fetch_tile(self, z: int, x: int, y: int) -> np.ndarray:
        from PIL import Image

        url = _TILE_URL.format(z=z, x=x, y=y)
        resp = self._session.get(url, timeout=self._timeout)
        resp.raise_for_status()
        img = np.asarray(Image.open(io.BytesIO(resp.content)).convert("RGB"),
                         dtype="float64")
        return img[:, :, 0] * 256.0 + img[:, :, 1] + img[:, :, 2] / 256.0 - 32768.0

    def fetch(self, bbox: BBox, resolution_m: float) -> DemRaster:
        zoom = _zoom_for_resolution(bbox, resolution_m)
        n = 2 ** zoom
        x0, y0 = _lonlat_to_tile(bbox.west, bbox.north, zoom)  # NW tile
        x1, y1 = _lonlat_to_tile(bbox.east, bbox.south, zoom)  # SE tile
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        rows = (y1 - y0 + 1) * _TILE_PX
        cols = (x1 - x0 + 1) * _TILE_PX
        mosaic = np.full((rows, cols), np.nan, dtype="float64")
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                tile = self._fetch_tile(zoom, tx, ty)
                r = (ty - y0) * _TILE_PX
                c = (tx - x0) * _TILE_PX
                mosaic[r:r + _TILE_PX, c:c + _TILE_PX] = tile

        # Mosaic georeferencing in EPSG:3857.
        tile_merc = 2 * _MERC_MAX / n
        px = tile_merc / _TILE_PX
        origin_x = -_MERC_MAX + x0 * tile_merc
        origin_y = _MERC_MAX - y0 * tile_merc
        transform = Affine.translation(origin_x, origin_y) * Affine.scale(px, -px)

        dem = DemRaster(data=mosaic.astype("float32"), transform=transform,
                        crs="EPSG:3857", nodata=None)
        return _crop_to_bbox(dem, bbox)


def _crop_to_bbox(dem: DemRaster, bbox: BBox) -> DemRaster:
    """Window-crop a 3857 mosaic to the bbox extent (reprojected to 3857)."""
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds(
        "EPSG:4326", dem.crs, bbox.west, bbox.south, bbox.east, bbox.north
    )
    inv = ~dem.transform
    c0, r0 = inv * (left, top)
    c1, r1 = inv * (right, bottom)
    col_lo, col_hi = sorted((int(math.floor(c0)), int(math.ceil(c1))))
    row_lo, row_hi = sorted((int(math.floor(r0)), int(math.ceil(r1))))
    rows, cols = dem.shape
    col_lo, col_hi = max(0, col_lo), min(cols, col_hi)
    row_lo, row_hi = max(0, row_lo), min(rows, row_hi)
    if col_hi <= col_lo or row_hi <= row_lo:
        return dem  # degenerate crop; return uncropped rather than empty
    window = dem.data[row_lo:row_hi, col_lo:col_hi]
    new_transform = dem.transform * Affine.translation(col_lo, row_lo)
    return DemRaster(data=window, transform=new_transform, crs=dem.crs,
                     nodata=dem.nodata)
