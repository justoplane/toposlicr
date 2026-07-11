"""Lightweight geographic helpers.

Phase 0 only needs a rough real-world extent for a lon/lat bounding box so the
scale math has meters to work with. The precise UTM reprojection lives in
Phase 2 (``pyproj``); here we use an equirectangular approximation evaluated at
the box's mean latitude, which is accurate to well under a percent at
mountain-range scale — far tighter than the map's physical tolerances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# WGS84 mean Earth radius (meters). Good enough for a scale estimate.
_EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class BBox:
    """A geographic bounding box in decimal degrees, ordered W, S, E, N."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.east <= self.west:
            raise ValueError(f"bbox east ({self.east}) must be > west ({self.west})")
        if self.north <= self.south:
            raise ValueError(f"bbox north ({self.north}) must be > south ({self.south})")
        for name, lon in (("west", self.west), ("east", self.east)):
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"bbox {name} longitude {lon} out of range [-180, 180]")
        for name, lat in (("south", self.south), ("north", self.north)):
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"bbox {name} latitude {lat} out of range [-90, 90]")

    @classmethod
    def from_list(cls, values: list[float]) -> BBox:
        if len(values) != 4:
            raise ValueError(f"bbox needs exactly 4 values [W, S, E, N], got {len(values)}")
        return cls(*(float(v) for v in values))

    @property
    def mean_lat(self) -> float:
        return (self.south + self.north) / 2.0

    @property
    def mean_lon(self) -> float:
        return (self.west + self.east) / 2.0

    def extent_m(self) -> tuple[float, float]:
        """Return the (width, height) of the box in meters.

        Width is the east–west span at the mean latitude; height is the
        north–south span. Both use the equirectangular approximation.
        """
        lat_rad = math.radians(self.mean_lat)
        m_per_deg_lat = _EARTH_RADIUS_M * math.pi / 180.0
        m_per_deg_lon = m_per_deg_lat * math.cos(lat_rad)
        width_m = (self.east - self.west) * m_per_deg_lon
        height_m = (self.north - self.south) * m_per_deg_lat
        return width_m, height_m

    def utm_epsg(self) -> int:
        """EPSG code of the UTM zone containing the box center.

        Northern-hemisphere zones are 326xx, southern 327xx. Reprojecting into
        this zone gives true meters with minimal distortion at map scale.
        """
        zone = int((self.mean_lon + 180.0) // 6.0) + 1
        zone = min(max(zone, 1), 60)
        return (32600 if self.mean_lat >= 0 else 32700) + zone


def reproject_geom(geom, src_epsg: int, dst_epsg: int):
    """Reproject a Shapely geometry between EPSG codes (e.g. 4326 → UTM).

    Imported lazily so modules that never touch vector features don't pull in
    pyproj at import time.
    """
    if src_epsg == dst_epsg:
        return geom
    from pyproj import Transformer
    from shapely.ops import transform

    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}",
                                       always_xy=True)
    return transform(transformer.transform, geom)


def is_in_usa(bbox: BBox) -> bool:
    """Rough check whether a box lies fully within the contiguous US + AK/HI.

    Used only to pick default data providers (3DEP/GNIS/NHD vs COP30/OSM); the
    bounds are generous and a false negative merely falls back to global sources.
    """
    # Contiguous US, plus generous Alaska and Hawaii windows.
    windows = (
        (-125.0, 24.0, -66.5, 49.5),   # CONUS
        (-170.0, 51.0, -129.0, 72.0),  # Alaska
        (-161.0, 18.5, -154.0, 22.5),  # Hawaii
    )
    for w, s, e, n in windows:
        if bbox.west >= w and bbox.east <= e and bbox.south >= s and bbox.north <= n:
            return True
    return False
