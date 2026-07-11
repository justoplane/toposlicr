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
