"""Lightweight geographic helpers.

Phase 0 only needs a rough real-world extent for a lon/lat bounding box so the
scale math has meters to work with; that uses an equirectangular approximation
evaluated at the box's mean latitude, accurate to well under a percent at
mountain-range scale — far tighter than the map's physical tolerances.

The precise projection used by the layer model is a **local transverse
Mercator centered on the box** (:meth:`BBox.local_crs`). A stock UTM zone would
do for true meters, but grid north in UTM only points straight up on the zone's
central meridian; anywhere else the grid is rotated by the *convergence angle*
(≈ Δλ·sin φ — about 0.8° for the Sierra example, 2° near a zone edge), which
turns a lon/lat box into a visibly tilted quadrilateral. Centering the
projection on the box itself makes the convergence zero at the map's center, so
north is up and the cut frame is an axis-aligned rectangle.
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
        """EPSG code of the UTM zone containing the box center (informational).

        Northern-hemisphere zones are 326xx, southern 327xx. The pipeline no
        longer projects into this zone (see :meth:`local_crs`); it is kept for
        reports and for callers that want a standard, named CRS.
        """
        zone = int((self.mean_lon + 180.0) // 6.0) + 1
        zone = min(max(zone, 1), 60)
        return (32600 if self.mean_lat >= 0 else 32700) + zone

    def local_crs(self) -> str:
        """PROJ string of a transverse Mercator centered on this box.

        Same ellipsoid, scale factor and units as UTM, but with the central
        meridian (and latitude of origin) at the box center, so grid north is
        true north at the map's center and the box projects to an axis-aligned
        near-rectangle instead of a tilted one. Distortion is lower than UTM's
        because the map sits on the central meridian rather than up to 3° away.
        """
        return (f"+proj=tmerc +lat_0={self.mean_lat:.8f} +lon_0={self.mean_lon:.8f} "
                "+k=0.9996 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def crs_string(crs: int | str) -> str:
    """Normalize an EPSG integer or any CRS string to a form pyproj accepts."""
    return f"EPSG:{crs}" if isinstance(crs, int) else str(crs)


def reproject_geom(geom, src_crs: int | str, dst_crs: int | str):
    """Reproject a Shapely geometry between CRSs (EPSG ints or PROJ/WKT strings).

    Imported lazily so modules that never touch vector features don't pull in
    pyproj at import time.
    """
    src, dst = crs_string(src_crs), crs_string(dst_crs)
    if src == dst:
        return geom
    from pyproj import Transformer
    from shapely.ops import transform

    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return transform(transformer.transform, geom)


def projected_frame(bbox: BBox, crs: int | str, samples: int = 64
                    ) -> tuple[float, float, float, float]:
    """Largest axis-aligned rectangle inside ``bbox`` projected into ``crs``.

    A lon/lat box projects to a slightly curved quadrilateral (meridians
    converge, parallels bow). The cut frame must be a true rectangle, so we
    densify each edge, project it, and take the tightest extent: the west edge's
    largest x, the east edge's smallest x, and likewise for south/north. Returns
    ``(x_min, y_min, x_max, y_max)`` in projected units. With :meth:`BBox.local_crs`
    the trimmed slivers are on the order of meters — far below a DEM pixel.
    """
    import numpy as np
    from pyproj import Transformer

    fwd = Transformer.from_crs("EPSG:4326", crs_string(crs), always_xy=True)
    lats = np.linspace(bbox.south, bbox.north, samples)
    lons = np.linspace(bbox.west, bbox.east, samples)
    wx, _ = fwd.transform(np.full(samples, bbox.west), lats)
    ex, _ = fwd.transform(np.full(samples, bbox.east), lats)
    _, sy = fwd.transform(lons, np.full(samples, bbox.south))
    _, ny = fwd.transform(lons, np.full(samples, bbox.north))
    x_min, x_max = float(np.max(wx)), float(np.min(ex))
    y_min, y_max = float(np.max(sy)), float(np.min(ny))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox collapses under projection; it is too large or "
                         "too close to a pole for a local transverse Mercator")
    return x_min, y_min, x_max, y_max


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
