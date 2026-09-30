"""DEM raster model and the pluggable provider interface.

``DemRaster`` is the single in-memory representation of elevation data that
flows through the pipeline. ``DemProvider`` is the abstraction the plan calls
for (Section 1.1): concrete backends (OpenTopography, AWS terrain tiles, a
synthetic generator for tests) all return a ``DemRaster`` for a bounding box,
and the rest of the pipeline neither knows nor cares which one produced it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine

from ..geo import BBox

# Sentinel CRS for flat/fictional data (terrain bundles): the pipeline skips
# reprojection and treats the raster's units as linear world units.
FLAT_CRS = "FLAT"


@dataclass
class DemRaster:
    """A north-up elevation grid with georeferencing.

    ``data`` is a 2D float array of elevations in meters, row 0 at the north
    edge. ``transform`` maps (col, row) pixel coordinates to world coordinates
    in ``crs``. ``nodata`` cells are excluded from statistics and contouring.
    """

    data: np.ndarray
    transform: Affine
    crs: str
    nodata: float | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"DEM data must be 2D, got shape {self.data.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # (rows, cols)

    @property
    def resolution(self) -> tuple[float, float]:
        """(x, y) pixel size in the raster's CRS units."""
        return (abs(self.transform.a), abs(self.transform.e))

    def masked(self) -> np.ndarray:
        """Elevations as a masked array with nodata masked out."""
        if self.nodata is None:
            return np.ma.masked_invalid(self.data)
        return np.ma.masked_invalid(np.ma.masked_equal(self.data, self.nodata))

    def elevation_range(self) -> tuple[float, float]:
        """(min, max) finite elevation in meters, ignoring nodata."""
        m = self.masked()
        if m.count() == 0:
            raise ValueError("DEM has no valid elevation cells")
        return float(m.min()), float(m.max())

    def pixel_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """1D arrays of cell-center x and y world coordinates (for contouring)."""
        rows, cols = self.shape
        xs = np.array([self.transform * (c + 0.5, 0.5) for c in range(cols)])[:, 0]
        ys = np.array([self.transform * (0.5, r + 0.5) for r in range(rows)])[:, 1]
        return xs, ys

    def to_crs(self, dst_crs: str, resolution: float | None = None) -> DemRaster:
        """Reproject to ``dst_crs`` (an EPSG code or PROJ string, e.g. the
        box-centered transverse Mercator from ``BBox.local_crs``) so units are
        true meters.

        ``resolution`` optionally sets the output pixel size in destination
        units; by default rasterio picks one preserving the pixel count.
        """
        # Imported lazily so the geometry-only parts of the package don't need
        # the full rasterio warp stack at import time.
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        if dst_crs == self.crs:
            return self
        rows, cols = self.shape
        left, top = self.transform * (0, 0)
        right, bottom = self.transform * (cols, rows)
        dst_transform, dst_w, dst_h = calculate_default_transform(
            self.crs, dst_crs, cols, rows, left, bottom, right, top,
            resolution=resolution,
        )
        dst = np.full((dst_h, dst_w), np.nan, dtype="float32")
        reproject(
            source=self.data.astype("float32"),
            destination=dst,
            src_transform=self.transform,
            src_crs=self.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=self.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        return DemRaster(data=dst, transform=dst_transform, crs=dst_crs, nodata=None)

    def pad_edges(self, cells: int = 1) -> DemRaster:
        """Grow the grid by ``cells`` on every side, replicating edge values.

        Contours run between cell *centers*, so a band can never reach closer
        than half a pixel to the raster edge. Padding by one replicated cell
        pushes that limit half a pixel *outside* the original extent, so bands
        that touch the map edge can be clipped flush to the frame instead of
        stopping a fraction of a millimeter short of it. Nodata (NaN) at the edge
        is replicated too, so missing data never turns into fake terrain.
        """
        if cells <= 0:
            return self
        data = np.pad(self.data, cells, mode="edge")
        transform = self.transform * Affine.translation(-cells, -cells)
        return DemRaster(data=data, transform=transform, crs=self.crs,
                         nodata=self.nodata)

    def save_geotiff(self, path: str | Path) -> Path:
        import rasterio

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows, cols = self.shape
        with rasterio.open(
            p, "w", driver="GTiff", height=rows, width=cols, count=1,
            dtype="float32", crs=self.crs, transform=self.transform,
            nodata=self.nodata if self.nodata is not None else np.nan,
            compress="deflate",
        ) as dst:
            dst.write(self.data.astype("float32"), 1)
        return p

    @classmethod
    def from_geotiff(cls, path: str | Path) -> DemRaster:
        import rasterio

        with rasterio.open(path) as src:
            data = src.read(1).astype("float32")
            return cls(
                data=data,
                transform=src.transform,
                crs=str(src.crs),
                nodata=src.nodata,
            )


class DemProvider(ABC):
    """A source of elevation data for a bounding box."""

    #: short stable id used in cache keys and logs
    name: str = "abstract"

    @abstractmethod
    def fetch(self, bbox: BBox, resolution_m: float) -> DemRaster:
        """Return a DEM covering ``bbox`` at roughly ``resolution_m`` meters.

        Implementations return data in any georeferenced CRS (lon/lat or a
        projected one); the pipeline reprojects into a box-centered transverse
        Mercator. ``resolution_m`` is a target, not a guarantee.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<DemProvider {self.name}>"
