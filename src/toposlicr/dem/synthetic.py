"""A synthetic DEM provider — no network, no keys, deterministic.

This is what makes the whole pipeline runnable and unit-testable offline: it
generates a smooth analytic terrain (a sum of Gaussian hills plus a gentle
plane) over the requested bbox. Because the surface is known in closed form,
tests can assert exact properties (e.g. contour bands are nested rings).
"""

from __future__ import annotations

import numpy as np
from affine import Affine

from ..geo import BBox
from .base import DemProvider, DemRaster


class SyntheticDemProvider(DemProvider):
    """Deterministic Gaussian-hill terrain for tests and offline demos."""

    name = "synthetic"

    def __init__(self, base_elev_m: float = 500.0, relief_m: float = 2000.0,
                 hills: int = 3, seed: int = 0):
        self.base_elev_m = base_elev_m
        self.relief_m = relief_m
        self.hills = hills
        self.seed = seed

    def fetch(self, bbox: BBox, resolution_m: float) -> DemRaster:
        width_m, height_m = bbox.extent_m()
        cols = max(16, int(round(width_m / max(resolution_m, 1.0))))
        rows = max(16, int(round(height_m / max(resolution_m, 1.0))))
        # Cap grid size so tests stay fast; detail beyond this is irrelevant here.
        cols, rows = min(cols, 512), min(rows, 512)

        xs = np.linspace(0.0, 1.0, cols)
        ys = np.linspace(0.0, 1.0, rows)
        gx, gy = np.meshgrid(xs, ys)

        rng = np.random.default_rng(self.seed)
        z = np.zeros((rows, cols), dtype="float64")
        # A dominant central hill guarantees clean nested bands, plus a few
        # deterministic satellites for shape variety.
        centers = [(0.5, 0.5, 0.28)]
        for _ in range(max(0, self.hills - 1)):
            centers.append((rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                            rng.uniform(0.08, 0.18)))
        for i, (cx, cy, sigma) in enumerate(centers):
            amp = 1.0 if i == 0 else rng.uniform(0.3, 0.7)
            z += amp * np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2)))

        z = z / z.max()
        elev = self.base_elev_m + self.relief_m * z

        # North-up transform in EPSG:4326.
        px = (bbox.east - bbox.west) / cols
        py = (bbox.north - bbox.south) / rows
        transform = Affine.translation(bbox.west, bbox.north) * Affine.scale(px, -py)
        return DemRaster(data=elev.astype("float32"), transform=transform,
                         crs="EPSG:4326", nodata=None)
