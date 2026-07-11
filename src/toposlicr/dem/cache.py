"""Disk cache and provider selection for DEM fetches.

A ``CachingDemProvider`` wraps any provider and keys cached GeoTIFFs on
(provider, dataset, bbox, resolution), so re-runs and iterative curation never
re-hit the network. ``select_provider`` implements the plan's auto-selection:
3DEP inside the US, COP30 globally, with a keyless terrarium fallback.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

from ..geo import BBox, is_in_usa
from .base import DemProvider, DemRaster
from .opentopography import OpenTopographyDemProvider
from .synthetic import SyntheticDemProvider
from .terrarium import TerrariumDemProvider

DEFAULT_CACHE_DIR = Path("cache") / "dem"


def _cache_key(provider: str, bbox: BBox, resolution_m: float, dataset: str) -> str:
    raw = f"{provider}|{dataset}|{bbox.west:.6f},{bbox.south:.6f},{bbox.east:.6f}," \
          f"{bbox.north:.6f}|{resolution_m:.2f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class CachingDemProvider(DemProvider):
    """Wrap a provider with a keyed on-disk GeoTIFF cache."""

    def __init__(self, inner: DemProvider, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                 dataset: str = "auto"):
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.cache_dir = Path(cache_dir)
        self.dataset = dataset

    def fetch(self, bbox: BBox, resolution_m: float) -> DemRaster:
        key = _cache_key(self.inner.name, bbox, resolution_m, self.dataset)
        path = self.cache_dir / f"{key}.tif"
        if path.is_file():
            return DemRaster.from_geotiff(path)
        dem = self.inner.fetch(bbox, resolution_m)
        # A cache write failure must never break a run.
        with contextlib.suppress(Exception):
            dem.save_geotiff(path)
        return dem


def make_provider(dem_pref: str, bbox: BBox, *, api_key: str | None = None) -> DemProvider:
    """Construct a concrete provider from a config ``dem`` preference.

    ``dem_pref`` values: ``auto`` (US→3DEP, else COP30, terrarium fallback),
    ``terrarium``, ``synthetic``, or any OpenTopography dataset id (``COP30``,
    ``USGS10m``, ...).
    """
    pref = (dem_pref or "auto").strip()
    if pref == "synthetic":
        return SyntheticDemProvider()
    if pref == "terrarium":
        return TerrariumDemProvider()
    if pref == "auto":
        if api_key:
            dataset = "USGS10m" if is_in_usa(bbox) else "COP30"
            return OpenTopographyDemProvider(dataset=dataset, api_key=api_key)
        return TerrariumDemProvider()
    # An explicit OpenTopography dataset id.
    return OpenTopographyDemProvider(dataset=pref, api_key=api_key)


def resolve_dem(dem_pref: str, bbox: BBox, resolution_m: float, *,
                api_key: str | None = None, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                use_cache: bool = True) -> DemRaster:
    """Top-level convenience: pick a provider, wrap in cache, fetch."""
    provider = make_provider(dem_pref, bbox, api_key=api_key)
    if use_cache:
        provider = CachingDemProvider(provider, cache_dir=cache_dir, dataset=dem_pref)
    return provider.fetch(bbox, resolution_m)
