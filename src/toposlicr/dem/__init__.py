"""DEM acquisition subsystem (plan Section 1.1)."""

from __future__ import annotations

from .base import DemProvider, DemRaster
from .cache import CachingDemProvider, make_provider, resolve_dem
from .opentopography import OpenTopographyDemProvider
from .synthetic import SyntheticDemProvider
from .terrarium import TerrariumDemProvider

__all__ = [
    "DemProvider",
    "DemRaster",
    "CachingDemProvider",
    "make_provider",
    "resolve_dem",
    "OpenTopographyDemProvider",
    "SyntheticDemProvider",
    "TerrariumDemProvider",
]
