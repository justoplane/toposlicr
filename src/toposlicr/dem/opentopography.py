"""OpenTopography Global/USGS DEM provider.

One HTTP endpoint returns a GeoTIFF for a bounding box (plan Section 1.1). A
free API key is required (rate-limited, but results are cached locally so the
limit is rarely a concern). The key is read from the ``OPENTOPOGRAPHY_API_KEY``
environment variable unless passed explicitly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import requests

from ..geo import BBox, is_in_usa
from .base import DemProvider, DemRaster

_GLOBAL_URL = "https://portal.opentopography.org/API/globaldem"
_USGS_URL = "https://portal.opentopography.org/API/usgsdem"

# Map our short dataset ids onto OpenTopography's parameters.
_GLOBAL_DATASETS = {"COP30": "COP30", "SRTMGL1": "SRTMGL1", "NASADEM": "NASADEM"}
_USGS_DATASETS = {"USGS10m": "USGS10m", "USGS1m": "USGS1m", "USGS30m": "USGS30m"}


class OpenTopographyDemProvider(DemProvider):
    """DEM via the OpenTopography REST API (COP30/SRTM globally, 3DEP in the US)."""

    name = "opentopography"

    def __init__(self, dataset: str = "auto", api_key: str | None = None,
                 session: requests.Session | None = None, timeout: float = 120.0):
        self.dataset = dataset
        self._api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
        self._session = session or requests.Session()
        self._timeout = timeout

    def _resolve_dataset(self, bbox: BBox) -> str:
        if self.dataset != "auto":
            return self.dataset
        return "USGS10m" if is_in_usa(bbox) else "COP30"

    def fetch(self, bbox: BBox, resolution_m: float) -> DemRaster:
        if not self._api_key:
            raise RuntimeError(
                "OpenTopography requires an API key; set OPENTOPOGRAPHY_API_KEY "
                "or use the keyless 'terrarium' provider."
            )
        dataset = self._resolve_dataset(bbox)
        if dataset in _USGS_DATASETS:
            url, ds_param = _USGS_URL, _USGS_DATASETS[dataset]
        elif dataset in _GLOBAL_DATASETS:
            url, ds_param = _GLOBAL_URL, _GLOBAL_DATASETS[dataset]
        else:
            raise ValueError(f"unknown OpenTopography dataset '{dataset}'")

        params = {
            "demtype": ds_param,
            "west": bbox.west, "south": bbox.south,
            "east": bbox.east, "north": bbox.north,
            "outputFormat": "GTiff",
            "API_Key": self._api_key,
        }
        resp = self._session.get(url, params=params, timeout=self._timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"OpenTopography {dataset} request failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
        # rasterio needs a path/file; write the GeoTIFF bytes to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)
        try:
            return DemRaster.from_geotiff(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
