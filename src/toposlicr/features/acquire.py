"""Feature acquisition with an on-disk cache (mirrors the DEM cache).

Fetches OSM features for a bbox via Overpass and caches the raw JSON response,
keyed on the bbox, so re-runs and the ``features.csv`` curation loop never re-hit
the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..geo import BBox
from .overpass import OverpassFeatureProvider, _query, parse_overpass_json
from .schema import FeatureCollection

DEFAULT_CACHE_DIR = Path("cache") / "features"


def _cache_key(bbox: BBox) -> str:
    raw = f"osm|{bbox.west:.6f},{bbox.south:.6f},{bbox.east:.6f},{bbox.north:.6f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def fetch_features(bbox: BBox, *, cache_dir: Path | str = DEFAULT_CACHE_DIR,
                   use_cache: bool = True,
                   provider: OverpassFeatureProvider | None = None) -> FeatureCollection:
    """Fetch (or load cached) OSM features for ``bbox``."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{_cache_key(bbox)}.json"
    if use_cache and path.is_file():
        return parse_overpass_json(json.loads(path.read_text()))

    provider = provider or OverpassFeatureProvider()
    # Issue the request here so the raw JSON can be cached before parsing.
    resp = provider._session.post(provider._url, data={"data": _query(bbox)},
                                  timeout=provider._timeout)
    resp.raise_for_status()
    payload = resp.json()
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    return parse_overpass_json(payload)
