"""OpenStreetMap vector features via the Overpass API (plan Section 1.2).

OSM is the universal backbone: global, one query interface, names and geometry
together. This fetches water bodies, waterways, peaks and place names for a
bounding box and normalizes them into ``GeoFeature`` objects. US enrichment
(GNIS/NHD) can be layered on later behind the same interface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import requests
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from ..geo import BBox
from .schema import FeatureCollection, FeatureType, GeoFeature

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Overpass rejects the default python-requests UA with 406; identify ourselves.
_USER_AGENT = "toposlicr/0.1 (laser-cut topo map pipeline)"

# Pseudo stream-order for OSM waterway classes, so `min_stream_order` filtering
# works globally even without NHD's real stream order.
_WATERWAY_ORDER = {"river": 5, "canal": 4, "stream": 2, "tidal_channel": 3}

_EARTH_R = 6_371_008.8


def _query(bbox: BBox) -> str:
    s, w, n, e = bbox.south, bbox.west, bbox.north, bbox.east
    b = f"({s},{w},{n},{e})"
    return f"""[out:json][timeout:90];
(
  way["natural"="water"]{b};
  relation["natural"="water"]{b};
  way["landuse"="reservoir"]{b};
  way["waterway"~"river|stream|canal|tidal_channel"]{b};
  node["natural"="peak"]{b};
  node["natural"="saddle"]{b};
  node["mountain_pass"="yes"]{b};
  node["natural"="glacier"]{b};
  node["place"~"town|village|hamlet|locality"]{b};
);
out geom;"""


def _poly_area_km2(poly: Polygon, lat: float) -> float:
    m_per_deg = _EARTH_R * math.pi / 180.0
    # area in deg² → m² using local scale, then km².
    return poly.area * (m_per_deg ** 2) * math.cos(math.radians(lat)) / 1e6


class OverpassFeatureProvider:
    """Fetches and normalizes OSM features for a bbox."""

    name = "overpass"

    def __init__(self, session: requests.Session | None = None, timeout: float = 120.0,
                 url: str = _OVERPASS_URL):
        self._session = session or requests.Session()
        # Overwrite the default "python-requests/x" UA, which Overpass 406s.
        self._session.headers["User-Agent"] = _USER_AGENT
        self._timeout = timeout
        self._url = url

    def fetch(self, bbox: BBox) -> FeatureCollection:
        resp = self._session.post(self._url, data={"data": _query(bbox)},
                                  timeout=self._timeout)
        resp.raise_for_status()
        return parse_overpass_json(resp.json())


def parse_overpass_json(payload: dict) -> FeatureCollection:
    """Turn a raw Overpass JSON response into a ``FeatureCollection``."""
    coll = FeatureCollection(epsg=4326)
    for el in payload.get("elements", []):
        feature = _element_to_feature(el)
        if feature is not None and not feature.geometry.is_empty:
            coll.add(feature)
    return coll


def _element_to_feature(el: dict) -> GeoFeature | None:
    tags = el.get("tags", {}) or {}
    etype = el.get("type")
    osm_id = f"{etype}/{el.get('id')}"
    name = tags.get("name")

    if etype == "node":
        return _node_feature(el, tags, name, osm_id)
    if etype == "way":
        return _way_feature(el, tags, name, osm_id)
    if etype == "relation":
        return _relation_feature(el, tags, name, osm_id)
    return None


def _node_feature(el: dict, tags: dict, name: str | None, osm_id: str) -> GeoFeature | None:
    if "lat" not in el or "lon" not in el:
        return None
    pt = Point(el["lon"], el["lat"])
    ele = _parse_float(tags.get("ele"))
    if tags.get("natural") == "peak":
        return GeoFeature(FeatureType.PEAK, pt, name=name, elevation=ele,
                          importance=ele, osm_id=osm_id, tags=tags)
    return GeoFeature(FeatureType.PLACE, pt, name=name, elevation=ele,
                      osm_id=osm_id, tags=tags)


def _way_feature(el: dict, tags: dict, name: str | None, osm_id: str) -> GeoFeature | None:
    coords = [(g["lon"], g["lat"]) for g in el.get("geometry", []) if g]
    if len(coords) < 2:
        return None
    is_water = tags.get("natural") == "water" or tags.get("landuse") == "reservoir" \
        or "water" in tags
    if is_water and len(coords) >= 4 and coords[0] == coords[-1]:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        lat = poly.centroid.y
        return GeoFeature(FeatureType.LAKE, poly, name=name,
                          importance=_poly_area_km2(poly, lat), osm_id=osm_id, tags=tags)
    if "waterway" in tags:
        order = _WATERWAY_ORDER.get(tags["waterway"], 1)
        return GeoFeature(FeatureType.RIVER, LineString(coords), name=name,
                          importance=order, osm_id=osm_id, tags=tags)
    return None


def _relation_feature(el: dict, tags: dict, name: str | None,
                      osm_id: str) -> GeoFeature | None:
    # Best-effort multipolygon: union outer rings, subtract inner rings.
    outers, inners = [], []
    for member in el.get("members", []):
        if member.get("type") != "way":
            continue
        coords = [(g["lon"], g["lat"]) for g in member.get("geometry", []) if g]
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        (inners if member.get("role") == "inner" else outers).append(poly)
    if not outers:
        return None
    geom = unary_union(outers)
    if inners:
        geom = geom.difference(unary_union(inners))
    if geom.is_empty:
        return None
    lat = geom.centroid.y
    return GeoFeature(FeatureType.LAKE, geom, name=name,
                      importance=_poly_area_km2(geom, lat), osm_id=osm_id, tags=tags)


def _parse_float(val) -> float | None:
    if val is None:
        return None
    try:
        return float(str(val).split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return None


def load_overpass_file(path: str | Path) -> FeatureCollection:
    """Parse a saved Overpass JSON file (useful for offline tests/fixtures)."""
    return parse_overpass_json(json.loads(Path(path).read_text()))
