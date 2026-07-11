"""Normalized vector-feature schema (plan Section 1.2).

All symbolic data — lakes, rivers, peaks, place names — is normalized into one
simple structure regardless of source (OSM, GNIS, NHD), so downstream symbology
code is source-agnostic. Geometry is stored in EPSG:4326 (lon/lat) until the
pipeline reprojects it alongside the DEM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from shapely.geometry.base import BaseGeometry


class FeatureType(StrEnum):
    LAKE = "lake"          # water polygon (natural=water, reservoir, ...)
    RIVER = "river"        # waterway line
    PEAK = "peak"          # natural=peak point (has ele)
    PLACE = "place"        # town/valley/pass/glacier point
    COASTLINE = "coastline"


@dataclass
class GeoFeature:
    """One map feature in a source CRS (default EPSG:4326)."""

    feature_type: FeatureType
    geometry: BaseGeometry
    name: str | None = None
    elevation: float | None = None      # meters, when known (peaks, lake surface)
    importance: float | None = None     # stream order / prominence / area proxy
    osm_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or ""


@dataclass
class FeatureCollection:
    """A bag of features plus the CRS they live in."""

    features: list[GeoFeature] = field(default_factory=list)
    epsg: int = 4326

    def __iter__(self):
        return iter(self.features)

    def __len__(self) -> int:
        return len(self.features)

    def of_type(self, ftype: FeatureType) -> list[GeoFeature]:
        return [f for f in self.features if f.feature_type == ftype]

    def add(self, feature: GeoFeature) -> None:
        self.features.append(feature)
