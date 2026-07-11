"""Vector feature subsystem: acquisition, schema, and selection (Section 1.2)."""

from __future__ import annotations

from .acquire import fetch_features
from .overpass import OverpassFeatureProvider, parse_overpass_json
from .schema import FeatureCollection, FeatureType, GeoFeature
from .select import (
    FeatureChoice,
    auto_choices,
    read_features_csv,
    resolve_features,
    write_features_csv,
)

__all__ = [
    "fetch_features",
    "OverpassFeatureProvider",
    "parse_overpass_json",
    "FeatureCollection",
    "FeatureType",
    "GeoFeature",
    "FeatureChoice",
    "auto_choices",
    "read_features_csv",
    "resolve_features",
    "write_features_csv",
]
