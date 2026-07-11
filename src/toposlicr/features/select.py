"""Automatic feature selection + the ``features.csv`` review loop (Section 1.2).

The pipeline auto-selects features by threshold (peaks by prominence/elevation,
lakes by area, rivers by stream order) and proceeds without stopping — a
zero-touch default. It also writes ``features.csv`` (name, type, include, label
override); if that file exists on a re-run it overrides the automatic choice, so
curation is an optional edit-and-rerun loop rather than a blocking checkpoint.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from ..config import SymbologyConfig
from .schema import FeatureCollection, FeatureType, GeoFeature

_CSV_FIELDS = ["osm_id", "type", "name", "include", "label"]


@dataclass
class FeatureChoice:
    feature: GeoFeature
    include: bool
    label: str


def _auto_include(f: GeoFeature, sym: SymbologyConfig) -> bool:
    if f.feature_type is FeatureType.PEAK:
        if not f.name or f.elevation is None:
            return False
        prom = _parse_prominence(f)
        if prom is not None and prom < sym.peaks.min_prominence_m:
            return False
        return not (sym.peaks.min_elevation_m is not None
                    and f.elevation < sym.peaks.min_elevation_m)
    if f.feature_type is FeatureType.RIVER:
        return (f.importance or 0) >= sym.rivers.min_stream_order
    if f.feature_type is FeatureType.LAKE:
        return (f.importance or 0.0) >= sym.lakes.min_area_km2
    if f.feature_type is FeatureType.PLACE:
        return sym.include_places and bool(f.name)
    return bool(f.name)


def _parse_prominence(f: GeoFeature) -> float | None:
    val = f.tags.get("prominence")
    if val is None:
        return None
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return None


def auto_choices(coll: FeatureCollection, sym: SymbologyConfig) -> list[FeatureChoice]:
    """Apply thresholds to produce include/exclude choices for every feature."""
    choices = [
        FeatureChoice(feature=f, include=_auto_include(f, sym), label=f.label)
        for f in coll.features
    ]
    # Optional cap: keep only the N highest peaks.
    if sym.peaks.max_count is not None:
        peaks = sorted(
            (c for c in choices if c.feature.feature_type is FeatureType.PEAK
             and c.include),
            key=lambda c: c.feature.elevation or 0.0, reverse=True,
        )
        for c in peaks[sym.peaks.max_count:]:
            c.include = False
    return choices


def write_features_csv(choices: list[FeatureChoice], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for c in sorted(choices, key=lambda c: (c.feature.feature_type.value,
                                                c.label or "~")):
            writer.writerow({
                "osm_id": c.feature.osm_id or "",
                "type": c.feature.feature_type.value,
                "name": c.feature.name or "",
                "include": "yes" if c.include else "no",
                "label": c.label or "",
            })
    return p


def read_features_csv(path: str | Path) -> dict[str, tuple[bool, str]]:
    """Return ``{osm_id: (include, label_override)}`` from an edited CSV."""
    overrides: dict[str, tuple[bool, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            osm_id = (row.get("osm_id") or "").strip()
            if not osm_id:
                continue
            include = (row.get("include") or "yes").strip().lower() in {"yes", "true", "1"}
            overrides[osm_id] = (include, (row.get("label") or "").strip())
    return overrides


def resolve_features(coll: FeatureCollection, sym: SymbologyConfig,
                     csv_path: str | Path) -> tuple[FeatureCollection, bool]:
    """Auto-select, apply an existing ``features.csv``, and return the kept set.

    Returns ``(collection, csv_existed)``. When the CSV does not exist it is
    written with the automatic choices; when it does, it overrides them.
    """
    choices = auto_choices(coll, sym)
    csv_p = Path(csv_path)
    csv_existed = csv_p.is_file()
    if csv_existed:
        overrides = read_features_csv(csv_p)
        for c in choices:
            osm_id = c.feature.osm_id or ""
            if osm_id in overrides:
                c.include, override_label = overrides[osm_id]
                if override_label:
                    c.label = override_label
    else:
        write_features_csv(choices, csv_p)

    kept = FeatureCollection(epsg=coll.epsg)
    for c in choices:
        if c.include:
            # Carry the (possibly overridden) label as the feature's name.
            c.feature.name = c.label or c.feature.name
            kept.add(c.feature)
    return kept, csv_existed
