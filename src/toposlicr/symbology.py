"""Symbology generation — water, rivers, labels (plan Section 4).

All symbology is computed against the **visible band** of each layer,
``polygon(k) − footprint(k+1)``: the part of a layer that stays exposed after
the next layer is glued on top. Anything outside it would be hidden, so labels,
rivers and lake outlines are clipped to it. The complementary **hidden zone**,
``polygon(k) ∩ footprint(k+1)``, is where seams and part IDs go later.

Feature → layer assignment is purely geometric: a feature belongs to the topmost
layer whose visible band contains it — the surface you would actually see — so no
DEM re-sampling is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import MultiLineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import Config
from .features.schema import FeatureCollection, FeatureType, GeoFeature
from .geo import reproject_geom
from .labels import LabelReport, LabelRequest, place_labels
from .layers import LayerModel, apply_world_matrix


def visible_band(model: LayerModel, k: int) -> BaseGeometry:
    """Exposed part of layer k after layer k+1 is stacked on it."""
    here = model.footprint(k)
    above = model.footprint(k + 1)
    if here.is_empty:
        return here
    return here if above.is_empty else here.difference(above)


def hidden_zone(model: LayerModel, k: int) -> BaseGeometry:
    """Part of layer k covered by layer k+1 (where seams / IDs can hide)."""
    here = model.footprint(k)
    above = model.footprint(k + 1)
    if here.is_empty or above.is_empty:
        from shapely.geometry import Polygon
        return Polygon()
    return here.intersection(above)


@dataclass
class LayerSymbology:
    layer_index: int
    visible_band: BaseGeometry
    rivers: BaseGeometry                     # clipped score lines (model mm)
    lake_outlines: BaseGeometry              # score-mode lake boundaries
    lakes: list[GeoFeature] = field(default_factory=list)  # source features
    lake_polys: list[BaseGeometry] = field(default_factory=list)  # model-mm polys (Phase 3)


@dataclass
class SymbologyResult:
    per_layer: dict[int, LayerSymbology]
    labels: LabelReport
    warnings: list[str] = field(default_factory=list)

    def for_layer(self, k: int) -> LayerSymbology | None:
        return self.per_layer.get(k)


def _to_model(model: LayerModel, geom: BaseGeometry) -> BaseGeometry:
    """Feature geometry: source EPSG:4326 → UTM → model mm."""
    utm = reproject_geom(geom, 4326, model.utm_epsg)
    return apply_world_matrix(model.world_to_model, utm)


def _assign_point_layer(pt: Point, bands: dict[int, BaseGeometry],
                        model: LayerModel) -> int:
    """Topmost layer whose visible band contains the point (the visible surface)."""
    for k in range(model.layer_count - 1, -1, -1):
        band = bands.get(k)
        if band is not None and not band.is_empty and band.contains(pt):
            return k
    # Fall back to the highest layer whose footprint contains it.
    for k in range(model.layer_count - 1, -1, -1):
        if model.footprint(k).contains(pt):
            return k
    return 0


def build_symbology(model: LayerModel, features: FeatureCollection, cfg: Config,
                    label_overrides: dict[str, tuple[int, float, float]] | None = None
                    ) -> SymbologyResult:
    """Clip rivers/lakes and place labels against each layer's visible band.

    ``label_overrides`` maps label text → (layer, cx, cy) for manually-nudged
    labels (from an edited labels file); those are placed verbatim.
    """
    label_overrides = label_overrides or {}
    warnings: list[str] = []
    bands = {k: visible_band(model, k) for k in range(model.layer_count)}

    rivers_by_layer: dict[int, list[BaseGeometry]] = {k: [] for k in bands}
    lake_outlines_by_layer: dict[int, list[BaseGeometry]] = {k: [] for k in bands}
    lakes_by_layer: dict[int, list[GeoFeature]] = {k: [] for k in bands}
    lake_polys_by_layer: dict[int, list[BaseGeometry]] = {k: [] for k in bands}
    label_requests: list[LabelRequest] = []
    sym = cfg.symbology

    for feat in features.features:
        geom_m = _to_model(model, feat.geometry)
        if geom_m.is_empty:
            continue

        if feat.feature_type is FeatureType.RIVER:
            # A river is scored on every layer it crosses, only where exposed.
            for k, band in bands.items():
                clipped = geom_m.intersection(band)
                if not clipped.is_empty and clipped.length > 0:
                    rivers_by_layer[k].append(clipped)

        elif feat.feature_type is FeatureType.LAKE:
            k = _assign_lake_layer(geom_m, bands, model)
            lakes_by_layer[k].append(feat)
            lake_polys_by_layer[k].append(geom_m)
            if sym.lakes.mode == "score":
                outline = geom_m.boundary.intersection(bands[k])
                if not outline.is_empty:
                    lake_outlines_by_layer[k].append(outline)
            label_requests.append(LabelRequest(
                text=feat.name or "", anchor=geom_m.centroid, layer_index=k,
                is_point=False, cap_height_mm=sym.labels.cap_height_mm,
                font=sym.labels.font))

        elif feat.feature_type in (FeatureType.PEAK, FeatureType.PLACE):
            pt = geom_m if isinstance(geom_m, Point) else geom_m.centroid
            k = _assign_point_layer(pt, bands, model)
            label_requests.append(LabelRequest(
                text=_peak_text(feat, cfg), anchor=pt, layer_index=k,
                is_point=True, cap_height_mm=sym.labels.cap_height_mm,
                font=sym.labels.font))

    live_requests = [r for r in label_requests if r.text.strip()]
    for req in live_requests:
        if req.text in label_overrides:
            layer, cx, cy = label_overrides[req.text]
            req.layer_index = layer
            req.forced_center = (cx, cy)
    placed = place_labels(live_requests, bands)
    report = LabelReport(placed=placed)
    for p in report.unplaced:
        warnings.append(f"label '{p.text}' could not be placed on layer {p.layer_index}")

    per_layer = {}
    for k in bands:
        per_layer[k] = LayerSymbology(
            layer_index=k,
            visible_band=bands[k],
            rivers=_merge_lines(rivers_by_layer[k]),
            lake_outlines=_merge_lines(lake_outlines_by_layer[k]),
            lakes=lakes_by_layer[k],
            lake_polys=lake_polys_by_layer[k],
        )
    return SymbologyResult(per_layer=per_layer, labels=report, warnings=warnings)


def _assign_lake_layer(geom_m: BaseGeometry, bands: dict[int, BaseGeometry],
                       model: LayerModel) -> int:
    """Surface layer = the band with the greatest overlap with the lake.

    If the lake overlaps no visible band (it sits entirely under a higher layer),
    fall back to the topmost layer whose footprint contains its centroid — the
    same rule used for point features — rather than defaulting to layer 0.
    """
    best_k, best_area = -1, 0.0
    for k, band in bands.items():
        if band.is_empty:
            continue
        area = geom_m.intersection(band).area
        if area > best_area:
            best_k, best_area = k, area
    if best_k >= 0:
        return best_k
    centroid = geom_m.centroid
    for k in range(model.layer_count - 1, -1, -1):
        if model.footprint(k).contains(centroid):
            return k
    return 0


def _peak_text(feat: GeoFeature, cfg: Config) -> str:
    text = feat.name or ""
    if (feat.feature_type is FeatureType.PEAK and cfg.symbology.peaks.label_elevation
            and feat.elevation is not None):
        text = f"{text} {feat.elevation:,.0f} m".strip()
    return text


def _merge_lines(lines: list[BaseGeometry]) -> BaseGeometry:
    if not lines:
        return MultiLineString()
    merged = unary_union(lines)
    return merged


def label_geometry_for_layer(report: LabelReport, k: int) -> BaseGeometry:
    """Union of all placed label glyphs on layer k (for engraving)."""
    geoms = [p.geometry for p in report.placed if p.placed and p.layer_index == k]
    return unary_union(geoms) if geoms else MultiLineString()


def leader_geometry_for_layer(report: LabelReport, k: int) -> BaseGeometry:
    geoms = [p.leader for p in report.placed
             if p.placed and p.layer_index == k and p.leader is not None]
    return unary_union(geoms) if geoms else MultiLineString()
