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
    icons: BaseGeometry | None = None        # engraved icon glyphs (fictional maps)
    trails: BaseGeometry | None = None       # dashed trail linework (score_trail)
    trail_labels: BaseGeometry | None = None  # engraved trail names


@dataclass
class SymbologyResult:
    per_layer: dict[int, LayerSymbology]
    labels: LabelReport
    warnings: list[str] = field(default_factory=list)

    def for_layer(self, k: int) -> LayerSymbology | None:
        return self.per_layer.get(k)


def _to_model(model: LayerModel, geom: BaseGeometry) -> BaseGeometry:
    """Feature geometry → model mm.

    Real features (EPSG:4326) reproject into the model's projected CRS (the
    box-centered transverse Mercator the DEM was projected into) first;
    fictional features are already in flat world units, so only the
    world→model affine applies.
    """
    world = geom if model.flat else reproject_geom(geom, 4326, model.crs or model.utm_epsg)
    return apply_world_matrix(model.world_to_model, world)


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
    icons_by_layer: dict[int, list[BaseGeometry]] = {k: [] for k in bands}
    trails_by_layer: dict[int, list[BaseGeometry]] = {k: [] for k in bands}
    named_trails: dict[str, BaseGeometry] = {}   # name → longest model-mm geometry
    label_requests: list[LabelRequest] = []
    sym = cfg.symbology

    for feat in features.features:
        geom_m = _to_model(model, feat.geometry)
        if geom_m.is_empty:
            continue

        if feat.feature_type is FeatureType.RIVER:
            if geom_m.length == 0:
                # A point 'river' row (e.g. an Azgaar named river with no
                # linework) — label it like a place rather than dropping it.
                pt = geom_m if isinstance(geom_m, Point) else geom_m.centroid
                k = _assign_point_layer(pt, bands, model)
                if feat.name:
                    label_requests.append(LabelRequest(
                        text=feat.name, anchor=pt, layer_index=k, is_point=True,
                        cap_height_mm=sym.labels.cap_height_mm, font=sym.labels.font))
                continue
            # A river is scored on every layer it crosses, only where exposed.
            for k, band in bands.items():
                clipped = geom_m.intersection(band)
                if not clipped.is_empty and clipped.length > 0:
                    rivers_by_layer[k].append(clipped)

        elif feat.feature_type is FeatureType.TRAIL:
            # Trails read continuously up the stack, scored (dashed) only where
            # exposed — same visible-band clipping as rivers.
            for k, band in bands.items():
                clipped = geom_m.intersection(band)
                if not clipped.is_empty and clipped.length > 0:
                    trails_by_layer[k].append(clipped)
            if feat.name and sym.trails.label:
                prev = named_trails.get(feat.name)
                if prev is None or geom_m.length > prev.length:
                    named_trails[feat.name] = geom_m

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
            # Optional engraved icon glyph, clipped to the visible band.
            icon_name = feat.tags.get("icon")
            if icon_name:
                from .icons import icon_glyph
                glyph = icon_glyph(icon_name, pt.x, pt.y, sym.labels.icon_size_mm)
                glyph = glyph.intersection(bands[k])
                if not glyph.is_empty:
                    icons_by_layer[k].append(glyph)

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

    # Trail names: label each named trail once, on the layer where its exposed
    # segment is longest, curved along the path (with a horizontal fallback).
    trail_labels_by_layer = _place_trail_labels(named_trails, bands, cfg)

    per_layer = {}
    for k in bands:
        trail_geom = _merge_lines(trails_by_layer[k])
        dashed = dash_line(trail_geom, sym.trails.dash_mm, sym.trails.gap_mm) \
            if not trail_geom.is_empty else None
        per_layer[k] = LayerSymbology(
            layer_index=k,
            visible_band=bands[k],
            rivers=_merge_lines(rivers_by_layer[k]),
            lake_outlines=_merge_lines(lake_outlines_by_layer[k]),
            lakes=lakes_by_layer[k],
            lake_polys=lake_polys_by_layer[k],
            icons=unary_union(icons_by_layer[k]) if icons_by_layer[k] else None,
            trails=dashed,
            trail_labels=trail_labels_by_layer.get(k),
        )
    return SymbologyResult(per_layer=per_layer, labels=report, warnings=warnings)


def _longest_exposed_layer(geom_m: BaseGeometry, bands: dict[int, BaseGeometry]
                           ) -> tuple[int, BaseGeometry]:
    """Layer whose visible band holds the longest exposed piece of ``geom_m``."""
    best_k, best_len, best_geom = -1, 0.0, None
    for k, band in bands.items():
        clip = geom_m.intersection(band)
        if not clip.is_empty and clip.length > best_len:
            best_k, best_len, best_geom = k, clip.length, clip
    return best_k, best_geom


def _place_trail_labels(named_trails: dict[str, BaseGeometry],
                        bands: dict[int, BaseGeometry], cfg: Config
                        ) -> dict[int, BaseGeometry]:
    """Engrave each named trail's name on its most-exposed layer."""
    from .labels import trail_name_geometry

    sym = cfg.symbology
    by_layer: dict[int, list[BaseGeometry]] = {}
    for name, geom in named_trails.items():
        k, seg = _longest_exposed_layer(geom, bands)
        if k < 0 or seg is None:
            continue
        glyphs = trail_name_geometry(name, seg, cap_height_mm=sym.labels.cap_height_mm,
                                     font=sym.labels.font,
                                     curved=sym.trails.curved_labels)
        clipped = glyphs.intersection(bands[k]) if not glyphs.is_empty else glyphs
        if not clipped.is_empty:
            by_layer.setdefault(k, []).append(clipped)
    return {k: unary_union(v) for k, v in by_layer.items()}


def dash_line(geom: BaseGeometry, dash_mm: float, gap_mm: float) -> BaseGeometry:
    """Break line(s) into dash segments (curved-following) for a dashed score."""
    from shapely.geometry import MultiLineString
    from shapely.ops import substring

    if geom.is_empty or dash_mm <= 0:
        return geom
    lines = geom.geoms if hasattr(geom, "geoms") else [geom]
    period = dash_mm + max(gap_mm, 0.0)
    segments = []
    for line in lines:
        if getattr(line, "geom_type", "") != "LineString":
            continue
        length = line.length
        d = 0.0
        while d < length:
            end = min(d + dash_mm, length)
            if end - d > 0.05:
                seg = substring(line, d, end)
                if not seg.is_empty and seg.length > 0:
                    segments.append(seg)
            d += period
    return MultiLineString(segments) if segments else MultiLineString()


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
