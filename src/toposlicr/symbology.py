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

import math
from dataclasses import dataclass, field

from shapely.geometry import LineString, MultiLineString, Point
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
            # A lake name is anchored at the water but must sit on the shore:
            # the water itself is an inset (or scored outline), not a surface.
            label_requests.append(LabelRequest(
                text=feat.name or "", anchor=geom_m.centroid, layer_index=k,
                is_point=True, cap_height_mm=sym.labels.cap_height_mm,
                font=sym.labels.font, min_cap_height_mm=sym.labels.min_cap_height_mm,
                priority=_priority(feat, features), allow_higher=True))

        elif feat.feature_type in (FeatureType.PEAK, FeatureType.PLACE):
            pt = geom_m if isinstance(geom_m, Point) else geom_m.centroid
            k = _assign_point_layer(pt, bands, model)
            full = _peak_text(feat, cfg)
            label_requests.append(LabelRequest(
                text=full, anchor=pt, layer_index=k,
                is_point=True, cap_height_mm=sym.labels.cap_height_mm,
                font=sym.labels.font, min_cap_height_mm=sym.labels.min_cap_height_mm,
                short_text=(feat.name or "") if full != (feat.name or "") else None,
                priority=_priority(feat, features)))
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
    # Trail names go first: they are bound to their path, so place names must
    # steer around them. Each named trail is labelled once, on the layer where
    # its exposed segment is longest, curved along the path (horizontal fallback).
    trail_labels_by_layer = _place_trail_labels(named_trails, bands, cfg, warnings)

    # Text must not cover scored water, the water insets, icons or trail names.
    obstacles: dict[int, BaseGeometry] = {}
    for k in bands:
        items = (rivers_by_layer[k] + lake_outlines_by_layer[k]
                 + lake_polys_by_layer[k] + icons_by_layer[k])
        if trail_labels_by_layer.get(k) is not None:
            items.append(trail_labels_by_layer[k])
        if items:
            obstacles[k] = unary_union(items)
    placed = place_labels(live_requests, bands, obstacles=obstacles,
                          search_radius_mm=sym.labels.search_radius_mm,
                          curved=sym.labels.curved)
    report = LabelReport(placed=placed)
    for p in report.unplaced:
        warnings.append(f"label '{p.text}' could not be placed on layer {p.layer_index}")

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


def _exposed_segments(geom_m: BaseGeometry, bands: dict[int, BaseGeometry]
                      ) -> list[tuple[int, LineString]]:
    """Every exposed piece of ``geom_m``, as (layer, LineString), longest first."""
    out: list[tuple[int, LineString]] = []
    for k, band in bands.items():
        clip = geom_m.intersection(band)
        if clip.is_empty:
            continue
        parts = [clip] if isinstance(clip, LineString) else \
            [g for g in getattr(clip, "geoms", []) if isinstance(g, LineString)]
        out.extend((k, g) for g in parts if g.length > 0)
    return sorted(out, key=lambda t: -t[1].length)


def _longest_exposed_layer(geom_m: BaseGeometry, bands: dict[int, BaseGeometry]
                           ) -> tuple[int, BaseGeometry]:
    """Layer whose visible band holds the longest exposed piece of ``geom_m``."""
    best_k, best_len, best_geom = -1, 0.0, None
    for k, band in bands.items():
        clip = geom_m.intersection(band)
        if not clip.is_empty and clip.length > best_len:
            best_k, best_len, best_geom = k, clip.length, clip
    return best_k, best_geom


# Trail names keep this much clear of cut edges, like place names do.
_TRAIL_LABEL_GAP_MM = 1.5


def _place_trail_labels(named_trails: dict[str, BaseGeometry],
                        bands: dict[int, BaseGeometry], cfg: Config,
                        warnings: list[str] | None = None) -> dict[int, BaseGeometry]:
    """Engrave each named trail's name along an exposed stretch of its path.

    The name slides along every exposed segment (longest first) until the whole
    of it lies inside that layer's visible band, above or below the path,
    stepping down a size ladder (never below ``min_cap_height_mm``) when the
    full size finds no room; it is never clipped mid-glyph. Segments are
    simplified first so the jitter of a surveyed GPS track doesn't read as
    curvature. A name with no such stretch is skipped with a warning rather
    than engraved truncated.
    """
    from shapely.affinity import translate as _tr
    from shapely.ops import substring

    from .labels import text_along_path, text_to_polygons
    from .placement import SIZE_LADDER

    sym = cfg.symbology
    font = sym.labels.font
    by_layer: dict[int, list[BaseGeometry]] = {}
    inner = {k: band.buffer(-_TRAIL_LABEL_GAP_MM) for k, band in bands.items()}
    occupied: dict[int, BaseGeometry] = {}

    def accept(k: int, geom: BaseGeometry) -> bool:
        if geom is None or geom.is_empty or not geom.within(inner[k]):
            return False
        occ = occupied.get(k)
        if occ is not None and geom.intersects(occ):
            return False
        by_layer.setdefault(k, []).append(geom)
        occupied[k] = geom if occ is None else unary_union([occ, geom])
        return True

    sizes = [sym.labels.cap_height_mm * f for f in SIZE_LADDER]
    sizes = [c for c in sizes if c >= sym.labels.min_cap_height_mm - 1e-9] or sizes[:1]

    for name, geom in named_trails.items():
        placed = False
        segments = _exposed_segments(geom, bands)
        for cap in sizes:
            glyphs = text_to_polygons(name, font=font, cap_height_mm=cap)
            if glyphs.is_empty:
                break
            _minx, _miny, maxx, _maxy = glyphs.bounds
            width = maxx - _minx
            # Perpendicular offsets: just above the path, then just below it.
            offsets = [0.55 * cap, -(0.3 * cap + _maxy)]
            for k, raw_seg in segments:
                seg = raw_seg.simplify(0.3 * cap, preserve_topology=False)
                if seg.is_empty or seg.length <= 0:
                    continue
                length = seg.length
                half = width * 0.55
                mid = length / 2.0
                steps = int(max(0.0, mid - half) / cap) + 1
                starts = [mid] + [v for i in range(1, steps + 1)
                                  for v in (mid + i * cap, mid - i * cap)]
                if sym.trails.curved_labels and length >= width * 1.1:
                    for s0 in starts:
                        if s0 - half < 0 or s0 + half > length:
                            continue
                        window = substring(seg, s0 - half, s0 + half)
                        for off in offsets:
                            flowed = text_along_path(name, window, cap_height_mm=cap,
                                                     font=font, offset_mm=off)
                            if accept(k, flowed):
                                placed = True
                                break
                        if placed:
                            break
                if placed:
                    break
                # Horizontal fallback: slide along the segment, above then below.
                for s0 in starts:
                    if s0 < 0 or s0 > length:
                        continue
                    pt = seg.interpolate(s0)
                    a = seg.interpolate(max(0.0, s0 - 1.0))
                    b = seg.interpolate(min(length, s0 + 1.0))
                    ang = math.atan2(b.y - a.y, b.x - a.x)
                    perp = (-math.sin(ang), math.cos(ang))
                    for off in offsets:
                        moved = _tr(glyphs, pt.x + perp[0] * off, pt.y + perp[1] * off)
                        if accept(k, moved):
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed and warnings is not None:
            warnings.append(f"trail name '{name}' has no exposed stretch long enough "
                            "to engrave")
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


def _priority(feat: GeoFeature, coll: FeatureCollection) -> float:
    """Importance as a rank within the feature's own type (0 = least, 1 = most),
    so peaks (prominence) and lakes (area) compare on the same footing."""
    if feat.importance is None:
        return 0.0
    peers = sorted(f.importance for f in coll.features
                   if f.feature_type is feat.feature_type and f.importance is not None)
    if len(peers) <= 1:
        return 1.0
    below = sum(1 for v in peers if v < feat.importance)
    return below / (len(peers) - 1)


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
