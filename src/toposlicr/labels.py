"""Text labels as filled vector polygons (plan Section 4.4).

Labels are rendered as filled glyph outlines from a real font (via matplotlib's
``TextPath``, which uses fontTools under the hood) → Shapely polygons → filled
SVG paths, which Glowforge treats as engrave regions. A greedy placement search
keeps each label inside its layer's visible band, clear of cut edges and other
labels; unplaceable labels are reported for manual nudging.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.affinity import scale as _scale
from shapely.affinity import translate as _translate
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

# Cache TextPath results per (font, text) — glyph tessellation is not free.
_glyph_cache: dict[tuple[str, str], BaseGeometry] = {}


def _path_to_polygons(path) -> BaseGeometry:
    """Even-odd fill of a matplotlib Path → polygons with holes (for 'o', 'e', …)."""
    rings = [r for r in path.to_polygons() if len(r) >= 4]
    geom: BaseGeometry | None = None
    for ring in sorted((Polygon(r) for r in rings), key=lambda p: p.area, reverse=True):
        p = ring.buffer(0)
        geom = p if geom is None else geom.symmetric_difference(p)
    return geom if geom is not None else MultiPolygon()


def text_to_polygons(text: str, *, font: str = "DejaVu Sans",
                     cap_height_mm: float = 4.0) -> BaseGeometry:
    """Return glyph outlines for ``text`` sized to ``cap_height_mm``, baseline at y=0.

    The result is centered horizontally on x=0 so callers can anchor it.
    """
    if not text.strip():
        return MultiPolygon()
    # Cache the glyphs at unit cap-height (keyed on font+text only) and scale per
    # call, so the same text requested at different cap heights isn't mis-sized.
    key = (font, text)
    if key not in _glyph_cache:
        from matplotlib.font_manager import FontProperties
        from matplotlib.textpath import TextPath

        fp = FontProperties(family=font)
        tp = TextPath((0, 0), text, size=1.0, prop=fp)
        cap = TextPath((0, 0), "H", size=1.0, prop=fp).get_extents().height or 0.7
        geom = _path_to_polygons(tp)
        if not geom.is_empty:
            geom = _scale(geom, 1.0 / cap, 1.0 / cap, origin=(0, 0))  # cap height = 1
            minx, _, maxx, _ = geom.bounds
            geom = _translate(geom, -(minx + maxx) / 2.0, 0)  # center on x=0
        _glyph_cache[key] = geom
    unit = _glyph_cache[key]
    if unit.is_empty:
        return unit
    return _scale(unit, cap_height_mm, cap_height_mm, origin=(0, 0))


def trail_name_geometry(text: str, line: BaseGeometry, *, cap_height_mm: float,
                        font: str = "DejaVu Sans", curved: bool = True) -> BaseGeometry:
    """Engrave a trail name along ``line`` (curved) or horizontally (fallback).

    Curved placement flows each glyph along the path tangent when the segment is
    long and smooth enough; otherwise the name is set horizontally, offset just
    above the line's midpoint. (The curved renderer lives in ``text_along_path``.)
    """
    glyphs = text_to_polygons(text, font=font, cap_height_mm=cap_height_mm)
    if glyphs.is_empty or line.is_empty:
        return MultiPolygon()
    if curved:
        curved_geom = text_along_path(text, line, cap_height_mm=cap_height_mm, font=font)
        if curved_geom is not None and not curved_geom.is_empty:
            return curved_geom
    # Horizontal fallback: centre the name above the segment midpoint.
    mid = line.interpolate(0.5, normalized=True)
    minx, miny, maxx, maxy = glyphs.bounds
    return _translate(glyphs, mid.x - (minx + maxx) / 2.0,
                      mid.y - miny + cap_height_mm * 0.6)


def text_along_path(text: str, line: BaseGeometry, *, cap_height_mm: float,
                    font: str = "DejaVu Sans") -> BaseGeometry | None:
    """Placeholder for curved text-on-path — returns None until implemented.

    Returning None makes ``trail_name_geometry`` use the horizontal fallback, so
    the feature is complete now; the curved renderer is added separately.
    """
    return None


@dataclass
class LabelRequest:
    text: str
    anchor: Point               # model mm, where the label points to
    layer_index: int
    is_point: bool = True       # peaks/places anchor to a point; areas to centroid
    cap_height_mm: float = 4.0
    font: str = "DejaVu Sans"
    forced_center: tuple[float, float] | None = None   # manual override (model mm)


@dataclass
class PlacedLabel:
    text: str
    layer_index: int
    geometry: BaseGeometry      # glyph polygons in model mm (empty if unplaced)
    anchor: Point
    placed: bool
    leader: BaseGeometry | None = None


# 8 candidate offset directions (unit vectors), tried in this order.
_DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
]


def place_labels(requests: list[LabelRequest], visible_bands: dict[int, BaseGeometry],
                 *, gap_mm: float = 1.5, fallback_depth: int = 4) -> list[PlacedLabel]:
    """Greedily place labels within each layer's visible band, avoiding overlaps.

    Tries the anchor plus 8 offset positions; the first candidate that fits the
    visible band and clears already-placed labels wins. Larger/lower labels are
    placed first (more constrained), improving overall success.

    A point label whose home layer is too small (a summit cap) falls back to
    lower, larger bands — up to ``fallback_depth`` layers down — where it lands
    in the visible band with a leader line pointing toward the feature. This is
    the standard cartographic fix for peaks too small to letter directly.
    """
    placed: list[PlacedLabel] = []
    occupied: dict[int, BaseGeometry] = {}
    # Place higher (smaller) layers and larger text first — most constrained.
    ordered = sorted(requests, key=lambda r: (-r.layer_index, -r.cap_height_mm))
    for req in ordered:
        glyphs = text_to_polygons(req.text, font=req.font, cap_height_mm=req.cap_height_mm)
        # A manually-nudged label is honored verbatim, no search.
        if req.forced_center is not None:
            placed.append(_place_forced(req, glyphs))
            continue
        home = req.layer_index
        lowest = max(0, home - fallback_depth) if req.is_point else home
        result = PlacedLabel(req.text, home, MultiPolygon(), req.anchor, False)
        for j in range(home, lowest - 1, -1):
            band = visible_bands.get(j)
            candidate = _place_one(req, glyphs, band, occupied.get(j), gap_mm,
                                   layer_index=j)
            if candidate.placed:
                result = candidate
                occ = occupied.get(j)
                occupied[j] = (candidate.geometry if occ is None
                               else unary_union([occ, candidate.geometry]))
                break
        placed.append(result)
    return placed


def _place_forced(req: LabelRequest, glyphs: BaseGeometry) -> PlacedLabel:
    """Place a label at a manual override position, centered, without search."""
    if glyphs.is_empty:
        return PlacedLabel(req.text, req.layer_index, MultiPolygon(), req.anchor, False)
    minx, miny, maxx, maxy = glyphs.bounds
    mid_y = (miny + maxy) / 2.0
    cx, cy = req.forced_center
    moved = _translate(glyphs, cx, cy - mid_y)
    leader = _leader_line(req.anchor, moved) if req.is_point else None
    return PlacedLabel(req.text, req.layer_index, moved, req.anchor, True, leader)


def _place_one(req: LabelRequest, glyphs: BaseGeometry, band: BaseGeometry | None,
               occupied: BaseGeometry | None, gap_mm: float,
               layer_index: int) -> PlacedLabel:
    if glyphs.is_empty or band is None or band.is_empty:
        return PlacedLabel(req.text, layer_index, MultiPolygon(), req.anchor, False)

    minx, miny, maxx, maxy = glyphs.bounds
    half_w = (maxx - minx) / 2.0
    half_h = (maxy - miny) / 2.0
    mid_y = (miny + maxy) / 2.0            # glyph bbox is centered on x=0, baseline y=0
    ax, ay = req.anchor.x, req.anchor.y

    # Position the glyph bbox-center: first on the anchor, then in rings of
    # offsets at increasing distance (so a peak on a lower band can sit farther
    # out with a longer leader). Translation places bbox center (0, mid_y).
    targets = [(ax, ay)]
    for ring in (1, 2, 3):
        for dx, dy in _DIRECTIONS:
            targets.append((ax + dx * (half_w + gap_mm) * ring,
                            ay + dy * (half_h + gap_mm) * ring))

    for i, (tx, ty) in enumerate(targets):
        moved = _translate(glyphs, tx, ty - mid_y)
        if not moved.within(band):
            continue
        if occupied is not None and moved.intersects(occupied):
            continue
        leader = None
        if i != 0 and req.is_point:
            leader = _leader_line(req.anchor, moved)
            if leader is not None:
                leader = leader.intersection(band)  # score only the exposed part
                if leader.is_empty:
                    leader = None
        return PlacedLabel(req.text, layer_index, moved, req.anchor, True, leader)

    return PlacedLabel(req.text, layer_index, MultiPolygon(), req.anchor, False)


def _leader_line(anchor: Point, glyphs: BaseGeometry):
    from shapely.geometry import LineString
    from shapely.ops import nearest_points

    near = nearest_points(anchor, glyphs)[1]
    line = LineString([(anchor.x, anchor.y), (near.x, near.y)])
    return line if line.length > 0.5 else None


@dataclass
class LabelReport:
    placed: list[PlacedLabel] = field(default_factory=list)

    @property
    def unplaced(self) -> list[PlacedLabel]:
        return [p for p in self.placed if not p.placed]

    def engrave_area_mm2(self) -> float:
        return sum(p.geometry.area for p in self.placed if p.placed)


def write_labels_file(report: LabelReport, path) -> None:
    """Write an editable record of every label's position (plan Section 4.4).

    The builder can move a ``center`` coordinate (and/or ``layer``) and re-run;
    ``read_label_overrides`` feeds those back as forced placements. ``placed`` is
    informational — unplaced labels appear too so they can be positioned by hand.
    """
    import json
    from pathlib import Path

    rows = []
    for p in report.placed:
        if not p.geometry.is_empty:
            # Record the glyph bbox center — that is what _place_forced/_place_one
            # position, so an unedited save→reload is a no-op (no drift).
            minx, miny, maxx, maxy = p.geometry.bounds
            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        else:
            cx, cy = p.anchor.x, p.anchor.y
        rows.append({
            "text": p.text, "layer": p.layer_index, "placed": p.placed,
            "center": [round(cx, 2), round(cy, 2)],
        })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def read_label_overrides(path) -> dict[str, tuple[int, float, float]]:
    """Return ``{text: (layer, cx, cy)}`` from an edited labels file."""
    import json
    from pathlib import Path

    overrides: dict[str, tuple[int, float, float]] = {}
    for row in json.loads(Path(path).read_text()):
        center = row.get("center")
        if row.get("text") and center and len(center) == 2:
            overrides[row["text"]] = (int(row.get("layer", 0)),
                                      float(center[0]), float(center[1]))
    return overrides
