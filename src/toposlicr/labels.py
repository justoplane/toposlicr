"""Text labels as filled vector polygons (plan Section 4.4).

Labels are rendered as filled glyph outlines from a real font (via matplotlib's
``TextPath``, which uses fontTools under the hood) → Shapely polygons → filled
SVG paths, which Glowforge treats as engrave regions. A greedy placement search
keeps each label inside its layer's visible band, clear of cut edges and other
labels; unplaceable labels are reported for manual nudging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.affinity import rotate as _rotate
from shapely.affinity import scale as _scale
from shapely.affinity import translate as _translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
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


# Curved-label tuning (degrees). A single sharp turn or too much cumulative bend
# over the text span reads as broken, so we fall back to a horizontal label.
_MAX_LOCAL_TURN_DEG = 40.0
_MAX_CUMULATIVE_TURN_DEG = 120.0


def _angle_diff(a: float, b: float) -> float:
    """Signed smallest difference a-b, wrapped to (-180, 180] degrees."""
    return (a - b + 180.0) % 360.0 - 180.0


def _tangent_deg(line: LineString, d: float, length: float, eps: float) -> float:
    a = line.interpolate(max(0.0, d - eps))
    b = line.interpolate(min(length, d + eps))
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def _path_smooth_enough(line: LineString, s0: float, span: float, eps: float) -> bool:
    """True if the path over [s0, s0+span] is gentle enough to letter along."""
    length = line.length
    samples = 20
    step = span / samples
    if step <= 0:
        return False
    angles = [_tangent_deg(line, s0 + i * step, length, eps) for i in range(samples + 1)]
    max_turn = 0.0
    cumulative = 0.0
    for i in range(1, len(angles)):
        turn = abs(_angle_diff(angles[i], angles[i - 1]))
        max_turn = max(max_turn, turn)
        cumulative += turn
    return max_turn <= _MAX_LOCAL_TURN_DEG and cumulative <= _MAX_CUMULATIVE_TURN_DEG


def text_along_path(text: str, line: BaseGeometry, *, cap_height_mm: float,
                    font: str = "DejaVu Sans") -> BaseGeometry | None:
    """Flow ``text`` glyph-by-glyph along ``line`` (model mm), tangent to the path.

    Each glyph is placed at its arc-length position, rotated to the local tangent
    and offset just above the line so the name sits over the trail. Returns the
    union of glyph polygons, or ``None`` when the segment is too short or too
    curvy — signalling the caller to use a horizontal label instead.
    """
    if not text.strip() or line is None or line.is_empty:
        return None

    # 1. Work on the longest LineString component.
    if isinstance(line, MultiLineString):
        parts = [g for g in line.geoms if g.geom_type == "LineString" and not g.is_empty]
        if not parts:
            return None
        line = max(parts, key=lambda g: g.length)
    if not isinstance(line, LineString) or line.length <= 0:
        return None

    # Read left-to-right along the dominant direction (flip if it runs leftward).
    coords = list(line.coords)
    if coords[-1][0] < coords[0][0]:
        line = LineString(coords[::-1])

    length = line.length
    cap = cap_height_mm
    spacing = 0.12 * cap

    # 2. Per-glyph geometry (centred on x=0, baseline y=0) + proportional advances.
    glyphs: list[tuple[BaseGeometry | None, float]] = []
    for ch in text:
        g = text_to_polygons(ch, font=font, cap_height_mm=cap)
        if g.is_empty:
            glyphs.append((None, 0.4 * cap if ch == " " else 0.3 * cap))
        else:
            minx, _, maxx, _ = g.bounds
            glyphs.append((g, (maxx - minx) + spacing))

    total = sum(adv for _, adv in glyphs)
    if total <= 0 or total > 0.95 * length:
        return None

    s0 = (length - total) / 2.0
    eps = min(0.5, length / 100.0) or 0.5

    # 3. Bail to horizontal if the run is too sharp/curvy to letter cleanly.
    if not _path_smooth_enough(line, s0, total, eps):
        return None

    # 4. Place each glyph at its arc-length centre, rotated to the tangent and
    #    offset above the line so text clears the (dashed) trail.
    off = 0.55 * cap
    placed = []
    cursor = s0
    for g, adv in glyphs:
        centre = cursor + adv / 2.0
        cursor += adv
        if g is None:
            continue
        pt = line.interpolate(centre)
        ang = _tangent_deg(line, centre, length, eps)
        rad = math.radians(ang)
        perp = (-math.sin(rad), math.cos(rad))         # left of travel = above
        moved = _rotate(g, ang, origin=(0, 0), use_radians=False)
        moved = _translate(moved, pt.x + perp[0] * off, pt.y + perp[1] * off)
        placed.append(moved)

    if not placed:
        return None
    result = unary_union(placed)
    return result if not result.is_empty else None


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
