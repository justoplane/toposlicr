"""Label placement: find the open space first, then fit the text to it.

The visible band of a layer (the part not covered by the layer above) is a
ribbon: long along the contour, a few millimeters across. A horizontal label
at a fixed size almost never fits such a ribbon, so the placer works the other
way round from a "try positions around the anchor" search:

1. **Open space.** For a label of cap height ``h`` the band (minus rivers,
   lakes and already-placed text) is eroded by ``h/2 + gap``. Every point on
   the boundary of that eroded region has clearance ≥ ``h/2 + gap`` in all
   directions, so text centered there fits across the ribbon; the boundary
   curves are therefore the natural centerlines for both straight (rotated to
   the local tangent) and curved (flowed along the curve) text. Open plateaus
   get a second family of candidates: rings around the anchor with horizontal
   text, so a name in the middle of a flat doesn't get pushed to its edge.
2. **Ladder.** Each request is tried at full size, then 85% and 70% (never
   below ``min_cap_height_mm``), and for peaks with the elevation dropped once
   the full text finds no room.
3. **Score.** Every fitting candidate costs its distance from the feature plus
   penalties for shrinking, dropping text, rotating away from horizontal,
   curving, and falling to a lower layer. The cheapest wins; ties go to the
   nearest. Features are placed greedily by priority, each placed label
   becoming an obstacle for the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.affinity import rotate as _rotate
from shapely.affinity import translate as _translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring, unary_union

from .labels import LabelRequest, PlacedLabel, text_along_path, text_to_polygons

# Size ladder (fractions of the requested cap height), largest first.
SIZE_LADDER = (1.0, 0.85, 0.7)
# Cost weights, all in "mm of distance from the feature".
SHRINK_PENALTY_MM = 30.0        # × (1 − size fraction)
# Dropping the elevation sits between an 85% and a 70% shrink in preference.
SHORT_TEXT_PENALTY_MM = 6.0
ROTATION_PENALTY_MM_PER_DEG = 0.05
STEEP_PENALTY_MM = 6.0          # extra when |angle| > 60°
CURVE_PENALTY_MM = 3.0
# A curved place name may bend this much in total; more reads as wrapping
# around a corner (trail names, bound to their path, allow more).
MAX_CURVE_TURN_DEG = 60.0
LAYER_DROP_PENALTY_MM = 15.0    # per layer below the feature's own
# Geometry sampling.
CURVE_STEP_FRACTION = 0.6       # sample step along centerlines, × cap height
RING_STEP_FRACTION = 1.0        # ring spacing around the anchor, × cap height
RING_DIRECTIONS = 12
# Extra erosion so text centered on a centerline clears the gap with margin
# rather than touching it at floating-point precision.
CENTERLINE_SLACK_MM = 0.15
# A leader is drawn when the label is farther than this (× cap) from the anchor.
LEADER_BEYOND_CAPS = 2.0


@dataclass
class _Candidate:
    cost: float
    geometry: BaseGeometry
    layer: int
    rotation_deg: float
    cap_height_mm: float
    text: str
    curved: bool
    leader: BaseGeometry | None


@dataclass
class _Site:
    """The open space of one layer at one text height, cached per request."""

    band: BaseGeometry                 # visible band minus static obstacles
    curves: list[LineString] = field(default_factory=list)


class _Layer:
    def __init__(self, band: BaseGeometry, obstacles: BaseGeometry | None, gap: float):
        base = band
        if obstacles is not None and not obstacles.is_empty:
            base = band.difference(obstacles.buffer(gap))
        self.base = base
        # Text keeps `gap` clear of every cut edge and obstacle, not just of
        # other labels: engraving that touches an edge chars it.
        self.inner = base.buffer(-gap)
        shapely.prepare(self.inner)
        self.occupied: BaseGeometry | None = None
        self._sites: dict[float, _Site] = {}

    def site(self, half: float) -> _Site:
        key = round(half, 3)
        if key not in self._sites:
            eroded = self.base.buffer(-half)
            curves: list[LineString] = []
            for poly in _polys(eroded):
                curves.append(LineString(poly.exterior.coords))
                curves.extend(LineString(r.coords) for r in poly.interiors)
            self._sites[key] = _Site(self.base, [c for c in curves if c.length > 0])
        return self._sites[key]

    def fits(self, geom: BaseGeometry) -> bool:
        if geom.is_empty or not self.inner.contains(geom):
            return False
        return self.occupied is None or not geom.intersects(self.occupied)

    def occupy(self, geom: BaseGeometry, gap: float) -> None:
        g = geom.buffer(gap)
        self.occupied = g if self.occupied is None else unary_union([self.occupied, g])


def place_labels(requests: list[LabelRequest], visible_bands: dict[int, BaseGeometry],
                 *, gap_mm: float = 1.5, fallback_depth: int = 4,
                 obstacles: dict[int, BaseGeometry] | None = None,
                 search_radius_mm: float = 60.0, curved: bool = True
                 ) -> list[PlacedLabel]:
    """Place every request; see the module docstring for the strategy.

    ``visible_bands`` maps layer → exposed surface; ``obstacles`` maps layer →
    linework/areas text must not cover (rivers, lakes, icons). A point label
    may fall up to ``fallback_depth`` layers below its own, with a leader.
    """
    layers = {k: _Layer(band, (obstacles or {}).get(k), gap_mm)
              for k, band in visible_bands.items()
              if band is not None and not band.is_empty}
    placed: list[PlacedLabel] = []
    # Most important first; among equals, higher (smaller) layers and larger text.
    ordered = sorted(requests, key=lambda r: (-(r.priority or 0.0), -r.layer_index,
                                              -r.cap_height_mm))
    for req in ordered:
        if req.forced_center is not None:
            result = _place_forced(req)
        else:
            result = _place_request(req, layers, gap_mm, fallback_depth,
                                    search_radius_mm, curved)
        if result.placed and result.layer_index in layers:
            layers[result.layer_index].occupy(result.geometry, gap_mm)
        placed.append(result)
    return placed


def _place_forced(req: LabelRequest) -> PlacedLabel:
    """A manually-nudged label is honored verbatim: centered, horizontal."""
    glyphs = text_to_polygons(req.text, font=req.font, cap_height_mm=req.cap_height_mm)
    if glyphs.is_empty:
        return PlacedLabel(req.text, req.layer_index, MultiPolygon(), req.anchor, False)
    minx, miny, maxx, maxy = glyphs.bounds
    cx, cy = req.forced_center
    moved = _translate(glyphs, cx, cy - (miny + maxy) / 2.0)
    leader = _leader(req.anchor, moved) if req.is_point else None
    return PlacedLabel(req.text, req.layer_index, moved, req.anchor, True, leader,
                       rendered_text=req.text, cap_height_mm=req.cap_height_mm)


def _ladder(req: LabelRequest) -> list[tuple[str, float, float]]:
    """(text, cap height, penalty) variants in preference order."""
    out = []
    texts = [(req.text, 0.0)]
    if req.short_text and req.short_text.strip() and req.short_text != req.text:
        texts.append((req.short_text, SHORT_TEXT_PENALTY_MM))
    for text, tpen in texts:
        for frac in SIZE_LADDER:
            cap = req.cap_height_mm * frac
            if cap + 1e-9 < req.min_cap_height_mm and frac < 1.0:
                continue
            out.append((text, cap, tpen + SHRINK_PENALTY_MM * (1.0 - frac)))
    return out


def _place_request(req: LabelRequest, layers: dict[int, _Layer], gap: float,
                   fallback_depth: int, radius: float, curved: bool) -> PlacedLabel:
    home = req.layer_index
    # Layers to try, nearest first: home, then one down, (one up), two down, …
    order = [home]
    if req.is_point:
        for d in range(1, fallback_depth + 1):
            order.append(home - d)
            if req.allow_higher:
                order.append(home + d)
    best: _Candidate | None = None
    for text, cap, penalty in _ladder(req):
        if best is not None and penalty >= best.cost:
            continue                                   # can't beat what we have
        glyphs = text_to_polygons(text, font=req.font, cap_height_mm=cap)
        if glyphs.is_empty:
            continue
        for k in order:
            layer = layers.get(k)
            if layer is None:
                continue
            drop = LAYER_DROP_PENALTY_MM * abs(home - k)
            if best is not None and penalty + drop >= best.cost:
                break
            cand = _best_on_layer(req, layer, k, text, glyphs, cap, gap, radius,
                                  penalty + drop, curved,
                                  best.cost if best else math.inf)
            if cand is not None and (best is None or cand.cost < best.cost):
                best = cand
    if best is None:
        return PlacedLabel(req.text, home, MultiPolygon(), req.anchor, False)
    return PlacedLabel(req.text, best.layer, best.geometry, req.anchor, True, best.leader,
                       rotation_deg=best.rotation_deg, cap_height_mm=best.cap_height_mm,
                       rendered_text=best.text, curved=best.curved)


def _best_on_layer(req, layer: _Layer, k: int, text: str, glyphs: BaseGeometry,
                   cap: float, gap: float, radius: float, base_penalty: float,
                   curved: bool, bound: float) -> _Candidate | None:
    ax, ay = req.anchor.x, req.anchor.y
    anchor = req.anchor
    minx, miny, maxx, maxy = glyphs.bounds
    width = maxx - minx
    mid_y = (miny + maxy) / 2.0
    half = (maxy - miny) / 2.0 + gap + CENTERLINE_SLACK_MM
    site = layer.site(half)
    best: _Candidate | None = None
    needs_leader = req.is_point
    far = LEADER_BEYOND_CAPS * cap

    def consider(geom: BaseGeometry, dist: float, angle: float, is_curved: bool):
        nonlocal best
        cost = base_penalty + dist + ROTATION_PENALTY_MM_PER_DEG * abs(angle)
        cost += STEEP_PENALTY_MM if abs(angle) > 60 else 0.0
        cost += CURVE_PENALTY_MM if is_curved else 0.0
        if cost >= min(bound, best.cost if best else math.inf):
            return
        if not layer.fits(geom):
            return
        leader = None
        if needs_leader and (dist > far or k != req.layer_index):
            leader = _leader(anchor, geom)
            if leader is not None:
                if layer.occupied is not None and leader.intersects(layer.occupied):
                    return
                leader = leader.intersection(layer.base)
                if leader.is_empty:
                    leader = None
        best = _Candidate(cost, geom, k, angle, cap, text, is_curved, leader)

    # --- family 1: rings around the anchor, horizontal (open plateaus) -------
    step = RING_STEP_FRACTION * cap
    r = 0.0
    while r <= radius:
        dirs = [(0.0, 0.0)] if r == 0 else [
            (math.cos(2 * math.pi * i / RING_DIRECTIONS),
             math.sin(2 * math.pi * i / RING_DIRECTIONS))
            for i in range(RING_DIRECTIONS)]
        for dx, dy in dirs:
            cx, cy = ax + dx * r, ay + dy * r
            if base_penalty + r >= min(bound, best.cost if best else math.inf):
                break
            consider(_translate(glyphs, cx, cy - mid_y), r, 0.0, False)
        r += step

    # --- family 2: along the open-space centerlines (ribbons) ----------------
    cstep = max(CURVE_STEP_FRACTION * cap, 1.0)
    near_pt = Point(ax, ay)
    for curve in site.curves:
        if curve.distance(near_pt) > radius:
            continue
        length = curve.length
        s_near = curve.project(near_pt)
        samples = np.arange(-radius, radius + cstep, cstep)
        for ds in samples:
            s = s_near + ds
            if s < 0 or s > length:
                continue
            pt = curve.interpolate(s)
            dist = math.hypot(pt.x - ax, pt.y - ay)
            if dist > radius:
                continue
            if base_penalty + dist >= min(bound, best.cost if best else math.inf):
                continue
            angle = _upright(_tangent(curve, s, length))
            # straight, rotated to the local tangent
            moved = _rotate(glyphs, angle, origin=(0, mid_y), use_radians=False)
            consider(_translate(moved, pt.x, pt.y - mid_y), dist, angle, False)
            # curved, flowed along the centerline window
            if curved and width > 1.5 * cap and length >= width * 1.1:
                s0, s1 = s - width * 0.55, s + width * 0.55
                if s0 >= 0 and s1 <= length:
                    window = substring(curve, s0, s1)
                    if _total_turn(window, 0.3 * cap) > MAX_CURVE_TURN_DEG:
                        continue
                    flowed = text_along_path(text, window, cap_height_mm=cap,
                                             font=req.font, offset_mm=-mid_y)
                    if flowed is not None and not flowed.is_empty:
                        consider(flowed, dist, _mean_angle(window), True)
    return best


def _tangent(curve: LineString, s: float, length: float) -> float:
    eps = min(1.0, length / 50.0) or 1.0
    a = curve.interpolate(max(0.0, s - eps))
    b = curve.interpolate(min(length, s + eps))
    return math.degrees(math.atan2(b.y - a.y, b.x - a.x))


def _total_turn(line: LineString, tol: float) -> float:
    """Sum of absolute heading changes along the (simplified) line, degrees."""
    coords = list(line.simplify(tol, preserve_topology=False).coords)
    if len(coords) < 3:
        return 0.0
    heads = [math.degrees(math.atan2(y1 - y0, x1 - x0))
             for (x0, y0), (x1, y1) in zip(coords, coords[1:], strict=False)]
    turn = 0.0
    for a, b in zip(heads, heads[1:], strict=False):
        d = (b - a + 180.0) % 360.0 - 180.0
        turn += abs(d)
    return turn


def _mean_angle(line: LineString) -> float:
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    return _upright(math.degrees(math.atan2(y1 - y0, x1 - x0)))


def _upright(angle: float) -> float:
    """Fold an angle into (-90, 90] so text never reads upside down."""
    a = (angle + 180.0) % 360.0 - 180.0
    if a > 90.0:
        a -= 180.0
    elif a <= -90.0:
        a += 180.0
    return a


def _leader(anchor: Point, glyphs: BaseGeometry) -> LineString | None:
    from shapely.ops import nearest_points

    near = nearest_points(anchor, glyphs)[1]
    line = LineString([(anchor.x, anchor.y), (near.x, near.y)])
    return line if line.length > 0.5 else None


def _polys(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
