"""Parametric icon glyphs for fictional-map iconography (spec Tier 1/3).

Fictional maps invite iconography — shrine/tower/settlement stamps engraved onto
the plywood. Icons are simple filled shapes generated at a given size and centre
(model mm), so they engrave cleanly and scale with the machine. The name comes
from a feature's ``icon`` column in the bundle's ``features.csv``.
"""

from __future__ import annotations

import math

from shapely.affinity import translate
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


def _regular(cx: float, cy: float, r: float, n: int, rot: float = 0.0) -> Polygon:
    pts = [(cx + r * math.cos(rot + 2 * math.pi * i / n),
            cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]
    return Polygon(pts)


def _star(cx: float, cy: float, r: float, points: int = 5) -> Polygon:
    pts = []
    for i in range(points * 2):
        rr = r if i % 2 == 0 else r * 0.42
        a = -math.pi / 2 + math.pi * i / points
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return Polygon(pts)


def _rect(cx: float, cy: float, w: float, h: float) -> Polygon:
    return Polygon([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                    (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)])


def _peak(cx: float, cy: float, s: float) -> Polygon:
    return Polygon([(cx - s / 2, cy - s / 2), (cx + s / 2, cy - s / 2), (cx, cy + s / 2)])


def _tower(cx: float, cy: float, s: float) -> BaseGeometry:
    body = _rect(cx, cy - s * 0.1, s * 0.4, s * 0.8)
    # three merlons on top
    merlons = [_rect(cx + dx, cy + s * 0.35, s * 0.12, s * 0.2)
               for dx in (-s * 0.16, 0.0, s * 0.16)]
    return unary_union([body, *merlons])


def _shrine(cx: float, cy: float, s: float) -> Polygon:
    # teardrop-ish diamond
    return Polygon([(cx, cy + s / 2), (cx + s / 3, cy), (cx, cy - s / 2), (cx - s / 3, cy)])


_BUILDERS = {
    "peak": _peak, "mountain": _peak, "summit": _peak,
    "tower": _tower, "castle": _tower, "fort": _tower,
    "shrine": _shrine, "temple": _shrine,
    "settlement": lambda cx, cy, s: _rect(cx, cy, s * 0.7, s * 0.7),
    "town": lambda cx, cy, s: _rect(cx, cy, s * 0.7, s * 0.7),
    "village": lambda cx, cy, s: _regular(cx, cy, s * 0.42, 3, rot=math.pi / 2),
    "city": lambda cx, cy, s: _regular(cx, cy, s * 0.5, 6),
    "star": lambda cx, cy, s: _star(cx, cy, s * 0.5),
    "poi": lambda cx, cy, s: _regular(cx, cy, s * 0.45, 16),   # circle-ish
    "marker": lambda cx, cy, s: _regular(cx, cy, s * 0.45, 16),
}

#: names callers can rely on existing
AVAILABLE = sorted(_BUILDERS)


def icon_glyph(name: str, cx: float, cy: float, size_mm: float) -> BaseGeometry:
    """Return a filled icon glyph of ``name`` centred at (cx, cy), ~size_mm across.

    Unknown names fall back to a circle marker so a stray icon value never fails.
    """
    builder = _BUILDERS.get((name or "").strip().lower(), _BUILDERS["marker"])
    glyph = builder(0.0, 0.0, size_mm)
    if not glyph.is_valid:
        glyph = glyph.buffer(0)
    return translate(glyph, cx, cy)
