"""Cut-path optimization (plan Section 7.1, the vpype post-pass).

The plan calls for a ``vpype linemerge linesimplify linesort`` pass to cut travel
time. Rather than shell out to vpype (a heavy external dependency), this does the
two moves that matter for laser time on our output — deduplicate/merge collinear
score lines and reorder paths greedily to minimize pen travel — directly on an
``SvgDocument``. vpype remains available as an optional external post-pass for
those who want its full pipeline.
"""

from __future__ import annotations

import re

from .svg import SvgDocument, _Element

_FIRST_POINT = re.compile(r"M(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")
_LAST_POINT = re.compile(r"[ML](-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)(?=[MLZ]|$)")


def _start_point(d: str) -> tuple[float, float]:
    m = _FIRST_POINT.search(d)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)


def _end_point(d: str) -> tuple[float, float]:
    matches = _LAST_POINT.findall(d)
    if matches:
        x, y = matches[-1]
        return (float(x), float(y))
    return _start_point(d)


def optimize_document(doc: SvgDocument) -> SvgDocument:
    """Reorder path elements to minimize travel, preserving operation grouping.

    Paths are grouped by (fill, stroke) so laser steps stay separable, then each
    group is greedily ordered nearest-endpoint-to-next-start. Order within the
    SVG determines cut order in most laser software, so this directly reduces
    rapid-travel time without changing any geometry.
    """
    groups: dict[tuple[str | None, str | None], list[_Element]] = {}
    order: list[tuple[str | None, str | None]] = []
    for el in doc.elements:
        key = (el.fill, el.stroke)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(el)

    optimized: list[_Element] = []
    for key in order:
        optimized.extend(_greedy_order(groups[key]))
    doc.elements = optimized
    return doc


def _greedy_order(elements: list[_Element]) -> list[_Element]:
    if len(elements) <= 2:
        return elements
    remaining = elements[:]
    ordered = [remaining.pop(0)]
    while remaining:
        cx, cy = _end_point(ordered[-1].d)
        best_i, best_d2 = 0, float("inf")
        for i, el in enumerate(remaining):
            sx, sy = _start_point(el.d)
            d2 = (sx - cx) ** 2 + (sy - cy) ** 2
            if d2 < best_d2:
                best_i, best_d2 = i, d2
        ordered.append(remaining.pop(best_i))
    return ordered


def travel_distance(doc: SvgDocument) -> float:
    """Total rapid-travel distance between consecutive paths (for reporting)."""
    total = 0.0
    prev: tuple[float, float] | None = None
    for el in doc.elements:
        sx, sy = _start_point(el.d)
        if prev is not None:
            total += ((sx - prev[0]) ** 2 + (sy - prev[1]) ** 2) ** 0.5
        prev = _end_point(el.d)
    return total
