"""SVG output with laser-cutter conventions (plan Section 7.1).

Two conventions matter for a Glowforge and most laser software:

* **1:1 physical units** — explicit ``width``/``height`` in ``mm`` with a
  matching ``viewBox``, sidestepping the 72/96 DPI ambiguity between tools.
* **Color = operation** — each cut/score/engrave intent gets its own stroke or
  fill color, taken from the machine profile, which the laser app maps to steps.

Model coordinates are y-up with the origin at the map's SW corner; SVG is y-down,
so every geometry is flipped on the way out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

# Thin hairline so vector ops aren't misread as fills; Glowforge ignores width
# for cut/score but other tools render it.
_HAIRLINE_MM = 0.1


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _ring_to_d(coords, height_mm: float) -> str:
    parts = []
    for i, (x, y) in enumerate(coords):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{_fmt(x)},{_fmt(height_mm - y)}")
    parts.append("Z")
    return "".join(parts)


def polygon_to_path_d(geom: BaseGeometry, height_mm: float) -> str:
    """Serialize a (multi)polygon to an SVG path ``d`` (exterior + holes)."""
    polys = _polys(geom)
    subpaths = []
    for poly in polys:
        subpaths.append(_ring_to_d(poly.exterior.coords, height_mm))
        for interior in poly.interiors:
            subpaths.append(_ring_to_d(interior.coords, height_mm))
    return "".join(subpaths)


def line_to_path_d(geom: BaseGeometry, height_mm: float) -> str:
    """Serialize a (multi)linestring to an open SVG path ``d``."""
    lines = _lines(geom)
    subpaths = []
    for line in lines:
        pts = list(line.coords)
        if len(pts) < 2:
            continue
        seg = [f"M{_fmt(pts[0][0])},{_fmt(height_mm - pts[0][1])}"]
        seg += [f"L{_fmt(x)},{_fmt(height_mm - y)}" for x, y in pts[1:]]
        subpaths.append("".join(seg))
    return "".join(subpaths)


def _polys(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]


def _lines(geom: BaseGeometry) -> list[LineString]:
    if isinstance(geom, LineString):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]


@dataclass
class _Element:
    d: str
    stroke: str | None
    fill: str | None
    label: str


@dataclass
class SvgDocument:
    """Accumulates path elements grouped by operation, then serializes to SVG."""

    width_mm: float
    height_mm: float
    title: str = "toposlicr"
    elements: list[_Element] = field(default_factory=list)

    def add_cut(self, geom: BaseGeometry, color: str, label: str = "cut") -> None:
        d = polygon_to_path_d(geom, self.height_mm)
        if d:
            self.elements.append(_Element(d, stroke=color, fill="none", label=label))

    def add_score(self, geom: BaseGeometry, color: str, label: str = "score") -> None:
        """Score linework — accepts lines or polygon boundaries."""
        if isinstance(geom, (Polygon, MultiPolygon)):
            d = polygon_to_path_d(geom, self.height_mm)
        else:
            d = line_to_path_d(geom, self.height_mm)
        if d:
            self.elements.append(_Element(d, stroke=color, fill="none", label=label))

    def add_engrave(self, geom: BaseGeometry, color: str, label: str = "engrave") -> None:
        """Filled region — Glowforge treats closed filled paths as engrave."""
        d = polygon_to_path_d(geom, self.height_mm)
        if d:
            self.elements.append(_Element(d, stroke="none", fill=color, label=label))

    def to_svg(self) -> str:
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{_fmt(self.width_mm)}mm" height="{_fmt(self.height_mm)}mm" '
            f'viewBox="0 0 {_fmt(self.width_mm)} {_fmt(self.height_mm)}">',
            f"<title>{escape(self.title)}</title>",
        ]
        for el in self.elements:
            attrs = [f'd="{el.d}"']
            if el.fill and el.fill != "none":
                attrs.append(f'fill="{el.fill}"')
                if el.stroke and el.stroke != "none":
                    attrs.append(f'stroke="{el.stroke}"')
            else:
                attrs.append('fill="none"')
                attrs.append(f'stroke="{el.stroke}"')
                attrs.append(f'stroke-width="{_HAIRLINE_MM}"')
            attrs.append(f'data-op="{escape(el.label)}"')
            lines.append(f"<path {' '.join(attrs)}/>")
        lines.append("</svg>")
        return "\n".join(lines)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_svg(), encoding="utf-8")
        return p
