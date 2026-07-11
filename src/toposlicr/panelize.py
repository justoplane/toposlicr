"""Panelization — split oversized layers and hide seams (plan Section 5).

When a layer's part is larger than the laser bed it must be split into pieces
that each fit. The trick is to route the seam through the layer's **hidden
zone** — the area covered by the layer above — so the finished, glued-up stack
never shows a seam. This is a computable constraint: a candidate cut is scored
partly by how much of it falls inside that hidden corridor.

v1 uses a guillotine strategy: recursively pick the best straight cut (vertical
or horizontal) that makes the pieces closer to fitting the bed while keeping the
seam inside the corridor and the two areas balanced. Each part's score/engrave
linework is distributed to the sub-part it falls within, so symbology follows
its piece.
"""

from __future__ import annotations

from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from .config import Config
from .layers import LayerModel
from .parts import Part, part_id
from .symbology import hidden_zone

# Guard rails for the recursive guillotine.
_MAX_DEPTH = 14
_N_CANDIDATES = 19
_EDGE_EPS = 1e-6


def panelize_parts(parts: list[Part], model: LayerModel, cfg: Config) -> list[Part]:
    """Split any part whose bbox exceeds the usable bed; return all pieces.

    Parts that already fit pass through unchanged. Split parts are re-numbered
    ``L{layer}-P0, P1, …`` with ``parent_id`` set, and every operation geometry
    is clipped to the sub-part it belongs to.
    """
    bed = cfg.machine.usable_bed_mm
    corridor_cache: dict[int | None, BaseGeometry] = {}
    out: list[Part] = []
    for part in parts:
        if _fits(part.outline, bed):
            out.append(part)
            continue
        corridor = _corridor_for(part.layer_index, model, cfg, corridor_cache)
        pieces = _split_to_fit(_as_multipolygon(part.outline), corridor, bed)
        out.extend(_parts_from_pieces(part, pieces))
    return out


def _corridor_for(layer_index: int | None, model: LayerModel, cfg: Config,
                  cache: dict[int | None, BaseGeometry]) -> BaseGeometry:
    if layer_index in cache:
        return cache[layer_index]
    corridor: BaseGeometry = Polygon()
    if layer_index is not None:
        hz = hidden_zone(model, layer_index)
        if not hz.is_empty:
            eroded = hz.buffer(-cfg.panelization.seam_margin_mm)
            corridor = eroded if not eroded.is_empty else Polygon()
    cache[layer_index] = corridor
    return corridor


# --- geometry helpers ------------------------------------------------------

def _fits(geom: BaseGeometry, bed: tuple[float, float]) -> bool:
    minx, miny, maxx, maxy = geom.bounds
    return (maxx - minx) <= bed[0] + _EDGE_EPS and (maxy - miny) <= bed[1] + _EDGE_EPS


def _polys(geom: BaseGeometry) -> list[Polygon]:
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    polys = _polys(geom)
    return MultiPolygon(polys) if polys else MultiPolygon()


def _split_to_fit(outline: BaseGeometry, corridor: BaseGeometry,
                  bed: tuple[float, float], depth: int = 0) -> list[Polygon]:
    """Recursively guillotine ``outline`` until every piece fits the bed."""
    polys = _polys(outline)
    if not polys:
        return []
    geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]

    if _fits(geom, bed) or depth >= _MAX_DEPTH:
        return polys

    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    bw, bh = bed
    # Cut the dimension that overflows most (both if both overflow).
    if w > bw and h > bh:
        vertical = (w - bw) >= (h - bh)
    elif w > bw:
        vertical = True
    else:
        vertical = False

    halves = _best_cut(geom, corridor, bed, vertical)
    if halves is None:
        halves = _midline_cut(geom, vertical)

    result: list[Polygon] = []
    for half in halves:
        result.extend(_split_to_fit(half, corridor, bed, depth + 1))
    return result or polys


def _candidate_positions(lo: float, hi: float, bed_dim: float) -> list[float]:
    """Sweep positions in (lo, hi), biased toward cuts that let a piece fit."""
    span = hi - lo
    if span <= 0:
        return []
    positions = [lo + span * (i + 1) / (_N_CANDIDATES + 1) for i in range(_N_CANDIDATES)]
    # Always try the two cuts that make one side exactly bed-sized.
    for extra in (lo + bed_dim, hi - bed_dim):
        if lo + _EDGE_EPS < extra < hi - _EDGE_EPS:
            positions.append(extra)
    return sorted(positions)


def _best_cut(geom: BaseGeometry, corridor: BaseGeometry, bed: tuple[float, float],
              vertical: bool) -> list[Polygon] | None:
    minx, miny, maxx, maxy = geom.bounds
    bed_dim = bed[0] if vertical else bed[1]
    lo, hi = (minx, maxx) if vertical else (miny, maxy)
    best: tuple[float, list[Polygon]] | None = None
    for c in _candidate_positions(lo, hi, bed_dim):
        halves = _cut_at(geom, c, vertical, (minx, miny, maxx, maxy))
        if len(halves) < 2:
            continue
        score = _score_cut(halves, c, vertical, corridor, bed,
                            (minx, miny, maxx, maxy))
        if best is None or score > best[0]:
            best = (score, halves)
    return best[1] if best else None


def _cut_at(geom: BaseGeometry, c: float, vertical: bool,
            bounds: tuple[float, float, float, float]) -> list[Polygon]:
    minx, miny, maxx, maxy = bounds
    pad = 1.0
    if vertical:
        left = box(minx - pad, miny - pad, c, maxy + pad)
        right = box(c, miny - pad, maxx + pad, maxy + pad)
    else:
        left = box(minx - pad, miny - pad, maxx + pad, c)
        right = box(minx - pad, c, maxx + pad, maxy + pad)
    pieces: list[Polygon] = []
    for half in (left, right):
        pieces.extend(_polys(geom.intersection(half)))
    return pieces


def _score_cut(halves: list[Polygon], c: float, vertical: bool,
               corridor: BaseGeometry, bed: tuple[float, float],
               bounds: tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bounds
    fit_count = sum(1 for p in halves if _fits(p, bed))
    total_area = sum(p.area for p in halves) or 1.0
    # Balance: perfectly even split scores 1.0.
    balance = (1.0 - abs(halves[0].area - halves[1].area) / total_area
               if len(halves) == 2 else 0.5)

    # Corridor fraction of the seam line clipped to the geometry.
    seam = (LineString([(c, miny), (c, maxy)]) if vertical
            else LineString([(minx, c), (maxx, c)]))
    union = MultiPolygon(halves) if len(halves) > 1 else halves[0]
    seam_in_geom = seam.intersection(union)
    corridor_frac = 0.0
    if not corridor.is_empty and not seam_in_geom.is_empty and seam_in_geom.length > 0:
        corridor_frac = seam_in_geom.intersection(corridor).length / seam_in_geom.length

    return fit_count * 100.0 + corridor_frac * 10.0 + balance


def _midline_cut(geom: BaseGeometry, vertical: bool) -> list[Polygon]:
    minx, miny, maxx, maxy = geom.bounds
    c = (minx + maxx) / 2.0 if vertical else (miny + maxy) / 2.0
    return _cut_at(geom, c, vertical, (minx, miny, maxx, maxy))


# --- part assembly ---------------------------------------------------------

def _parts_from_pieces(parent: Part, pieces: list[Polygon]) -> list[Part]:
    if len(pieces) <= 1:
        # Nothing was gained; keep the original part unchanged.
        return [parent]
    new_parts: list[Part] = []
    for i, piece in enumerate(pieces):
        pid = (part_id(parent.layer_index, i, parent.kind)
               if parent.layer_index is not None else f"{parent.part_id}-{i}")
        sub = Part(part_id=pid, kind=parent.kind, material=parent.material,
                   outline=MultiPolygon([piece]), layer_index=parent.layer_index,
                   parent_id=parent.part_id)
        for role, geom in parent.ops.items():
            clipped = geom.intersection(piece)
            sub.add_op(role, clipped)
        new_parts.append(sub)
    return new_parts
