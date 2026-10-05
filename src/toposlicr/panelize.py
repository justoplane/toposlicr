"""Panelization — split oversized layers and hide seams (plan Section 5).

When a layer's part is larger than the laser bed it must be split into pieces
that each fit. The trick is to route the seam through the layer's **hidden
zone** — the area covered by the layer above — so the finished, glued-up stack
never shows a seam. That is a computable constraint: the *exposed length* of a
candidate seam (the part of it outside the hidden corridor) is the cost we
minimize.

Seams are straight guillotine lines or **Z seams** — a run at one position,
a jog across, and a run at a second position (an L when the second run exits
through an edge) — so a seam can turn a corner to stay inside a winding
corridor. What is optimized is *which* lines:

* **Exposed seam length is the objective**; fitting the bed is a constraint on
  the final pieces, not a per-cut bonus. A short beam search looks ahead through
  the follow-up cuts, so a hidden cut that needs one more cut beats an exposed
  cut that finishes immediately — each extra piece costs a fixed
  ``PIECE_PENALTY_MM`` of "seam-equivalent" so the splitter doesn't fragment
  layers for a few millimeters.
* **Bent seams cost a little extra** (``BEND_PENALTY_MM``) so a straight seam
  wins when hiding is a wash; a Z only appears when its jog buys hidden length.
* **Islands are split independently.** A layer made of several disjoint blobs
  only cuts the blobs that don't fit; a line across the whole layer would seam
  every island it crossed.
* **Seams stagger between adjacent layers** (brickwork): a cut within
  ``STAGGER_MM`` of the layer below's seam on the same axis is penalized, for
  glue strength rather than looks.
* **Fit counts either orientation** — the nester may rotate a part 90°, so a
  tall narrow piece that fits the bed sideways is not split at all.

Each part's score/engrave linework is distributed to the sub-part it falls
within, so symbology follows its piece.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry

from .config import Config
from .layers import LayerModel
from .parts import Part, part_id
from .symbology import hidden_zone

# Guard rails for the recursive splitter.
_MAX_DEPTH = 14
_N_CANDIDATES = 23
_N_JOGS = 9                # jog positions tried along the other axis for Z seams
_BEND_MAX_DEPTH = 3        # Z seams are considered this deep in the split tree
_EDGE_EPS = 1e-6
_PAD = 1.0                 # dividers overshoot the bbox by this much
_UNFINISHED_COST = 1e6     # cost of a piece the splitter failed to make fit
# Lookahead width per level (shrinks with depth so deep trees stay cheap).
_BEAM = 4
# A Z seam adds two corners to align; prefer a straight seam when it's a wash.
BEND_PENALTY_MM = 8.0
# Jogs shorter than this are not worth a bend.
MIN_JOG_MM = 15.0
# A cut never leaves a piece thinner than this along the cut axis.
MIN_PIECE_MM = 30.0
# Cost of one extra piece, in mm of exposed seam it is worth avoiding.
PIECE_PENALTY_MM = 25.0
# Brickwork: seams on the same axis closer than this to the layer below's seam …
STAGGER_MM = 12.0
# … cost this much (mm-equivalent).
STAGGER_PENALTY_MM = 40.0


@dataclass
class _Ctx:
    corridor: BaseGeometry
    bed: tuple[float, float]
    below: list[tuple[bool, float]] = field(default_factory=list)  # (vertical, c)


def panelize_parts(parts: list[Part], model: LayerModel, cfg: Config) -> list[Part]:
    """Split any part whose bbox exceeds the usable bed; return all pieces.

    Parts that already fit (in either orientation) pass through unchanged.
    Split parts are re-numbered ``L{layer}-P0, P1, …`` with ``parent_id`` set,
    and every operation geometry is clipped to the sub-part it belongs to.
    """
    bed = cfg.machine.usable_bed_mm
    corridor_cache: dict[int | None, BaseGeometry] = {}
    seams_by_layer: dict[int | None, list[tuple[bool, float]]] = {}
    out: list[Part] = []
    for part in parts:
        if fits_bed(part.outline, bed):
            out.append(part)
            continue
        k = part.layer_index
        ctx = _Ctx(corridor=_corridor_for(k, model, cfg, corridor_cache), bed=bed,
                   below=seams_by_layer.get(k - 1 if k is not None else None, []))
        pieces: list[Polygon] = []
        seams: list[tuple[bool, float]] = []
        # Islands are physically separate pieces: only cut the ones that don't fit.
        for island in _polys(part.outline):
            if fits_bed(island, bed):
                pieces.append(island)
                continue
            _cost, sub_pieces, sub_seams = _solve(island, ctx, 0)
            pieces.extend(sub_pieces)
            seams.extend(sub_seams)
        seams_by_layer.setdefault(k, []).extend(seams)
        out.extend(_parts_from_pieces(part, pieces))
    return out


def fits_bed(geom: BaseGeometry, bed: tuple[float, float]) -> bool:
    """True if the geometry's bbox fits the bed as-is or rotated 90°."""
    minx, miny, maxx, maxy = geom.bounds
    w, h = maxx - minx, maxy - miny
    bw, bh = bed
    return ((w <= bw + _EDGE_EPS and h <= bh + _EDGE_EPS)
            or (w <= bh + _EDGE_EPS and h <= bw + _EDGE_EPS))


def seam_visibility(parts: list[Part], model: LayerModel, k: int
                    ) -> tuple[float, float]:
    """(total, exposed) seam length in mm for layer ``k`` — exposed meaning not
    covered by the layer above at all. Both zero when the layer wasn't split."""
    from .render import layer_seams

    seams = layer_seams(parts, k)
    if seams is None or seams.is_empty:
        return 0.0, 0.0
    hz = hidden_zone(model, k)
    exposed = seams.length - (0.0 if hz.is_empty else seams.intersection(hz).length)
    return float(seams.length), float(max(0.0, exposed))


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


def _solve(geom: Polygon, ctx: _Ctx, depth: int
           ) -> tuple[float, list[Polygon], list[tuple[bool, float]]]:
    """Best split of one polygon: (cost, pieces, seams).

    Cost = exposed seam mm + PIECE_PENALTY_MM per extra piece + bend and
    stagger penalties, summed over the whole subtree. Candidates are straight
    guillotine lines and Z-shaped seams (two parallel runs joined by a jog);
    every candidate is scored from cached line profiles, ranked by a lower
    bound on the total cost, only the best few are actually cut, and a beam of
    those is explored recursively so the choice accounts for the cuts it forces.
    """
    if fits_bed(geom, ctx.bed):
        return 0.0, [geom], []
    if depth >= _MAX_DEPTH:
        # Out of depth with an oversized piece: never let this look cheap.
        return _UNFINISHED_COST, [geom], []

    bounds = geom.bounds
    minx, miny, maxx, maxy = bounds
    w, h = maxx - minx, maxy - miny
    bw, bh = ctx.bed
    # A cut only helps along an axis that overflows; when both do, try both.
    orientations = []
    if w > bw:
        orientations.append(True)
    if h > bh:
        orientations.append(False)
    if not orientations:
        orientations.append(w >= h)

    prof = _Profiles(geom, ctx.corridor, bounds)
    est = _Estimator(prof, ctx.bed)
    beam = max(1, _BEAM - depth)
    pool = max(2 * beam + 2, 4)
    cands: list[_Cand] = []
    for vertical in orientations:
        cands.extend(_straight_candidates(vertical, bounds, ctx, prof, est))
        if depth <= _BEND_MAX_DEPTH:
            cands.extend(_bent_candidates(vertical, bounds, ctx, prof, est, pool))
    if not cands:
        return 0.0, [geom], []

    # Rank by the lower bound on total cost, then by how much overflow the
    # cut leaves (so equally hidden cuts prefer the one that makes progress).
    # The best straight cuts are always cut and explored: a bent seam only
    # wins by beating them on the true recursive total.
    cands.sort(key=lambda c: (c.bound, c.leftover))
    straight = [c for c in cands if len(c.seams) == 1][:2]
    chosen = cands[:pool] + [c for c in straight if c not in cands[:pool]]
    cut: list[tuple[tuple[float, float], _Cand, list[Polygon]]] = []
    for cand in chosen:
        halves = _cut_with(geom, cand.divider)
        if len(halves) < 2:
            continue
        real = [hh.bounds for hh in halves]
        bound = cand.cost + PIECE_PENALTY_MM * (len(halves) - 1) + est.remaining(real)
        cut.append(((bound, est.leftover(real)), cand, halves))
    if not cut:
        vert, mid = orientations[0], _mid(geom, orientations[0])
        halves = _cut_with(geom, _straight_divider(vert, mid, bounds))
        if len(halves) < 2:
            return _UNFINISHED_COST, [geom], []
        return _recurse(halves, 0.0, [(vert, mid)], ctx, depth)
    cut.sort(key=lambda t: t[0])
    explore = cut[:beam]
    best_straight = next((t for t in cut if len(t[1].seams) == 1), None)
    if best_straight is not None and best_straight not in explore:
        explore.append(best_straight)

    best: tuple[float, list[Polygon], list[tuple[bool, float]]] | None = None
    for _key, cand, halves in explore:
        result = _recurse(halves, cand.cost, cand.seams, ctx, depth)
        if best is None or result[0] < best[0]:
            best = result
    return best


def _recurse(halves, local, seams_here, ctx, depth):
    cost = local + PIECE_PENALTY_MM * (len(halves) - 1)
    pieces: list[Polygon] = []
    seams: list[tuple[bool, float]] = list(seams_here)
    for half in halves:
        sub_cost, sub_pieces, sub_seams = _solve(half, ctx, depth + 1)
        cost += sub_cost
        pieces.extend(sub_pieces)
        seams.extend(sub_seams)
    return cost, pieces, seams


@dataclass
class _Cand:
    cost: float                            # this seam's own cost (exposed + penalties)
    bound: float                           # cost + lower bound of what remains
    leftover: float                        # overflow the halves still carry (tie-break)
    divider: Polygon                       # region on one side of the seam
    seams: list[tuple[bool, float]]        # (vertical, position) per straight run


def _min_pieces(w: float, h: float, bed: tuple[float, float]) -> int:
    """Fewest bed-sized pieces a w×h box can be cut into (either orientation)."""
    bw, bh = bed

    def n(a, b):
        return max(1, math.ceil((a - _EDGE_EPS) / b))
    return min(n(w, bw) * n(h, bh), n(w, bh) * n(h, bw))


class _Estimator:
    """Lower bound on what a set of pieces still costs to finish.

    Uses only line profiles already cached for the parent polygon, so ranking
    thousands of candidates costs no new geometry operations.
    """

    def __init__(self, prof: _Profiles, bed: tuple[float, float]):
        self.prof, self.bed = prof, bed
        self._memo: dict[tuple, float] = {}

    def remaining(self, bboxes) -> float:
        return sum(self._one(bb) for bb in bboxes)

    def leftover(self, bboxes) -> float:
        bw, bh = self.bed
        total = 0.0
        for (x0, y0, x1, y1) in bboxes:
            if not fits_bed(box(x0, y0, x1, y1), self.bed):
                total += max(0.0, (x1 - x0) - bw) + max(0.0, (y1 - y0) - bh)
        return total

    def _one(self, bb) -> float:
        key = tuple(round(v, 3) for v in bb)
        if key not in self._memo:
            x0, y0, x1, y1 = bb
            pieces = _min_pieces(x1 - x0, y1 - y0, self.bed)
            self._memo[key] = 0.0 if pieces <= 1 else (
                PIECE_PENALTY_MM * (pieces - 1) + self._cheapest_line(x0, y0, x1, y1))
        return self._memo[key]

    def _cheapest_line(self, x0, y0, x1, y1) -> float:
        """Least exposed straight cut across an overflowing axis of the box,
        among the parent's already-profiled lines that cross it."""
        bw, bh = self.bed
        best = math.inf
        if x1 - x0 > bw:
            for c in self.prof.cached_positions(True):
                if x0 + MIN_PIECE_MM <= c <= x1 - MIN_PIECE_MM:
                    best = min(best, self.prof.exposed(True, c, y0, y1))
        if y1 - y0 > bh:
            for c in self.prof.cached_positions(False):
                if y0 + MIN_PIECE_MM <= c <= y1 - MIN_PIECE_MM:
                    best = min(best, self.prof.exposed(False, c, x0, x1))
        return 0.0 if best is math.inf else best


class _Profiles:
    """Exposed-length profiles of axis-aligned lines through a polygon.

    For a line at position ``c`` (vertical → x = c, else y = c) this caches the
    intervals along the line that lie inside the polygon but outside the
    hidden corridor. Any sub-span's exposed length is then an interval sum, so
    thousands of candidate seams score without touching the polygon again.
    """

    def __init__(self, geom: BaseGeometry, corridor: BaseGeometry,
                 bounds: tuple[float, float, float, float]):
        self.geom, self.corridor, self.bounds = geom, corridor, bounds
        self._cache: dict[tuple[bool, float], list[tuple[float, float]]] = {}

    def _intervals(self, vertical: bool, c: float) -> list[tuple[float, float]]:
        key = (vertical, round(c, 6))
        if key in self._cache:
            return self._cache[key]
        minx, miny, maxx, maxy = self.bounds
        pad = _PAD
        line = (LineString([(c, miny - pad), (c, maxy + pad)]) if vertical
                else LineString([(minx - pad, c), (maxx + pad, c)]))
        inside = line.intersection(self.geom)
        exposed = inside if self.corridor.is_empty else inside.difference(self.corridor)
        out: list[tuple[float, float]] = []
        for seg in _segments(exposed):
            coords = list(seg.coords)
            vals = [pt[1] if vertical else pt[0] for pt in coords]
            out.append((min(vals), max(vals)))
        self._cache[key] = out
        return out

    def cached_positions(self, vertical: bool) -> list[float]:
        return sorted(c for v, c in self._cache if v == vertical)

    def exposed(self, vertical: bool, c: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in self._intervals(vertical, c))

    def cumulative(self, vertical: bool, c: float, ts: np.ndarray) -> np.ndarray:
        """Exposed length of the line at ``c`` from its start up to each t."""
        out = np.zeros(len(ts))
        for a, b in self._intervals(vertical, c):
            out += np.clip(ts - a, 0.0, b - a)
        return out


def _segments(geom: BaseGeometry) -> list[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, LineString)]


def _axes(vertical: bool, bounds):
    """(lo, hi) along the cut axis and (olo, ohi) along the other."""
    minx, miny, maxx, maxy = bounds
    return ((minx, maxx), (miny, maxy)) if vertical else ((miny, maxy), (minx, maxx))


def _side_bboxes(vertical, lo, low_hi, high_lo, hi, olo, ohi):
    """Estimated bboxes of the two sides: low side spans [lo, low_hi] on the cut
    axis, high side [high_lo, hi]; both span the full other axis."""
    if vertical:
        return [(lo, olo, low_hi, ohi), (high_lo, olo, hi, ohi)]
    return [(olo, lo, ohi, low_hi), (olo, high_lo, ohi, hi)]


def _positions(lo, hi, bed_dim):
    """Candidate cut positions that leave at least MIN_PIECE_MM on each side."""
    return [c for c in _candidate_positions(lo, hi, bed_dim)
            if c - lo >= MIN_PIECE_MM and hi - c >= MIN_PIECE_MM]


def _straight_candidates(vertical, bounds, ctx, prof, est) -> list[_Cand]:
    (lo, hi), (olo, ohi) = _axes(vertical, bounds)
    bed_dim = ctx.bed[0] if vertical else ctx.bed[1]
    out = []
    for c in _positions(lo, hi, bed_dim):
        cost = prof.exposed(vertical, c, olo - _PAD, ohi + _PAD)
        cost += _stagger_penalty([(vertical, c)], ctx.below)
        bbs = _side_bboxes(vertical, lo, c, c, hi, olo, ohi)
        out.append(_Cand(cost, cost + PIECE_PENALTY_MM + est.remaining(bbs),
                         est.leftover(bbs), _straight_divider(vertical, c, bounds),
                         [(vertical, c)]))
    return out


def _bent_candidates(vertical, bounds, ctx, prof, est, top: int) -> list[_Cand]:
    """Z seams: run at c1 up to the jog at t0, jog across to c2, run on to the
    far edge (c2 may be an edge of the bbox, which makes an L). All
    (c1, t0, c2) combinations are scored as one numpy tensor from the cached
    profiles; only the ``top`` by bound are materialized."""
    (lo, hi), (olo, ohi) = _axes(vertical, bounds)
    bed_dim = ctx.bed[0] if vertical else ctx.bed[1]
    P = np.array(_positions(lo, hi, bed_dim))
    if len(P) == 0:
        return []
    Q = np.concatenate([P, [lo - _PAD, hi + _PAD]])
    ospan = ohi - olo
    T = np.array([olo + ospan * (i + 1) / (_N_JOGS + 1) for i in range(_N_JOGS)])
    T = T[(T - olo >= MIN_PIECE_MM) & (ohi - T >= MIN_PIECE_MM)]
    if len(T) == 0:
        return []
    # first[i, j]: run at P[i] from the near edge to T[j]
    first = np.stack([prof.cumulative(vertical, c, T) for c in P])              # (nP, nT)
    # second[k, j]: run at Q[k] from T[j] to the far edge
    second = np.stack([
        prof.exposed(vertical, c, olo - _PAD, ohi + _PAD) - prof.cumulative(vertical, c, T)
        for c in Q])                                                             # (nQ, nT)
    # jog[j, i, k]: along the line at T[j] between P[i] and Q[k]
    cumP = np.stack([prof.cumulative(not vertical, t, P) for t in T])           # (nT, nP)
    cumQ = np.stack([prof.cumulative(not vertical, t, Q) for t in T])           # (nT, nQ)
    jog = np.abs(cumQ[:, None, :] - cumP[:, :, None])                            # (nT, nP, nQ)

    cost = (first.T[:, :, None] + jog + second.T[:, None, :]) + BEND_PENALTY_MM   # (nT, nP, nQ)
    stag_p = np.array([_stagger_penalty([(vertical, c)], ctx.below) for c in P])
    stag_q = np.array([_stagger_penalty([(vertical, c)], ctx.below) if lo <= c <= hi else 0.0
                       for c in Q])
    stag_t = np.array([_stagger_penalty([(not vertical, t)], ctx.below) for t in T])
    cost += np.maximum(np.maximum(stag_p[None, :, None], stag_q[None, None, :]),
                       stag_t[:, None, None])
    cost[:, np.abs(Q[None, :] - P[:, None]) < MIN_JOG_MM] = np.inf

    # Lower bound: the low side ends at max(P, clip(Q)), the high side starts at
    # min(P, clip(Q)); both spans are memoized per distinct position.
    Qc = np.clip(Q, lo, hi)
    hi_end = np.maximum(P[:, None], Qc[None, :])                                 # (nP, nQ)
    lo_end = np.minimum(P[:, None], Qc[None, :])
    vals = np.unique(np.concatenate([hi_end.ravel(), lo_end.ravel()]))
    rem_low = {v: est.remaining([_side_bboxes(vertical, lo, v, v, hi, olo, ohi)[0]]) for v in vals}
    rem_high = {v: est.remaining([_side_bboxes(vertical, lo, v, v, hi, olo, ohi)[1]]) for v in vals}
    left_low = {v: est.leftover([_side_bboxes(vertical, lo, v, v, hi, olo, ohi)[0]]) for v in vals}
    left_high = {v: est.leftover([_side_bboxes(vertical, lo, v, v, hi, olo, ohi)[1]]) for v in vals}
    rem = (np.vectorize(rem_low.get)(hi_end) + np.vectorize(rem_high.get)(lo_end))
    left = (np.vectorize(left_low.get)(hi_end) + np.vectorize(left_high.get)(lo_end))
    bound = cost + PIECE_PENALTY_MM + rem[None, :, :]

    flat = np.lexsort((np.broadcast_to(left[None], bound.shape).ravel(), bound.ravel()))
    out = []
    for idx in flat[:top]:
        j, i, k = np.unravel_index(idx, bound.shape)
        if not np.isfinite(bound[j, i, k]):
            break
        c1, t0, c2 = float(P[i]), float(T[j]), float(Q[k])
        seams = [(vertical, c1), (not vertical, t0)]
        if lo <= c2 <= hi:
            seams.append((vertical, c2))
        out.append(_Cand(float(cost[j, i, k]), float(bound[j, i, k]), float(left[i, k]),
                         _z_divider(vertical, c1, t0, c2, bounds), seams))
    return out


def _straight_divider(vertical: bool, c: float, bounds) -> Polygon:
    minx, miny, maxx, maxy = bounds
    p = _PAD
    if vertical:
        return box(minx - p, miny - p, c, maxy + p)
    return box(minx - p, miny - p, maxx + p, c)


def _z_divider(vertical: bool, c1: float, t0: float, c2: float, bounds) -> Polygon:
    """Region on the low side of a Z seam (in cut-axis / other-axis coords)."""
    (lo, hi), (olo, ohi) = _axes(vertical, bounds)
    p = _PAD
    pts_uv = [(lo - p, olo - p), (c1, olo - p), (c1, t0),
              (c2, t0), (c2, ohi + p), (lo - p, ohi + p)]
    pts = pts_uv if vertical else [(v, u) for u, v in pts_uv]
    return Polygon(pts)


def _cut_with(geom: BaseGeometry, divider: Polygon) -> list[Polygon]:
    pieces = _polys(geom.intersection(divider)) + _polys(geom.difference(divider))
    return [p for p in pieces if p.area > _EDGE_EPS]


def _mid(geom: BaseGeometry, vertical: bool) -> float:
    minx, miny, maxx, maxy = geom.bounds
    return (minx + maxx) / 2.0 if vertical else (miny + maxy) / 2.0


def _candidate_positions(lo: float, hi: float, bed_dim: float) -> list[float]:
    """Sweep positions in (lo, hi), plus the two cuts that make one side exactly
    bed-sized."""
    span = hi - lo
    if span <= 0:
        return []
    positions = [lo + span * (i + 1) / (_N_CANDIDATES + 1) for i in range(_N_CANDIDATES)]
    for extra in (lo + bed_dim, hi - bed_dim):
        if lo + _EDGE_EPS < extra < hi - _EDGE_EPS:
            positions.append(extra)
    return sorted(positions)


def _stagger_penalty(seams: list[tuple[bool, float]], below: list[tuple[bool, float]]) -> float:
    for vertical, c in seams:
        for v, bc in below:
            if v == vertical and abs(bc - c) < STAGGER_MM:
                return STAGGER_PENALTY_MM
    return 0.0


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
