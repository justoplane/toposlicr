"""Contour extraction and geometry cleanup (plan Section 3).

Turns a projected DEM into clean, nested Shapely polygons — one (multi)polygon
per elevation band — expressed in **model millimeters** ready for SVG. All the
2D polygon math the rest of the pipeline relies on starts here.

The band model: layer ``k`` is the set of terrain with elevation ≥
``base + k*interval``. Higher layers are subsets of lower ones, so the stack is
strictly nested — which we assert and, if needed, gently enforce.
"""

from __future__ import annotations

import numpy as np
from contourpy import FillType, contour_generator
from scipy.ndimage import gaussian_filter
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .dem.base import DemRaster


def smooth_dem(data: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian pre-smoothing. Smoothing the raster (not the vectors afterwards)
    yields far more organic contour curves (plan Section 3.1).

    NaN nodata cells (e.g. slivers a reprojection leaves outside the source
    footprint, or source voids/ocean) are preserved through smoothing via a *normalized*
    convolution: filling them with a constant before blurring would bleed that
    value into the terrain and make ``contourpy`` treat them as real ground —
    contaminating the base-layer footprint with the full raster rectangle.
    """
    arr = data.astype("float64")
    if sigma_px <= 0:
        return arr  # NaN passes through; contourpy masks nodata cells.
    mask = np.isnan(arr)
    if not mask.any():
        return gaussian_filter(arr, sigma=sigma_px)
    # Normalized convolution: blur values and a validity weight, then divide,
    # so nodata neither bleeds in nor drags edges toward a fill constant.
    filled = np.where(mask, 0.0, arr)
    weight = (~mask).astype("float64")
    smoothed = gaussian_filter(filled, sigma=sigma_px)
    norm = gaussian_filter(weight, sigma=sigma_px)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed / norm
    out[mask] = np.nan            # restore nodata so contourpy keeps masking it
    return out


def _filled_to_polygons(cg, lower: float, upper: float) -> list[Polygon]:
    """Convert one contourpy filled level into Shapely polygons with holes."""
    points_list, offsets_list, outer_list = cg.filled(lower, upper)
    polys: list[Polygon] = []
    for points, offsets, outer in zip(points_list, offsets_list, outer_list, strict=True):
        if points is None or outer is None:
            continue
        for j in range(len(outer) - 1):
            o_start, o_end = int(outer[j]), int(outer[j + 1])
            exterior = points[offsets[o_start]:offsets[o_start + 1]]
            holes = [
                points[offsets[r]:offsets[r + 1]]
                for r in range(o_start + 1, o_end)
            ]
            if len(exterior) < 4:
                continue
            poly = Polygon(exterior, holes)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 0:
                polys.append(poly)
    return polys


def band_polygon(dem: DemRaster, threshold_m: float, xs: np.ndarray,
                 ys: np.ndarray, z: np.ndarray) -> BaseGeometry:
    """Region with elevation ≥ ``threshold_m`` as a (multi)polygon in DEM CRS."""
    zmax = float(np.nanmax(z))
    upper = max(zmax + 1.0, threshold_m + 1.0)
    cg = contour_generator(
        x=xs, y=ys, z=z, fill_type=FillType.ChunkCombinedOffsetOffset,
    )
    polys = _filled_to_polygons(cg, threshold_m, upper)
    if not polys:
        return MultiPolygon()
    return unary_union(polys)


def chaikin(geom: BaseGeometry, iterations: int = 1) -> BaseGeometry:
    """Chaikin corner-cutting for smooth flowing edges (plan Section 3.4)."""
    if iterations <= 0 or geom.is_empty:
        return geom
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for poly in polys:
        ext = _chaikin_ring(list(poly.exterior.coords), iterations)
        holes = [_chaikin_ring(list(r.coords), iterations) for r in poly.interiors]
        if len(ext) >= 4:
            out.append(Polygon(ext, [h for h in holes if len(h) >= 4]))
    return _as_multipolygon(unary_union(out) if out else MultiPolygon())


def _chaikin_ring(coords: list[tuple[float, float]], iterations: int) -> list:
    pts = coords[:-1] if coords[0] == coords[-1] else coords[:]
    for _ in range(iterations):
        new = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            new.append((0.75 * p[0] + 0.25 * q[0], 0.75 * p[1] + 0.25 * q[1]))
            new.append((0.25 * p[0] + 0.75 * q[0], 0.25 * p[1] + 0.75 * q[1]))
        pts = new
    pts.append(pts[0])
    return pts


def clean_geometry(geom: BaseGeometry, *, simplify_tol_mm: float,
                   min_area_mm2: float, min_hole_mm2: float,
                   close_radius_mm: float = 0.0) -> BaseGeometry:
    """Simplify, drop slivers/pinholes, and optionally morphologically close.

    Tolerances are in model mm; ``simplify_tol_mm`` sits below the kerf so there
    is no visible fidelity loss but a large node-count reduction for the laser.
    """
    if geom.is_empty:
        return geom
    if close_radius_mm > 0:
        geom = geom.buffer(close_radius_mm).buffer(-close_radius_mm)
    if simplify_tol_mm > 0:
        geom = geom.simplify(simplify_tol_mm, preserve_topology=True)
    return _drop_small(geom, min_area_mm2, min_hole_mm2)


def _drop_small(geom: BaseGeometry, min_area_mm2: float,
                min_hole_mm2: float) -> BaseGeometry:
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    kept = []
    for poly in polys:
        if not isinstance(poly, Polygon) or poly.area < min_area_mm2:
            continue
        holes = [
            r for r in poly.interiors
            if Polygon(r).area >= min_hole_mm2
        ]
        kept.append(Polygon(poly.exterior, holes))
    return _as_multipolygon(unary_union(kept) if kept else MultiPolygon())


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    if geom.is_empty:
        return MultiPolygon()
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    # GeometryCollection etc.: keep only polygonal parts.
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(parts)
