"""The layer model — the pipeline's central data structure.

``build_layer_model`` runs Phase 1's geometry: reproject the DEM into a local
transverse Mercator centered on the box (true meters, north up — see
``geo.BBox.local_crs`` for why not plain UTM), fix the cut frame as the
axis-aligned rectangle inscribed in the projected box, derive the scale from the
DEM's own elevation range, slice into nested elevation bands, clip each band to
the frame, clean it, and express everything in model millimeters. Downstream
phases (symbology, panelization, nesting) consume ``LayerModel`` and never touch
rasters again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, box
from shapely.geometry.base import BaseGeometry

from .config import Config
from .contour import band_polygon, chaikin, clean_geometry, smooth_dem
from .dem.base import DemRaster
from .geo import BBox, projected_frame
from .scale import ScaleResult, solve_scale


@dataclass
class Layer:
    """One physical plywood layer: the terrain at or above ``threshold_m``."""

    index: int                     # k; 0 is the base layer
    threshold_m: float             # elevation ≥ this belongs to the layer
    geometry: MultiPolygon         # footprint in model mm (origin at map SW, y-up)
    warnings: list[str] = field(default_factory=list)

    @property
    def area_mm2(self) -> float:
        return self.geometry.area


@dataclass
class LayerModel:
    """The full nested stack plus the metadata needed to lay it out and cut it."""

    layers: list[Layer]
    interval_m: float
    base_elev_m: float
    max_elev_m: float
    model_width_mm: float
    model_height_mm: float
    scale: ScaleResult
    utm_epsg: int                       # nearest UTM zone, informational; 0 when flat
    world_to_model: tuple[float, float, float, float, float, float]
    flat: bool = False                  # fictional bundle → no CRS / no reprojection
    crs: str = ""                       # projected CRS of the world frame ("" when flat)
    warnings: list[str] = field(default_factory=list)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def footprint(self, k: int) -> BaseGeometry:
        """Geometry of layer ``k``; empty polygon if out of range."""
        if 0 <= k < len(self.layers):
            return self.layers[k].geometry
        return MultiPolygon()


def _world_to_model_matrix(x_min: float, y_min: float, k_mm_per_m: float
                           ) -> tuple[float, float, float, float, float, float]:
    """Affine (for shapely) mapping world meters → model mm, SW corner at origin."""
    # shapely affine format: [a, b, d, e, xoff, yoff] => x' = a*x + b*y + xoff
    return (k_mm_per_m, 0.0, 0.0, k_mm_per_m,
            -x_min * k_mm_per_m, -y_min * k_mm_per_m)


def build_layer_model(dem_geographic: DemRaster, cfg: Config,
                      bbox: BBox | None = None) -> LayerModel:
    """Build the nested layer stack from a DEM (real or fictional) and config.

    Real DEMs (lon/lat or any projected CRS) are reprojected into a transverse
    Mercator centered on the bbox, and the cut frame is the axis-aligned
    rectangle inscribed in the projected bbox — so the map is north-up and
    rectangular rather than tilted by the UTM grid convergence. Fictional data
    from a terrain bundle arrives flat (``crs == FLAT_CRS``); its units are
    already linear world units, so the raster extent *is* the frame and
    everything downstream is identical.
    """
    from .dem.base import FLAT_CRS

    warnings: list[str] = []
    flat = dem_geographic.crs == FLAT_CRS

    # 1. Project real DEMs into the box-centered CRS; fictional data stays flat.
    if flat:
        epsg, crs = 0, ""
        utm = dem_geographic
    else:
        if bbox is None:
            raise ValueError("a real-world DEM requires a bbox for projection")
        epsg, crs = bbox.utm_epsg(), bbox.local_crs()
        utm = dem_geographic.to_crs(crs)

    # 2. Elevation range → base datum and solved scale. Optional percentile clip
    #    keeps anomalous spires/pits from stretching the banding without amputating
    #    the base footprint: the LOW percentile becomes the floor (and the field is
    #    clamped up to it below, so layer 0 stays a full solid slab), while the HIGH
    #    percentile caps the span so a lone spire merges into the top band.
    import numpy as np

    true_zmin, true_zmax = utm.elevation_range()
    if cfg.physical.clip_percentiles:
        valid = utm.masked().compressed()
        lo, hi = cfg.physical.clip_percentiles
        low_clip = float(np.percentile(valid, lo))
        zmax = float(np.percentile(valid, hi))
    else:
        low_clip, zmax = true_zmin, true_zmax
    base = low_clip if cfg.physical.base_datum == "auto" else float(cfg.physical.base_datum)
    if base > zmax:
        raise ValueError(f"base datum {base} is above the terrain max {zmax:.0f}")

    # The cut frame, in world units. Flat data: the raster extent. Real data:
    # the largest axis-aligned rectangle inside the projected bbox, further
    # limited to what the raster actually covers (a source that stops short of
    # the bbox shrinks the map rather than leaving a bite out of it).
    rows, cols = utm.shape
    left, top = utm.transform * (0, 0)
    right, bottom = utm.transform * (cols, rows)
    fx0, fy0 = min(left, right), min(top, bottom)
    fx1, fy1 = max(left, right), max(top, bottom)
    if not flat:
        bx0, by0, bx1, by1 = projected_frame(bbox, crs)
        fx0, fy0 = max(fx0, bx0), max(fy0, by0)
        fx1, fy1 = min(fx1, bx1), min(fy1, by1)
        if fx1 <= fx0 or fy1 <= fy0:
            raise ValueError("the DEM does not overlap the requested bbox")
    frame = box(fx0, fy0, fx1, fy1)
    utm_width = fx1 - fx0
    utm_height = fy1 - fy0

    # normalize_layers (fictional) uses the same math as layer_count.
    layer_count = cfg.physical.layer_count or cfg.physical.normalize_layers
    scale = solve_scale(
        model_width_mm=cfg.physical.model_width_mm,
        real_width_m=utm_width,
        ply_thickness_mm=cfg.physical.ply_thickness_mm,
        exaggeration=cfg.physical.exaggeration,
        interval_m=cfg.physical.interval_m,
        layer_count=layer_count,
        base_elev_m=base,
        max_elev_m=zmax,
    )
    warnings.extend(scale.warnings)

    # 3. Model transform. Scale exactly to the requested model width.
    k_mm_per_m = cfg.physical.model_width_mm / utm_width
    matrix = _world_to_model_matrix(fx0, fy0, k_mm_per_m)
    model_height_mm = utm_height * k_mm_per_m

    # 4. Smooth once, then slice each band from the shared field. Clamp up to the
    #    low floor so terrain below it joins the base slab instead of vanishing.
    #    Contours stop half a cell inside the grid, so pad one replicated cell
    #    first: bands then overshoot the frame and are clipped flush to it.
    padded = utm.pad_edges(1)
    z = smooth_dem(padded.data, cfg.contour.smoothing_px)
    if cfg.physical.clip_percentiles and cfg.physical.base_datum == "auto":
        z = np.where(np.isnan(z), z, np.maximum(z, low_clip))
    xs, ys = padded.pixel_coords()

    interval = scale.interval_m
    n_layers = max(1, int((zmax - base) // interval) + 1)
    cc = cfg.contour

    layers: list[Layer] = []
    prev_geom_model: BaseGeometry | None = None
    for k in range(n_layers):
        threshold = base + k * interval
        # Filled contours exclude cells sitting exactly at the lower level, so a
        # flat floor at the minimum (ocean/plain) would leave the base slab holed.
        # Nudge the base-layer threshold just below the minimum to capture it.
        lower = threshold - interval * 1e-3 if k == 0 else threshold
        world_geom = band_polygon(padded, lower, xs, ys, z)
        if world_geom.is_empty:
            continue
        # Smooth BEFORE clipping to the frame: corner-cutting on the dense
        # contour vertices is harmless, but on the frame's long straight edges
        # it would chamfer the map's corners.
        if cc.chaikin_iterations > 0:
            world_geom = chaikin(world_geom, cc.chaikin_iterations)
        world_geom = _to_multipolygon(world_geom.intersection(frame))
        if world_geom.is_empty:
            continue
        model_geom = affine_transform(world_geom, matrix)
        model_geom = clean_geometry(
            model_geom,
            simplify_tol_mm=cc.simplify_tol_mm,
            min_area_mm2=cc.min_area_mm2,
            min_hole_mm2=cc.min_area_mm2,
            close_radius_mm=cc.close_radius_mm,
        )
        if model_geom.is_empty:
            continue

        lyr_warnings: list[str] = []
        # 5. Nesting validity: keep layer k inside layer k-1 for glue support.
        # The band model guarantees nesting mathematically; independent smoothing
        # of adjacent contours introduces only sub-millimeter boundary crossings,
        # so clip quietly and warn only when a structurally real area protrudes.
        if prev_geom_model is not None:
            overhang = model_geom.difference(prev_geom_model)
            if not overhang.is_empty and overhang.area > 0:
                clipped = model_geom.intersection(prev_geom_model)
                if not clipped.is_empty and clipped.area > 0.5 * model_geom.area:
                    model_geom = _to_multipolygon(clipped)
                    if overhang.area > cc.min_area_mm2:
                        lyr_warnings.append(
                            f"layer {k} protruded {overhang.area:.0f} mm² beyond "
                            f"layer {k - 1} and was clipped — check base datum/data"
                        )
        layers.append(Layer(index=len(layers), threshold_m=threshold,
                            geometry=_to_multipolygon(model_geom),
                            warnings=lyr_warnings))
        warnings.extend(lyr_warnings)
        prev_geom_model = model_geom

    if not layers:
        raise ValueError("no layers produced; check interval, base datum and bbox")

    # Warn when the stack collapsed far below the requested count — usually a
    # lone spire dominating the range. Percentile clipping is the fix.
    if layer_count and len(layers) < max(2, layer_count // 2):
        warnings.append(
            f"requested {layer_count} layers but only {len(layers)} produced — the "
            "elevation range is likely dominated by a small area; set "
            "clip_percentiles (e.g. [0.5, 99.5]) to normalize the banding."
        )

    return LayerModel(
        layers=layers,
        interval_m=interval,
        base_elev_m=base,
        max_elev_m=zmax,
        model_width_mm=cfg.physical.model_width_mm,
        model_height_mm=model_height_mm,
        scale=scale,
        utm_epsg=epsg,
        world_to_model=matrix,
        flat=flat,
        crs=crs,
        warnings=warnings,
    )


def _to_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    from shapely.geometry import Polygon

    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(parts)


def apply_world_matrix(matrix: tuple[float, float, float, float, float, float],
                       geom: BaseGeometry) -> BaseGeometry:
    """Apply a stored world→model affine to any geometry (used by symbology)."""
    return affine_transform(geom, matrix)
