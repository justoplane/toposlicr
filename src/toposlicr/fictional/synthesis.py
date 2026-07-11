"""Heightfield synthesis — the algorithmic core of Tier 3 (spec Section 3.2).

A stack of composable, deterministic (seeded) float-array field operations, each
cheap so iteration is instant:

    coastal ramp → ridge stamping → hydrological carving → band-aware noise → normalize

The inputs (land/sea mask, painted terrain-class zones, traced rivers) come from
the artwork-extraction stage; the output is a normalized [0, 1] heightfield that
the neutral terrain bundle maps to 16-bit and the core bands with
``normalize_layers``. Fictional elevation units are meaningless by design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --- individual field operations ------------------------------------------

def coastal_ramp(land_mask: np.ndarray, amplitude: float = 1.0,
                 exponent: float = 0.6) -> np.ndarray:
    """Base elevation rising inland from the coast.

    ``d`` is the distance (in cells) from each land cell to the nearest sea
    cell; ``E0 = amplitude * d**exponent``. ``exponent < 1`` is concave — gentle
    coasts that steepen inland. Sea cells are 0.
    """
    land = np.asarray(land_mask, dtype=bool)
    d = distance_transform_edt(land)
    field = amplitude * np.power(np.maximum(d, 0.0), exponent)
    field[~land] = 0.0
    return field.astype("float64")


def _bathymetry(land_mask: np.ndarray, bands: int, amplitude: float) -> np.ndarray:
    """Negative, layered ocean depth (the classic depth-band lake aesthetic)."""
    sea = ~np.asarray(land_mask, dtype=bool)
    if not sea.any() or bands <= 0:
        return np.zeros(land_mask.shape, dtype="float64")
    d = distance_transform_edt(sea)
    dmax = d.max() or 1.0
    depth = np.ceil(d / dmax * bands) / bands            # quantized 0..1
    out = np.zeros(land_mask.shape, dtype="float64")
    out[sea] = -amplitude * depth[sea]
    return out


def stamp_ridges(field: np.ndarray, zone_mask: np.ndarray, amplitude: float,
                 falloff_exponent: float = 1.0, profile: str = "mountain"
                 ) -> np.ndarray:
    """Add a massif whose crest follows the drawn zone's medial axis.

    Inside ``zone_mask``, the distance transform peaks along the medial axis, so
    ``amplitude * dn**falloff_exponent`` makes ranges whose crests trace the drawn
    shape rather than isolated blobs. ``profile`` shapes the cross-section:
    ``mountain`` (peaked), ``hill`` (rounded), ``plateau`` (flat-top). A tiny
    (point) zone becomes a clamped Gaussian summit.
    """
    out = np.array(field, dtype="float64", copy=True)
    zone = np.asarray(zone_mask, dtype=bool)
    area = int(zone.sum())
    if area == 0:
        return out
    if area <= 9:                                        # point peak → Gaussian
        ys, xs = np.nonzero(zone)
        cy, cx = ys.mean(), xs.mean()
        h, w = out.shape
        sigma = max(3.0, 0.03 * min(h, w))
        yy, xx = np.mgrid[0:h, 0:w]
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        bump = amplitude * np.exp(-(d2 / (2.0 * sigma ** 2)))
        # Bound the support so a point peak lifts only its neighbourhood, never
        # the whole grid (which would raise far-away sea cells above 0).
        bump[d2 > (4.0 * sigma) ** 2] = 0.0
        out += bump
        return out

    d = distance_transform_edt(zone)
    dmax = d.max() or 1.0
    dn = d / dmax                                        # 0 at edge, 1 at crest
    if profile == "plateau":
        contrib = amplitude * _smoothstep(dn / 0.5)     # saturates → flat top
    elif profile == "hill":
        contrib = amplitude * _smoothstep(dn)           # rounded
    else:                                               # mountain (peaked)
        contrib = amplitude * np.power(dn, falloff_exponent)
    out[zone] += contrib[zone]
    return out


def _order_source_to_mouth(polyline, mouth, field) -> list:
    pts = [tuple(p) for p in polyline]
    if len(pts) < 2:
        return pts
    if mouth is None:                                   # mouth = lower endpoint
        h, w = field.shape
        r0, c0 = pts[0]
        r1, c1 = pts[-1]
        z0 = field[_clamp(r0, h), _clamp(c0, w)]
        z1 = field[_clamp(r1, h), _clamp(c1, w)]
        mouth = 0 if z0 < z1 else 1
    return list(reversed(pts)) if mouth == 0 else pts    # mouth ends up last


def _clamp(v, hi: int) -> int:
    """Round to an integer index clamped to [0, hi-1] (no negative wraparound)."""
    return max(0, min(int(round(v)), hi - 1))


def _densify(pts) -> list:
    dense = []
    for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:], strict=False):
        n = max(1, int(round(np.hypot(r1 - r0, c1 - c0))))
        for i in range(n):
            dense.append((r0 + (r1 - r0) * i / n, c0 + (c1 - c0) * i / n))
    dense.append(tuple(pts[-1]))
    return dense


def carve_rivers(field: np.ndarray, polylines, mouths=None,
                 valley_width_px: float = 3.0, depth: float = 0.05, lakes=None):
    """Carve monotonically-downhill valleys along river polylines.

    Each polyline is walked source→mouth applying a running minimum, so the bed
    is monotonically non-increasing downstream — a scored river can never climb
    across a contour band. A Gaussian-falloff valley is then feathered around it.
    Where a river crosses rising terrain (a stamped ridge) the carve wins and a
    warning is emitted. ``lakes`` (list of ``{mask, outlet?}``) flatten to their
    outlet elevation. Returns ``(new_field, warnings)``.
    """
    out = np.array(field, dtype="float64", copy=True)
    h, w = out.shape
    warnings: list[str] = []
    rng = float(out.max() - out.min()) or 1.0

    river_mask = np.zeros((h, w), dtype=bool)
    # +inf so shared cells take the running MINIMUM across crossing rivers
    # (last-writer-wins would break per-river monotonicity at intersections).
    bed_grid = np.full((h, w), np.inf, dtype="float64")
    if mouths is None:
        mouths = [None] * len(polylines)

    for idx, (pl, mouth) in enumerate(zip(polylines, mouths, strict=False)):
        ordered = _order_source_to_mouth(pl, mouth, out)
        if len(ordered) < 2:
            continue
        dense = _densify(ordered)
        sampled = np.array([out[_clamp(r, h), _clamp(c, w)] for r, c in dense])
        if np.any(np.diff(sampled) > 1e-9 * rng):
            warnings.append(f"river {idx} climbs across rising terrain "
                            "(a ridge?); carving it downhill anyway")
        bed = np.minimum.accumulate(sampled) - depth * rng      # monotone channel
        for (r, c), b in zip(dense, bed, strict=False):
            ri, ci = _clamp(r, h), _clamp(c, w)
            river_mask[ri, ci] = True
            bed_grid[ri, ci] = min(bed_grid[ri, ci], b)

    if river_mask.any():
        dist, (ir, ic) = distance_transform_edt(~river_mask, return_indices=True)
        nearest_bed = bed_grid[ir, ic]
        weight = np.exp(-((dist / max(valley_width_px, 1e-6)) ** 2))
        carved = np.minimum(out, nearest_bed)
        out = out + weight * (carved - out)

    for lake in (lakes or []):
        mask = np.asarray(lake["mask"], dtype=bool)
        if not mask.any():
            continue
        outlet = lake.get("outlet")
        out[mask] = float(outlet) if outlet is not None else float(out[mask].min())

    return out, warnings


def _octave_noise(shape, octaves: int, seed: int) -> np.ndarray:
    """fBm noise in ~[-1, 1] via opensimplex, or a seeded numpy fallback."""
    h, w = shape
    try:
        from opensimplex import OpenSimplex

        gen = OpenSimplex(seed=int(seed))
        total = np.zeros((h, w), dtype="float64")
        norm = 0.0
        for o in range(octaves):
            freq = 2.0 ** o
            amp = 0.5 ** o
            xs = np.linspace(0.0, 6.0 * freq, w)
            ys = np.linspace(0.0, 6.0 * freq * h / max(w, 1), h)
            total += amp * gen.noise2array(xs, ys)
            norm += amp
        return total / (norm or 1.0)
    except Exception:
        # Seeded value-noise fallback: sum upsampled smooth random grids.
        from scipy.ndimage import gaussian_filter, zoom

        rng = np.random.default_rng(int(seed))
        total = np.zeros((h, w), dtype="float64")
        norm = 0.0
        for o in range(octaves):
            amp = 0.5 ** o
            gh, gw = max(2, h // (2 ** (octaves - o))), max(2, w // (2 ** (octaves - o)))
            grid = rng.random((gh, gw)) * 2.0 - 1.0
            up = zoom(grid, (h / gh, w / gw), order=1)[:h, :w]
            total += amp * gaussian_filter(up, sigma=1.0)
            norm += amp
        return total / (norm or 1.0)


def band_aware_noise(field: np.ndarray, layers: int, octaves: int = 3,
                     amplitude_fraction: float = 0.4, seed: int = 0) -> np.ndarray:
    """Add elevation-weighted fBm noise, capped below a fraction of one band.

    Amplitude is proportional to local elevation (mountains rougher than plains)
    and the *total* noise magnitude is capped strictly below
    ``amplitude_fraction`` of one layer interval (``range / layers``), so noise
    can never spawn speckle islands or pinholes across a band boundary.
    """
    out = np.array(field, dtype="float64", copy=True)
    rng = float(out.max() - out.min())
    if rng <= 0 or layers <= 0:
        return out
    interval = rng / layers
    cap = amplitude_fraction * interval

    noise = _octave_noise(out.shape, octaves, seed)
    weight = (out - out.min()) / rng                    # rougher where higher
    combined = noise * weight
    peak = float(np.max(np.abs(combined))) or 1.0
    combined *= (cap * 0.99) / peak                     # strictly below the cap
    return out + combined


def normalize(field: np.ndarray, percentiles=(0.5, 99.5)) -> np.ndarray:
    """Percentile-clip and scale to [0, 1] (the bundle maps this to 16-bit)."""
    arr = np.asarray(field, dtype="float64")
    lo, hi = np.percentile(arr, percentiles[0]), np.percentile(arr, percentiles[1])
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


# --- the full stack --------------------------------------------------------

def synthesize(land_mask: np.ndarray, *, zones=None, rivers=None, layers: int = 12,
               seed: int = 0, coastal_exponent: float = 0.6, noise_octaves: int = 3,
               noise_max_fraction: float = 0.4, bathymetry_bands: int = 0):
    """Run the full synthesis stack. Returns ``(field[0,1], warnings)``.

    ``zones``: list of ``{mask, class in mountain|hill|plateau, amplitude}``.
    ``rivers``: list of ``{polyline, mouth?}`` (polyline = list of (row, col)).
    """
    warnings: list[str] = []
    field = coastal_ramp(land_mask, amplitude=1.0, exponent=coastal_exponent)

    for zone in (zones or []):
        field = stamp_ridges(field, zone["mask"], float(zone.get("amplitude", 1.0)),
                             falloff_exponent=float(zone.get("falloff_exponent", 1.0)),
                             profile=str(zone.get("class", "mountain")))

    if rivers:
        polylines = [r["polyline"] for r in rivers]
        mouths = [r.get("mouth") for r in rivers]
        field, river_warnings = carve_rivers(field, polylines, mouths)
        warnings.extend(river_warnings)

    if bathymetry_bands > 0:
        field = field + _bathymetry(land_mask, bathymetry_bands, amplitude=field.max() or 1.0)

    field = band_aware_noise(field, layers, octaves=noise_octaves,
                             amplitude_fraction=noise_max_fraction, seed=seed)
    return normalize(field), warnings


def synthesize_to_bundle(out_dir, land_mask: np.ndarray, *, zones=None, rivers=None,
                         layers: int = 12, seed: int = 0, coastal_exponent: float = 0.6,
                         noise_octaves: int = 3, noise_max_fraction: float = 0.4,
                         bathymetry_bands: int = 0, units_per_pixel: float = 100.0,
                         source: str = "", write_water: bool = True) -> Path:
    """Synthesize a heightfield and write a complete terrain bundle."""
    import csv

    from ..bundle import BundleMeta, save_hillshade, write_gray16

    field, warnings = synthesize(
        land_mask, zones=zones, rivers=rivers, layers=layers, seed=seed,
        coastal_exponent=coastal_exponent, noise_octaves=noise_octaves,
        noise_max_fraction=noise_max_fraction, bathymetry_bands=bathymetry_bands)

    out = Path(out_dir)
    if out.suffix != ".terrainbundle":
        out = out.with_name(out.name + ".terrainbundle") if out.suffix else out
    out.mkdir(parents=True, exist_ok=True)

    write_gray16(out / "heightmap.png", field)
    if write_water:
        sea = (~np.asarray(land_mask, dtype=bool)).astype("uint16") * 65535
        write_gray16(out / "water.png", sea)
    save_hillshade(field, out / "preview_hillshade.png")

    h = field.shape[0]
    meta = BundleMeta(units_per_pixel=units_per_pixel, height_scale=1.0 / 65535.0,
                      height_offset=0.0, world_origin=(0.0, h * units_per_pixel),
                      vertical_range=[0.0, 1.0], source=source or "synthesized",
                      attribution="synthesized heightfield")
    import json
    (out / "meta.json").write_text(json.dumps(meta.to_dict()))
    with (out / "features.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=["name", "type", "x", "y", "elev", "include",
                                       "icon", "label_override"]).writeheader()
    return out
