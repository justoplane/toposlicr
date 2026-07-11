"""Azgaar's Fantasy Map Generator → terrain bundle (spec Section 3.2).

Azgaar is *data*, not artwork — the cleanest fictional tier legally (its maps are
free for commercial use) and technically. It maintains a true per-cell heightmap
(0-100, water below 20) plus procedurally consistent rivers, settlements and
labels, all exportable as GeoJSON/CSV. This adapter rasterizes the cell heights
into a bundle heightmap and turns burgs/rivers into ``features.csv`` — no
synthesis, no tracing.

Inputs (from Azgaar's export menu):
- Cells GeoJSON: FeatureCollection of polygons, each ``properties.height`` 0-100.
- Rivers GeoJSON (optional): LineString/MultiLineString features.
- Burgs (optional): a point GeoJSON FeatureCollection, or a CSV with
  ``Burg,Latitude,Longitude,Population`` columns, or a list of dicts.

All inputs share one coordinate frame (Azgaar's map/lon-lat space); the bundle's
heightmap and every feature x,y are written in that same frame so the core's
world→model transform lines them up exactly.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import click
from affine import Affine

from ..bundle import BundleMeta, save_hillshade, write_gray16, write_mask

# Azgaar convention: cell height < 20 is water (sea/lake).
DEFAULT_WATER_THRESHOLD = 20
# Population at/above which a burg is drawn as a city rather than a town.
_CITY_POPULATION = 20.0


def azgaar_to_bundle(cells_geojson, out_dir, *, rivers_geojson=None, burgs=None,
                     cells_across: int = 1000, water_threshold: int = DEFAULT_WATER_THRESHOLD,
                     source: str = "Azgaar FMG") -> Path:
    """Convert an Azgaar cells export (+ optional rivers/burgs) into a bundle.

    Returns the created ``*.terrainbundle`` directory path.
    """
    from rasterio.features import rasterize
    from shapely.geometry import shape

    cells = _load_geojson(cells_geojson)
    feats = cells.get("features") or []
    if not feats:
        raise ValueError("cells GeoJSON has no features")

    shapes_heights = []
    minx = miny = math.inf
    maxx = maxy = -math.inf
    for f in feats:
        raw = f.get("geometry")
        if not raw:
            continue
        geom = shape(raw)
        if geom.is_empty:
            continue
        height = float((f.get("properties") or {}).get("height", 0) or 0)
        shapes_heights.append((geom, height))
        bx0, by0, bx1, by1 = geom.bounds
        minx, miny = min(minx, bx0), min(miny, by0)
        maxx, maxy = max(maxx, bx1), max(maxy, by1)

    width_world = maxx - minx
    height_world = maxy - miny
    if width_world <= 0 or height_world <= 0:
        raise ValueError("cells GeoJSON has a degenerate bounding box")

    cols = max(1, int(cells_across))
    upp = width_world / cols
    rows = max(1, round(height_world / upp))
    # North-up: pixel (0,0) is the NW corner (minx, maxy). Matches TerrainBundle.
    transform = Affine.translation(minx, maxy) * Affine.scale(upp, -upp)

    height_grid = rasterize(
        shapes_heights, out_shape=(rows, cols), transform=transform,
        fill=0.0, dtype="float32",
    )

    water_mask = height_grid < water_threshold

    out = Path(out_dir)
    if out.suffix != ".terrainbundle":
        out = out.with_suffix(".terrainbundle")
    out.mkdir(parents=True, exist_ok=True)

    write_gray16(out / "heightmap.png", height_grid)
    if water_mask.any():
        # write_mask, not write_gray16: an all-water export must not normalize to 0.
        write_mask(out / "water.png", water_mask)

    lo, hi = float(height_grid.min()), float(height_grid.max())
    # write_gray16 normalises by the array's own [lo, hi] to fill 16-bit, so the
    # inverse mapping stored pixel → original height is scale/offset below.
    scale = (hi - lo) / 65535.0 if hi > lo else 1.0
    meta = BundleMeta(
        units_per_pixel=upp, height_scale=scale, height_offset=lo,
        world_origin=(minx, maxy), vertical_range=[lo, hi], source=source,
        attribution="Azgaar FMG (maps free for commercial use)",
    )
    (out / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2))

    rows_out = _burg_rows(burgs) + _river_rows(rivers_geojson)
    _write_features(out / "features.csv", rows_out)

    save_hillshade(height_grid.astype("float64") * scale + lo,
                   out / "preview_hillshade.png")
    return out


# --- feature extraction ----------------------------------------------------

def _burg_rows(burgs) -> list[dict]:
    if burgs is None:
        return []
    rows = []
    for name, x, y, pop in _iter_burgs(burgs):
        is_city = pop is not None and pop >= _CITY_POPULATION
        rows.append({
            "name": name or "", "type": "city" if is_city else "town",
            "x": x, "y": y, "elev": "", "include": "yes",
            "icon": "city" if is_city else "tower", "label_override": "",
        })
    return rows


def _iter_burgs(burgs):
    """Yield (name, x, y, population) from GeoJSON points / CSV / list of dicts."""
    # GeoJSON FeatureCollection of points.
    if isinstance(burgs, dict) and burgs.get("type") == "FeatureCollection":
        for f in burgs.get("features") or []:
            geom = f.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom.get("coordinates") or []
            if len(coords) < 2:
                continue
            x, y = coords[0], coords[1]
            props = f.get("properties") or {}
            yield (_first(props, "name", "Burg", "burg"), float(x), float(y),
                   _maybe_num(_first(props, "population", "Population", "pop")))
        return
    # A CSV path.
    if isinstance(burgs, (str, Path)) and Path(burgs).is_file():
        with Path(burgs).open(newline="", encoding="utf-8") as fh:
            yield from _iter_burg_rows(csv.DictReader(fh))
        return
    # An iterable of dict rows (already-parsed CSV, or plain dicts).
    if isinstance(burgs, (list, tuple)):
        yield from _iter_burg_rows(burgs)


def _iter_burg_rows(rows):
    for r in rows:
        x = _first(r, "x", "Longitude", "longitude", "lon")
        y = _first(r, "y", "Latitude", "latitude", "lat")
        if x is None or y is None:
            continue
        yield (_first(r, "name", "Burg", "burg"), float(x), float(y),
               _maybe_num(_first(r, "population", "Population", "pop")))


def _river_rows(rivers_geojson) -> list[dict]:
    """Rivers → one labelled point per named river at its midpoint.

    Simplification (v1): ``features.csv`` is point-per-row, so river *linework*
    is not carried — only the river name is placed. Full river polylines would
    need a bundle-format extension (a rivers geometry file).
    """
    if rivers_geojson is None:
        return []
    from shapely.geometry import shape

    data = _load_geojson(rivers_geojson)
    rows = []
    for f in data.get("features") or []:
        raw = f.get("geometry")
        if not raw:
            continue
        geom = shape(raw)
        if geom.is_empty:
            continue
        pt = geom.interpolate(0.5, normalized=True) if geom.geom_type == "LineString" \
            else geom.representative_point()
        name = _first(f.get("properties") or {}, "name", "River", "river")
        rows.append({"name": name or "", "type": "river", "x": pt.x, "y": pt.y,
                     "elev": "", "include": "yes", "icon": "", "label_override": ""})
    return rows


def _write_features(path: Path, rows: list[dict]) -> None:
    fields = ["name", "type", "x", "y", "elev", "include", "icon", "label_override"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# --- helpers ---------------------------------------------------------------

def _load_geojson(src):
    if isinstance(src, dict):
        return src
    return json.loads(Path(src).read_text(encoding="utf-8"))


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _maybe_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- CLI -------------------------------------------------------------------

@click.command("azgaar")
@click.argument("cells_geojson", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", default="world.terrainbundle", show_default=True)
@click.option("--rivers", "rivers_geojson", type=click.Path(exists=True, dir_okay=False),
              default=None, help="Azgaar rivers GeoJSON.")
@click.option("--burgs", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Burgs GeoJSON or CSV.")
@click.option("--cells-across", default=1000, show_default=True, type=int)
@click.option("--water-threshold", default=DEFAULT_WATER_THRESHOLD, show_default=True,
              type=int)
def main(cells_geojson, out_dir, rivers_geojson, burgs, cells_across, water_threshold):
    """Convert an Azgaar FMG export into a terrain bundle."""
    path = azgaar_to_bundle(cells_geojson, out_dir, rivers_geojson=rivers_geojson,
                            burgs=burgs, cells_across=cells_across,
                            water_threshold=water_threshold)
    click.echo(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
