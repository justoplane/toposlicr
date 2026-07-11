"""Tier 1 adapter — The Legend of Zelda: Breath of the Wild → terrain bundle.

This is the reference Tier 1 adapter (spec Section 1.2): a game that stores real
terrain grids, with a modding community that documents the formats and ships
extraction tools. The payoff is elevation as good as a real DEM plus *exact*
named-feature coordinates in the same coordinate system, so registration is free.

Legal stance (enforced here and in docs): this adapter NEVER downloads, bundles,
or distributes game assets. It operates only on files the user supplies from
their own legally-owned copy — a heightmap PNG they extracted with the community
**BotWHeightMapConverter**, raw ``.hght`` tiles they dumped, and an objmap
GeoJSON they exported. ``meta.json.attribution`` records the source; what the
user does with a map of a copyrighted world is their own IP question.

What is implemented here vs. delegated:
- ``read_hght_tile`` fully parses a single ``.hght`` (256×256 16-bit LE grid),
  documented on the ZeldaMods **HGHT** wiki page.
- ``assemble_terrain_dir`` mosaics ``.hght`` tiles when a simple grid layout is
  discernible from filenames; full **TSCB**-driven tile placement (a binary
  descriptor) is out of scope and delegated to BotWHeightMapConverter — pass the
  resulting ``heightmap_png`` instead.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..bundle import (
    BundleError,
    BundleMeta,
    read_gray,
    save_hillshade,
    write_gray16,
    write_mask,
)

DEFAULT_SOURCE = "The Legend of Zelda: Breath of the Wild (user-supplied)"
_ATTRIBUTION = ("Terrain and named locations from a user-supplied, legally-owned "
                "copy of BotW (extracted via community tools: BotWHeightMapConverter, "
                "ZeldaMods objmap). No game assets are distributed by toposlicr.")

_HGHT_DIM = 256                      # each .hght tile is 256×256
_HGHT_BYTES = _HGHT_DIM * _HGHT_DIM * 2

_CSV_FIELDS = ["name", "type", "x", "y", "elev", "include", "icon", "label_override"]


# --- raw .hght parsing (ZeldaMods HGHT) ------------------------------------

def read_hght_tile(path: str | Path) -> np.ndarray:
    """Parse one ``.hght`` file → a (256, 256) uint16 height grid.

    Format (ZeldaMods HGHT): a raw 256×256 grid of 16-bit unsigned
    little-endian height values, row-major, 131072 bytes.
    """
    data = Path(path).read_bytes()
    if len(data) != _HGHT_BYTES:
        raise BundleError(
            f"{path}: expected {_HGHT_BYTES} bytes for a 256×256 16-bit .hght tile, "
            f"got {len(data)}")
    return np.frombuffer(data, dtype="<u2").reshape(_HGHT_DIM, _HGHT_DIM).copy()


def assemble_terrain_dir(terrain_dir: str | Path, region=None, lod: int = 0
                         ) -> np.ndarray:
    """Mosaic ``.hght`` tiles into one heightmap when the layout is discernible.

    Full TSCB-driven placement is not reimplemented; instead we infer a grid from
    filenames (two integers ⇒ column,row) or fall back to a perfect-square
    row-major arrangement. If neither applies, we refuse rather than guess and
    point the user at the community extractor.
    """
    tiles = sorted(Path(terrain_dir).glob("*.hght"))
    if not tiles:
        raise BundleError(f"no .hght tiles found in {terrain_dir}")

    # 1) grid from filename indices, e.g. "3_5.hght" → column 3, row 5.
    coords = []
    for t in tiles:
        ints = re.findall(r"\d+", t.stem)
        if len(ints) >= 2:
            coords.append((int(ints[-2]), int(ints[-1])))
    if len(coords) == len(tiles):
        cols = sorted({c for c, _ in coords})
        rows = sorted({r for _, r in coords})
        if len(coords) == len(cols) * len(rows):
            grid = {cr: read_hght_tile(t)
                    for t, cr in zip(tiles, coords, strict=True)}
            return np.vstack([np.hstack([grid[(c, r)] for c in cols]) for r in rows])

    # 2) perfect-square count → row-major n×n.
    n = int(round(len(tiles) ** 0.5))
    if n * n == len(tiles):
        arrs = [read_hght_tile(t) for t in tiles]
        return np.vstack([np.hstack(arrs[i * n:(i + 1) * n]) for i in range(n)])

    raise BundleError(
        "cannot infer .hght tile placement without the binary TSCB descriptor; "
        "run the community BotWHeightMapConverter and pass heightmap_png= instead.")


# --- objmap → features -----------------------------------------------------

def _map_category(category: str) -> tuple[str, str]:
    """Map an objmap category to (bundle feature type, icon name)."""
    c = (category or "").lower()
    if any(k in c for k in ("mountain", "peak", "summit", "volcano")):
        return "peak", "peak"
    if any(k in c for k in ("lake", "water", "sea", "pond", "spring")):
        return "lake", ""
    if "tower" in c:
        return "poi", "tower"
    if any(k in c for k in ("shrine", "temple")):
        return "poi", "shrine"
    if any(k in c for k in ("village", "town", "city", "stable", "settlement")):
        return "settlement", "tower"
    return "poi", ""


def _objmap_points(objmap_geojson) -> list[dict]:
    """Read an objmap GeoJSON (path or dict) → normalized point records."""
    if objmap_geojson is None:
        return []
    data = (json.loads(Path(objmap_geojson).read_text())
            if isinstance(objmap_geojson, (str, Path)) else objmap_geojson)
    out = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feat.get("properties") or {}
        name = (props.get("name") or props.get("label") or props.get("Name")
                or props.get("MessageID") or "")
        category = (props.get("type") or props.get("category")
                    or props.get("Category") or "")
        out.append({"x": float(coords[0]), "y": float(coords[1]),
                    "name": str(name).strip(), "category": str(category)})
    return out


# --- world frame -----------------------------------------------------------

def _world_frame(shape, world_bounds, units_per_pixel, points):
    """Return (world_origin, units_per_pixel) tying features to the heightmap.

    Preference order: an explicit ``world_bounds`` (exact registration, the
    in-game coordinate frame) → derive from the objmap extent (self-consistent
    approximation) → a plain pixel grid.
    """
    h, w = shape
    if world_bounds is not None:
        minx, miny, maxx, maxy = world_bounds
        # Reconcile the y extent with the image height: sizing cells from x alone
        # vertically misregisters features when the PNG aspect ratio differs from
        # the declared bounds. Square cells over the larger extent, centred.
        upp = max((maxx - minx) / w, (maxy - miny) / h)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        return (cx - w * upp / 2.0, cy + h * upp / 2.0), upp
    if points:
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        mx = (maxx - minx) * 0.05 or 1.0
        my = (maxy - miny) * 0.05 or 1.0
        minx, maxx, miny, maxy = minx - mx, maxx + mx, miny - my, maxy + my
        # Square cells sized to cover the larger extent, points centred in the
        # grid, so every feature lands inside regardless of aspect ratio.
        upp = max((maxx - minx) / w, (maxy - miny) / h)
        cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
        return (cx - w * upp / 2.0, cy + h * upp / 2.0), upp
    return (0.0, h * units_per_pixel), units_per_pixel


# --- main entry ------------------------------------------------------------

def botw_to_bundle(out_dir, *, heightmap_png=None, terrain_dir=None,
                   objmap_geojson=None, water_png=None, region=None, lod: int = 0,
                   units_per_pixel: float = 1.0, world_bounds=None,
                   source: str = DEFAULT_SOURCE) -> Path:
    """Assemble a terrain bundle from user-supplied BotW terrain + objmap.

    Provide either ``heightmap_png`` (already extracted with the community tool)
    or ``terrain_dir`` (raw ``.hght`` tiles). ``objmap_geojson`` adds named
    features in the same coordinate frame as the heightmap.
    """
    if heightmap_png is not None:
        height = read_gray(heightmap_png).astype("float64")
    elif terrain_dir is not None:
        height = assemble_terrain_dir(terrain_dir, region, lod).astype("float64")
    else:
        raise ValueError("provide heightmap_png (from BotWHeightMapConverter) "
                         "or terrain_dir (raw .hght tiles)")

    h, w = height.shape
    points = _objmap_points(objmap_geojson)
    origin, upp = _world_frame((h, w), world_bounds, units_per_pixel, points)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_gray16(out / "heightmap.png", height)

    if water_png is not None:
        wm = read_gray(water_png)
        if wm.shape != height.shape:
            raise BundleError("water_png grid does not match the heightmap")
        # write_mask (not write_gray16) so an all-water mask isn't normalized to 0.
        write_mask(out / "water.png", wm > 0)

    lo, hi = float(np.nanmin(height)), float(np.nanmax(height))
    meta = BundleMeta(
        units_per_pixel=upp, world_origin=(origin[0], origin[1]),
        height_scale=(hi - lo) / 65535.0 if hi > lo else 1.0, height_offset=lo,
        source=source, attribution=_ATTRIBUTION)
    (out / "meta.json").write_text(json.dumps(meta.to_dict(), indent=2))

    _write_features(out / "features.csv", points)
    save_hillshade(height, out / "preview_hillshade.png")
    return out


def _write_features(path: Path, points: list[dict]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for p in points:
            ftype, icon = _map_category(p["category"])
            named = bool(p["name"])
            writer.writerow({
                "name": p["name"], "type": ftype,
                "x": f"{p['x']:.3f}", "y": f"{p['y']:.3f}", "elev": "",
                "include": "yes" if named else "no",
                "icon": icon, "label_override": "",
            })


# --- CLI -------------------------------------------------------------------

def main() -> None:  # pragma: no cover - thin CLI wrapper
    import click

    @click.command("botw")
    @click.option("--out", required=True, type=click.Path(file_okay=False))
    @click.option("--heightmap", type=click.Path(exists=True, dir_okay=False),
                  help="16-bit PNG from BotWHeightMapConverter.")
    @click.option("--terrain-dir", type=click.Path(exists=True, file_okay=False),
                  help="Folder of raw .hght tiles (simple grid layouts only).")
    @click.option("--objmap", type=click.Path(exists=True, dir_okay=False),
                  help="ZeldaMods objmap GeoJSON of named locations.")
    @click.option("--water", type=click.Path(exists=True, dir_okay=False))
    @click.option("--units-per-pixel", default=1.0, show_default=True)
    @click.option("--source", default=DEFAULT_SOURCE)
    def _cmd(out, heightmap, terrain_dir, objmap, water, units_per_pixel, source):
        """Build a terrain bundle from user-supplied BotW terrain + objmap."""
        path = botw_to_bundle(
            out, heightmap_png=heightmap, terrain_dir=terrain_dir,
            objmap_geojson=objmap, water_png=water,
            units_per_pixel=units_per_pixel, source=source)
        click.secho(f"✓ wrote bundle → {path}", fg="green")

    _cmd()


if __name__ == "__main__":  # pragma: no cover
    main()
