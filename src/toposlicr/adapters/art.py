"""Tier 3 artwork → terrain bundle (spec Section 3): extract → synthesize → refine.

Orchestrates the extraction (``fictional.extract``) and synthesis
(``fictional.synthesis``) stages into a bundle: segment land/sea, read an
optional painted terrain-class overlay, trace rivers, synthesize a plausible
heightfield, and package it — with lakes (not the whole ocean) as acrylic-inset
candidates and rivers as real scored linework.

This tier is authoring assisted by automation: the class overlay and adjustment
layer are cheap manual steps that keep the human in control of artistic
judgement. Every run emits hillshade + relief-over-art previews for the refine
loop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..bundle import BundleMeta, save_hillshade, write_gray16, write_mask

# Amplitude/profile per painted terrain class (spec 3.2 step 2).
_ZONE_AMPLITUDE = {"mountain": 1.0, "hill": 0.4, "plateau": 0.5}


def art_to_bundle(art_path, out_dir, *, class_overlay=None, adjustment_layer=None,
                  water_segmentation="auto", ocean_point=None, normalize_layers=12,
                  coastal_exponent=0.6, noise_octaves=3, noise_max_fraction=0.4,
                  cells_across=1000, units_per_pixel=100.0, seed=0, ocr=False,
                  river_color=None, source="") -> Path:
    """Convert 2D artwork into a terrain bundle. Returns the bundle directory."""
    import cv2

    from ..fictional import extract as ex
    from ..fictional import synthesis as syn

    rgb, _ = ex._read_rgb(art_path)
    h0, w0 = rgb.shape[:2]
    cells_h = max(16, round(cells_across * h0 / w0))
    small = cv2.resize(rgb, (cells_across, cells_h), interpolation=cv2.INTER_AREA)
    warnings: list[str] = []

    # 1. Land / sea.
    land = ex.segment_land_sea(small, method=water_segmentation, ocean_point=ocean_point)

    # 2. Terrain-class zones from the painted overlay (highest-leverage step).
    zones = []
    if class_overlay is not None:
        masks = ex.read_class_overlay(class_overlay)
        for cls, mask in masks.items():
            m = _resize_mask(mask, cells_across, cells_h)
            if m.any():
                zones.append({"mask": m & land, "class": cls,
                              "amplitude": _ZONE_AMPLITUDE.get(cls, 0.6)})

    # 3. Rivers → polylines + mouth classification.
    rivers_syn, river_lines = [], []
    try:
        polylines, rwarn = ex.extract_rivers(small, river_color=river_color)
        warnings.extend(rwarn)
        mouths = ex.classify_river_mouths(polylines, land)
        for pl, mo in zip(polylines, mouths, strict=False):
            rivers_syn.append({"polyline": [tuple(p) for p in pl], "mouth": mo})
            river_lines.append(pl)
    except Exception as exc:  # river tracing is best-effort
        warnings.append(f"river extraction skipped ({type(exc).__name__}: {exc})")

    # 4. Synthesize the heightfield.
    field, swarn = syn.synthesize(
        land, zones=zones, rivers=rivers_syn, layers=normalize_layers, seed=seed,
        coastal_exponent=coastal_exponent, noise_octaves=noise_octaves,
        noise_max_fraction=noise_max_fraction)
    warnings.extend(swarn)

    # 5. Optional manual adjustment layer (raise/lower), then renormalize.
    if adjustment_layer is not None:
        delta = _resize_field(ex.read_adjustment_layer(adjustment_layer),
                              cells_across, cells_h)
        field = syn.normalize(field + 0.25 * delta)

    # 6. Package the bundle.
    out = Path(out_dir)
    if out.suffix != ".terrainbundle":
        out = out.with_suffix(".terrainbundle") if not out.suffix else \
            out.with_name(out.name + ".terrainbundle")
    out.mkdir(parents=True, exist_ok=True)
    write_gray16(out / "heightmap.png", field)

    # Water: interior lakes only — the ocean is the base, not an acrylic inset.
    lakes = _interior_water(~land)
    if lakes.any():
        write_mask(out / "water.png", lakes)

    transform_hw = (cells_h, units_per_pixel)
    if river_lines:
        _write_rivers_geojson(out / "rivers.geojson", river_lines, transform_hw)

    _write_features(out / "features.csv", small if ocr else None, transform_hw)

    meta = BundleMeta(units_per_pixel=units_per_pixel, height_scale=1.0 / 65535.0,
                      height_offset=0.0, world_origin=(0.0, cells_h * units_per_pixel),
                      vertical_range=[0.0, 1.0], source=source or "synthesized from artwork",
                      attribution="synthesized heightfield; source artwork is the user's")
    (out / "meta.json").write_text(json.dumps(meta.to_dict()))

    # 7. Refine-loop previews.
    save_hillshade(field, out / "preview_hillshade.png")
    _save_relief_over_art(field, small, out / "preview_relief_over_art.png")
    if warnings:
        (out / "report.txt").write_text("\n".join(warnings))
    return out


def _resize_mask(mask, w, h):
    import cv2
    return cv2.resize(mask.astype("uint8"), (w, h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def _resize_field(field, w, h):
    import cv2
    return cv2.resize(field.astype("float32"), (w, h), interpolation=cv2.INTER_LINEAR)


def _interior_water(water: np.ndarray) -> np.ndarray:
    """Water regions not connected to the map border (i.e. lakes, not ocean)."""
    from scipy.ndimage import label

    lbl, n = label(water)
    if n == 0:
        return np.zeros_like(water, dtype=bool)
    border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border.discard(0)
    lakes = water.copy()
    for c in border:
        lakes[lbl == c] = False
    return lakes


def _pix_to_world(row, col, cells_h, upp):
    # Pixel centers (+0.5), matching the core's contour sampling, so features and
    # rivers register with the terrain rather than sitting half a cell off.
    return ((col + 0.5) * upp, (cells_h - row - 0.5) * upp)


def _write_rivers_geojson(path, polylines, transform_hw):
    cells_h, upp = transform_hw
    feats = []
    for pl in polylines:
        coords = [_pix_to_world(r, c, cells_h, upp) for r, c in pl]
        if len(coords) >= 2:
            feats.append({"type": "Feature", "properties": {"type": "river"},
                          "geometry": {"type": "LineString", "coordinates": coords}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _write_features(path, ocr_img, transform_hw):
    import csv

    cells_h, upp = transform_hw
    rows = []
    if ocr_img is not None:
        from ..fictional.extract import ocr_labels
        for lab in ocr_labels(ocr_img):
            # OCR reports pixel positions; convert to the heightmap's world frame.
            px, py = lab.get("x", 0), lab.get("y", 0)
            wx, wy = _pix_to_world(py, px, cells_h, upp)
            rows.append({"name": lab.get("name") or lab.get("text", ""),
                         "type": "place", "x": wx, "y": wy, "include": "yes"})
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "type", "x", "y", "elev",
                                           "include", "icon", "label_override"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _save_relief_over_art(field, art_rgb, path):
    """Composite the synthesized hillshade over the source art (refine preview)."""
    from PIL import Image

    from ..bundle import hillshade

    hs = hillshade(field).astype("float32")
    hs_rgb = np.stack([hs, hs, hs], axis=-1)
    blend = (0.55 * hs_rgb + 0.45 * art_rgb.astype("float32")).clip(0, 255).astype("uint8")
    Image.fromarray(blend).save(path)


try:
    import click

    @click.command()
    @click.argument("art_path", type=click.Path(exists=True))
    @click.option("-o", "--out", "out_dir", default="world.terrainbundle")
    @click.option("--class-overlay", type=click.Path(exists=True), default=None)
    @click.option("--adjust", "adjustment_layer", type=click.Path(exists=True),
                  default=None)
    @click.option("--segmentation", "water_segmentation", default="auto")
    @click.option("--cells-across", default=1000, type=int)
    @click.option("--layers", "normalize_layers", default=12, type=int)
    @click.option("--ocr", is_flag=True)
    def main(art_path, out_dir, class_overlay, adjustment_layer, water_segmentation,
             cells_across, normalize_layers, ocr):
        """Synthesize a terrain bundle from 2D map artwork."""
        path = art_to_bundle(
            art_path, out_dir, class_overlay=class_overlay,
            adjustment_layer=adjustment_layer, water_segmentation=water_segmentation,
            cells_across=cells_across, normalize_layers=normalize_layers, ocr=ocr)
        click.secho(f"✓ bundle → {path}", fg="green")
except ImportError:  # pragma: no cover
    main = None
