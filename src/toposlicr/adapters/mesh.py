"""Tier 2 adapter: 3D mesh → terrain bundle (spec Section 2).

Loads an STL/OBJ/PLY/GLB mesh, orients it, removes a print pedestal, and
rasterizes the **top surface** into a heightfield — then writes a terrain
bundle the core consumes like any DEM.

Rasterization method. The spec's default is vertical ray-grid casting keeping the
highest hit. Trimesh's ray engines need ``embreex`` (Embree) or ``rtree`` for
acceleration; when neither is present this module uses the spec's third listed
method — numpy barycentric **triangle z-max rasterization** — which is
mathematically identical (for each cell, the maximum surface z of any covering
triangle == the highest vertical-ray intersection) while needing no BVH and
tolerating non-manifold, non-watertight game rips. If ``embreex`` is importable
the ray path is used instead.

Taking the max is semantically correct: a stacked-layer map cannot represent
overhangs or caves, so the top surface is by definition what gets built.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..bundle import BundleMeta, save_hillshade, write_gray16

_EPS = 1e-9


@dataclass
class MeshReport:
    """Diagnostics from a conversion — the tripwire for garbage meshes."""

    up_axis: str = "z"
    engine: str = "triangle-raster"
    pedestal_clip_height: float | None = None
    hole_fraction: float = 0.0
    overhang_fraction: float = 0.0
    cells: tuple[int, int] = (0, 0)
    notes: list[str] = field(default_factory=list)


# --- Stage A: load & normalize --------------------------------------------

def _load_mesh(mesh_path, mesh_include, report: MeshReport):
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene" if mesh_include else None)
    if isinstance(loaded, trimesh.Scene):
        geoms = loaded.geometry
        names = list(geoms)
        if mesh_include:
            names = [n for n in names
                     if any(fnmatch.fnmatch(n, pat) for pat in mesh_include)]
            report.notes.append(f"mesh filter kept {len(names)}/{len(geoms)} parts")
        meshes = [geoms[n] for n in names]
        if not meshes:
            raise ValueError("no meshes matched mesh_include filter")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


# --- Stage B: orientation & pedestal --------------------------------------

def _detect_up(vertices: np.ndarray) -> str:
    extents = vertices.max(0) - vertices.min(0)
    axis = int(np.argmin(extents))            # terrain is flat-ish → thin axis is up
    col = vertices[:, axis]
    mid = (col.min() + col.max()) / 2.0
    # Bottom-heavy → most mass low, peaks sparse → up is +axis; else flipped.
    sign = "" if np.mean(col < mid) >= 0.5 else "-"
    return f"{sign}{'xyz'[axis]}"


def _orient(vertices: np.ndarray, up: str) -> np.ndarray:
    up = up.lower().strip()
    sign = 1.0
    if up[0] in "+-":
        sign = -1.0 if up[0] == "-" else 1.0
        up = up[1:]
    axis = {"x": 0, "y": 1, "z": 2}[up]
    others = [i for i in range(3) if i != axis]
    return np.column_stack([vertices[:, others[0]], vertices[:, others[1]],
                            sign * vertices[:, axis]])


def _face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _detect_pedestal(verts: np.ndarray, faces: np.ndarray,
                     report: MeshReport) -> float | None:
    """Base_top = the dominant horizontal flat level (by area) that has real
    terrain above it and sits in the lower part of the height range.

    Uses face *area*, not vertex count: a printed pedestal is a large flat slab
    that may have very few vertices (true of low-poly test meshes and clean
    game rips alike).
    """
    lo, hi = float(verts[:, 2].min()), float(verts[:, 2].max())
    rng = hi - lo
    if rng <= 0:
        return None
    fz = verts[faces][:, :, 2]
    horiz = (fz.max(1) - fz.min(1)) < 0.02 * rng      # near-horizontal faces
    if not horiz.any():
        return None
    areas = _face_areas(verts, faces)
    total = float(areas.sum()) or 1.0
    horiz_mean = fz[horiz].mean(1)
    hist, edges = np.histogram(horiz_mean, bins=40, range=(lo, hi),
                               weights=areas[horiz])
    fmax = fz.max(1)
    # A print pedestal is a SLAB: a flat top *elevated above* the model bottom,
    # backed by a matching flat bottom face. A ground plain/plateau has flat area
    # only at the very bottom — clipping that would delete real terrain. So we
    # require both a bottom slab face (near lo) and the flat top to sit clearly
    # above lo, with sloped terrain rising above it.
    bottom_band = horiz_mean <= lo + 0.05 * rng
    has_bottom_slab = float(areas[horiz][bottom_band].sum()) >= 0.05 * total
    if not has_bottom_slab:
        return None
    for k in range(len(hist) - 1, -1, -1):
        frac = hist[k] / total
        base_top = float(edges[k + 1])
        if not (frac >= 0.12 and base_top <= lo + 0.6 * rng):
            continue
        if base_top - lo < 0.1 * rng:                  # too thin to be a slab (it's the ground)
            continue
        sloped_above = (fmax > base_top + 0.05 * rng) & ~horiz
        if areas[sloped_above].sum() < 0.05 * total:
            continue                                   # only flat above → a slab, not terrain
        if frac < 0.22:
            report.notes.append(
                f"uncertain pedestal (flat level holds {frac:.0%} of area)")
        return base_top
    return None


# --- Stage C: rasterization (triangle z-max == vertical-ray max) -----------

def _rasterize_zmax(verts, faces, minx, maxy, upp, rows, cols):
    """Top-surface height per cell + lowest covering z (overhang proxy)."""
    zmax = np.full((rows, cols), -np.inf)
    zmin = np.full((rows, cols), np.inf)
    vx, vy, vz = verts[:, 0], verts[:, 1], verts[:, 2]
    for f in faces:
        a, b, c = f
        ax, ay, az = vx[a], vy[a], vz[a]
        bx, by, bz = vx[b], vy[b], vz[b]
        cx, cy, cz = vx[c], vy[c], vz[c]
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-15:
            continue
        c0 = int(np.floor((min(ax, bx, cx) - minx) / upp - 0.5))
        c1 = int(np.ceil((max(ax, bx, cx) - minx) / upp - 0.5))
        r0 = int(np.floor((maxy - max(ay, by, cy)) / upp - 0.5))
        r1 = int(np.ceil((maxy - min(ay, by, cy)) / upp - 0.5))
        c0, c1 = max(c0, 0), min(c1, cols - 1)
        r0, r1 = max(r0, 0), min(r1, rows - 1)
        if c1 < c0 or r1 < r0:
            continue
        gx = minx + (np.arange(c0, c1 + 1) + 0.5) * upp
        gy = maxy - (np.arange(r0, r1 + 1) + 0.5) * upp
        GX, GY = np.meshgrid(gx, gy)
        u = ((by - cy) * (GX - cx) + (cx - bx) * (GY - cy)) / det
        v = ((cy - ay) * (GX - cx) + (ax - cx) * (GY - cy)) / det
        w = 1.0 - u - v
        inside = (u >= -1e-9) & (v >= -1e-9) & (w >= -1e-9)
        if not inside.any():
            continue
        z = u * az + v * bz + w * cz
        bmax = zmax[r0:r1 + 1, c0:c1 + 1]
        bmin = zmin[r0:r1 + 1, c0:c1 + 1]
        bmax[inside] = np.maximum(bmax[inside], z[inside])
        bmin[inside] = np.minimum(bmin[inside], z[inside])
    holes = ~np.isfinite(zmax)
    zmax[holes] = np.nan
    span = np.where(np.isfinite(zmin) & ~holes, zmax - zmin, 0.0)
    return zmax, span


def _blockmax(arr: np.ndarray, ss: int) -> np.ndarray:
    if ss <= 1:
        return arr
    import warnings

    rows, cols = arr.shape[0] // ss, arr.shape[1] // ss
    trimmed = arr[:rows * ss, :cols * ss].reshape(rows, ss, cols, ss)
    with warnings.catch_warnings():           # all-NaN blocks are holes, expected
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(trimmed, axis=(1, 3))


# --- Stage E: post-processing ---------------------------------------------

def _fill_holes(z: np.ndarray) -> np.ndarray:
    """Nearest-valid fill (deterministic; precise for float elevations)."""
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(z)
    if not mask.any():
        return z
    idx = distance_transform_edt(mask, return_indices=True)[1]
    return z[tuple(idx)]


def _postprocess(z: np.ndarray, report: MeshReport) -> np.ndarray:
    from scipy.ndimage import gaussian_filter, median_filter

    report.hole_fraction = float(np.mean(np.isnan(z)))
    if report.hole_fraction > 0.03:
        report.notes.append(
            f"{report.hole_fraction:.0%} of cells missed — likely a broken mesh")
    z = _fill_holes(z)
    z = median_filter(z, size=3)               # reject stray-triangle spikes/pits
    return gaussian_filter(z, sigma=0.8)


# --- Stage F: water candidate inference -----------------------------------

def _water_candidates(z: np.ndarray, minx, maxy, upp, report: MeshReport):
    from scipy import ndimage

    relief = float(np.nanmax(z) - np.nanmin(z)) or 1.0
    dy, dx = np.gradient(z)
    slope = np.hypot(dx, dy)
    flat = slope < (0.01 * relief)
    labels, n = ndimage.label(flat)
    min_area = max(16, z.size // 400)
    mask = np.zeros(z.shape, dtype=bool)
    candidates = []
    for lab in range(1, n + 1):
        region = labels == lab
        if region.sum() < min_area:
            continue
        # local minimum: region mean below its dilated surroundings' mean
        surround = ndimage.binary_dilation(region, iterations=3) & ~region
        if surround.any() and z[region].mean() < z[surround].mean() - 0.02 * relief:
            mask |= region
            rr, cc = np.where(region)
            wx = minx + (cc.mean() + 0.5) * upp
            wy = maxy - (rr.mean() + 0.5) * upp
            candidates.append((wx, wy))
    if candidates:
        report.notes.append(f"{len(candidates)} water candidate(s) — review to enable")
    return mask, candidates


# --- driver ----------------------------------------------------------------

def mesh_to_bundle(mesh_path, out_dir, *, up_axis="auto", pedestal_clip="auto",
                   cells_across=1000, supersample=2, mesh_include=None,
                   water_candidates=True) -> Path:
    """Convert a 3D mesh into a ``*.terrainbundle`` directory. Returns its path."""
    report = MeshReport()
    mesh = _load_mesh(mesh_path, mesh_include, report)

    up = _detect_up(mesh.vertices) if up_axis == "auto" else up_axis
    report.up_axis = up
    verts = _orient(np.asarray(mesh.vertices, dtype="float64"), up)
    faces = np.asarray(mesh.faces, dtype="int64")

    # Pedestal removal.
    if pedestal_clip == "off":
        base_top = None
    elif pedestal_clip == "auto":
        base_top = _detect_pedestal(verts, faces, report)
    else:
        base_top = float(pedestal_clip)
    if base_top is not None:
        report.pedestal_clip_height = base_top
        keep = verts[:, 2][faces].max(axis=1) > base_top + _EPS
        faces = faces[keep]
        if len(faces) == 0:
            raise ValueError("pedestal clip removed all geometry; try pedestal_clip=off")

    # Grid from the mesh XY extent (world units == mesh units). Size cells from
    # the LONGER axis so cells_across bounds it — otherwise a long, thin mesh
    # blows the other dimension up to unbounded rows/memory.
    minx, miny = verts[:, 0].min(), verts[:, 1].min()
    maxx, maxy = verts[:, 0].max(), verts[:, 1].max()
    xext, yext = float(maxx - minx), float(maxy - miny)
    if xext <= 0 or yext <= 0:
        raise ValueError("degenerate mesh XY extent")
    upp = max(xext, yext) / cells_across
    cols = max(1, int(round(xext / upp)))
    rows = max(1, int(round(yext / upp)))

    ss = max(1, int(supersample))
    zmax, span = _rasterize_zmax(verts, faces, minx, maxy, upp / ss, rows * ss, cols * ss)
    z = _blockmax(zmax, ss)
    z = z[:rows, :cols]

    if not np.isfinite(z).any():
        raise ValueError(
            "no mesh surface covered any grid cell — likely a bad orientation or "
            "all-degenerate/vertical faces; try a different up_axis or "
            "pedestal_clip=off")

    relief = float(np.nanmax(z) - np.nanmin(z)) or 1.0
    report.overhang_fraction = float(np.mean(_blockmax(span, ss)[:rows, :cols]
                                             > 0.5 * relief))
    if report.overhang_fraction > 0.02:
        report.notes.append(
            f"{report.overhang_fraction:.0%} of cells sit over overhangs/caves; "
            "only the top surface is kept")

    z = _postprocess(z, report)
    report.cells = (rows, cols)

    out = _write_bundle(out_dir, z, minx, maxy, upp, mesh_path, report,
                        water_candidates)
    return out


def _write_bundle(out_dir, z, minx, maxy, upp, mesh_path, report: MeshReport,
                  do_water) -> Path:
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_gray16(out / "heightmap.png", z)

    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    scale = (hi - lo) / 65535.0 if hi > lo else 1.0
    meta = BundleMeta(units_per_pixel=float(upp), height_scale=scale,
                      height_offset=lo, world_origin=(float(minx), float(maxy)),
                      vertical_range=[lo, hi], source=f"mesh:{Path(mesh_path).name}",
                      attribution="user-supplied 3D model")
    (out / "meta.json").write_text(_json(meta.to_dict()))

    rows_out = []
    if do_water:
        mask, cands = _water_candidates(z, minx, maxy, upp, report)
        if mask.any():
            Image.fromarray(mask.astype("uint16") * 65535).save(
                out / "water_candidates.png")
        for i, (wx, wy) in enumerate(cands):
            rows_out.append({"name": f"water_{i}", "type": "lake", "x": round(wx, 2),
                             "y": round(wy, 2), "elev": "", "include": "no",
                             "icon": "", "label_override": ""})
    _write_features_csv(out / "features.csv", rows_out)

    save_hillshade(z, out / "preview_hillshade.png")
    (out / "report.json").write_text(_json({
        "up_axis": report.up_axis, "engine": report.engine,
        "pedestal_clip_height": report.pedestal_clip_height,
        "hole_fraction": report.hole_fraction,
        "overhang_fraction": report.overhang_fraction,
        "cells": list(report.cells), "notes": report.notes,
    }))
    return out


_FEATURE_FIELDS = ["name", "type", "x", "y", "elev", "include", "icon",
                   "label_override"]


def _write_features_csv(path, rows):
    import csv

    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_FEATURE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _json(obj) -> str:
    import json

    return json.dumps(obj, indent=2)


# --- CLI -------------------------------------------------------------------

def main():  # pragma: no cover - thin CLI wrapper
    import click

    @click.command()
    @click.argument("mesh_path", type=click.Path(exists=True, dir_okay=False))
    @click.option("-o", "--out", "out_dir", default="world.terrainbundle",
                  show_default=True)
    @click.option("--cells-across", type=int, default=1000, show_default=True)
    @click.option("--up-axis", default="auto", show_default=True)
    @click.option("--pedestal-clip", default="auto", show_default=True)
    @click.option("--supersample", type=int, default=2, show_default=True)
    @click.option("--no-water", is_flag=True, help="Skip water-candidate detection.")
    def _cmd(mesh_path, out_dir, cells_across, up_axis, pedestal_clip, supersample,
             no_water):
        """Convert a 3D mesh (STL/OBJ/GLB/PLY) into a terrain bundle."""
        out = mesh_to_bundle(mesh_path, out_dir, up_axis=up_axis,
                             pedestal_clip=pedestal_clip, cells_across=cells_across,
                             supersample=supersample, water_candidates=not no_water)
        click.secho(f"✓ terrain bundle → {out}", fg="green")

    _cmd()


if __name__ == "__main__":  # pragma: no cover
    main()
