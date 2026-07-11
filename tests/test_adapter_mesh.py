"""Tier 2 mesh → terrain-bundle adapter tests (offline; trimesh primitives)."""

import json

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from toposlicr.adapters.mesh import mesh_to_bundle  # noqa: E402
from toposlicr.bundle import FLAT_CRS, load_bundle  # noqa: E402


def _export(mesh, path):
    mesh.export(path)
    return path


def _report(bundle_dir):
    return json.loads((bundle_dir / "report.json").read_text())


def test_flat_plate_gives_uniform_heightmap(tmp_path):
    plate = trimesh.creation.box(extents=[4.0, 4.0, 0.05])
    src = _export(plate, tmp_path / "plate.stl")
    out = mesh_to_bundle(src, tmp_path / "plate.terrainbundle",
                         cells_across=60, supersample=1)
    b = load_bundle(out)
    assert b.shape[1] == 60
    assert b.to_dem().crs == FLAT_CRS
    # a flat plate is nearly uniform
    assert float(np.nanstd(b.elevation())) < 1e-3


def test_cone_peaks_at_centre(tmp_path):
    cone = trimesh.creation.cone(radius=2.0, height=3.0)
    src = _export(cone, tmp_path / "cone.stl")
    out = mesh_to_bundle(src, tmp_path / "cone.terrainbundle",
                         cells_across=81, supersample=2, pedestal_clip="off")
    b = load_bundle(out)
    elev = b.elevation()
    lo, hi = float(np.nanmin(elev)), float(np.nanmax(elev))
    assert hi > lo                                  # positive relief
    r, c = np.unravel_index(np.nanargmax(elev), elev.shape)
    rows, cols = b.shape
    # apex is near the grid centre
    assert abs(r - rows / 2) <= rows * 0.1
    assert abs(c - cols / 2) <= cols * 0.1


def test_bundle_round_trips_and_slices(tmp_path):
    cone = trimesh.creation.cone(radius=2.0, height=3.0)
    src = _export(cone, tmp_path / "cone.stl")
    out = mesh_to_bundle(src, tmp_path / "cone.terrainbundle",
                         cells_across=70, supersample=1, pedestal_clip="off")
    # the core can build a nested layer model straight from the bundle
    from toposlicr.config import parse_config
    from toposlicr.layers import build_layer_model
    cfg = parse_config({
        "region": {"bundle": str(out)},
        "physical": {"model_width_mm": 200, "normalize_layers": 6},
    })
    model = build_layer_model(load_bundle(out).to_dem(), cfg)
    assert model.flat and model.layer_count >= 3
    for k in range(1, model.layer_count):
        assert model.footprint(k).area <= model.footprint(k - 1).area + 1e-6


def test_pedestal_detected_and_clipped(tmp_path):
    base = trimesh.creation.box(extents=[6.0, 6.0, 1.0])
    base.apply_translation([0, 0, -0.5])            # z in [-1, 0]
    cone = trimesh.creation.cone(radius=1.5, height=3.0)  # z in [0, 3]
    island = trimesh.util.concatenate([base, cone])
    src = _export(island, tmp_path / "island.stl")

    auto = mesh_to_bundle(src, tmp_path / "auto.terrainbundle",
                          cells_across=70, supersample=1, pedestal_clip="auto")
    off = mesh_to_bundle(src, tmp_path / "off.terrainbundle",
                         cells_across=70, supersample=1, pedestal_clip="off")
    ra, ro = _report(auto), _report(off)

    # auto finds the pedestal near the slab top (z≈0); off does not clip
    assert ra["pedestal_clip_height"] is not None
    assert ra["pedestal_clip_height"] == pytest.approx(0.0, abs=0.3)
    assert ro["pedestal_clip_height"] is None
    # clipping the slab leaves only the cone → the flat border becomes holes
    assert ra["hole_fraction"] > ro["hole_fraction"] + 0.2
    # the cone peak still survives the clip
    assert float(np.nanmax(load_bundle(auto).elevation())) > \
        float(np.nanmin(load_bundle(auto).elevation()))


def test_flat_slab_is_not_mistaken_for_pedestal(tmp_path):
    plate = trimesh.creation.box(extents=[4.0, 4.0, 0.05])
    src = _export(plate, tmp_path / "plate.stl")
    out = mesh_to_bundle(src, tmp_path / "p.terrainbundle",
                         cells_across=50, supersample=1, pedestal_clip="auto")
    assert _report(out)["pedestal_clip_height"] is None


def test_up_axis_override(tmp_path):
    cone = trimesh.creation.cone(radius=1.5, height=3.0)
    # rotate so the cone axis lies along X
    cone.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    src = _export(cone, tmp_path / "conex.stl")
    out = mesh_to_bundle(src, tmp_path / "x.terrainbundle", cells_across=70,
                         supersample=1, up_axis="x", pedestal_clip="off")
    elev = load_bundle(out).elevation()
    assert _report(out)["up_axis"] == "x"
    # a sensible peaked heightfield (real relief, not flat)
    assert float(np.nanmax(elev)) - float(np.nanmin(elev)) > 1.0
    assert float(np.nanstd(elev)) > 0.1


def test_scene_mesh_include_filter(tmp_path):
    cone = trimesh.creation.cone(radius=1.5, height=3.0)
    cube = trimesh.creation.box(extents=[1, 1, 1])
    cube.apply_translation([5, 5, 0])
    scene = trimesh.Scene({"terrain_cone": cone, "prop_cube": cube})
    src = tmp_path / "scene.glb"
    scene.export(src)
    out = mesh_to_bundle(src, tmp_path / "s.terrainbundle", cells_across=60,
                         supersample=1, mesh_include=["terrain*"], pedestal_clip="off")
    b = load_bundle(out)
    # only the cone was kept → its peak dominates, prop excluded
    assert float(np.nanmax(b.elevation())) > float(np.nanmin(b.elevation()))
    assert any("filter kept 1/2" in n for n in _report(out)["notes"])
