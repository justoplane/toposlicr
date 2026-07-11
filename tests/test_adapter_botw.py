"""Tier 1 BotW → terrain-bundle adapter (offline; no game assets)."""

import json

import numpy as np
import pytest

from toposlicr.adapters.botw import (
    assemble_terrain_dir,
    botw_to_bundle,
    read_hght_tile,
)
from toposlicr.bundle import BundleError, load_bundle, write_gray16

# --- raw .hght parsing -----------------------------------------------------

def test_read_hght_tile_roundtrip(tmp_path):
    vals = (np.arange(256 * 256, dtype="uint16") % 65535).reshape(256, 256)
    p = tmp_path / "A.hght"
    p.write_bytes(vals.astype("<u2").tobytes())
    back = read_hght_tile(p)
    assert back.shape == (256, 256)
    assert back.dtype == np.uint16
    assert np.array_equal(back, vals)


def test_read_hght_tile_bad_size(tmp_path):
    p = tmp_path / "bad.hght"
    p.write_bytes(b"\x00\x01\x02")
    with pytest.raises(BundleError, match="256"):
        read_hght_tile(p)


def _write_tile(path, value):
    arr = np.full((256, 256), value, dtype="uint16")
    path.write_bytes(arr.astype("<u2").tobytes())


def test_assemble_terrain_dir_grid_from_names(tmp_path):
    # tile (col, row) filled with r*2 + c so placement is verifiable
    for c in (0, 1):
        for r in (0, 1):
            _write_tile(tmp_path / f"{c}_{r}.hght", r * 2 + c)
    mosaic = assemble_terrain_dir(tmp_path)
    assert mosaic.shape == (512, 512)
    assert mosaic[0, 0] == 0        # (c0, r0)
    assert mosaic[0, 300] == 1      # (c1, r0)
    assert mosaic[300, 0] == 2      # (c0, r1)
    assert mosaic[300, 300] == 3    # (c1, r1)


def test_assemble_terrain_dir_undiscernible_raises(tmp_path):
    for name in ("a", "b", "c"):
        _write_tile(tmp_path / f"{name}.hght", 1)
    with pytest.raises(BundleError, match="community"):
        assemble_terrain_dir(tmp_path)


# --- objmap → features + coordinate consistency ----------------------------

def _objmap():
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [500, 950]},
         "properties": {"name": "Death Mountain", "type": "Mountain"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [300, 300]},
         "properties": {"name": "Lake Hylia", "type": "Lake"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [700, 600]},
         "properties": {"name": "Lakeside Shrine", "type": "Shrine"}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [100, 100]},
         "properties": {"type": "Region"}},  # unnamed
    ]}


def _peaked_heightmap(tmp_path, high_at_top=True):
    """100×100 heightmap, high in the top rows (north)."""
    rows = np.linspace(0, 1, 100)[:, None] * np.ones((1, 100))
    grid = (1 - rows) if high_at_top else rows      # top rows high
    return write_gray16(tmp_path / "height.png", grid * 60000 + 100)


def test_botw_bundle_feature_types_and_include(tmp_path):
    hp = _peaked_heightmap(tmp_path)
    (tmp_path / "obj.geojson").write_text(json.dumps(_objmap()))
    bdir = botw_to_bundle(tmp_path / "out.terrainbundle", heightmap_png=hp,
                          objmap_geojson=tmp_path / "obj.geojson",
                          world_bounds=(0, 0, 1000, 1000))
    b = load_bundle(bdir)
    feats = {f.name: f for f in b.features()}
    assert "Death Mountain" in feats and "Lake Hylia" in feats
    assert "Lakeside Shrine" in feats
    assert feats["Death Mountain"].feature_type.value == "peak"
    assert feats["Lake Hylia"].feature_type.value == "lake"
    assert feats["Lakeside Shrine"].tags.get("icon") == "shrine"
    # attribution recorded, no assets referenced
    assert "user-supplied" in b.meta.attribution.lower()


def test_botw_unnamed_feature_excluded(tmp_path):
    hp = _peaked_heightmap(tmp_path)
    (tmp_path / "obj.geojson").write_text(json.dumps(_objmap()))
    bdir = botw_to_bundle(tmp_path / "out.terrainbundle", heightmap_png=hp,
                          objmap_geojson=tmp_path / "obj.geojson",
                          world_bounds=(0, 0, 1000, 1000))
    b = load_bundle(bdir)
    # the unnamed Region point is include=no → absent from the loaded collection
    assert len(list(b.features())) == 3


def test_botw_coordinate_consistency(tmp_path):
    """Death Mountain (high y) must land on a high heightmap cell."""
    hp = _peaked_heightmap(tmp_path, high_at_top=True)
    (tmp_path / "obj.geojson").write_text(json.dumps(_objmap()))
    bdir = botw_to_bundle(tmp_path / "out.terrainbundle", heightmap_png=hp,
                          objmap_geojson=tmp_path / "obj.geojson",
                          world_bounds=(0, 0, 1000, 1000))
    b = load_bundle(bdir)
    dm = next(f for f in b.features() if f.name == "Death Mountain")
    x, y = dm.geometry.x, dm.geometry.y
    # within the heightmap's world bounds
    assert 0 <= x <= 1000
    assert 0 <= y <= 1000
    # the cell under the feature is in the top (high) half of the terrain
    col, row = (~b.transform()) * (x, y)
    row, col = int(row), int(col)
    assert row < 50                                  # top half
    assert b.heightmap[row, col] > np.median(b.heightmap)


def test_botw_derives_frame_from_objmap_when_no_bounds(tmp_path):
    hp = _peaked_heightmap(tmp_path)
    (tmp_path / "obj.geojson").write_text(json.dumps(_objmap()))
    bdir = botw_to_bundle(tmp_path / "out.terrainbundle", heightmap_png=hp,
                          objmap_geojson=tmp_path / "obj.geojson")
    b = load_bundle(bdir)
    # every included feature falls within the derived world bounds
    left, top = b.transform() * (0, 0)
    right, bottom = b.transform() * (b.shape[1], b.shape[0])
    for f in b.features():
        assert min(left, right) <= f.geometry.x <= max(left, right)
        assert min(top, bottom) <= f.geometry.y <= max(top, bottom)


def test_botw_requires_a_source():
    with pytest.raises(ValueError, match="heightmap_png|terrain_dir"):
        botw_to_bundle("/tmp/nope.terrainbundle")


# --- full core run ---------------------------------------------------------

def test_bundle_runs_through_core(tmp_path):
    from toposlicr.config import parse_config
    from toposlicr.layers import build_layer_model

    ys, xs = np.mgrid[0:80, 0:80] / 80
    z = np.exp(-(((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / (2 * 0.04)))
    hp = write_gray16(tmp_path / "h.png", z * 50000 + 100)
    bdir = botw_to_bundle(tmp_path / "w.terrainbundle", heightmap_png=hp,
                          units_per_pixel=5.0)
    b = load_bundle(bdir)
    cfg = parse_config({"region": {"bundle": str(bdir)},
                        "physical": {"model_width_mm": 300, "normalize_layers": 6}})
    model = build_layer_model(b.to_dem(), cfg)
    assert model.flat and model.layer_count >= 2
