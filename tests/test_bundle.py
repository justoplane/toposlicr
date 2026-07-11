"""Terrain-bundle contract + fictional-map core support (foundation tests)."""

import csv
import json

import numpy as np
import pytest

from toposlicr.bundle import (
    FLAT_CRS,
    BundleError,
    BundleMeta,
    hillshade,
    load_bundle,
    write_gray16,
)
from toposlicr.config import ConfigError, parse_config
from toposlicr.icons import AVAILABLE, icon_glyph


def _make_bundle(root, *, with_water=True, with_features=True, n=120):
    bdir = root / "world.terrainbundle"
    bdir.mkdir(parents=True, exist_ok=True)
    ys, xs = np.mgrid[0:n, 0:n] / n
    z = np.exp(-(((xs - 0.4) ** 2 + (ys - 0.45) ** 2) / (2 * 0.03)))
    z += 0.1 * np.exp(-(((xs - 0.5) ** 2 + (ys - 0.5) ** 2) / (2 * 0.4)))
    write_gray16(bdir / "heightmap.png", z)
    if with_water:
        water = ((xs - 0.65) ** 2 + (ys - 0.3) ** 2 < 0.01).astype("uint16") * 65535
        write_gray16(bdir / "water.png", water)
    meta = BundleMeta(units_per_pixel=100.0, world_origin=(0.0, n * 100.0),
                      source="test world")
    (bdir / "meta.json").write_text(json.dumps(meta.to_dict()))
    if with_features:
        with (bdir / "features.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["name", "type", "x", "y", "elev",
                                              "include", "icon", "label_override"])
            w.writeheader()
            w.writerow({"name": "Vaelspire", "type": "peak", "x": 4800, "y": 6600,
                        "include": "yes", "icon": "peak"})
            w.writerow({"name": "Hidden", "type": "poi", "x": 100, "y": 100,
                        "include": "no"})
            w.writerow({"name": "Old", "type": "town", "x": 6000, "y": 3000,
                        "include": "yes", "icon": "tower", "label_override": "Newkeep"})
    return bdir


# --- format --------------------------------------------------------------

def test_meta_roundtrip():
    m = BundleMeta(units_per_pixel=50, height_scale=2.0, world_origin=(1, 2),
                   source="x")
    assert BundleMeta.from_dict(m.to_dict()).units_per_pixel == 50


def test_write_and_read_gray16_roundtrip(tmp_path):
    data = np.linspace(0, 1000, 64 * 64).reshape(64, 64)
    from toposlicr.bundle import read_gray
    p = write_gray16(tmp_path / "h.png", data)
    back = read_gray(p)
    assert back.shape == (64, 64)
    assert back.dtype == np.uint16
    # monotonic mapping preserved (corners are the extremes)
    assert back[0, 0] == 0 and back[-1, -1] == 65535


def test_load_bundle_and_dem(tmp_path):
    bdir = _make_bundle(tmp_path)
    b = load_bundle(bdir)
    assert b.shape == (120, 120)
    dem = b.to_dem()
    assert dem.crs == FLAT_CRS
    lo, hi = dem.elevation_range()
    assert hi > lo


def test_missing_bundle_and_heightmap(tmp_path):
    with pytest.raises(BundleError, match="not found"):
        load_bundle(tmp_path / "nope.terrainbundle")
    empty = tmp_path / "empty.terrainbundle"
    empty.mkdir()
    with pytest.raises(BundleError, match="heightmap"):
        load_bundle(empty)


def test_water_grid_mismatch_raises(tmp_path):
    bdir = _make_bundle(tmp_path, with_water=False)
    write_gray16(bdir / "water.png", np.zeros((10, 10)))
    with pytest.raises(BundleError, match="water"):
        load_bundle(bdir)


def test_water_polygons_in_world_coords(tmp_path):
    b = load_bundle(_make_bundle(tmp_path))
    polys = b.water_polygons()
    assert len(polys) >= 1
    # world coords: origin (0, 12000), so y within [0, 12000]
    minx, miny, maxx, maxy = polys[0].bounds
    assert minx >= 0 and maxx <= 120 * 100
    assert polys[0].area > 0


def test_features_respect_include_and_icon_and_override(tmp_path):
    b = load_bundle(_make_bundle(tmp_path))
    feats = list(b.features())
    names = {f.name for f in feats}
    assert "Vaelspire" in names
    assert "Hidden" not in names           # include=no dropped
    assert "Newkeep" in names              # label_override applied
    peak = next(f for f in feats if f.name == "Vaelspire")
    assert peak.tags.get("icon") == "peak"


def test_hillshade_shape_and_dtype():
    z = np.random.default_rng(0).random((32, 32))
    hs = hillshade(z)
    assert hs.shape == (32, 32) and hs.dtype == np.uint8


# --- icons ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["peak", "tower", "shrine", "town", "city", "poi"])
def test_icon_glyphs_are_valid_sized_polygons(name):
    g = icon_glyph(name, cx=10.0, cy=20.0, size_mm=6.0)
    assert not g.is_empty and g.is_valid and g.area > 0
    minx, miny, maxx, maxy = g.bounds
    assert max(maxx - minx, maxy - miny) <= 6.0 + 1e-6
    assert 7 < (minx + maxx) / 2 < 13     # centred near cx


def test_unknown_icon_falls_back_to_marker():
    assert not icon_glyph("dragon", 0, 0, 5).is_empty
    assert "peak" in AVAILABLE


# --- config --------------------------------------------------------------

def test_config_accepts_bundle_region(tmp_path):
    cfg = parse_config({
        "region": {"bundle": "world.terrainbundle"},
        "physical": {"model_width_mm": 300, "normalize_layers": 8},
    })
    assert cfg.region.is_fictional
    assert cfg.region.dem == "terrain_bundle"
    assert cfg.region.bbox is None
    assert cfg.physical.normalize_layers == 8


def test_config_requires_bbox_or_bundle():
    with pytest.raises(ConfigError, match="bbox.*bundle"):
        parse_config({"region": {}, "physical": {"model_width_mm": 300}})


def test_config_clip_percentiles_parsed():
    cfg = parse_config({
        "region": {"bundle": "w"},
        "physical": {"model_width_mm": 300, "normalize_layers": 6,
                     "clip_percentiles": [1, 99]},
    })
    assert cfg.physical.clip_percentiles == [1.0, 99.0]


def test_config_rejects_two_drivers():
    with pytest.raises(ConfigError, match="only one"):
        parse_config({
            "region": {"bundle": "w"},
            "physical": {"model_width_mm": 300, "normalize_layers": 6,
                         "interval_m": 100},
        })


# --- flat layer model + end-to-end ---------------------------------------

def test_flat_layer_model_needs_no_bbox(tmp_path):
    from toposlicr.layers import build_layer_model
    cfg = parse_config({
        "region": {"bundle": str(_make_bundle(tmp_path))},
        "physical": {"model_width_mm": 300, "normalize_layers": 6},
    })
    b = load_bundle(cfg.region.bundle)
    model = build_layer_model(b.to_dem(), cfg, bbox=None)
    assert model.flat is True
    assert model.utm_epsg == 0
    assert 4 <= model.layer_count <= 8
    # nested + within model bounds
    for k in range(1, model.layer_count):
        assert model.footprint(k).area <= model.footprint(k - 1).area + 1e-6


def test_normalize_layers_hits_target_count(tmp_path):
    from toposlicr.layers import build_layer_model
    cfg = parse_config({
        "region": {"bundle": str(_make_bundle(tmp_path))},
        "physical": {"model_width_mm": 300, "normalize_layers": 10},
    })
    b = load_bundle(cfg.region.bundle)
    model = build_layer_model(b.to_dem(), cfg)
    assert abs(model.layer_count - 10) <= 1


def test_fictional_pipeline_end_to_end(tmp_path):
    from toposlicr.pipeline import run_pipeline
    bdir = _make_bundle(tmp_path)
    cfg = parse_config({
        "region": {"bundle": str(bdir)},
        "physical": {"model_width_mm": 300, "normalize_layers": 8},
        "machine": {"bed_mm": [495, 279]},
        "materials": {"default": "birch_3mm", "water": "blue_acrylic_3mm"},
        "symbology": {"lakes": {"min_area_km2": 0.0, "mode": "inset"},
                      "labels": {"cap_height_mm": 4.0, "icon_size_mm": 6.0}},
    })
    res = run_pipeline(cfg, tmp_path / "out", write_debug=False)
    assert res.model.flat
    assert len(res.boards) >= 1
    # water became an acrylic inset on its own material board
    materials = {b.material for b in res.boards}
    assert "blue_acrylic_3mm" in materials
    assert any(p.kind == "acrylic" for p in res.parts)
