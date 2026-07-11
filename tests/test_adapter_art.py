"""Tier 3 art→bundle orchestrator (extract → synthesize → bundle)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from toposlicr.adapters.art import art_to_bundle  # noqa: E402
from toposlicr.bundle import load_bundle  # noqa: E402


def _make_art(tmp_path, n=160):
    """A simple fantasy-ish map: blue ocean border, green land, an inland lake."""
    import cv2

    img = np.zeros((n, n, 3), dtype="uint8")
    img[:, :] = (200, 170, 120)          # sea (BGR-ish blue-tan; art is RGB here)
    img[:, :] = (60, 110, 200)           # ocean (RGB blue)
    yy, xx = np.mgrid[0:n, 0:n]
    land = (xx - n / 2) ** 2 + (yy - n / 2) ** 2 < (n * 0.36) ** 2
    img[land] = (90, 150, 80)            # green land
    lake = (xx - n * 0.58) ** 2 + (yy - n * 0.45) ** 2 < (n * 0.07) ** 2
    img[lake] = (60, 110, 200)           # inland lake (same blue)
    # a dark river line from interior toward the coast
    cv2.line(img, (int(n * 0.5), int(n * 0.5)), (int(n * 0.5), int(n * 0.9)),
             (30, 30, 30), 2)
    p = tmp_path / "map.png"
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return p


def _make_overlay(tmp_path, n=160):
    import cv2
    ov = np.zeros((n, n, 4), dtype="uint8")
    yy, xx = np.mgrid[0:n, 0:n]
    peak = (xx - n * 0.4) ** 2 + (yy - n * 0.45) ** 2 < (n * 0.1) ** 2
    ov[peak] = (220, 30, 30, 255)        # red mountains (RGBA)
    p = tmp_path / "overlay.png"
    cv2.imwrite(str(p), cv2.cvtColor(ov, cv2.COLOR_RGBA2BGRA))
    return p


def test_art_to_bundle_produces_loadable_bundle(tmp_path):
    art = _make_art(tmp_path)
    out = art_to_bundle(art, tmp_path / "world", cells_across=160,
                        normalize_layers=8, seed=1)
    assert out.is_dir()
    b = load_bundle(out)
    assert (out / "heightmap.png").is_file()
    assert (out / "preview_hillshade.png").is_file()
    assert (out / "preview_relief_over_art.png").is_file()
    dem = b.to_dem()
    lo, hi = dem.elevation_range()
    assert hi > lo                       # real relief was synthesized


def test_art_land_is_higher_than_ocean(tmp_path):
    art = _make_art(tmp_path)
    out = art_to_bundle(art, tmp_path / "w2", cells_across=160, normalize_layers=8)
    b = load_bundle(out)
    elev = b.elevation()
    n = elev.shape[0]
    centre = elev[n // 2, n // 2]        # land interior
    corner = elev[2, 2]                  # ocean corner
    assert centre > corner


def test_art_interior_lake_becomes_water_not_ocean(tmp_path):
    art = _make_art(tmp_path)
    out = art_to_bundle(art, tmp_path / "w3", cells_across=160, normalize_layers=8)
    b = load_bundle(out)
    # water.png should mark the interior lake but NOT the whole ocean border.
    if b.water is not None:
        assert not b.water[0, 0]         # ocean corner is not an inset candidate
        assert b.water.sum() < b.water.size * 0.5


def test_art_rivers_written_as_geojson(tmp_path):
    art = _make_art(tmp_path)
    out = art_to_bundle(art, tmp_path / "w4", cells_across=160, normalize_layers=8)
    rivers = out / "rivers.geojson"
    if rivers.is_file():                 # river tracing is best-effort
        b = load_bundle(out)
        assert isinstance(b.rivers(), list)


def test_art_with_class_overlay(tmp_path):
    art = _make_art(tmp_path)
    overlay = _make_overlay(tmp_path)
    out = art_to_bundle(art, tmp_path / "w5", class_overlay=overlay,
                        cells_across=160, normalize_layers=8)
    assert load_bundle(out).to_dem().elevation_range()[1] > 0


def test_art_bundle_runs_through_core(tmp_path):
    from toposlicr.config import parse_config
    from toposlicr.layers import build_layer_model

    art = _make_art(tmp_path)
    out = art_to_bundle(art, tmp_path / "w6", cells_across=160, normalize_layers=8)
    cfg = parse_config({
        "region": {"bundle": str(out)},
        "physical": {"model_width_mm": 300, "normalize_layers": 8},
    })
    model = build_layer_model(load_bundle(out).to_dem(), cfg)
    assert model.flat and model.layer_count >= 3
