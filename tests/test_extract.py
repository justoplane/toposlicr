"""Tier 3 artwork feature-extraction tests — offline, synthetic images only."""

import cv2
import numpy as np
import pytest

from toposlicr.fictional.extract import (
    classify_river_mouths,
    coastline,
    extract_rivers,
    ocr_labels,
    read_adjustment_layer,
    read_class_overlay,
    segment_land_sea,
)


def _blue_sea_green_island(n=100, radius=30):
    """RGB image: blue ocean everywhere, a green land circle in the middle."""
    img = np.empty((n, n, 3), np.uint8)
    img[:, :] = (30, 60, 200)                       # blue (RGB)
    cv2.circle(img, (n // 2, n // 2), radius, (40, 160, 60), -1)  # green land
    return img


def _write_png(path, rgb):
    """Write an RGB array to a PNG file (cv2 stores BGR, so flip channels)."""
    cv2.imwrite(str(path), rgb[:, :, ::-1] if rgb.ndim == 3 else rgb)


# --- land / sea -----------------------------------------------------------

def test_segment_land_sea_kmeans():
    img = _blue_sea_green_island()
    land = segment_land_sea(img, method="auto")
    assert land.dtype == bool and land.shape == (100, 100)
    assert land[50, 50]           # interior = land
    assert not land[0, 0]         # border = sea
    assert not land[99, 99]
    assert 0.05 < land.mean() < 0.5


def test_segment_land_sea_flood_fill():
    img = _blue_sea_green_island()
    land = segment_land_sea(img, method="flood_fill", ocean_point=(0, 0), tolerance=40)
    assert land[50, 50]
    assert not land[0, 0]


def test_segment_sam_not_installed():
    with pytest.raises(NotImplementedError, match="SAM"):
        segment_land_sea(_blue_sea_green_island(), method="sam")


# --- coastline ------------------------------------------------------------

def test_coastline_traces_circle():
    n, r = 100, 30
    mask = np.zeros((n, n), bool)
    yy, xx = np.mgrid[0:n, 0:n]
    mask[(xx - 50) ** 2 + (yy - 50) ** 2 <= r ** 2] = True
    contours = coastline(mask)
    assert len(contours) >= 1
    big = max(contours, key=len)
    rmin, rmax = big[:, 0].min(), big[:, 0].max()
    cmin, cmax = big[:, 1].min(), big[:, 1].max()
    assert 15 < rmin < 25 and 75 < rmax < 85
    assert 15 < cmin < 25 and 75 < cmax < 85


# --- rivers ---------------------------------------------------------------

def test_extract_rivers_traces_a_line():
    img = np.full((100, 120, 3), 255, np.uint8)
    pts = [(x, int(50 + 20 * np.sin(x / 15.0))) for x in range(10, 110)]
    for a, b in zip(pts[:-1], pts[1:], strict=False):
        cv2.line(img, a, b, (20, 20, 20), 2)
    dark_px = int((cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) < 100).sum())

    polylines, warnings = extract_rivers(img)
    assert len(polylines) >= 1
    longest = max(polylines, key=len)
    # traces most of the drawn span (columns 10..109)
    assert longest[:, 1].min() < 20 and longest[:, 1].max() > 100
    # simplification: far fewer vertices than lit pixels
    assert len(longest) < dark_px


def test_extract_rivers_empty_when_no_ink():
    polylines, warnings = extract_rivers(np.full((40, 40, 3), 255, np.uint8))
    assert polylines == []
    assert warnings


def test_classify_river_mouths_picks_sea_end():
    land = np.ones((100, 100), bool)
    land[90:, :] = False                              # sea along the bottom
    poly = np.array([[10, 50], [50, 50], [95, 50]], float)  # top→bottom (toward sea)
    assert classify_river_mouths([poly], land) == [1]


# --- overlays -------------------------------------------------------------

def test_read_class_overlay(tmp_path):
    overlay = np.zeros((60, 60, 3), np.uint8)         # black = plains
    overlay[5:15, 5:15] = (210, 40, 40)               # red = mountain
    overlay[5:15, 40:50] = (230, 140, 30)             # orange = hill
    overlay[40:50, 5:15] = (230, 220, 40)             # yellow = plateau
    p = tmp_path / "overlay.png"
    _write_png(p, overlay)

    masks = read_class_overlay(p)
    assert set(masks) == {"mountain", "hill", "plateau"}
    assert masks["mountain"][10, 10] and masks["hill"][10, 45]
    assert masks["plateau"][45, 10]
    assert not masks["mountain"][45, 45]              # plains untouched
    # classes don't overlap
    assert not (masks["mountain"] & masks["hill"]).any()


def test_read_adjustment_layer(tmp_path):
    gray = np.full((80, 80), 128, np.uint8)           # neutral
    gray[10:25, 10:25] = 220                          # raise
    gray[55:70, 55:70] = 40                           # lower
    p = tmp_path / "adjust.png"
    cv2.imwrite(str(p), gray)

    delta = read_adjustment_layer(p, feather_px=3.0)
    assert delta.shape == (80, 80)
    assert delta[17, 17] > 0.3                        # light patch raises
    assert delta[62, 62] < -0.3                       # dark patch lowers
    assert abs(delta[40, 40]) < 0.15                  # neutral middle
    # feathered: values vary smoothly (no single hard step to full range)
    assert 0 < delta.std() < 1.0


# --- OCR (gated) ----------------------------------------------------------

def test_ocr_labels_requires_extra():
    try:
        import pytesseract  # noqa: F401
        installed = True
    except ImportError:
        installed = False
    if installed:
        pytest.skip("pytesseract is installed; the gated-error path can't be tested")
    with pytest.raises(RuntimeError, match="OCR"):
        ocr_labels(np.full((30, 80, 3), 255, np.uint8))


def test_single_color_art_is_all_land():
    # Solid-colour art has no separable water cluster → must be treated as all
    # land, not an arbitrary k-means split.
    img = np.full((60, 60, 3), (80, 150, 80), dtype="uint8")
    land = segment_land_sea(img, method="auto")
    assert land.mean() > 0.95
