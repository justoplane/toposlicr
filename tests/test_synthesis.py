"""Tier 3 heightfield synthesis stack (spec Section 3.2)."""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from toposlicr.fictional.synthesis import (
    band_aware_noise,
    carve_rivers,
    coastal_ramp,
    normalize,
    stamp_ridges,
    synthesize,
    synthesize_to_bundle,
)


def _island(n=80, r=0.35):
    ys, xs = np.mgrid[0:n, 0:n] / n
    return ((xs - 0.5) ** 2 + (ys - 0.5) ** 2) < r ** 2


# --- coastal ramp ---------------------------------------------------------

def test_coastal_ramp_rises_inland_and_sea_is_zero():
    n = 80
    land = _island(n)
    field = coastal_ramp(land, amplitude=1.0, exponent=0.6)
    c = n // 2
    assert field[~land].max() == 0.0                    # sea flat at 0
    assert field[c, c] == field.max()                   # highest at centre
    # a coastal land cell is lower than the interior
    edge = np.argwhere(land)[0]
    assert field[edge[0], edge[1]] < field[c, c]


def test_coastal_ramp_is_concave():
    n = 100
    land = _island(n, r=0.45)
    field = coastal_ramp(land, exponent=0.6)
    c = n // 2
    # along a radius the value at half-distance exceeds half the max (concave)
    row = field[c, c:]
    peak = row[0]
    half = row[len(row) // 2]
    assert half > 0.4 * peak


# --- ridge stamping -------------------------------------------------------

def test_stamp_ridges_crest_follows_medial_axis():
    n = 80
    field = np.zeros((n, n))
    zone = np.zeros((n, n), dtype=bool)
    zone[20:31, 10:70] = True                           # long, 11 rows thick
    out = stamp_ridges(field, zone, amplitude=1.0, profile="mountain")
    mid_col = 40
    crest = out[25, mid_col]                            # centerline row
    edge = out[21, mid_col]                             # near zone edge
    assert crest > edge > 0
    assert not np.shares_memory(out, field)             # input not mutated


def test_stamp_ridges_plateau_has_flat_top():
    n = 90
    zone = np.zeros((n, n), dtype=bool)
    zone[20:70, 20:70] = True
    out = stamp_ridges(np.zeros((n, n)), zone, amplitude=1.0, profile="plateau")
    d = distance_transform_edt(zone)
    interior = zone & (d > 0.6 * d.max())
    vals = out[interior]
    assert vals.std() < 0.05                            # flat interior
    assert vals.mean() > 0.9                            # saturated near amplitude


def test_stamp_ridges_point_zone_is_gaussian_peak():
    n = 60
    zone = np.zeros((n, n), dtype=bool)
    zone[30, 30] = True
    out = stamp_ridges(np.zeros((n, n)), zone, amplitude=2.0)
    assert out[30, 30] == pytest.approx(out.max())
    assert out.max() > 1.5                              # clamped Gaussian summit


# --- river carving --------------------------------------------------------

def test_carve_rivers_enforces_monotonic_downhill():
    n = 40
    field = np.tile(np.arange(n, dtype="float64"), (n, 1))   # rises with column
    # source at low col (5), mouth at high col (30) → river "climbs" downstream
    polyline = [(10, 5), (10, 30)]
    out, warnings = carve_rivers(field, [polyline], mouths=[1])
    along = out[10, 5:31]
    assert np.all(np.diff(along) <= 1e-6)               # non-increasing bed
    assert warnings                                     # flagged the climb


def test_carve_rivers_downhill_case_no_warning():
    n = 40
    field = np.tile(np.arange(n, dtype="float64")[::-1], (n, 1))  # falls with col
    out, warnings = carve_rivers(field, [[(10, 5), (10, 30)]], mouths=[1])
    assert out[10, 5:31].max() <= field[10, 5:31].max() + 1e-9
    assert not warnings


def test_carve_lake_flattens_to_outlet():
    field = np.random.default_rng(0).random((30, 30)) + 1.0
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:15, 10:15] = True
    out, _ = carve_rivers(field, [], lakes=[{"mask": mask, "outlet": 0.5}])
    assert np.allclose(out[mask], 0.5)


# --- band-aware noise -----------------------------------------------------

def test_band_aware_noise_capped_below_band_fraction():
    n = 70
    field = coastal_ramp(_island(n), amplitude=10.0)
    layers, frac = 8, 0.4
    out = band_aware_noise(field, layers, amplitude_fraction=frac, seed=1)
    rng = field.max() - field.min()
    added = out - field
    assert np.max(np.abs(added)) <= frac * (rng / layers) + 1e-9


def test_band_aware_noise_is_deterministic():
    field = coastal_ramp(_island(60), amplitude=5.0)
    a = band_aware_noise(field, 8, seed=7)
    b = band_aware_noise(field, 8, seed=7)
    c = band_aware_noise(field, 8, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_band_aware_noise_no_op_on_flat_field():
    flat = np.ones((20, 20))
    assert np.array_equal(band_aware_noise(flat, 8), flat)


# --- normalize ------------------------------------------------------------

def test_normalize_range_and_spike_clipping():
    field = np.random.default_rng(1).random((50, 50))
    field[0, 0] = 1000.0                                # anomalous spire
    out = normalize(field, percentiles=(0.5, 99.5))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0, 0] == pytest.approx(1.0)              # spike clipped to top
    assert out.mean() > 0.1                             # bulk not crushed to ~0


# --- full stack + bundle --------------------------------------------------

def test_synthesize_end_to_end_is_finite_and_normalized():
    n = 80
    land = _island(n)
    zone = np.zeros((n, n), dtype=bool)
    zone[30:36, 25:55] = True
    rivers = [{"polyline": [(40, 40), (55, 55), (60, 60)], "mouth": 1}]
    field, warnings = synthesize(land, zones=[{"mask": zone, "class": "mountain",
                                               "amplitude": 1.5}],
                                 rivers=rivers, layers=10, seed=3)
    assert field.shape == (n, n)
    assert np.isfinite(field).all()
    assert field.min() >= 0.0 and field.max() <= 1.0


def test_synthesize_is_deterministic():
    land = _island(60)
    f1, _ = synthesize(land, layers=8, seed=5)
    f2, _ = synthesize(land, layers=8, seed=5)
    assert np.array_equal(f1, f2)


def test_synthesize_to_bundle_loads_and_bands(tmp_path):
    from toposlicr.bundle import FLAT_CRS, load_bundle
    from toposlicr.config import parse_config
    from toposlicr.layers import build_layer_model

    n = 90
    land = _island(n)
    zone = np.zeros((n, n), dtype=bool)
    zone[30:40, 30:60] = True
    bdir = synthesize_to_bundle(
        tmp_path / "world", land, zones=[{"mask": zone, "class": "mountain",
                                          "amplitude": 1.5}],
        layers=8, seed=2, units_per_pixel=100.0, source="test")
    b = load_bundle(bdir)
    assert b.to_dem().crs == FLAT_CRS
    assert (bdir / "preview_hillshade.png").is_file()

    cfg = parse_config({
        "region": {"bundle": str(bdir)},
        "physical": {"model_width_mm": 300, "normalize_layers": 8},
    })
    model = build_layer_model(b.to_dem(), cfg)
    assert model.flat
    assert 4 <= model.layer_count <= 10
