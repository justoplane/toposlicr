import math

import pytest

from toposlicr.geo import BBox
from toposlicr.scale import (
    _NICE_INTERVALS_M,
    SolveMode,
    exaggeration_from_interval,
    scale_denominator,
    snap_interval,
    solve_scale,
)


def test_scale_denominator_matches_plan_example():
    # Plan Section 2: 40 km extent at 500 mm wide -> 1:80,000.
    n = scale_denominator(model_width_mm=500, real_width_m=40_000)
    assert n == pytest.approx(80_000)


def test_snap_interval_picks_nearest_nice_on_log_scale():
    assert snap_interval(60) == 50
    assert snap_interval(90) == 100
    assert snap_interval(11) == 10
    assert snap_interval(240) == 250


def test_interval_mode_derives_exaggeration_exactly():
    # S = 1/80000, thickness 3mm, interval 100m -> E = t/(S*interval).
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        interval_m=100,
    )
    assert res.mode is SolveMode.INTERVAL
    # E = thickness_mm / (S * interval_m * 1000); mm→m factor included.
    expected_e = 3.0 / ((1 / 80_000) * 100 * 1000)
    assert res.exaggeration == pytest.approx(expected_e)
    assert res.exaggeration == pytest.approx(2.4)
    assert res.interval_m == 100  # interval honored, no snap


def test_units_true_scale_matches_plan_example():
    # Plan Section 2: 40 km @ 500 mm (1:80000), 3 mm ply, E=1 → 240 m interval,
    # and one 240 m band scaled down is exactly the 3 mm ply thickness.
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        exaggeration=1.0,
        snap=False,
    )
    assert res.interval_m == pytest.approx(240.0)
    assert res.interval_m * res.scale_ratio * 1000 == pytest.approx(3.0)  # → mm


def test_exaggeration_mode_snaps_interval_and_reports_actual_e():
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        exaggeration=2.0,
    )
    assert res.mode is SolveMode.EXAGGERATION
    assert res.requested_exaggeration == 2.0
    # Interval was snapped, so actual E is recomputed from the snapped interval.
    recomputed = exaggeration_from_interval(
        res.ply_thickness_mm, res.scale_ratio, res.interval_m
    )
    assert res.exaggeration == pytest.approx(recomputed)
    assert res.interval_m in _NICE_INTERVALS_M


def test_layer_count_mode_requires_elevation_range():
    with pytest.raises(ValueError, match="requires base_elev_m"):
        solve_scale(
            model_width_mm=500,
            real_width_m=40_000,
            ply_thickness_mm=3.0,
            layer_count=10,
        )


def test_layer_count_mode_solves_interval_and_reports_count():
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        layer_count=10,
        base_elev_m=1000,
        max_elev_m=3000,
    )
    assert res.mode is SolveMode.LAYER_COUNT
    # 2000 m over 10 layers = 200 m raw, which is already a nice value.
    assert res.interval_m == 200
    assert res.layer_count == 10


def test_requires_exactly_one_driver():
    with pytest.raises(ValueError, match="exactly one"):
        solve_scale(
            model_width_mm=500,
            real_width_m=40_000,
            ply_thickness_mm=3.0,
            exaggeration=1.5,
            interval_m=100,
        )
    with pytest.raises(ValueError, match="exactly one"):
        solve_scale(model_width_mm=500, real_width_m=40_000, ply_thickness_mm=3.0)


def test_too_many_layers_warns():
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        interval_m=50,
        base_elev_m=0,
        max_elev_m=4000,  # 80 layers
    )
    assert res.layer_count == 80
    assert any("practical limit" in w for w in res.warnings)


def test_snapping_large_e_drift_warns():
    # Choose an exaggeration whose ideal interval falls between nice values.
    res = solve_scale(
        model_width_mm=500,
        real_width_m=40_000,
        ply_thickness_mm=3.0,
        exaggeration=3.1,
    )
    # If the snap moved E by >15%, a warning is present; otherwise it isn't.
    drift = abs(res.exaggeration - 3.1) / 3.1
    assert (drift > 0.15) == any("exaggeration" in w for w in res.warnings)


def test_bbox_extent_reasonable_for_known_box():
    # ~0.15° lon at 36.5°N should be ~13-14 km wide.
    bbox = BBox.from_list([-118.35, 36.52, -118.20, 36.62])
    width_m, height_m = bbox.extent_m()
    assert 12_000 < width_m < 15_000
    assert 10_000 < height_m < 12_000
    assert math.isfinite(width_m)
