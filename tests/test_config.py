import pytest

from toposlicr.config import ConfigError, load_config, parse_config


def _base(**overrides):
    data = {
        "region": {"bbox": [-118.35, 36.52, -118.20, 36.62], "dem": "USGS10m"},
        "physical": {"model_width_mm": 500, "ply_thickness_mm": 3.0, "exaggeration": 1.5},
        "machine": {"profile": "glowforge", "bed_mm": [495, 279], "kerf_mm": 0.15},
    }
    for section, patch in overrides.items():
        data.setdefault(section, {}).update(patch)
    return data


def test_parses_minimal_valid_config():
    cfg = parse_config(_base())
    assert cfg.region.bbox.west == -118.35
    assert cfg.physical.model_width_mm == 500
    assert cfg.physical.exaggeration == 1.5
    assert cfg.machine.profile == "glowforge"
    assert cfg.machine.colors["cut"] == "#000000"


def test_missing_region_raises():
    data = _base()
    del data["region"]
    with pytest.raises(ConfigError, match="region"):
        parse_config(data)


def test_missing_model_width_raises():
    data = _base()
    del data["physical"]["model_width_mm"]
    with pytest.raises(ConfigError, match="model_width_mm"):
        parse_config(data)


def test_multiple_scale_drivers_raise():
    data = _base()
    data["physical"] = {"model_width_mm": 500, "exaggeration": 1.5, "interval_m": 100}
    with pytest.raises(ConfigError, match="only one"):
        parse_config(data)


def test_no_driver_defaults_to_exaggeration_with_warning():
    data = _base()
    data["physical"] = {"model_width_mm": 500}
    cfg = parse_config(data)
    assert cfg.physical.exaggeration == 1.5
    assert any("scale driver" in w for w in cfg.warnings)


def test_interval_driver_clears_exaggeration():
    data = _base()
    data["physical"] = {"model_width_mm": 500, "interval_m": 100}
    cfg = parse_config(data)
    assert cfg.physical.interval_m == 100
    assert cfg.physical.exaggeration is None


def test_colors_merge_over_defaults():
    data = _base()
    data["machine"]["colors"] = {"cut": "#111111", "custom_op": "#abcdef"}
    cfg = parse_config(data)
    assert cfg.machine.colors["cut"] == "#111111"
    assert cfg.machine.colors["score_registration"] == "#FF0000"  # default kept
    assert cfg.machine.colors["custom_op"] == "#abcdef"


def test_unknown_keys_warn_but_load():
    data = _base()
    data["physical"]["bogus"] = 1
    data["nonsense"] = {"x": 1}
    cfg = parse_config(data)
    assert any("bogus" in w for w in cfg.warnings)
    assert any("nonsense" in w for w in cfg.warnings)


def test_usable_bed_accounts_for_margin():
    cfg = parse_config(_base(machine={"margin_mm": 8}))
    w, h = cfg.machine.usable_bed_mm
    assert w == 495 - 16
    assert h == 279 - 16


def test_bad_bbox_raises():
    data = _base()
    data["region"]["bbox"] = [10, 20, 5, 25]  # east < west
    with pytest.raises(ConfigError, match="bbox"):
        parse_config(data)


def test_load_config_from_file(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        "[region]\nbbox = [-118.35, 36.52, -118.20, 36.62]\n"
        "[physical]\nmodel_width_mm = 500\nexaggeration = 1.5\n"
    )
    cfg = load_config(p)
    assert cfg.source_path == p
    assert cfg.physical.model_width_mm == 500
