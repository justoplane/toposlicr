"""Project configuration schema and loader.

Maps the TOML sketch in plan Section 9 onto typed dataclasses. Phase 0 fully
models the ``[region]``, ``[physical]`` and ``[machine]`` blocks (everything the
scale math and future SVG output depend on) and keeps ``[materials]``,
``[symbology]`` and ``[panelization]`` as light, forgiving structures so later
phases can flesh them out without breaking existing configs.

Design notes:
* Loading is pure: ``load_config(path) -> Config``. No global state.
* Unknown keys are collected as warnings rather than hard errors, so a config
  written for a newer pipeline still loads on an older one (and vice versa).
* Validation that needs data we don't have yet (elevation range, precise UTM
  extent) is deferred to the stage that owns it.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .geo import BBox


class ConfigError(ValueError):
    """Raised when a config file is structurally invalid."""


# --- machine profile -------------------------------------------------------

# Default Glowforge-flavored operation→color map (plan Section 7.1). Colors live
# in config so nothing is hardcoded; this ships as the fallback.
_DEFAULT_COLORS = {
    "cut": "#000000",
    "score_registration": "#FF0000",
    "score_hydro": "#0000FF",
    "score_ids": "#00A0A0",
    "engrave_fill": "#333333",
}


@dataclass
class MachineProfile:
    profile: str = "glowforge"
    bed_mm: tuple[float, float] = (495.0, 279.0)
    margin_mm: float = 8.0
    kerf_mm: float = 0.15
    colors: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_COLORS))

    @property
    def usable_bed_mm(self) -> tuple[float, float]:
        """Bed dimensions minus the margin on all sides."""
        return (
            max(0.0, self.bed_mm[0] - 2 * self.margin_mm),
            max(0.0, self.bed_mm[1] - 2 * self.margin_mm),
        )


@dataclass
class RegionConfig:
    bbox: BBox
    dem: str = "auto"


@dataclass
class PhysicalConfig:
    model_width_mm: float
    ply_thickness_mm: float = 3.0
    # Exactly one of these three drives the scale solve (validated at load).
    exaggeration: float | None = 1.5
    interval_m: float | None = None
    layer_count: int | None = None
    base_datum: str | float = "auto"


@dataclass
class ContourConfig:
    """Geometry-cleanup knobs (plan Section 3). Lengths in model mm unless noted."""

    smoothing_px: float = 1.5           # Gaussian sigma, in DEM pixels
    simplify_tol_mm: float = 0.2        # Douglas-Peucker tolerance (below kerf)
    chaikin_iterations: int = 1         # corner-cutting passes
    min_feature_mm: float = 4.0         # drop plywood slivers/pinholes below this
    close_radius_mm: float = 0.0        # morphological close to fuse near-touching blobs
    ledge_margin_mm: float = 2.0        # min glue-ledge between stacked layers

    @property
    def min_area_mm2(self) -> float:
        return self.min_feature_mm ** 2


@dataclass
class PeaksConfig:
    min_prominence_m: float = 150.0    # applied where OSM/GNIS supplies prominence
    min_elevation_m: float | None = None
    label_elevation: bool = True
    max_count: int | None = None       # keep the N highest, None = no cap


@dataclass
class RiversConfig:
    min_stream_order: int = 3           # OSM waterway class → pseudo stream order
    widen_major: bool = False           # engrave major rivers as thin polygons


@dataclass
class LakesConfig:
    min_area_km2: float = 0.05
    mode: str = "score"                 # "score" | "inset"
    fit_gap_mm: float = 0.1             # acrylic press-fit gap (Phase 3)


@dataclass
class LabelsConfig:
    font: str = "DejaVu Sans"
    mode: str = "engrave_fill"          # "engrave_fill" | "score"
    cap_height_mm: float = 4.0
    append_elevation: bool = True


@dataclass
class SymbologyConfig:
    peaks: PeaksConfig = field(default_factory=PeaksConfig)
    rivers: RiversConfig = field(default_factory=RiversConfig)
    lakes: LakesConfig = field(default_factory=LakesConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    include_places: bool = True


@dataclass
class PanelizationConfig:
    seam_margin_mm: float = 4.0          # erode the hidden corridor so seams don't peek
    seam_joint: str = "butt"             # "butt" | "puzzle"
    tab_size_mm: float | None = None     # puzzle tab size; defaults to ~3× thickness


@dataclass
class Config:
    region: RegionConfig
    physical: PhysicalConfig
    machine: MachineProfile = field(default_factory=MachineProfile)
    contour: ContourConfig = field(default_factory=ContourConfig)
    symbology: SymbologyConfig = field(default_factory=SymbologyConfig)
    panelization: PanelizationConfig = field(default_factory=PanelizationConfig)
    # Materials stays a light dict (default/water/overrides map); resolved in parts.py.
    materials: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source_path: Path | None = None


_KNOWN_SECTIONS = {
    "region",
    "physical",
    "machine",
    "contour",
    "materials",
    "symbology",
    "panelization",
}


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"[{where}] is missing required key '{key}'")
    return section[key]


def _parse_machine(raw: dict[str, Any], warnings: list[str]) -> MachineProfile:
    mp = MachineProfile()
    if "profile" in raw:
        mp.profile = str(raw["profile"])
    if "bed_mm" in raw:
        bed = raw["bed_mm"]
        if not isinstance(bed, (list, tuple)) or len(bed) != 2:
            raise ConfigError("[machine] bed_mm must be a 2-element list [width, height]")
        mp.bed_mm = (float(bed[0]), float(bed[1]))
    if "margin_mm" in raw:
        mp.margin_mm = float(raw["margin_mm"])
    if "kerf_mm" in raw:
        mp.kerf_mm = float(raw["kerf_mm"])
    if "colors" in raw:
        colors = raw["colors"]
        if not isinstance(colors, dict):
            raise ConfigError("[machine.colors] must be a table of name = hex")
        merged = dict(_DEFAULT_COLORS)
        merged.update({str(k): str(v) for k, v in colors.items()})
        mp.colors = merged
    for key in raw:
        if key not in {"profile", "bed_mm", "margin_mm", "kerf_mm", "colors"}:
            warnings.append(f"[machine] unknown key '{key}' ignored")
    return mp


def _parse_physical(raw: dict[str, Any], warnings: list[str]) -> PhysicalConfig:
    model_width_mm = float(_require(raw, "model_width_mm", "physical"))
    phys = PhysicalConfig(model_width_mm=model_width_mm)
    if "ply_thickness_mm" in raw:
        phys.ply_thickness_mm = float(raw["ply_thickness_mm"])
    if "base_datum" in raw:
        bd = raw["base_datum"]
        phys.base_datum = bd if isinstance(bd, str) else float(bd)

    # Determine which scale driver was pinned. Default to exaggeration=1.5 only
    # when the user pinned none of the three.
    drivers = {k: raw[k] for k in ("exaggeration", "interval_m", "layer_count") if k in raw}
    if len(drivers) > 1:
        raise ConfigError(
            "[physical] pin only one of exaggeration / interval_m / layer_count; "
            f"got {sorted(drivers)}"
        )
    phys.exaggeration = None
    phys.interval_m = None
    phys.layer_count = None
    if not drivers:
        phys.exaggeration = 1.5
        warnings.append("[physical] no scale driver set; defaulting to exaggeration = 1.5")
    elif "exaggeration" in drivers:
        phys.exaggeration = float(drivers["exaggeration"])
    elif "interval_m" in drivers:
        phys.interval_m = float(drivers["interval_m"])
    else:
        phys.layer_count = int(drivers["layer_count"])

    known = {"model_width_mm", "ply_thickness_mm", "base_datum",
             "exaggeration", "interval_m", "layer_count"}
    for key in raw:
        if key not in known:
            warnings.append(f"[physical] unknown key '{key}' ignored")
    return phys


def _parse_contour(raw: dict[str, Any], warnings: list[str]) -> ContourConfig:
    cc = ContourConfig()
    fields = {
        "smoothing_px": float, "simplify_tol_mm": float, "chaikin_iterations": int,
        "min_feature_mm": float, "close_radius_mm": float, "ledge_margin_mm": float,
    }
    for key, val in raw.items():
        if key in fields:
            setattr(cc, key, fields[key](val))
        else:
            warnings.append(f"[contour] unknown key '{key}' ignored")
    return cc


def _fill_dataclass(obj: Any, raw: dict[str, Any], where: str,
                    warnings: list[str]) -> None:
    """Set known dataclass fields from a raw table; warn on unknown keys."""
    from dataclasses import fields as dc_fields

    known = {f.name: f.type for f in dc_fields(obj)}
    for key, val in raw.items():
        if key in known:
            setattr(obj, key, val)
        else:
            warnings.append(f"[{where}] unknown key '{key}' ignored")


def _parse_symbology(raw: dict[str, Any], warnings: list[str]) -> SymbologyConfig:
    sym = SymbologyConfig()
    subtables = {"peaks": sym.peaks, "rivers": sym.rivers,
                 "lakes": sym.lakes, "labels": sym.labels}
    for key, val in raw.items():
        if key in subtables:
            if not isinstance(val, dict):
                raise ConfigError(f"[symbology] '{key}' must be a table")
            _fill_dataclass(subtables[key], val, f"symbology.{key}", warnings)
        elif key == "include_places":
            sym.include_places = bool(val)
        else:
            warnings.append(f"[symbology] unknown key '{key}' ignored")
    return sym


def parse_config(data: dict[str, Any], source_path: Path | None = None) -> Config:
    """Build a ``Config`` from an already-parsed TOML mapping."""
    warnings: list[str] = []

    for key in data:
        if key not in _KNOWN_SECTIONS:
            warnings.append(f"unknown top-level section '[{key}]' ignored")

    if "region" not in data:
        raise ConfigError("config is missing the required [region] section")
    if "physical" not in data:
        raise ConfigError("config is missing the required [physical] section")

    region_raw = data["region"]
    bbox_raw = _require(region_raw, "bbox", "region")
    try:
        bbox = BBox.from_list(list(bbox_raw))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"[region] bbox invalid: {exc}") from exc
    region = RegionConfig(bbox=bbox, dem=str(region_raw.get("dem", "auto")))
    for key in region_raw:
        if key not in {"bbox", "dem"}:
            warnings.append(f"[region] unknown key '{key}' ignored")

    physical = _parse_physical(data["physical"], warnings)
    machine = _parse_machine(data.get("machine", {}), warnings)
    contour = _parse_contour(data.get("contour", {}), warnings)
    symbology = _parse_symbology(data.get("symbology", {}), warnings)
    panelization = PanelizationConfig()
    _fill_dataclass(panelization, data.get("panelization", {}), "panelization", warnings)

    return Config(
        region=region,
        physical=physical,
        machine=machine,
        contour=contour,
        symbology=symbology,
        panelization=panelization,
        materials=dict(data.get("materials", {})),
        warnings=warnings,
        source_path=source_path,
    )


def load_config(path: str | Path) -> Config:
    """Load and validate a TOML config file."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        with p.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: invalid TOML — {exc}") from exc
    return parse_config(data, source_path=p)
