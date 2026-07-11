"""End-to-end pipeline orchestration.

Wires the stage functions together into a single run. Stages are added here as
each phase lands; the run is fully end-to-end by default, writing outputs (and
non-blocking debug artifacts) under an output directory. Progress is reported
through an optional ``log`` callback so the CLI and a future web front-end can
present it however they like.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, load_config
from .dem.base import DemRaster
from .dem.cache import resolve_dem
from .features.acquire import fetch_features
from .features.select import resolve_features
from .geo import BBox
from .layers import LayerModel, build_layer_model
from .render import render_composite_preview, render_layer_svgs, write_layer_geojson
from .symbology import SymbologyResult, build_symbology

Logger = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


@dataclass
class PipelineResult:
    """Everything a run produced, for inspection or a report."""

    config: Config
    model: LayerModel
    out_dir: Path
    symbology: SymbologyResult | None = None
    layer_svgs: list[Path] = field(default_factory=list)
    debug_artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(cfg: Config, out_dir: str | Path, *, log: Logger = _noop,
                 dem_resolution_m: float = 30.0, api_key: str | None = None,
                 use_cache: bool = True, write_debug: bool = True,
                 fetch_symbology: bool = True) -> PipelineResult:
    """Run the implemented pipeline stages and write outputs to ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bbox = cfg.region.bbox

    # Stage 1 — data acquisition.
    log(f"[1/3] fetching DEM ({cfg.region.dem}) for {bbox_str(bbox)} …")
    dem: DemRaster = resolve_dem(
        cfg.region.dem, bbox, dem_resolution_m,
        api_key=api_key, cache_dir=out.parent / "cache" / "dem", use_cache=use_cache,
    )
    log(f"      DEM {dem.shape[1]}×{dem.shape[0]} px, "
        f"elevation {dem.elevation_range()[0]:.0f}–{dem.elevation_range()[1]:.0f} m")

    # Stages 2–3 — projection/scale + contouring into the nested layer model.
    log("[2/4] building layer model (project → scale → contour) …")
    model = build_layer_model(dem, cfg, bbox)
    log(f"      {model.layer_count} layers, interval {model.interval_m:.0f} m, "
        f"scale {model.scale.scale_label()}, exaggeration {model.scale.exaggeration:.2f}×")

    # Stage 4 — symbology (rivers, lakes, labels) against each visible band.
    warnings = list(model.warnings)
    symbology: SymbologyResult | None = None
    if fetch_symbology:
        symbology = _run_symbology(model, cfg, bbox, out, log, use_cache, warnings)

    # Output — per-layer SVGs (cut + registration + symbology).
    log("[4/4] rendering per-layer SVGs …")
    layer_svgs = render_layer_svgs(model, cfg, out / "layers", symbology)

    debug: list[Path] = []
    if write_debug:
        debug.append(render_composite_preview(model, cfg, out / "preview.svg"))
        debug.extend(write_layer_geojson(model, out / "debug"))

    log(f"done → {out}")
    return PipelineResult(config=cfg, model=model, out_dir=out, symbology=symbology,
                          layer_svgs=layer_svgs, debug_artifacts=debug,
                          warnings=warnings)


def _run_symbology(model, cfg, bbox, out: Path, log: Logger, use_cache: bool,
                   warnings: list[str]) -> SymbologyResult | None:
    """Fetch OSM features, apply the features.csv loop, and build symbology.

    Network/parse failures degrade gracefully: the run continues terrain-only
    with a warning, matching the 'proceed without stopping' design.
    """
    log("[3/4] fetching OSM features + building symbology …")
    try:
        raw = fetch_features(bbox, cache_dir=out.parent / "cache" / "features",
                             use_cache=use_cache)
    except Exception as exc:
        warnings.append(f"feature fetch failed ({type(exc).__name__}: {exc}); "
                        "continuing terrain-only")
        return None
    kept, csv_existed = resolve_features(raw, cfg.symbology, out / "features.csv")
    log(f"      {len(raw)} features fetched, {len(kept)} selected"
        + ("" if csv_existed else " (wrote features.csv)"))

    # Manual label-nudge loop: read overrides if present, write positions after.
    from .labels import read_label_overrides, write_labels_file
    labels_path = out / "labels.json"
    overrides = read_label_overrides(labels_path) if labels_path.is_file() else {}
    symbology = build_symbology(model, kept, cfg, label_overrides=overrides)
    if not labels_path.is_file():
        write_labels_file(symbology.labels, labels_path)
    warnings.extend(symbology.warnings)
    n_labels = len([p for p in symbology.labels.placed if p.placed])
    log(f"      {n_labels} labels placed, "
        f"{len(symbology.labels.unplaced)} unplaced, "
        f"engrave area {symbology.labels.engrave_area_mm2():.0f} mm²")
    return symbology


def run_from_config_path(config_path: str | Path, out_dir: str | Path,
                         **kwargs) -> PipelineResult:
    return run_pipeline(load_config(config_path), out_dir, **kwargs)


def bbox_str(bbox: BBox) -> str:
    return f"[{bbox.west:.3f},{bbox.south:.3f},{bbox.east:.3f},{bbox.north:.3f}]"
