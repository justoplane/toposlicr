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
from .geo import BBox
from .layers import LayerModel, build_layer_model
from .render import render_composite_preview, render_layer_svgs, write_layer_geojson

Logger = Callable[[str], None]


def _noop(_msg: str) -> None:
    pass


@dataclass
class PipelineResult:
    """Everything a run produced, for inspection or a report."""

    config: Config
    model: LayerModel
    out_dir: Path
    layer_svgs: list[Path] = field(default_factory=list)
    debug_artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(cfg: Config, out_dir: str | Path, *, log: Logger = _noop,
                 dem_resolution_m: float = 30.0, api_key: str | None = None,
                 use_cache: bool = True, write_debug: bool = True) -> PipelineResult:
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
    log("[2/3] building layer model (project → scale → contour) …")
    model = build_layer_model(dem, cfg, bbox)
    log(f"      {model.layer_count} layers, interval {model.interval_m:.0f} m, "
        f"scale {model.scale.scale_label()}, exaggeration {model.scale.exaggeration:.2f}×")

    # Output — per-layer SVGs (cut + registration score).
    log("[3/3] rendering per-layer SVGs …")
    layer_svgs = render_layer_svgs(model, cfg, out / "layers")

    debug: list[Path] = []
    if write_debug:
        debug.append(render_composite_preview(model, cfg, out / "preview.svg"))
        debug.extend(write_layer_geojson(model, out / "debug"))

    log(f"done → {out}")
    return PipelineResult(config=cfg, model=model, out_dir=out,
                          layer_svgs=layer_svgs, debug_artifacts=debug,
                          warnings=list(model.warnings))


def run_from_config_path(config_path: str | Path, out_dir: str | Path,
                         **kwargs) -> PipelineResult:
    return run_pipeline(load_config(config_path), out_dir, **kwargs)


def bbox_str(bbox: BBox) -> str:
    return f"[{bbox.west:.3f},{bbox.south:.3f},{bbox.east:.3f},{bbox.north:.3f}]"
