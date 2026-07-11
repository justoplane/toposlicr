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

from .acrylic import apply_acrylic
from .boards import render_boards
from .config import Config, load_config
from .dem.base import DemRaster
from .dem.cache import resolve_dem
from .features.acquire import fetch_features
from .features.select import resolve_features
from .geo import BBox
from .guide import write_assembly_guide
from .layers import LayerModel, build_layer_model
from .nest import nest_parts
from .panelize import panelize_parts
from .parts import Board, Part, build_parts_from_model
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
    parts: list[Part] = field(default_factory=list)
    boards: list[Board] = field(default_factory=list)
    layer_svgs: list[Path] = field(default_factory=list)
    board_svgs: list[Path] = field(default_factory=list)
    guide_path: Path | None = None
    debug_artifacts: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_pipeline(cfg: Config, out_dir: str | Path, *, log: Logger = _noop,
                 dem_resolution_m: float = 30.0, api_key: str | None = None,
                 use_cache: bool = True, write_debug: bool = True,
                 fetch_symbology: bool = True, panelize: bool = True,
                 nest: bool = True, project_name: str = "toposlicr",
                 optimize: bool = True) -> PipelineResult:
    """Run the full pipeline end-to-end and write outputs to ``out_dir``.

    Stages: DEM → layer model → symbology → per-layer SVGs → parts (acrylic +
    panelization) → nested cut boards → assembly guide. Board nesting can be
    disabled to stop at per-layer SVGs.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bbox = cfg.region.bbox
    fictional = cfg.region.is_fictional

    # Stage 1 — data acquisition (real DEM, or a fictional terrain bundle).
    bundle = None
    if fictional:
        from .bundle import load_bundle
        log(f"[1/6] loading terrain bundle {cfg.region.bundle} …")
        bundle = load_bundle(cfg.region.bundle)
        dem: DemRaster = bundle.to_dem()
        log(f"      bundle {dem.shape[1]}×{dem.shape[0]} px "
            f"(flat, {bundle.meta.source or 'fictional'})")
    else:
        log(f"[1/6] fetching DEM ({cfg.region.dem}) for {bbox_str(bbox)} …")
        dem = resolve_dem(
            cfg.region.dem, bbox, dem_resolution_m,
            api_key=api_key, cache_dir=out.parent / "cache" / "dem",
            use_cache=use_cache)
        log(f"      DEM {dem.shape[1]}×{dem.shape[0]} px, "
            f"elevation {dem.elevation_range()[0]:.0f}–{dem.elevation_range()[1]:.0f} m")

    # Stages 2–3 — projection/scale + contouring into the nested layer model.
    log("[2/6] building layer model (project → scale → contour) …")
    model = build_layer_model(dem, cfg, bbox)
    log(f"      {model.layer_count} layers, interval {model.interval_m:.1f}, "
        f"scale {model.scale.scale_label()}")

    # Stage 4 — symbology (rivers, lakes, labels/icons) against each visible band.
    warnings = list(model.warnings)
    symbology: SymbologyResult | None = None
    if fictional:
        log("[3/6] building symbology from bundle features + water …")
        symbology = _bundle_symbology(model, bundle, cfg, warnings)
    elif fetch_symbology:
        log("[3/6] fetching OSM features + building symbology …")
        symbology = _run_symbology(model, cfg, bbox, out, log, use_cache, warnings)

    # Per-layer SVGs (cut + registration + symbology) — always emitted.
    log("[4/6] rendering per-layer SVGs …")
    layer_svgs = render_layer_svgs(model, cfg, out / "layers", symbology)

    # Stages 5–6 — parts (acrylic insets + panelization) → nested boards → guide.
    parts: list[Part] = []
    boards: list[Board] = []
    board_svgs: list[Path] = []
    guide_path: Path | None = None
    if nest:
        parts, boards, board_svgs, guide_path = _run_layout(
            model, symbology, cfg, out, log, warnings, project_name, panelize, optimize)

    debug: list[Path] = []
    if write_debug:
        debug.append(render_composite_preview(model, cfg, out / "preview.svg"))
        debug.extend(write_layer_geojson(model, out / "debug"))

    log(f"done → {out}")
    return PipelineResult(config=cfg, model=model, out_dir=out, symbology=symbology,
                          parts=parts, boards=boards, layer_svgs=layer_svgs,
                          board_svgs=board_svgs, guide_path=guide_path,
                          debug_artifacts=debug, warnings=warnings)


def _run_layout(model, symbology, cfg, out: Path, log: Logger, warnings: list[str],
                project_name: str, panelize: bool, optimize: bool):
    """Parts → acrylic → panelization → nesting → boards + guide."""
    log("[5/6] building parts (acrylic insets + panelization) …")
    parts = build_parts_from_model(model, symbology, cfg)
    # Acrylic runs BEFORE panelization so lake holes are cut into whole layers
    # and the emitted acrylic parts are themselves panelized if oversized —
    # otherwise a lake wider than the bed would reach nesting unsplit and abort.
    if symbology is not None:
        parts, acr_warnings = apply_acrylic(parts, model, symbology, cfg)
        warnings.extend(acr_warnings)
    if panelize:
        parts = panelize_parts(parts, model, cfg)
    log(f"      {len(parts)} parts "
        f"({sum(1 for p in parts if p.kind == 'acrylic')} acrylic)")

    log("[6/6] nesting parts onto cut boards …")
    boards = nest_parts(parts, cfg)
    board_svgs = render_boards(boards, cfg, out / "boards", project_name=project_name,
                              optimize=optimize)
    log(f"      {len(boards)} boards across "
        f"{len({b.material for b in boards})} material(s)")

    guide_path = write_assembly_guide(model, boards, cfg, out / "assembly_guide.html",
                                      project_name=project_name)
    return parts, boards, board_svgs, guide_path


def _bundle_symbology(model, bundle, cfg, warnings: list[str]) -> SymbologyResult:
    """Symbology for a fictional bundle: features.csv + water mask (no network).

    Water-mask polygons become LAKE features (so the acrylic-inset path works
    unchanged); named features (peaks/settlements/POIs, with optional icons)
    come from the bundle's features.csv. Everything is already in flat world
    units, so no reprojection happens.
    """
    from .bundle import FLAT_EPSG
    from .features.schema import FeatureType, GeoFeature

    coll = bundle.features()
    for poly in bundle.water_polygons():
        coll.add(GeoFeature(feature_type=FeatureType.LAKE, geometry=poly,
                            importance=poly.area))
    for line, name in bundle.rivers():
        coll.add(GeoFeature(feature_type=FeatureType.RIVER, geometry=line,
                            name=name, importance=5))
    coll.epsg = FLAT_EPSG
    symbology = build_symbology(model, coll, cfg)
    warnings.extend(symbology.warnings)
    return symbology


def _run_symbology(model, cfg, bbox, out: Path, log: Logger, use_cache: bool,
                   warnings: list[str]) -> SymbologyResult | None:
    """Fetch OSM features, apply the features.csv loop, and build symbology.

    Network/parse failures degrade gracefully: the run continues terrain-only
    with a warning, matching the 'proceed without stopping' design.
    """
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
