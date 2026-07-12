"""Command-line interface — a thin wrapper over the core library.

Every command loads a project config and calls stateless functions in the
``toposlicr`` package. The same functions back a future FastAPI/web front-end,
so no pipeline logic lives here — only argument parsing and presentation.
"""

from __future__ import annotations

import click

from . import __version__
from .config import Config, ConfigError, load_config
from .scale import solve_scale


@click.group()
@click.version_option(__version__, prog_name="toposlicr")
def main() -> None:
    """toposlicr — laser-cut topographic map pipeline."""


def _load(config_path: str) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("scale")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--min-elev", type=float, default=None,
              help="Base elevation (m). Provisional until the DEM is fetched in Phase 1.")
@click.option("--max-elev", type=float, default=None,
              help="Summit elevation (m). Enables the layer-count report.")
@click.option("--no-snap", is_flag=True,
              help="Do not snap the derived contour interval to a round value.")
def scale_cmd(config_path: str, min_elev: float | None, max_elev: float | None,
              no_snap: bool) -> None:
    """Report scale, exaggeration, contour interval and layer count for a config."""
    cfg = _load(config_path)
    for w in cfg.warnings:
        click.secho(f"  config: {w}", fg="yellow")

    width_m, height_m = cfg.region.bbox.extent_m()

    result = solve_scale(
        model_width_mm=cfg.physical.model_width_mm,
        real_width_m=width_m,
        ply_thickness_mm=cfg.physical.ply_thickness_mm,
        exaggeration=cfg.physical.exaggeration,
        interval_m=cfg.physical.interval_m,
        layer_count=cfg.physical.layer_count,
        base_elev_m=min_elev,
        max_elev_m=max_elev,
        snap=not no_snap,
    )

    model_height_mm = cfg.physical.model_width_mm * (height_m / width_m)

    click.secho("\nScale report", bold=True)
    click.echo(f"  solve mode        {result.mode.value}")
    click.echo(f"  real extent       {width_m / 1000:.2f} × {height_m / 1000:.2f} km")
    click.echo(
        f"  model size        {cfg.physical.model_width_mm:.0f} × "
        f"{model_height_mm:.0f} mm"
    )
    click.echo(f"  scale             {result.scale_label()}")
    click.echo(f"  ply thickness     {result.ply_thickness_mm:.1f} mm")
    click.echo(f"  contour interval  {result.interval_m:.0f} m")
    if result.requested_exaggeration is not None and abs(
        result.exaggeration - result.requested_exaggeration
    ) > 1e-9:
        click.echo(
            f"  exaggeration      {result.exaggeration:.2f}× "
            f"(requested {result.requested_exaggeration:.2f}×)"
        )
    else:
        click.echo(f"  exaggeration      {result.exaggeration:.2f}×")

    if result.layer_count is not None:
        click.echo(f"  layers            {result.layer_count}")
    else:
        click.secho(
            "  layers            n/a — pass --min-elev/--max-elev "
            "(auto-detected from the DEM in Phase 1)",
            fg="cyan",
        )

    for w in result.warnings:
        click.secho(f"  ⚠ {w}", fg="yellow")
    click.echo()


@main.command("run")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", type=click.Path(file_okay=False),
              default="output", show_default=True, help="Output directory.")
@click.option("--dem-resolution", type=float, default=30.0, show_default=True,
              help="Target DEM resolution in meters.")
@click.option("--no-cache", is_flag=True, help="Bypass the on-disk DEM cache.")
@click.option("--no-debug", is_flag=True, help="Skip preview + GeoJSON debug artifacts.")
@click.option("--no-symbology", is_flag=True,
              help="Skip OSM features (rivers/lakes/labels); terrain only.")
@click.option("--no-nest", is_flag=True,
              help="Stop at per-layer SVGs; skip parts/boards/guide.")
@click.option("--no-panelize", is_flag=True,
              help="Do not split oversized layers into bed-sized parts.")
@click.option("--name", "project_name", default=None,
              help="Project name for headers/guide (defaults to config filename).")
def run_cmd(config_path: str, out_dir: str, dem_resolution: float,
            no_cache: bool, no_debug: bool, no_symbology: bool, no_nest: bool,
            no_panelize: bool, project_name: str | None) -> None:
    """Run the full pipeline: bbox + config → laser-ready nested cut boards."""
    import os
    from pathlib import Path

    from .pipeline import run_pipeline

    cfg = _load(config_path)
    for w in cfg.warnings:
        click.secho(f"  config: {w}", fg="yellow")

    try:
        result = run_pipeline(
            cfg, out_dir,
            log=lambda m: click.secho(m, fg="cyan"),
            dem_resolution_m=dem_resolution,
            api_key=os.environ.get("OPENTOPOGRAPHY_API_KEY"),
            use_cache=not no_cache,
            write_debug=not no_debug,
            fetch_symbology=not no_symbology,
            nest=not no_nest,
            panelize=not no_panelize,
            project_name=project_name or Path(config_path).stem,
        )
    except Exception as exc:  # surface pipeline failures cleanly
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc

    for w in result.warnings:
        click.secho(f"  ⚠ {w}", fg="yellow")
    click.secho(f"\n✓ {len(result.layer_svgs)} layer SVG(s) → {result.out_dir}/layers",
                fg="green")
    if result.boards:
        click.secho(f"✓ {len(result.boards)} cut board(s) → {result.out_dir}/boards",
                    fg="green")
    if result.guide_path:
        click.secho(f"✓ assembly guide → {result.guide_path}", fg="green")


@main.command("coupon")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_path", type=click.Path(dir_okay=False),
              default="coupon.svg", show_default=True)
@click.option("--start", type=float, default=-0.05, show_default=True,
              help="First press-fit offset (mm).")
@click.option("--stop", type=float, default=0.15, show_default=True,
              help="Last press-fit offset (mm).")
@click.option("--step", type=float, default=0.02, show_default=True)
def coupon_cmd(config_path: str, out_path: str, start: float, stop: float,
               step: float) -> None:
    """Generate a press-fit calibration coupon SVG for this machine/material."""
    from .coupon import offset_series, write_coupon

    cfg = _load(config_path)
    offsets = offset_series(start, stop, step)
    path = write_coupon(cfg, out_path, offsets=offsets)
    click.secho(f"✓ coupon with {len(offsets)} offsets → {path}", fg="green")


@main.group("adapt")
def adapt() -> None:
    """Convert a fictional source into a terrain bundle (needs the 'fictional' extra)."""


def _adapter_error(exc: ImportError) -> click.ClickException:
    return click.ClickException(
        f"missing dependency for this adapter ({exc}); install with: "
        "uv sync --extra fictional")


@adapt.command("mesh")
@click.argument("mesh_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", default="world.terrainbundle", show_default=True)
@click.option("--cells-across", default=1000, show_default=True, type=int)
@click.option("--up-axis", default="auto", show_default=True)
@click.option("--pedestal-clip", default="auto", show_default=True)
@click.option("--supersample", default=2, show_default=True, type=int)
@click.option("--no-water", is_flag=True, help="Skip flat-region water candidates.")
def adapt_mesh(mesh_path, out_dir, cells_across, up_axis, pedestal_clip,
               supersample, no_water) -> None:
    """Tier 2: 3D mesh (STL/OBJ/glTF) → heightfield bundle."""
    try:
        from .adapters.mesh import mesh_to_bundle
    except ImportError as exc:
        raise _adapter_error(exc) from exc
    path = mesh_to_bundle(mesh_path, out_dir, up_axis=up_axis,
                          pedestal_clip=pedestal_clip, cells_across=cells_across,
                          supersample=supersample, water_candidates=not no_water)
    click.secho(f"✓ bundle → {path}", fg="green")


@adapt.command("art")
@click.argument("art_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", default="world.terrainbundle", show_default=True)
@click.option("--class-overlay", type=click.Path(exists=True), default=None,
              help="Painted terrain-class overlay PNG (red=mtn, orange=hill, yellow=plateau).")
@click.option("--adjust", "adjustment_layer", type=click.Path(exists=True), default=None,
              help="Mid-gray height-adjustment layer PNG.")
@click.option("--segmentation", "water_segmentation", default="auto", show_default=True)
@click.option("--cells-across", default=1000, show_default=True, type=int)
@click.option("--layers", "normalize_layers", default=12, show_default=True, type=int)
@click.option("--ocr", is_flag=True, help="Lift place names via OCR (needs OCR extra).")
def adapt_art(art_path, out_dir, class_overlay, adjustment_layer, water_segmentation,
              cells_across, normalize_layers, ocr) -> None:
    """Tier 3: 2D map artwork → synthesized heightfield bundle."""
    try:
        from .adapters.art import art_to_bundle
    except ImportError as exc:
        raise _adapter_error(exc) from exc
    path = art_to_bundle(art_path, out_dir, class_overlay=class_overlay,
                         adjustment_layer=adjustment_layer,
                         water_segmentation=water_segmentation,
                         cells_across=cells_across, normalize_layers=normalize_layers,
                         ocr=ocr)
    click.secho(f"✓ bundle → {path}", fg="green")


@adapt.command("azgaar")
@click.argument("cells_geojson", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", "out_dir", default="world.terrainbundle", show_default=True)
@click.option("--rivers", "rivers_geojson", type=click.Path(exists=True), default=None)
@click.option("--burgs", type=click.Path(exists=True), default=None)
@click.option("--cells-across", default=1000, show_default=True, type=int)
@click.option("--water-threshold", default=20, show_default=True, type=int)
def adapt_azgaar(cells_geojson, out_dir, rivers_geojson, burgs, cells_across,
                 water_threshold) -> None:
    """Azgaar FMG cell export (heights + settlements) → bundle."""
    try:
        from .adapters.azgaar import azgaar_to_bundle
    except ImportError as exc:
        raise _adapter_error(exc) from exc
    path = azgaar_to_bundle(cells_geojson, out_dir, rivers_geojson=rivers_geojson,
                            burgs=burgs, cells_across=cells_across,
                            water_threshold=water_threshold)
    click.secho(f"✓ bundle → {path}", fg="green")


@adapt.command("botw")
@click.option("-o", "--out", "out_dir", default="world.terrainbundle", show_default=True)
@click.option("--heightmap", "heightmap_png", type=click.Path(exists=True), default=None,
              help="Pre-extracted 16-bit heightmap PNG (BotWHeightMapConverter output).")
@click.option("--terrain-dir", type=click.Path(exists=True), default=None,
              help="Folder of raw .hght tiles (partial support; prefer --heightmap).")
@click.option("--objmap", "objmap_geojson", type=click.Path(exists=True), default=None,
              help="ZeldaMods objmap GeoJSON of named locations.")
@click.option("--water", "water_png", type=click.Path(exists=True), default=None)
@click.option("--units-per-pixel", default=1.0, show_default=True, type=float)
def adapt_botw(out_dir, heightmap_png, terrain_dir, objmap_geojson, water_png,
               units_per_pixel) -> None:
    """Tier 1: Breath of the Wild (user-supplied files) → bundle."""
    try:
        from .adapters.botw import botw_to_bundle
    except ImportError as exc:
        raise _adapter_error(exc) from exc
    path = botw_to_bundle(out_dir, heightmap_png=heightmap_png, terrain_dir=terrain_dir,
                          objmap_geojson=objmap_geojson, water_png=water_png,
                          units_per_pixel=units_per_pixel)
    click.secho(f"✓ bundle → {path}", fg="green")


def _is_wsl() -> bool:
    """True when running under WSL (where a Windows browser can't reach 127.0.0.1)."""
    try:
        with open("/proc/version") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def _lan_ip() -> str | None:
    """Best-effort primary IP address (e.g. the WSL interface)."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


@main.command("serve")
@click.option("--host", default=None,
              help="Bind address (default: 0.0.0.0 under WSL, else 127.0.0.1).")
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--runs-dir", default="runs", show_default=True,
              help="Where pipeline runs write their output.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the GUI in a browser on start.")
def serve_cmd(host: str | None, port: int, runs_dir: str, open_browser: bool) -> None:
    """Launch the browser GUI (FastAPI) over the pipeline."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "web dependencies not installed — run: uv sync --extra web") from exc

    from .web.app import create_app

    wsl = _is_wsl()
    # Under WSL, 127.0.0.1 inside Linux isn't reachable from a Windows browser,
    # so bind all interfaces by default and print the reachable URLs.
    if host is None:
        host = "0.0.0.0" if wsl else "127.0.0.1"

    app = create_app(runs_dir=runs_dir)
    click.secho("\n  toposlicr GUI running — open one of:", fg="green", bold=True)
    if host in ("0.0.0.0", "::"):
        click.secho(f"    http://localhost:{port}", fg="green")
        ip = _lan_ip()
        if ip:
            click.secho(f"    http://{ip}:{port}"
                        + ("   (use this from Windows if localhost fails)" if wsl else ""),
                        fg="green")
    else:
        click.secho(f"    http://{host}:{port}", fg="green")
    click.echo()

    if open_browser and not wsl:   # auto-open can't reach a Windows browser from WSL
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


@main.command("validate")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def validate_cmd(config_path: str) -> None:
    """Load a config and report any problems without running the pipeline."""
    cfg = _load(config_path)
    if cfg.warnings:
        for w in cfg.warnings:
            click.secho(f"  ⚠ {w}", fg="yellow")
    else:
        click.secho("  no warnings", fg="green")
    click.secho(f"✓ {config_path} is valid", fg="green")


if __name__ == "__main__":  # pragma: no cover
    main()
