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


@main.command("serve")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True, type=int)
@click.option("--runs-dir", default="runs", show_default=True,
              help="Where pipeline runs write their output.")
@click.option("--open/--no-open", "open_browser", default=True,
              help="Open the GUI in a browser on start.")
def serve_cmd(host: str, port: int, runs_dir: str, open_browser: bool) -> None:
    """Launch the browser GUI (FastAPI) over the pipeline."""
    try:
        import uvicorn
    except ImportError as exc:
        raise click.ClickException(
            "web dependencies not installed — run: uv sync --extra web") from exc

    from .web.app import create_app

    app = create_app(runs_dir=runs_dir)
    url = f"http://{host}:{port}"
    click.secho(f"\n  toposlicr GUI → {url}\n", fg="green", bold=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
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
