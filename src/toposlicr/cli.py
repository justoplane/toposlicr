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
