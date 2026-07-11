# toposlicr

Automated pipeline that turns a geographic bounding box plus a handful of
physical parameters into laser-ready, layered topographic-map cut boards and an
assembly guide — replacing the manual *3D model → slicer → Inkscape tracing →
hand layout* workflow.

The core idea: **skip the 3D model and slicer**. Contour the Digital Elevation
Model (DEM) raster directly into polygons, and pull all symbology (rivers,
lakes, peaks, names) from georeferenced sources in the same coordinate system,
so matching a symbol to its place on a slice is a coordinate transform instead
of manual tracing. Everything downstream is 2D polygon math.

> **Status: Phase 0.** Scale/exaggeration math and the config schema are in
> place. DEM acquisition, contouring, symbology, panelization, nesting and SVG
> output land in later phases (see [the plan](#build-phases)).

## Architecture

`toposlicr` is a **core library of stateless stage functions** with a thin CLI
wrapper (`toposlicr.cli`). Stage boundaries are clean and JSON-serializable, so
the same core can later back a FastAPI + browser front-end (a stated long-term
goal) without restructuring.

```
config.toml
   ▼
[1] Data acquisition (DEM + vector features)      → cached GeoTIFF / GeoJSON
[2] Projection + scale/exaggeration math          → layer elevation bands   ← Phase 0 (here)
[3] Contour extraction + geometry cleanup         → per-layer polygons
[4] Symbology (water, rivers, labels, scores)
[5] Panelization (split oversized layers)
[6] Nesting into cut boards (+ part IDs)
[7] SVG output + assembly guide
```

## Install

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev      # create venv + install toposlicr and dev tools
```

## Usage

```bash
# Report scale, exaggeration, contour interval and layer count for a project:
uv run toposlicr scale examples/whitney.toml --min-elev 2500 --max-elev 4421

# Validate a config without running anything:
uv run toposlicr validate examples/whitney.toml
```

The scale command solves the relationship between physical scale, vertical
exaggeration and ply thickness three ways — pin any one of `exaggeration`,
`interval_m` or `layer_count` in `[physical]` and the rest is derived:

```
S = model_width_mm / (real_width_m × 1000)     # physical scale, 1:N
interval_real_m = ply_thickness_mm / (S × E)   # elevation covered by one layer
```

Elevation range (`--min-elev` / `--max-elev`) is provided manually for now; it
is auto-detected from the DEM once Phase 1 lands.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```

## Build phases

Each phase is independently useful:

- **Phase 0 — Scale calculator + config schema.** ← current
- **Phase 1 — Core slicer:** DEM fetch → cleaned layer polygons → per-layer SVGs
  with cut outlines + registration scores.
- **Phase 2 — Symbology:** rivers, lake outlines, labels, `features.csv` loop.
- **Phase 3 — Acrylic insets** + kerf-calibration coupon generator.
- **Phase 4 — Panelization** with hidden-seam routing.
- **Phase 5 — Nesting + boards + part IDs.**
- **Phase 6 — Assembly guide + cut-path optimization.**
