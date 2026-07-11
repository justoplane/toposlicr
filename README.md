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

> **Status: end-to-end.** All seven pipeline stages are implemented — one
> command takes a bounding box to laser-ready nested cut boards plus an assembly
> guide. Irregular (no-fit-polygon) nesting is the one v2 item; v1 uses
> rectangular bin-packing (see [build phases](#build-phases)).

## Architecture

`toposlicr` is a **core library of stateless stage functions** with a thin CLI
wrapper (`toposlicr.cli`). Stage boundaries are clean and JSON-serializable, so
the same core can later back a FastAPI + browser front-end (a stated long-term
goal) without restructuring.

```
config.toml
   ▼
[1] Data acquisition (DEM + vector features)      → cached GeoTIFF / GeoJSON
[2] Projection + scale/exaggeration math          → layer elevation bands
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
# Full pipeline: bbox + config → nested laser-ready boards + assembly guide.
uv run toposlicr run examples/whitney.toml -o output --name Whitney

# Just the scale/exaggeration/layer-count report (no data fetch):
uv run toposlicr scale examples/whitney.toml --min-elev 2500 --max-elev 4421

# Validate a config; generate a press-fit calibration coupon:
uv run toposlicr validate examples/whitney.toml
uv run toposlicr coupon examples/whitney.toml -o coupon.svg
```

`run` writes, under the output directory:

```
output/
  layers/layer_NN.svg        one SVG per layer (cut + registration + symbology)
  boards/board_MAT_NN.svg    nested cut boards, grouped by material
  assembly_guide.html        exploded stack, board index, material + acrylic lists
  features.csv               editable feature selection (re-run to apply edits)
  labels.json                editable label positions (manual-nudge loop)
  preview.svg, debug/*.geojson   non-blocking debug artifacts
```

DEM source is chosen by `[region].dem`: `auto` (US → 3DEP, else COP30, with an
OpenTopography key), the keyless `terrarium` (AWS terrain tiles), `synthetic`
(offline), or an explicit dataset id. Set `OPENTOPOGRAPHY_API_KEY` to use
OpenTopography. Results are cached under `cache/`.

The scale relationship (pin any one of `exaggeration`, `interval_m` or
`layer_count` in `[physical]`; the rest is derived):

```
S = model_width_mm / (real_width_m × 1000)          # physical scale, 1:N
interval_real_m = ply_thickness_mm / (S × E × 1000) # elevation covered by one layer
```

Colors map to laser operations via the machine profile (`[machine.colors]`):
black cut, red registration score, blue hydro score, teal part-ID score, filled
grey label engraving — the Glowforge/color-as-operation convention.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
```

## Build phases

All phases implemented:

- **Phase 0 — Scale calculator + config schema.** ✅
- **Phase 1 — Core slicer:** DEM fetch → cleaned nested layer polygons →
  per-layer SVGs with cut outlines + registration scores. ✅
- **Phase 2 — Symbology:** rivers, lake outlines, filled-font labels,
  `features.csv` curation loop. ✅
- **Phase 3 — Acrylic insets** (flush press-fit) + calibration coupon. ✅
- **Phase 4 — Panelization** with hidden-seam guillotine routing. ✅
- **Phase 5 — Nesting + boards + part IDs** (rectpack v1). ✅
- **Phase 6 — Assembly guide + cut-path optimization.** ✅

**v2 / future:** true irregular (no-fit-polygon) nesting to cut sheet waste;
puzzle-tab seam joints; a web GUI over the same core library.
