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
[2] Projection (box-centered TM) + scale math     → layer elevation bands
[3] Contour extraction + geometry cleanup         → per-layer polygons
[4] Symbology (water, rivers, labels, scores)
[5] Panelization (split oversized layers)
[6] Nesting into cut boards (+ part IDs)
[7] SVG output + assembly guide
```

## Install

Uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev            # core + dev tools
uv sync --extra web --extra dev  # also install the browser GUI deps
```

## Browser GUI

```bash
uv run toposlicr serve          # opens http://127.0.0.1:8000 in your browser
```

A single-page app over the same pipeline core: draw the bounding box on a map
(or type W/S/E/N), fill the guided form (with an advanced raw-TOML panel), watch
a live scale/interval/layer estimate as you edit, then **Run** to stream progress
and explore the result in the **stack viewer**: the layers drawn on top of each
other with shading, a slider/play button to build the map up one ply at a time,
hover to identify a layer, click one to open its cut file, zoom and pan, toggle
water/trails/names/seams, and export a PNG. Boards and Downloads tabs hold the
nested cut boards, the assembly guide and per-file/zip downloads. A legend under
each viewer explains every color and line style on stage. The FastAPI backend is a thin wrapper: it calls the exact same
stateless functions the CLI does.

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

**Projection.** The DEM and all features are projected into a transverse
Mercator centered on the bounding box, not the nearest UTM zone. UTM's grid
north only points straight up on the zone's central meridian; elsewhere the grid
is rotated by the convergence angle (≈ Δλ·sin φ — 0.8° for the Whitney example,
up to ~2° near a zone edge), which turns the lon/lat box into a tilted
quadrilateral. Centering the projection on the box keeps north up, and every
layer is clipped to the rectangle inscribed in the projected box so the cut
frame is a true rectangle with sharp corners.

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
black cut, red registration score, blue hydro score, brown **dashed trail**
score, teal part-ID score, filled grey label engraving — the Glowforge/
color-as-operation convention.

**Trails** (`[symbology].trails`): hiking paths and named routes from OSM
(`highway=path/footway/bridleway` + `route=hiking` relations), scored as dashed
lines with the route names engraved along the path. Filter by walking-network
grade (`min_network`) and `include_unnamed`; individual trails can be toggled in
the GUI or the `features.csv` loop. Fictional maps carry trails in a bundle
`trails.geojson`.

## Fictional maps

Cut maps of fictional worlds — games, 3D models, or fantasy artwork — not just
real terrain. Every source converges on one neutral **terrain bundle** that the
core consumes exactly like a DEM (flat coordinates, no CRS; vertical scale via
`normalize_layers` since fictional elevation units are meaningless):

```
world.terrainbundle/
  heightmap.png    16-bit grayscale, one value per cell
  water.png        optional water mask (→ acrylic lake insets)
  meta.json        units-per-pixel, height scale, origin, attribution
  features.csv     name, type, x, y, elev, include, icon, label_override
  rivers.geojson   optional river linework (scored per layer)
```

An adapter turns each source into a bundle; then you `run` it like any config:

```bash
uv run toposlicr adapt mesh   world.stl      -o world.terrainbundle   # Tier 2: 3D mesh
uv run toposlicr adapt art    map.png        -o world.terrainbundle   # Tier 3: 2D artwork
uv run toposlicr adapt azgaar cells.geojson  -o world.terrainbundle   # Azgaar FMG export
uv run toposlicr adapt botw   --heightmap h.png --objmap objs.geojson # Tier 1: BotW
uv run toposlicr run examples/fictional-bundle.toml -o output --name Mythwold
```

- **Tier 1 — extractable game terrain** (best case; reference = Breath of the
  Wild): user-supplied heightmap + ZeldaMods objmap → exact terrain, water and
  named features. Raw `.hght` tiles are read directly; full `.tscb` placement is
  delegated to the community `BotWHeightMapConverter`.
- **Tier 2 — 3D mesh** (STL/OBJ/glTF): ray/triangle top-surface rasterization
  with up-axis detection, pedestal removal and hole-fill.
- **Tier 3 — 2D artwork**: land/sea segmentation + a painted terrain-class
  overlay drive heightfield *synthesis* (coastal ramp, ridge stamping, river
  carving, band-aware noise). Authoring assisted by automation — every run emits
  hillshade + relief-over-art previews for a paint-and-rerun refine loop.
- **Azgaar Fantasy Map Generator**: its cell heightmap is data, not art — the
  cleanest path (and legally clean; Azgaar maps are free for commercial use).

Adapters operate only on files you supply from your own legally-owned copies;
they never download or bundle game assets or artwork. Needs the fictional extra:
`uv sync --extra fictional` (OCR/SAM/ML-depth assists are further optional extras).

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
- **Browser GUI** — FastAPI + no-build front-end over the same core. ✅
- **Phase 7/8 — Fictional maps:** terrain-bundle contract + flat-coordinate core
  (`normalize_layers`, icons, water insets), and Tier 1/2/3 + Azgaar adapters. ✅

**v2 / future:** true irregular (no-fit-polygon) nesting to cut sheet waste;
puzzle-tab seam joints; ML depth-assist and OCR/SAM extras for Tier 3.
