# toposlicr — User Guide

Turn a geographic area (or a fictional world) into **laser-ready, layered
plywood topographic-map cut boards** plus an assembly guide — from one command.
It replaces the manual *3D model → slicer → Inkscape tracing → hand-layout*
workflow: it contours a Digital Elevation Model (DEM) directly into stacked
polygons and pulls rivers, trails, lakes, peaks and place names from the same
coordinate system, so everything lines up automatically.

- **CLI**: `toposlicr run config.toml -o output`
- **GUI**: `toposlicr serve` → a browser app at `http://127.0.0.1:8000`

---

## Features

### Terrain → stacked layers
- **One command, real data.** Give a bounding box + a few physical parameters;
  out come nested per-layer SVGs, cut boards, and an assembly guide.
- **DEM sources, auto-selected.** Keyless AWS **terrarium** tiles (global, no
  setup), **OpenTopography** (COP30 globally, USGS 3DEP in the US — needs a free
  API key), or a **synthetic** provider for offline demos. Results are cached.
- **Scale/exaggeration math.** Pin any one of vertical *exaggeration*, *contour
  interval*, or *layer count* — the rest is derived, with warnings for
  impractical builds (too many layers, tiny summit caps).
- **Clean, organic contours.** DEM smoothing, Douglas-Peucker simplification,
  Chaikin corner-cutting, minimum-feature filtering (no fragile plywood
  slivers), and nesting validation so every layer has glue support.

### Symbology (scored & engraved onto the wood)
- **Registration scores** — the outline of the layer above is scored on each
  layer, so assembly is foolproof.
- **Rivers** — OSM waterways, scored per layer only where exposed, filtered by
  stream order.
- **Trails** — OSM hiking paths and named routes (e.g. the John Muir Trail),
  drawn as **dashed** lines on their own laser step, with route **names
  engraved curving along the path** (horizontal fallback on tight segments).
  Filter by walking-network grade; **toggle individual trails** in the GUI.
- **Lakes** — scored outlines, or **flush press-fit acrylic insets** (the lake
  is cut out and a blue-acrylic piece drops in). A calibration coupon dials in
  the press-fit tolerance for your machine/material.
- **Labels** — peak/place/lake names as filled-font engraving, placed in the
  nearest open space on the exposed surface: rotated along the band, curved
  when the band bends, shrunk down a size ladder (not below `min_cap_height_mm`,
  default 2.5 mm), with the elevation dropped when room is tight, or moved to a
  lower layer with a leader line. Water and icons are never covered.
- **Icons** — engraved glyphs (peak/tower/shrine/settlement…) for fictional maps.
- **Curation loop.** A `features.csv` (and `labels.json`) is written alongside
  the output; edit include/exclude or label overrides and re-run. The GUI edits
  trails interactively.

### Panelization, nesting & output
- **Oversized layers are split** to fit the laser bed, routing seams through the
  hidden zone under higher layers so the finished piece doesn't look fragmented.
- **Automatic nesting** onto bed-sized cut boards, grouped by material/color,
  with part IDs (`L04-P2`) scored where they'll be hidden. Post-nest validation
  rejects overlaps.
- **North-up, rectangular frame.** Terrain is projected into a transverse
  Mercator centered on your box (not the nearest UTM zone, whose grid
  convergence would tilt the map by up to ~2°), and every layer is clipped to
  the rectangle inscribed in the projected box — so the base slab is a true
  rectangle with north straight up.
- **Laser-ready SVGs** in true 1:1 mm units, with **color = operation** (the
  Glowforge convention): black cut, red registration, blue rivers, brown dashed
  trails, teal part-IDs, grey engraving. A cut-path optimizer reduces travel.
- **Assembly guide** (HTML): exploded stack, per-board part index, material list
  with sheet counts, acrylic-insert list, and stats.

### Fictional maps (games, models, art)
Cut maps of fictional worlds via a neutral **terrain bundle** the core consumes
like any DEM (flat coordinates; vertical scale via `normalize_layers`):
- **Tier 1 — extractable game terrain** (reference: Breath of the Wild): a
  user-supplied heightmap + ZeldaMods objmap → exact terrain, water and names.
- **Tier 2 — 3D mesh** (STL/OBJ/glTF): top-surface rasterization with up-axis
  detection and pedestal removal.
- **Tier 3 — 2D artwork**: land/sea segmentation + a painted terrain-class
  overlay drive heightfield *synthesis* (coastal ramp, ridge stamping, river
  carving, band-aware noise) with paint-and-rerun refine previews.
- **Azgaar Fantasy Map Generator**: its per-cell heightmap → bundle (cleanest,
  and free for commercial use).

Adapters operate only on files you supply; they never download game assets.

### Browser GUI
A no-build web app over the same core: draw the bbox on a map (or type it),
guided form with an advanced raw-TOML panel, a **Fictional tab** (upload a
mesh/artwork/Azgaar export), a **Trails checklist** for per-trail selection,
and one main view that swaps from the map to a progress bar to the interactive
**layer stack** (with Boards and Downloads tabs alongside).

---

## Install

Uses [uv](https://docs.astral.sh/uv/). From the project directory:

```bash
uv sync                                  # core (real-world maps + CLI)
uv sync --extra web                      # + browser GUI
uv sync --extra fictional                # + fictional-map adapters (mesh/art/azgaar/botw)
uv sync --extra web --extra fictional    # everything
uv sync --extra dev                      # + test/lint tools (add to any of the above)
```

Prefix commands with `uv run` (e.g. `uv run toposlicr …`), or activate the venv.

**Optional:** set `OPENTOPOGRAPHY_API_KEY` to use OpenTopography DEMs (3DEP/COP30);
otherwise the keyless `terrarium` source is used.

---

## Quick start

```bash
# 1. Real-world map from the example config (Mt Whitney):
uv run toposlicr run examples/whitney.toml -o output --name Whitney

# 2. Or just launch the GUI and click around:
uv run toposlicr serve

# 3. Offline demo (no network, no key):
uv run toposlicr run examples/synthetic-demo.toml -o output --no-symbology
```

`run` writes, under the output directory:

```
output/
  layers/layer_NN.svg        one SVG per layer — full layer outline + registration
                             + symbology + panelization seams (where a big layer
                             is split to fit the bed, in a distinct color)
  boards/board_MAT_NN.svg    nested cut boards, grouped by material
  assembly_guide.html        exploded stack, board index, material + acrylic lists
  features.csv               editable feature selection (re-run to apply edits)
  labels.json                editable label positions (manual-nudge loop)
  preview.svg, debug/*.geojson   non-blocking debug artifacts
```

Open the board SVGs in your laser software, map the path colors to
cut/score/engrave steps, and open `assembly_guide.html` in a browser to build.

---

## Using the CLI

### `run` — the full pipeline
```bash
uv run toposlicr run CONFIG.toml -o OUTDIR [options]
```
| Option | Meaning |
|---|---|
| `-o, --out DIR` | Output directory (default `output`). |
| `--name TEXT` | Project name for headers/guide (default: config filename). |
| `--dem-resolution FLOAT` | Target DEM resolution in meters (default 30). |
| `--no-symbology` | Terrain only — skip OSM rivers/trails/lakes/labels. |
| `--no-nest` | Stop at per-layer SVGs (skip parts/boards/guide). |
| `--no-panelize` | Don't split oversized layers into bed-sized parts. |
| `--no-cache` | Bypass the on-disk DEM/feature cache. |
| `--no-debug` | Skip the preview + GeoJSON debug artifacts. |

### `scale` — quick math report (no data fetch)
```bash
uv run toposlicr scale CONFIG.toml --min-elev 2500 --max-elev 4421
```
Reports the scale (1:N), contour interval, exaggeration, model size and layer
count for a config. `--min-elev/--max-elev` supply the elevation range (the
`run` command reads it from the DEM automatically). `--no-snap` keeps the exact
derived interval instead of rounding.

### `validate` — check a config
```bash
uv run toposlicr validate CONFIG.toml
```
Loads the config and reports warnings/errors without running anything.

### `coupon` — press-fit calibration
```bash
uv run toposlicr coupon CONFIG.toml -o coupon.svg --start -0.05 --stop 0.15 --step 0.02
```
Emits a strip of holes + plugs at stepped interference offsets. Cut it once per
machine + material, find the plug that press-fits cleanly, and put that offset in
`[symbology].lakes.fit_gap_mm`.

### `adapt` — build a fictional terrain bundle (needs `--extra fictional`)
```bash
uv run toposlicr adapt mesh   world.stl        -o world.terrainbundle --cells-across 1200
uv run toposlicr adapt art    map.png          -o world.terrainbundle --class-overlay zones.png
uv run toposlicr adapt azgaar cells.geojson    -o world.terrainbundle --burgs burgs.csv
uv run toposlicr adapt botw   -o world.terrainbundle --heightmap h.png --objmap objs.geojson
```
Then run it like any config that points `[region].bundle` at the bundle:
```bash
uv run toposlicr run examples/fictional-bundle.toml -o output --name Mythwold
```

### `serve` — launch the GUI
```bash
uv run toposlicr serve --port 8000        # opens the browser automatically
uv run toposlicr serve --no-open --host 0.0.0.0   # headless / LAN
```
It prints the URL(s) to open. **On WSL** it auto-binds all interfaces and prints
both `http://localhost:PORT` and your WSL IP — open one of those in your Windows
browser (try `localhost` first; use the IP if that fails). The browser
auto-open is skipped under WSL since it can't reach a Windows browser.

---

## Using the GUI

`toposlicr serve` opens a single page with the form on the left and a map +
results on the right.

**Real world tab**
1. **Region** — drag a rectangle on the map (or type W/S/E/N) and pick a DEM
   source.
2. **Physical** — model width, ply thickness, and **Set vertical scale by**:
   *Number of layers* (the default), *Vertical exaggeration*, or *Contour
   interval*. These three are linked, so you pick the one you care about and the
   others follow from the terrain and ply thickness.
3. **Machine / Materials / Symbology** — bed size, kerf, margin; material per
   layer range + the water/acrylic material; feature thresholds.
4. **Trails** — click **Load trails for this area** to fetch the trails in your
   bbox. They appear as a checklist (grouped by name, with network badge and
   length); tick/untick individual trails, or use *select all / none*. Your
   choices are applied on Run.
5. **Advanced** — an optional raw-TOML panel to hand-edit anything.
6. **Run pipeline** — the map swaps to a progress bar while the pipeline runs,
   then to the **Stack** tab: an interactive layer stack (build the map up layer
   by layer with a slider or play button, hover to identify layers, click a layer
   to open its cut file, zoom/pan, toggle water/trails/names/seams, export a
   PNG). The **Boards** tab shows the composite preview and nested cut boards,
   **Downloads** holds the assembly guide, per-file and **download-all (zip)**
   links, and the **Map** tab takes you back to adjust the box. A legend under
   each viewer explains the colors and line styles on stage. The full progress
   log and any warnings live in the collapsed **Progress** panel at the bottom
   of the page.

**Fictional tab**
1. Pick a **source type** (3D mesh, 2D artwork, Azgaar cells GeoJSON, or an
   existing bundle path) and **upload the file** (for artwork you can also add a
   painted terrain-class overlay).
2. Set the adapter options + `normalize_layers` and the shared machine/material
   fields.
3. **Run** uploads the source, runs the matching adapter into a terrain bundle,
   then the pipeline — same progress + results view.

Files you upload stay on your machine; nothing is sent off-box (the GUI runs
locally).

---

## Config file reference

A project is a TOML file. Minimal real-world example:

```toml
[region]
bbox = [-118.35, 36.52, -118.20, 36.62]   # W, S, E, N (lon/lat)
dem  = "terrarium"                         # auto | terrarium | COP30 | USGS10m | synthetic

[physical]
model_width_mm   = 500
ply_thickness_mm = 3.0
exaggeration     = 1.5      # OR interval_m = 100  OR layer_count = 20
# base_datum = "auto"       # or an elevation in meters (e.g. a lake surface)

[machine]
profile   = "glowforge"
bed_mm    = [495, 279]
margin_mm = 8
kerf_mm   = 0.15            # measure with `toposlicr coupon`

[materials]
default   = "birch_3mm"
water     = "blue_acrylic_3mm"
overrides = { "0-1" = "walnut_3mm" }   # layer-range → material

[symbology]
peaks  = { min_prominence_m = 150, label_elevation = true }
rivers = { min_stream_order = 3 }
trails = { include = true, min_network = "lwn", include_unnamed = true, dash_mm = 2.5, gap_mm = 1.5 }
lakes  = { min_area_km2 = 0.05, mode = "inset", fit_gap_mm = 0.1 }   # mode = score | inset
labels = { font = "DejaVu Sans", cap_height_mm = 4, icon_size_mm = 5 }

[panelization]
seam_margin_mm = 4
seam_joint     = "butt"     # butt | puzzle
```

**Fictional** projects replace `[region]` and use `normalize_layers`:
```toml
[region]
bundle = "world.terrainbundle"
[physical]
model_width_mm  = 400
normalize_layers = 12                 # N layers across the height range
clip_percentiles = [0.5, 99.5]        # trim spires/pits so banding stays even
```

**Colors = operations** (override per machine in `[machine.colors]`): `cut`
`#000000`, `score_registration` `#FF0000`, `score_hydro` `#0000FF`,
`score_trail` `#A05A2C`, `score_ids` `#00A0A0`, `engrave_fill` `#333333`,
`seam` `#CC00CC` (panelization-split preview on the per-layer SVGs). Set the
**number of layers** or **vertical exaggeration** in `[physical]` — pin
`layer_count`, `exaggeration`, or `interval_m` (any one; the rest are derived).

---

## Tips & troubleshooting

- **Too many layers / tiny summits.** Raise the contour interval, lower
  exaggeration, or use thicker stock — the `scale` command warns you first.
- **No rivers/trails showing.** OSM coverage varies; also check the feature
  thresholds. The run degrades gracefully (terrain-only) if Overpass is down.
- **Trails you don't want.** Untick them in the GUI Trails panel, or edit
  `include` in `features.csv` and re-run.
- **Acrylic doesn't press-fit.** Run `toposlicr coupon`, pick the clean offset,
  set `[symbology].lakes.fit_gap_mm`.
- **Wrong scale in your laser app.** The SVGs are true 1:1 mm with a matching
  `viewBox`; set your software's document units to millimeters.
- **Fictional units look off.** Fictional elevation is unitless — drive vertical
  scale with `normalize_layers`, not fake meters; `clip_percentiles` tames a
  lone spire.
