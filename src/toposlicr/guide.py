"""Auto-generated assembly guide (plan Section 7.2).

Produces a self-contained HTML guide with embedded SVG diagrams:
* an exploded stack diagram (layer order + the elevation each layer represents),
* a per-board layout index (board → parts, material),
* a material list with sheet counts,
* an acrylic-insert list, and
* per-board stats (cut length, utilization).

HTML keeps it dependency-free and printable; a PDF can be produced from it with
any browser or ``weasyprint`` if desired.
"""

from __future__ import annotations

import html
from pathlib import Path

from .config import Config
from .layers import LayerModel
from .parts import OP_CUT, Board
from .svg import SvgDocument


def write_assembly_guide(model: LayerModel, boards: list[Board], cfg: Config,
                         out_path: str | Path, *, project_name: str = "toposlicr"
                         ) -> Path:
    """Render the assembly guide HTML and return the written path."""
    stats = compute_stats(model, boards)
    sections = [
        _header(project_name, model, stats),
        _exploded_stack(model, cfg),
        _material_list(stats),
        _acrylic_list(boards),
        _board_index(boards, cfg),
    ]
    doc = _HTML_TEMPLATE.format(
        title=html.escape(f"{project_name} — assembly guide"),
        body="\n".join(sections),
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(doc, encoding="utf-8")
    return p


# --- stats -----------------------------------------------------------------

def compute_stats(model: LayerModel, boards: list[Board]) -> dict:
    materials: dict[str, dict] = {}
    total_parts = 0
    for board in boards:
        m = materials.setdefault(board.material, {"boards": 0, "parts": 0,
                                                  "cut_mm": 0.0})
        m["boards"] += 1
        for part in board.parts:
            total_parts += 1
            m["parts"] += 1
            m["cut_mm"] += _cut_length(part)
    board_stats = []
    for board in boards:
        area = board.width_mm * board.height_mm
        used = sum(p.area for p in board.parts)
        board_stats.append({
            "index": board.index, "material": board.material,
            "parts": board.part_count,
            "cut_mm": sum(_cut_length(p) for p in board.parts),
            "utilization": (used / area) if area else 0.0,
        })
    return {"materials": materials, "boards": board_stats,
            "total_parts": total_parts, "total_boards": len(boards)}


def _cut_length(part) -> float:
    geom = part.placed_geometries().get(OP_CUT)
    return geom.length if geom is not None and not geom.is_empty else 0.0


# --- sections --------------------------------------------------------------

def _header(project_name: str, model: LayerModel, stats: dict) -> str:
    return f"""<section>
  <h1>{html.escape(project_name)} — assembly guide</h1>
  <table class="kv">
    <tr><th>Layers</th><td>{model.layer_count}</td></tr>
    <tr><th>Scale</th><td>{model.scale.scale_label()}</td></tr>
    <tr><th>Vertical exaggeration</th><td>{model.scale.exaggeration:.2f}×</td></tr>
    <tr><th>Contour interval</th><td>{model.interval_m:.0f} m</td></tr>
    <tr><th>Elevation</th><td>{model.base_elev_m:.0f}–{model.max_elev_m:.0f} m</td></tr>
    <tr><th>Model size</th><td>{model.model_width_mm:.0f} × {model.model_height_mm:.0f} mm</td></tr>
    <tr><th>Total parts</th><td>{stats['total_parts']}</td></tr>
    <tr><th>Total boards</th><td>{stats['total_boards']}</td></tr>
  </table>
</section>"""


def _exploded_stack(model: LayerModel, cfg: Config) -> str:
    """Schematic side view: slabs bottom (base) → top (summit), with elevations."""
    n = model.layer_count
    slab_h, gap, pad = 10, 6, 40
    width = 420
    height = pad * 2 + n * (slab_h + gap)
    rows = []
    for i, layer in enumerate(model.layers):
        y = height - pad - (i + 1) * (slab_h + gap)
        inset = 30 * (i / max(1, n - 1))       # taper upward for a relief look
        x0, x1 = pad + inset, width - pad - inset
        shade = 220 - int(150 * (i / max(1, n - 1)))
        elev = layer.threshold_m
        rows.append(
            f'<rect x="{x0:.0f}" y="{y:.0f}" width="{x1 - x0:.0f}" height="{slab_h}" '
            f'fill="rgb({shade},{shade},{shade})" stroke="#333"/>'
            f'<text x="{width - pad + 4:.0f}" y="{y + slab_h - 1:.0f}" '
            f'class="lbl">L{layer.index:02d} · {elev:.0f} m</text>'
        )
    return f"""<section>
  <h2>Exploded stack (bottom = base, top = summit)</h2>
  <svg viewBox="0 0 {width + 90} {height}" class="stack">{''.join(rows)}</svg>
</section>"""


def _material_list(stats: dict) -> str:
    rows = []
    for mat, m in sorted(stats["materials"].items()):
        rows.append(
            f"<tr><td>{html.escape(mat)}</td><td>{m['boards']}</td>"
            f"<td>{m['parts']}</td><td>{m['cut_mm'] / 1000:.2f} m</td></tr>"
        )
    return f"""<section>
  <h2>Material list</h2>
  <table>
    <tr><th>Material</th><th>Sheets</th><th>Parts</th><th>Cut length</th></tr>
    {''.join(rows)}
  </table>
</section>"""


def _acrylic_list(boards: list[Board]) -> str:
    items = []
    for board in boards:
        for part in board.parts:
            if part.kind == "acrylic":
                items.append(f"<li>{html.escape(part.part_id)} → layer "
                             f"{part.layer_index} ({html.escape(part.material)})</li>")
    if not items:
        return ""
    return f"""<section>
  <h2>Acrylic inserts</h2>
  <ul>{''.join(items)}</ul>
</section>"""


def _board_index(boards: list[Board], cfg: Config) -> str:
    blocks = []
    for board in boards:
        thumb = _board_thumb(board, cfg)
        area = board.width_mm * board.height_mm
        used = sum(p.area for p in board.parts)
        util = (used / area * 100) if area else 0.0
        cut_m = sum(_cut_length(p) for p in board.parts) / 1000
        ids = ", ".join(html.escape(p.part_id) for p in board.parts)
        blocks.append(f"""<div class="board">
      <h3>{html.escape(board.material)} · board {board.index}</h3>
      <p>{board.part_count} parts · {util:.0f}% used · {cut_m:.2f} m cut</p>
      {thumb}
      <p class="ids">{ids}</p>
    </div>""")
    return f"""<section>
  <h2>Cut boards</h2>
  {''.join(blocks)}
</section>"""


def _board_thumb(board: Board, cfg: Config, max_px: int = 300) -> str:
    """Inline SVG thumbnail of a board's placed part outlines."""
    doc = SvgDocument(board.width_mm, board.height_mm, title=f"board {board.index}")
    for part in board.parts:
        geom = part.placed_geometries().get(OP_CUT)
        if geom is not None and not geom.is_empty:
            doc.add_cut(geom, "#000000", label="cut")
    scale = max_px / max(board.width_mm, board.height_mm)
    inner = doc.to_svg().split("\n", 1)[1].rsplit("</svg>", 1)[0]
    # Re-wrap with a pixel size for inline display, keeping the mm viewBox.
    return (f'<svg viewBox="0 0 {board.width_mm:.2f} {board.height_mm:.2f}" '
            f'width="{board.width_mm * scale:.0f}" '
            f'height="{board.height_mm * scale:.0f}" class="thumb">'
            f'{inner}</svg>')


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #222; }}
  h1 {{ font-size: 1.6rem; }} h2 {{ margin-top: 2rem; border-bottom: 1px solid #ccc; }}
  table {{ border-collapse: collapse; margin: 0.5rem 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
  table.kv th {{ background: #f5f5f5; }}
  .board {{ display: inline-block; vertical-align: top; margin: 0.5rem 1rem 1rem 0;
            padding: 0.5rem; border: 1px solid #ddd; border-radius: 6px; }}
  .thumb {{ border: 1px solid #eee; background: #fafafa; }}
  .thumb path {{ fill: #dfe6ee; stroke: #333; stroke-width: 0.4; }}
  .stack text.lbl {{ font-size: 9px; fill: #333; }}
  .ids {{ font-size: 0.75rem; color: #666; max-width: 300px; }}
</style></head>
<body>
{body}
</body></html>"""
