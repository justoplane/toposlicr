"""Cut-board SVG rendering (plan Sections 6–7).

Given nested ``Board`` objects, write one SVG per board named
``board_{material}_{n}.svg`` with every part's cut outline and score/engrave
linework placed by its ``Placement`` transform and colored by operation via the
machine profile. A small engraved header labels each board.
"""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .parts import OP_CUT, OP_ENGRAVE, Board
from .svg import SvgDocument


def render_boards(boards: list[Board], cfg: Config, out_dir: str | Path,
                  project_name: str = "toposlicr", optimize: bool = True) -> list[Path]:
    """Write one SVG per board; return the written paths.

    When ``optimize`` is set, each board's paths are reordered to minimize laser
    travel (the vpype-style post-pass) before writing.
    """
    out = Path(out_dir)
    colors = cfg.machine.colors
    paths: list[Path] = []
    for board in boards:
        doc = SvgDocument(board.width_mm, board.height_mm,
                          title=f"{project_name} {board.material} board {board.index}")
        for part in board.parts:
            geoms = part.placed_geometries()
            for role, geom in geoms.items():
                if geom.is_empty:
                    continue
                color = colors.get(role, colors[OP_CUT])
                if role == OP_CUT:
                    doc.add_cut(geom, color, label=OP_CUT)
                elif role == OP_ENGRAVE:
                    doc.add_engrave(geom, color, label=OP_ENGRAVE)
                else:
                    doc.add_score(geom, color, label=role)
        _add_header(doc, board, cfg, project_name)
        if optimize:
            from .optimize import optimize_document
            optimize_document(doc)
        name = f"board_{_safe(board.material)}_{board.index:02d}.svg"
        paths.append(doc.save(out / name))
    return paths


def _add_header(doc: SvgDocument, board: Board, cfg: Config, project_name: str) -> None:
    """Engrave a small header in the top margin scrap (Section 6.1)."""
    from .labels import text_to_polygons

    text = f"{project_name}  {board.material}  board {board.index}  ({board.part_count}p)"
    glyphs = text_to_polygons(text, font=cfg.symbology.labels.font, cap_height_mm=3.0)
    if glyphs.is_empty:
        return
    from shapely.affinity import translate

    margin = cfg.machine.margin_mm
    minx, miny, maxx, maxy = glyphs.bounds
    # Place near the top-left inside the margin (model y-up; top = height).
    placed = translate(glyphs, margin - minx, board.height_mm - margin - maxy)
    doc.add_engrave(placed, cfg.machine.colors.get(OP_ENGRAVE, "#333333"),
                    label="board_header")


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)
