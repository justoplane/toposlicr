"""Nesting parts into cut boards (plan Section 6).

Parts are grouped by material — each material nests independently onto its own
board sequence — then bin-packed onto usable-bed-sized sheets with a kerf-aware
gap between pieces. This is the deterministic v1 (rectangular bin packing on part
bounding boxes); true irregular no-fit-polygon nesting is a later upgrade behind
the same ``nest_parts`` interface.

Because nesting is unattended, every run ends with a hard post-nest validation:
no overlaps, all parts placed, everything within the bed. A bad layout fails the
run rather than being emitted.
"""

from __future__ import annotations

from collections import defaultdict

from rectpack import MaxRectsBssf, PackingBin, PackingMode, newPacker
from shapely.affinity import translate as _translate

from .config import Config
from .parts import OP_CUT, OP_SCORE_IDS, Board, Part

# Extra clear space between neighbouring parts, on top of 2×kerf, so adjacent
# cut kerfs never touch.
PART_GAP_MM = 1.0

# Part-ID score height (mm). Small — it is only an assembly aid.
PART_ID_CAP_MM = 3.0

# Absolute area tolerance (mm²) for the overlap check; below this counts as a
# shared edge / rounding artefact, not a real overlap.
_OVERLAP_TOL = 1e-3


def _spacing(cfg: Config) -> float:
    return 2 * cfg.machine.kerf_mm + PART_GAP_MM


def add_part_id_score(part: Part, cfg: Config) -> None:
    """Score the part's ID at an interior point so it can be read while building.

    Added to the part's local ops *before* placement, so it travels with the
    piece when nesting moves it.
    """
    from .labels import text_to_polygons

    glyphs = text_to_polygons(part.part_id, font=cfg.symbology.labels.font,
                              cap_height_mm=PART_ID_CAP_MM)
    if glyphs.is_empty:
        return
    point = part.outline.representative_point()
    minx, miny, maxx, maxy = glyphs.bounds
    cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    part.add_op(OP_SCORE_IDS, _translate(glyphs, point.x - cx, point.y - cy))


def nest_parts(parts: list[Part], cfg: Config) -> list[Board]:
    """Group parts by material and bin-pack each group onto its own boards."""
    usable_w, usable_h = cfg.machine.usable_bed_mm
    spacing = _spacing(cfg)

    # Score every part's ID first, so it is part of the piece being placed.
    for part in parts:
        add_part_id_score(part, cfg)

    by_material: dict[str, list[Part]] = defaultdict(list)
    for part in parts:
        by_material[part.material].append(part)

    boards: list[Board] = []
    for material in sorted(by_material):
        group = by_material[material]
        _guard_fits_bed(group, usable_w, usable_h, material)
        boards.extend(_nest_group(group, material, usable_w, usable_h, spacing))

    validate_boards(boards, cfg)
    return boards


def _guard_fits_bed(parts: list[Part], usable_w: float, usable_h: float,
                    material: str) -> None:
    for part in parts:
        w, h = part.size
        if w > usable_w + 1e-9 or h > usable_h + 1e-9:
            raise ValueError(
                f"part {part.part_id} ({w:.1f}×{h:.1f} mm) exceeds the usable bed "
                f"({usable_w:.1f}×{usable_h:.1f} mm) for material '{material}'; "
                "panelize oversized layers before nesting."
            )


def _nest_group(parts: list[Part], material: str, usable_w: float,
                usable_h: float, spacing: float) -> list[Board]:
    packer = newPacker(mode=PackingMode.Offline, bin_algo=PackingBin.BFF,
                       pack_algo=MaxRectsBssf, rotation=False)
    # Each rect is inflated by `spacing` for the inter-part gap, so the bin is
    # enlarged by the same amount — otherwise a part sized exactly to the usable
    # bed would fail to place. The outer bed margin absorbs the extra spacing,
    # and placement still keeps every part within the usable area.
    packer.add_bin(usable_w + spacing, usable_h + spacing, count=float("inf"))
    by_index = {}
    for i, part in enumerate(parts):
        w, h = part.size
        # Inflate by the gap so packed neighbours keep clear of each other.
        packer.add_rect(w + spacing, h + spacing, rid=i)
        by_index[i] = part
    packer.pack()

    placed_ids = set()
    board_parts: dict[int, list[Part]] = defaultdict(list)
    for bin_idx, x, y, _w, _h, rid in packer.rect_list():
        part = by_index[rid]
        minx, miny, _, _ = part.outline.bounds
        # Land the part's bbox-min at the rect corner; the gap sits on the
        # opposite sides, guaranteeing clearance to the next piece.
        from .parts import Placement

        part.placement = Placement(board_index=bin_idx, tx=x - minx, ty=y - miny)
        board_parts[bin_idx].append(part)
        placed_ids.add(rid)

    missing = [by_index[i].part_id for i in by_index if i not in placed_ids]
    if missing:
        raise ValueError(
            f"nesting failed to place {len(missing)} part(s) for material "
            f"'{material}': {', '.join(missing)}"
        )

    boards = []
    for bin_idx in sorted(board_parts):
        boards.append(Board(index=bin_idx, material=material, width_mm=usable_w,
                            height_mm=usable_h, parts=board_parts[bin_idx]))
    return boards


def validate_boards(boards: list[Board], cfg: Config) -> None:
    """Hard-fail the run if any board layout is invalid (Section 6.2).

    Checks: unique part IDs, every part placed, each part within the bed, and no
    two parts on the same board overlapping.
    """
    seen: set[str] = set()
    for board in boards:
        for part in board.parts:
            if part.part_id in seen:
                raise ValueError(f"part {part.part_id} placed on more than one board")
            seen.add(part.part_id)
            if part.placement is None:
                raise ValueError(f"part {part.part_id} has no placement")

        cuts = []
        for part in board.parts:
            cut = part.placed_geometries()[OP_CUT]
            minx, miny, maxx, maxy = cut.bounds
            if (minx < -1e-6 or miny < -1e-6
                    or maxx > board.width_mm + 1e-6 or maxy > board.height_mm + 1e-6):
                raise ValueError(
                    f"part {part.part_id} lies outside board "
                    f"{board.material}#{board.index} bounds"
                )
            cuts.append((part.part_id, cut))

        for i in range(len(cuts)):
            for j in range(i + 1, len(cuts)):
                (id_a, a), (id_b, b) = cuts[i], cuts[j]
                if a.intersection(b).area > _OVERLAP_TOL:
                    raise ValueError(
                        f"parts {id_a} and {id_b} overlap on board "
                        f"{board.material}#{board.index}"
                    )
