"""Kerf / press-fit calibration coupon generator (plan Section 4.3).

Because a flush press-fit tolerance is tight, the winning offset must be dialed
in once per machine + material pair. This emits a strip of holes with matching
plugs at stepped interference offsets (e.g. −0.05 to +0.15 mm); the builder cuts
it, finds the plug that press-fits cleanly, and stores that offset in config.

The holes are cut in the ply color, the plugs in the acrylic/cut color, each
engraved with its offset value in millimeters.
"""

from __future__ import annotations

from pathlib import Path

from shapely.affinity import translate
from shapely.geometry import Polygon

from .config import Config
from .labels import text_to_polygons
from .svg import SvgDocument


def offset_series(start: float = -0.05, stop: float = 0.15,
                  step: float = 0.02) -> list[float]:
    """Inclusive series of press-fit offsets in mm."""
    # Integer stepping avoids float drift; clamp so no value exceeds `stop` when
    # the step does not divide the range evenly.
    if step <= 0:
        raise ValueError("step must be positive")
    n = int((stop - start) / step + 1e-9)
    vals = [round(start + i * step, 3) for i in range(n + 1)]
    return [v for v in vals if v <= stop + 1e-9]


def build_coupon(cfg: Config, *, offsets: list[float] | None = None,
                 hole_mm: float = 12.0, gap_mm: float = 6.0) -> SvgDocument:
    """Build a calibration coupon SVG for the given offsets.

    Each cell holds a square hole (nominal ``hole_mm``) and, below it, a plug
    sized ``hole_mm + offset`` — positive offset = interference (tighter fit).
    """
    offsets = offsets or offset_series()
    colors = cfg.machine.colors
    cell = hole_mm + gap_mm
    cols = len(offsets)
    margin = cfg.machine.margin_mm

    width = margin * 2 + cols * cell
    height = margin * 2 + 2 * hole_mm + 3 * gap_mm + 6.0  # holes + plugs + label row
    doc = SvgDocument(width, height, title="toposlicr calibration coupon")

    for i, off in enumerate(offsets):
        x0 = margin + i * cell
        hole = _square(x0, margin + hole_mm + 2 * gap_mm, hole_mm)
        plug = _square(x0 + (hole_mm - (hole_mm + off)) / 2.0, margin, hole_mm + off)
        doc.add_cut(hole, colors["cut"], label="cut")
        doc.add_cut(plug, colors["cut"], label="cut")
        label = text_to_polygons(f"{off:+.2f}", font=cfg.symbology.labels.font,
                                 cap_height_mm=2.5)
        if not label.is_empty:
            lminx, lminy, lmaxx, lmaxy = label.bounds
            lx = x0 + (hole_mm - (lmaxx - lminx)) / 2.0 - lminx
            ly = margin + hole_mm + gap_mm - lminy
            doc.add_engrave(translate(label, lx, ly), colors["engrave_fill"],
                            label="engrave_fill")
    return doc


def _square(x: float, y: float, side: float) -> Polygon:
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def write_coupon(cfg: Config, out_path: str | Path, **kwargs) -> Path:
    return build_coupon(cfg, **kwargs).save(out_path)
