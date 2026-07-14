"""Render a ``LayerModel`` to per-layer SVGs and debug artifacts (Phase 1).

Per layer we emit the cut outline plus the registration score — the outline of
the layer above, ``boundary(layer k+1) ∩ layer k`` — which is free from the data
model and makes assembly foolproof (plan Section 4.1). A stacked composite
preview and per-layer GeoJSON are written as non-blocking debug artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from .config import Config
from .layers import LayerModel
from .svg import SvgDocument
from .symbology import (
    SymbologyResult,
    label_geometry_for_layer,
    leader_geometry_for_layer,
)


def registration_score(model: LayerModel, k: int) -> BaseGeometry:
    """Outline of layer k+1 as it sits on layer k (the assembly registration mark)."""
    above = model.footprint(k + 1)
    here = model.footprint(k)
    if above.is_empty or here.is_empty:
        return above.boundary  # empty
    return above.boundary.intersection(here)


def layer_seams(parts, layer_index: int) -> BaseGeometry | None:
    """Panelization seam lines for a layer — where an oversized layer is split.

    When a layer is split to fit the bed, adjacent ply sub-parts share a mating
    edge; those shared edges are the seams. Returns the merged seam linework
    (model mm), or None if the layer wasn't split.
    """
    from shapely.ops import unary_union

    ply = [p for p in (parts or [])
           if getattr(p, "kind", None) == "ply" and p.layer_index == layer_index
           and not p.outline.is_empty]
    if len(ply) < 2:
        return None
    segs = []
    for i in range(len(ply)):
        bi = ply[i].outline.boundary
        for j in range(i + 1, len(ply)):
            shared = bi.intersection(ply[j].outline.boundary)
            if not shared.is_empty and shared.length > 0.1:
                segs.append(shared)
    if not segs:
        return None
    merged = unary_union(segs)
    return merged if not merged.is_empty else None


def render_layer_svgs(model: LayerModel, cfg: Config, out_dir: str | Path,
                      symbology: SymbologyResult | None = None,
                      parts=None) -> list[Path]:
    """Write ``layer_{k}.svg`` for every layer; return the paths.

    Each layer carries: the cut outline, the registration score (layer above),
    and — when supplied — river/lake/trail score lines, label engraving, and the
    **panelization seams** (distinct color) showing where an oversized layer will
    be split to fit the cut boards.
    """
    out = Path(out_dir)
    colors = cfg.machine.colors
    paths: list[Path] = []
    for layer in model.layers:
        doc = SvgDocument(model.model_width_mm, model.model_height_mm,
                          title=f"layer {layer.index}")
        doc.add_cut(layer.geometry, colors["cut"], label="cut")
        reg = registration_score(model, layer.index)
        if not reg.is_empty:
            doc.add_score(reg, colors["score_registration"], label="score_registration")

        seams = layer_seams(parts, layer.index)
        if seams is not None and not seams.is_empty:
            doc.add_score(seams, colors.get("seam", "#CC00CC"), label="seam")

        if symbology is not None:
            sym = symbology.for_layer(layer.index)
            if sym is not None:
                if not sym.rivers.is_empty:
                    doc.add_score(sym.rivers, colors["score_hydro"], label="score_hydro")
                if not sym.lake_outlines.is_empty:
                    doc.add_score(sym.lake_outlines, colors["score_hydro"],
                                  label="score_hydro")
                if sym.trails is not None and not sym.trails.is_empty:
                    doc.add_score(sym.trails, colors.get("score_trail", "#A05A2C"),
                                  label="score_trail")
                if sym.trail_labels is not None and not sym.trail_labels.is_empty:
                    doc.add_engrave(sym.trail_labels, colors["engrave_fill"],
                                    label="engrave_fill")
                if sym.icons is not None and not sym.icons.is_empty:
                    doc.add_engrave(sym.icons, colors["engrave_fill"], label="icon")
            leaders = leader_geometry_for_layer(symbology.labels, layer.index)
            if not leaders.is_empty:
                doc.add_score(leaders, colors["score_ids"], label="leader")
            glyphs = label_geometry_for_layer(symbology.labels, layer.index)
            if not glyphs.is_empty:
                doc.add_engrave(glyphs, colors["engrave_fill"], label="engrave_fill")

        paths.append(doc.save(out / f"layer_{layer.index:02d}.svg"))
    return paths


def render_composite_preview(model: LayerModel, cfg: Config,
                             out_path: str | Path) -> Path:
    """A single stacked, color-graded SVG for a quick visual check."""
    doc = SvgDocument(model.model_width_mm, model.model_height_mm,
                      title="composite preview")
    n = max(1, model.layer_count - 1)
    for layer in model.layers:
        # Grade light (base) → dark (summit) so relief reads at a glance.
        shade = int(220 - 160 * (layer.index / n))
        doc.add_engrave(layer.geometry, f"#{shade:02x}{shade:02x}{shade:02x}",
                        label=f"layer_{layer.index}")
    return doc.save(out_path)


def write_layer_geojson(model: LayerModel, out_dir: str | Path) -> list[Path]:
    """Debug artifact: one GeoJSON per layer (model-mm coordinates)."""
    out = Path(out_dir)
    paths: list[Path] = []
    for layer in model.layers:
        feature = {
            "type": "Feature",
            "properties": {"index": layer.index, "threshold_m": layer.threshold_m},
            "geometry": mapping(layer.geometry),
        }
        p = out / f"layer_{layer.index:02d}.geojson"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(feature), encoding="utf-8")
        paths.append(p)
    return paths
