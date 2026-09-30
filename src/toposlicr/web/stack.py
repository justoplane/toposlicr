"""Build the ``stack.json`` payload the GUI's interactive stack viewer renders.

One document per run describing every layer as SVG path data in model
millimeters (y already flipped, so the viewer's viewBox is simply the model
size) together with the per-layer symbology grouped by laser operation. The
viewer draws layers on top of each other in order, so the payload carries no
registration scores — the layer above covers them — and instead the front-end
derives the "next layer goes here" ghost from the layer above's cut outline.
"""

from __future__ import annotations

import re
from typing import Any

from shapely import get_num_coordinates

from ..config import Config
from ..parts import material_for_layer
from ..pipeline import PipelineResult
from ..render import layer_seams
from ..svg import line_to_path_d, polygon_to_path_d
from ..symbology import label_geometry_for_layer, leader_geometry_for_layer

# Operation → path serializer. Lines and polygons serialize differently.
_LINE_OPS = {"score_hydro", "score_trail", "leader", "seam"}


def build_stack(result: PipelineResult, cfg: Config) -> dict[str, Any]:
    model = result.model
    height = model.model_height_mm
    parts = result.parts or []
    sym = result.symbology
    layer_warnings = _warnings_by_layer(result.warnings)

    layers: list[dict[str, Any]] = []
    for layer in model.layers:
        k = layer.index
        ops: dict[str, Any] = {}
        if sym is not None:
            per = sym.for_layer(k)
            if per is not None:
                _put(ops, "score_hydro", [per.rivers, per.lake_outlines], height)
                _put(ops, "score_trail", [per.trails], height)
                _put(ops, "engrave_fill", [per.trail_labels], height)
                _put(ops, "icon", [per.icons], height)
            _put(ops, "leader", [leader_geometry_for_layer(sym.labels, k)], height)
            _put(ops, "engrave_fill", [label_geometry_for_layer(sym.labels, k)], height)
        seams = layer_seams(parts, k)
        if seams is not None:
            _put(ops, "seam", [seams], height)
        panels = sum(1 for p in parts
                     if getattr(p, "kind", None) == "ply" and p.layer_index == k)
        layers.append({
            "index": k,
            "threshold_m": round(layer.threshold_m, 1),
            "material": material_for_layer(k, cfg.materials),
            "panels": max(panels, 1),
            "area_mm2": round(layer.geometry.area),
            "vertices": int(get_num_coordinates(layer.geometry)),
            "warnings": list(layer.warnings) + layer_warnings.get(k, []),
            "cut": polygon_to_path_d(layer.geometry, height),
            "ops": ops,
        })

    return {
        "model": {
            "width_mm": round(model.model_width_mm, 3),
            "height_mm": round(height, 3),
            "ply_thickness_mm": cfg.physical.ply_thickness_mm,
            "interval_m": round(model.interval_m, 1),
            "scale_label": model.scale.scale_label(),
            "base_elev_m": round(model.base_elev_m, 1),
            "max_elev_m": round(model.max_elev_m, 1),
            "layer_count": model.layer_count,
            "fictional": bool(model.flat),
        },
        "colors": dict(cfg.machine.colors),
        "layers": layers,
    }


def _put(ops: dict[str, Any], op: str, geoms, height: float) -> None:
    """Append serialized geometry to ``ops[op]`` (skipping empties)."""
    for g in geoms:
        if g is None or g.is_empty:
            continue
        d = line_to_path_d(g, height) if op in _LINE_OPS else polygon_to_path_d(g, height)
        if not d and op in _LINE_OPS:      # lake outlines may arrive as polygons
            d = polygon_to_path_d(g, height)
        if d:
            ops[op] = ops.get(op, "") + d


_LAYER_RE = re.compile(r"\bon layer (\d+)\b")


def _warnings_by_layer(warnings: list[str]) -> dict[int, list[str]]:
    """Pipeline warnings that name a layer ("… on layer 7"), keyed by index."""
    out: dict[int, list[str]] = {}
    for w in warnings:
        m = _LAYER_RE.search(w)
        if m:
            out.setdefault(int(m.group(1)), []).append(w)
    return out
