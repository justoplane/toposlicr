"""Acrylic lake insets — flush press-fit (plan Section 4.3).

When ``lakes.mode == "inset"``, each lake is cut *out* of its surface-layer ply
piece (becoming a hole) and a matching blue-acrylic part is emitted into the
water material group. The acrylic outline is the lake polygon grown by a small
press-fit offset so it wedges flush into the hole; the layer below forms the
shelf it rests on.

Because the hole (the lake polygon) and the acrylic (that same polygon buffered
outward) are generated from one base geometry, the two curves are congruent by
construction — which is what makes concave lakes press-fit evenly.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import Config
from .layers import LayerModel
from .parts import Part, part_id, water_material
from .symbology import SymbologyResult

# Manufacturing interference baked into the press-fit offset (mm). Tunable per
# machine/material via the calibration coupon; this is a sane default.
INTERFERENCE_MM = 0.05

# Lakes smaller than this (model mm²) aren't worth an acrylic insert.
_MIN_LAKE_AREA_MM2 = 1.0

# Nor are pieces narrower than this (mm) — e.g. the sliver left when a lake that
# is mostly off the map is clipped to it: too thin to cut, handle or press in.
_MIN_LAKE_WIDTH_MM = 3.0


def press_fit_offset(cfg: Config) -> float:
    """Outward offset for the acrylic outline: kerf_ply/2 + kerf_acrylic/2 + interference.

    Same machine cuts both, so the acrylic kerf equals the ply kerf here.
    """
    kerf_ply = cfg.machine.kerf_mm
    kerf_acrylic = cfg.machine.kerf_mm
    return kerf_ply / 2.0 + kerf_acrylic / 2.0 + INTERFERENCE_MM


def apply_acrylic(parts: list[Part], model: LayerModel, symbology: SymbologyResult,
                  cfg: Config) -> tuple[list[Part], list[str]]:
    """Cut lake holes into ply parts and append matching acrylic parts.

    Returns ``(parts, warnings)``. A no-op (parts unchanged, no warnings) when
    ``lakes.mode`` is not ``"inset"``.
    """
    if cfg.symbology.lakes.mode != "inset":
        return parts, []

    offset = press_fit_offset(cfg)
    warnings: list[str] = []
    result: list[Part] = list(parts)

    for k, sym in symbology.per_layer.items():
        acrylic_n = 0
        # The wood this layer is made of. A lake can run off the map edge or
        # past the layer's contour; only the part over wood becomes a hole, so
        # only that part may become acrylic.
        wood = unary_union([p.outline for p in result
                            if p.kind == "ply" and p.layer_index == k])
        for lake in sym.lake_polys:
            if lake.is_empty or lake.area < _MIN_LAKE_AREA_MM2:
                continue
            lake = _mp(lake.intersection(wood))
            if lake.is_empty:
                warnings.append(
                    f"no ply part found on layer {k} to host a lake inset")
                continue
            lake = _drop_thin(lake, _MIN_LAKE_WIDTH_MM)
            if lake.is_empty or lake.area < _MIN_LAKE_AREA_MM2:
                continue

            # The layer below must be solid under the lake to form the shelf.
            if k == 0:
                warnings.append(
                    f"lake on base layer 0 (area {lake.area:.0f} mm²) has no shelf "
                    "below; skipping inset")
                continue
            below = model.footprint(k - 1)
            if below.is_empty or not below.buffer(1e-6).contains(lake):
                warnings.append(
                    f"layer {k - 1} is not solid under a lake on layer {k}; "
                    "acrylic inset may lack a full shelf")

            # Cut the lake out of every ply sub-part on this layer it touches.
            cut_any = False
            for part in result:
                if not (part.kind == "ply" and part.layer_index == k
                        and part.outline.intersects(lake)):
                    continue
                new_outline = _mp(part.outline.difference(lake))
                if new_outline.is_empty:
                    # The lake fully covers this piece — skip rather than delete
                    # the part (an empty outline crashes downstream nesting).
                    warnings.append(
                        f"lake on layer {k} fully covers ply part {part.part_id}; "
                        "skipping this cut")
                    continue
                part.outline = new_outline
                # Re-clip the part's scores/engraving to the new outline so no
                # symbology floats over the now-missing (lake) material.
                for role, geom in list(part.ops.items()):
                    clipped = geom.difference(lake)
                    if clipped.is_empty:
                        del part.ops[role]
                    else:
                        part.ops[role] = clipped
                cut_any = True
            if not cut_any:
                warnings.append(
                    f"no ply part found on layer {k} to host a lake inset")
                continue

            # Emit the matching acrylic part: the lake grown by the press-fit
            # offset, clipped back to the layer so edges on the map border or a
            # contour sit flush instead of overhanging.
            acrylic_outline = _mp(lake.buffer(offset).intersection(wood))
            if acrylic_outline.is_empty:
                continue
            result.append(Part(
                part_id=part_id(k, acrylic_n, kind="acrylic"),
                kind="acrylic",
                material=water_material(cfg.materials),
                outline=acrylic_outline,
                layer_index=k,
            ))
            acrylic_n += 1

    return result, warnings


def _drop_thin(geom: MultiPolygon, min_width: float) -> MultiPolygon:
    """Drop the pieces narrower than ``min_width`` everywhere.

    A piece survives if shrinking it by half the width leaves anything, i.e. a
    ``min_width`` disc fits inside it somewhere.
    """
    return MultiPolygon([p for p in geom.geoms if not p.buffer(-min_width / 2).is_empty])


def _mp(geom: BaseGeometry) -> MultiPolygon:
    """Coerce any polygonal geometry to a MultiPolygon (dropping non-polygons)."""
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom]) if not geom.is_empty else MultiPolygon()
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(parts)
