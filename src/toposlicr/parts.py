"""The Part / Board data model — the spine for panelization, nesting and output.

A ``Part`` is one physical piece to be cut. It bundles its cut outline with all
associated score/engrave linework in a single local coordinate frame, so a rigid
placement transform (from nesting) moves the geometry and its symbology together.
Panelization splits a layer's part into sub-parts; acrylic insets add water
parts; nesting assigns each part a ``Placement`` on a board; board rendering
applies the placement and groups parts by material.

Operation geometry is keyed by machine-profile color role (``cut``,
``score_registration``, ``score_hydro``, ``score_ids``, ``engrave_fill``) so the
SVG writer maps each straight to a laser step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.affinity import rotate as _rotate
from shapely.affinity import translate as _translate
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry

# Operation roles (match MachineProfile.colors keys).
OP_CUT = "cut"
OP_SCORE_REGISTRATION = "score_registration"
OP_SCORE_HYDRO = "score_hydro"
OP_SCORE_TRAIL = "score_trail"
OP_SCORE_IDS = "score_ids"
OP_ENGRAVE = "engrave_fill"

SCORE_OPS = (OP_SCORE_REGISTRATION, OP_SCORE_HYDRO, OP_SCORE_TRAIL, OP_SCORE_IDS)


@dataclass
class Placement:
    """Where a part sits on its board (model→board transform)."""

    board_index: int
    tx: float = 0.0
    ty: float = 0.0
    rotation_deg: float = 0.0


@dataclass
class Part:
    """One cut piece: outline + per-operation linework in a local frame (mm)."""

    part_id: str                       # e.g. "L04-P2" (ply) or "W03-P1" (water)
    kind: str                          # "ply" | "acrylic"
    material: str                      # material key, e.g. "birch_3mm"
    outline: MultiPolygon              # the cut geometry (may contain holes)
    layer_index: int | None = None
    ops: dict[str, BaseGeometry] = field(default_factory=dict)
    placement: Placement | None = None
    parent_id: str | None = None       # set when produced by splitting

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.outline.bounds

    @property
    def size(self) -> tuple[float, float]:
        minx, miny, maxx, maxy = self.outline.bounds
        return (maxx - minx, maxy - miny)

    @property
    def area(self) -> float:
        return self.outline.area

    def add_op(self, role: str, geom: BaseGeometry | None) -> None:
        """Merge geometry into an operation role (ignores empty)."""
        if geom is None or geom.is_empty:
            return
        from shapely.ops import unary_union

        existing = self.ops.get(role)
        self.ops[role] = geom if existing is None else unary_union([existing, geom])

    def all_geometries(self) -> dict[str, BaseGeometry]:
        """Cut outline plus every operation, keyed by role (local coords)."""
        geoms: dict[str, BaseGeometry] = {OP_CUT: self.outline}
        geoms.update(self.ops)
        return geoms

    def placed_geometries(self) -> dict[str, BaseGeometry]:
        """Same as :meth:`all_geometries` but with the placement transform applied."""
        if self.placement is None:
            return self.all_geometries()
        out = {}
        for role, geom in self.all_geometries().items():
            out[role] = _apply_placement(geom, self.placement)
        return out


def _apply_placement(geom: BaseGeometry, p: Placement) -> BaseGeometry:
    g = geom
    if p.rotation_deg:
        g = _rotate(g, p.rotation_deg, origin="centroid", use_radians=False)
    if p.tx or p.ty:
        g = _translate(g, p.tx, p.ty)
    return g


@dataclass
class Board:
    """A bed-sized sheet of one material holding placed parts."""

    index: int
    material: str
    width_mm: float
    height_mm: float
    parts: list[Part] = field(default_factory=list)

    @property
    def part_count(self) -> int:
        return len(self.parts)


# --- material resolution ---------------------------------------------------

def material_for_layer(layer_index: int, materials: dict) -> str:
    """Resolve a layer's material from the ``[materials]`` config block.

    Supports ``default`` plus ``overrides`` keyed by an inclusive layer range
    string like ``"0-2"`` or a single index ``"3"`` (plan Section 6.1).
    """
    default = materials.get("default", "birch_3mm")
    overrides = materials.get("overrides", {}) or {}
    for key, mat in overrides.items():
        if _range_contains(str(key), layer_index):
            return mat
    return default


def _range_contains(key: str, index: int) -> bool:
    key = key.strip()
    if "-" in key:
        lo, hi = key.split("-", 1)
        try:
            return int(lo) <= index <= int(hi)
        except ValueError:
            return False
    try:
        return int(key) == index
    except ValueError:
        return False


def water_material(materials: dict) -> str:
    return materials.get("water", "blue_acrylic_3mm")


def part_id(layer_index: int, part_num: int, kind: str = "ply") -> str:
    """``L04-P2`` for ply, ``W04-P2`` for water/acrylic."""
    prefix = "W" if kind == "acrylic" else "L"
    return f"{prefix}{layer_index:02d}-P{part_num}"


def build_parts_from_model(model, symbology, cfg) -> list[Part]:
    """Assemble one ply Part per layer from the model + symbology (pre-panel).

    Each part carries its cut outline plus its registration score, hydro scores
    (rivers, lake outlines), label leaders and label engraving — all in model
    coordinates, ready to be split, nested and rendered.
    """
    from .symbology import label_geometry_for_layer, leader_geometry_for_layer

    parts: list[Part] = []
    for layer in model.layers:
        k = layer.index
        part = Part(part_id=part_id(k, 0), kind="ply",
                    material=material_for_layer(k, cfg.materials),
                    outline=_as_multipolygon(layer.geometry), layer_index=k)

        above = model.footprint(k + 1)
        if not above.is_empty:
            part.add_op(OP_SCORE_REGISTRATION, above.boundary.intersection(layer.geometry))

        if symbology is not None:
            sym = symbology.for_layer(k)
            if sym is not None:
                part.add_op(OP_SCORE_HYDRO, sym.rivers)
                part.add_op(OP_SCORE_HYDRO, sym.lake_outlines)
                part.add_op(OP_SCORE_TRAIL, sym.trails)
                part.add_op(OP_ENGRAVE, sym.trail_labels)
                part.add_op(OP_ENGRAVE, sym.icons)
            part.add_op(OP_SCORE_IDS, leader_geometry_for_layer(symbology.labels, k))
            part.add_op(OP_ENGRAVE, label_geometry_for_layer(symbology.labels, k))
        parts.append(part)
    return parts


def _as_multipolygon(geom: BaseGeometry) -> MultiPolygon:
    from shapely.geometry import Polygon

    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(parts)
