"""The terrain bundle — the neutral contract between fictional sources and core.

All three fictional tiers (extractable game terrain, 3D meshes, 2D artwork)
converge on one on-disk format that the core consumes like any DEM::

    world.terrainbundle/
      heightmap.png   16-bit grayscale, one value per cell
      water.png       optional water mask, same grid
      meta.json       units-per-pixel, height scale/offset, origin, attribution
      features.csv    name, type, x, y, elev?, include, icon?, label_override

Fictional coordinates are **flat** — a plain affine (world-units → mm), no CRS.
Fictional elevation units are meaningless, so vertical scale flows through the
core's ``normalize_layers`` mode, not fake meters. This module only *reads*
bundles (using core deps only); adapters write them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from affine import Affine
from shapely.geometry import Point

from .dem.base import FLAT_CRS, DemRaster
from .features.schema import FeatureCollection, FeatureType, GeoFeature

# Sentinel EPSG on a FeatureCollection meaning "already in flat world units".
FLAT_EPSG = 0

__all__ = ["FLAT_CRS", "FLAT_EPSG", "BundleMeta", "TerrainBundle", "BundleError",
           "load_bundle", "read_gray", "write_gray16", "write_mask", "hillshade",
           "save_hillshade"]

# Bundle feature-type strings → core FeatureType.
_FEATURE_TYPES = {
    "peak": FeatureType.PEAK, "mountain": FeatureType.PEAK,
    "lake": FeatureType.LAKE, "water": FeatureType.LAKE,
    "river": FeatureType.RIVER,
    "settlement": FeatureType.PLACE, "town": FeatureType.PLACE,
    "city": FeatureType.PLACE, "village": FeatureType.PLACE,
    "poi": FeatureType.PLACE, "place": FeatureType.PLACE,
    "region": FeatureType.PLACE,
}


class BundleError(ValueError):
    """Raised when a terrain bundle is missing or malformed."""


@dataclass
class BundleMeta:
    """Georeferencing + provenance for a bundle (all fictional, all flat)."""

    units_per_pixel: float = 1.0          # world units per cell (square cells)
    height_scale: float = 1.0             # elev = pixel * scale + offset
    height_offset: float = 0.0
    world_origin: tuple[float, float] = (0.0, 0.0)  # world (x, y) of pixel (0,0)
    vertical_range: list[float] | None = None       # suggested [lo, hi], advisory
    attribution: str = ""
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> BundleMeta:
        origin = d.get("world_origin", [0.0, 0.0])
        return cls(
            units_per_pixel=float(d.get("units_per_pixel", 1.0)),
            height_scale=float(d.get("height_scale", 1.0)),
            height_offset=float(d.get("height_offset", 0.0)),
            world_origin=(float(origin[0]), float(origin[1])),
            vertical_range=d.get("vertical_range"),
            attribution=str(d.get("attribution", "")),
            source=str(d.get("source", "")),
        )

    def to_dict(self) -> dict:
        return {
            "units_per_pixel": self.units_per_pixel,
            "height_scale": self.height_scale,
            "height_offset": self.height_offset,
            "world_origin": list(self.world_origin),
            "vertical_range": self.vertical_range,
            "attribution": self.attribution,
            "source": self.source,
        }


@dataclass
class TerrainBundle:
    meta: BundleMeta
    heightmap: np.ndarray                  # raw grid (float), pre scale/offset
    water: np.ndarray | None = None        # bool mask, same grid, True = water
    path: Path | None = None
    features_path: Path | None = None
    rivers_path: Path | None = None        # optional rivers.geojson (world coords)

    @property
    def shape(self) -> tuple[int, int]:
        return self.heightmap.shape

    def elevation(self) -> np.ndarray:
        """Heightmap mapped to world elevation via meta scale/offset."""
        return self.heightmap.astype("float64") * self.meta.height_scale \
            + self.meta.height_offset

    def transform(self) -> Affine:
        """Pixel → world affine (north-up: row 0 at max y)."""
        upp = self.meta.units_per_pixel
        ox, oy = self.meta.world_origin
        return Affine.translation(ox, oy) * Affine.scale(upp, -upp)

    def to_dem(self) -> DemRaster:
        """A flat DemRaster (crs=FLAT) the core builds layers from directly."""
        return DemRaster(data=self.elevation().astype("float32"),
                         transform=self.transform(), crs=FLAT_CRS, nodata=None)

    def water_polygons(self):
        """Water mask → shapely polygons in world coordinates (for lake insets)."""
        if self.water is None or not self.water.any():
            return []
        from rasterio.features import shapes
        from shapely.geometry import shape

        mask = self.water.astype("uint8")
        polys = []
        for geom, val in shapes(mask, mask=self.water, transform=self.transform()):
            if val:
                poly = shape(geom)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty and poly.area > 0:
                    polys.append(poly)
        return polys

    def features(self) -> FeatureCollection:
        """Load ``features.csv`` (world coords) into a flat FeatureCollection."""
        coll = FeatureCollection(epsg=FLAT_EPSG)
        if not self.features_path or not self.features_path.is_file():
            return coll
        with self.features_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                feat = _row_to_feature(row)
                if feat is not None:
                    coll.add(feat)
        return coll

    def rivers(self) -> list:
        """River linework (world coords) from an optional ``rivers.geojson``.

        features.csv is point-per-row, so real river polylines — which the core
        scores per layer and which drive carving — travel in a separate GeoJSON
        of LineString/MultiLineString features.
        """
        if not self.rivers_path or not self.rivers_path.is_file():
            return []
        from shapely.geometry import shape

        data = json.loads(self.rivers_path.read_text())
        feats = data.get("features", []) if isinstance(data, dict) else []
        lines = []
        for f in feats:
            geom = f.get("geometry") if isinstance(f, dict) else None
            if not geom:
                continue
            g = shape(geom)
            name = (f.get("properties") or {}).get("name")
            if g.geom_type in ("LineString", "MultiLineString") and not g.is_empty:
                lines.append((g, name))
        return lines


def _row_to_feature(row: dict) -> GeoFeature | None:
    include = (row.get("include") or "yes").strip().lower()
    if include in {"no", "false", "0"}:
        return None
    try:
        x, y = float(row["x"]), float(row["y"])
    except (KeyError, ValueError):
        return None
    ftype = _FEATURE_TYPES.get((row.get("type") or "").strip().lower(),
                               FeatureType.PLACE)
    elev = _maybe_float(row.get("elev"))
    tags = {}
    icon = (row.get("icon") or "").strip()
    if icon:
        tags["icon"] = icon
    label = (row.get("label_override") or "").strip() or (row.get("name") or "").strip()
    return GeoFeature(feature_type=ftype, geometry=Point(x, y),
                      name=label or None, elevation=elev, importance=elev,
                      tags=tags)


def _maybe_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --- loading & validation --------------------------------------------------

def load_bundle(path: str | Path) -> TerrainBundle:
    """Load and validate a ``*.terrainbundle`` directory."""
    p = Path(path)
    if not p.is_dir():
        raise BundleError(f"terrain bundle not found (expected a directory): {p}")
    hm = p / "heightmap.png"
    if not hm.is_file():
        raise BundleError(f"bundle missing heightmap.png: {p}")

    meta_path = p / "meta.json"
    meta = BundleMeta.from_dict(json.loads(meta_path.read_text())) \
        if meta_path.is_file() else BundleMeta()

    heightmap = read_gray(hm)
    water_path = p / "water.png"
    water = None
    if water_path.is_file():
        wm = read_gray(water_path)
        if wm.shape != heightmap.shape:
            raise BundleError("water.png grid does not match heightmap.png")
        water = wm > 0

    features_path = p / "features.csv"
    rivers_path = p / "rivers.geojson"
    return TerrainBundle(
        meta=meta, heightmap=heightmap, water=water, path=p,
        features_path=features_path if features_path.is_file() else None,
        rivers_path=rivers_path if rivers_path.is_file() else None)


def read_gray(path: str | Path) -> np.ndarray:
    """Read a grayscale PNG (8- or 16-bit) as a 2D numpy array."""
    from PIL import Image

    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:                      # collapse accidental RGB to luminance
        arr = arr[..., 0]
    return arr


def write_mask(path: str | Path, mask: np.ndarray) -> Path:
    """Write a boolean/0-1 mask as a 16-bit PNG (0 or 65535), no normalization.

    ``write_gray16`` min/max-normalizes, which would turn an all-True mask into
    all zeros — use this for water/binary masks so nonzero always means 'set'.
    """
    from PIL import Image

    arr = (np.asarray(mask).astype(bool)).astype("uint16") * 65535
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(p)
    return p


def write_gray16(path: str | Path, data: np.ndarray) -> Path:
    """Write a float/int array as a 16-bit grayscale PNG (used by adapters)."""
    from PIL import Image

    arr = np.asarray(data)
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    scaled = np.zeros_like(arr, dtype="float64") if hi <= lo else \
        (arr.astype("float64") - lo) / (hi - lo) * 65535.0
    img = Image.fromarray(np.clip(np.nan_to_num(scaled), 0, 65535).astype("uint16"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


# --- hillshade preview -----------------------------------------------------

def hillshade(elev: np.ndarray, azimuth_deg: float = 315.0,
              altitude_deg: float = 45.0) -> np.ndarray:
    """A numpy gradient-shaded relief image (uint8) for validation previews."""
    z = np.nan_to_num(elev.astype("float64"))
    dy, dx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth_deg + 90.0)
    alt = np.radians(altitude_deg)
    shaded = np.sin(alt) * np.sin(slope) + \
        np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shaded + 1) / 2 * 255, 0, 255).astype("uint8")


def save_hillshade(elev: np.ndarray, path: str | Path) -> Path:
    from PIL import Image

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(hillshade(elev)).save(p)
    return p
