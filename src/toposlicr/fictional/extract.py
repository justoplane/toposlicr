"""Tier 3 — feature extraction from 2D artwork (spec Section 3.1).

Lifts the raw ingredients of a heightfield out of a fantasy/game map image:
land/sea segmentation, coastlines, river polylines, a painted terrain-class
overlay, an adjustment layer, and (optionally) OCR'd labels. These are the
*inputs* to synthesis (`fictional.synthesis`); an art adapter orchestrates them.

Public functions operate on numpy image arrays in **RGB** order (H, W, 3),
uint8. Heavy optional backends (SAM, OCR) are gated behind lazy imports that
raise a clear "install the extra" error only when actually called.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "segment_land_sea",
    "coastline",
    "extract_rivers",
    "classify_river_mouths",
    "read_class_overlay",
    "read_adjustment_layer",
    "ocr_labels",
]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _to_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce an image to HxWx3 uint8 RGB (drop alpha, expand grayscale)."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _read_rgb(path) -> np.ndarray:
    """Read an image file as RGB (with alpha kept separately if present)."""
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if bgr.ndim == 2:
        return np.repeat(bgr[:, :, None], 3, axis=2).astype(np.uint8), None
    if bgr.shape[2] == 4:
        alpha = bgr[:, :, 3]
        rgb = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2RGB)
        return rgb.astype(np.uint8), alpha
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.uint8), None


def _morph_clean(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Open then close a boolean mask to remove speckle and fill pinholes."""
    import cv2

    m = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m.astype(bool)


# --------------------------------------------------------------------------
# land / sea segmentation
# --------------------------------------------------------------------------

def segment_land_sea(img: np.ndarray, method: str = "auto",
                     ocean_point: tuple[int, int] | None = None,
                     k: int = 5, tolerance: int = 30) -> np.ndarray:
    """Return a boolean **land** mask (True = land) from a map image.

    * ``auto``/``kmeans`` — LAB k-means; the water cluster is the one that both
      hugs the map border (oceans touch the edge) and is bluest.
    * ``flood_fill`` — flood-fill the sea from ``ocean_point`` (x, y).
    * ``sam`` — optional Segment Anything backend (not installed here).
    """
    rgb = _to_rgb(img)
    if method in ("auto", "kmeans"):
        water = _kmeans_water(rgb, k)
    elif method == "flood_fill":
        water = _floodfill_water(rgb, ocean_point, tolerance)
    elif method == "sam":
        raise NotImplementedError(
            "SAM land/sea segmentation needs the optional 'sam' extra "
            "(segment-anything); it is not installed. Use method='auto' or "
            "'flood_fill'.")
    else:
        raise ValueError(f"unknown segmentation method: {method!r}")
    return _morph_clean(~water)


def _kmeans_water(rgb: np.ndarray, k: int) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = max(2, min(k, len(np.unique(data, axis=0))))
    _, labels, _ = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(h, w)

    border = np.concatenate([labels[0, :], labels[-1, :],
                             labels[:, 0], labels[:, -1]])
    rgbf = rgb.astype(np.float32)
    best, best_score = 0, -1e9
    for c in range(k):
        m = labels == c
        if not m.any():
            continue
        border_frac = float((border == c).mean())
        mean = rgbf[m].mean(axis=0)
        blueness = float((mean[2] - (mean[0] + mean[1]) / 2.0) / 255.0)
        score = border_frac + 0.5 * max(0.0, blueness)
        if score > best_score:
            best, best_score = c, score
    return labels == best


def _floodfill_water(rgb: np.ndarray, ocean_point, tolerance: int) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    if ocean_point is None:
        ocean_point = (0, 0)
    seed = (int(ocean_point[0]), int(ocean_point[1]))
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(rgb.copy(), mask, seed, 0,
                  (tolerance,) * 3, (tolerance,) * 3, flags)
    return mask[1:-1, 1:-1].astype(bool)


# --------------------------------------------------------------------------
# coastline
# --------------------------------------------------------------------------

def coastline(land_mask: np.ndarray) -> list[np.ndarray]:
    """Coastline polygons as (N, 2) (row, col) arrays via marching squares."""
    from skimage import measure

    return measure.find_contours(land_mask.astype(float), 0.5)


# --------------------------------------------------------------------------
# rivers
# --------------------------------------------------------------------------

def extract_rivers(img: np.ndarray, river_color: tuple[int, int, int] | None = None,
                   darkness: float | None = None):
    """Trace river linework to simplified (N, 2) (row, col) polylines.

    Rivers are masked either by proximity to ``river_color`` (RGB) or, by
    default, by darkness (ink lines on lighter paper). The mask is skeletonized
    and the skeleton vectorized to polylines. Returns ``(polylines, warnings)``.
    """
    from skimage.measure import approximate_polygon
    from skimage.morphology import skeletonize

    rgb = _to_rgb(img)
    warnings: list[str] = []
    if river_color is not None:
        thr = 40.0 if darkness is None else float(darkness)
        dist = np.linalg.norm(rgb.astype(np.float32) - np.array(river_color, np.float32),
                              axis=2)
        mask = dist < thr
    else:
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        thr = 100.0 if darkness is None else float(darkness)
        mask = gray < thr

    if not mask.any():
        return [], ["no river pixels matched the mask threshold"]

    skel = skeletonize(mask)
    raw, w2 = _skeleton_to_polylines(skel)
    warnings.extend(w2)
    polylines = []
    for p in raw:
        simp = approximate_polygon(p, tolerance=2.0)
        polylines.append(simp if len(simp) >= 2 else p)
    return polylines, warnings


def _skeleton_to_polylines(skel: np.ndarray):
    """Vectorize a 1-px skeleton into polylines (own skeleton→graph→paths)."""
    import networkx as nx

    coords = [tuple(p) for p in np.argwhere(skel)]
    warnings: list[str] = []
    if not coords:
        return [], warnings
    coord_set = set(coords)
    g = nx.Graph()
    g.add_nodes_from(coords)
    for (r, c) in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_set:
                    g.add_edge((r, c), nb)

    polylines: list[np.ndarray] = []
    visited: set = set()

    def _edge(a, b):
        return (a, b) if a <= b else (b, a)

    specials = [n for n in g.nodes if g.degree(n) != 2]
    for s in specials:
        for nb in list(g.neighbors(s)):
            if _edge(s, nb) in visited:
                continue
            path = [s]
            prev, cur = s, nb
            visited.add(_edge(prev, cur))
            while True:
                path.append(cur)
                if g.degree(cur) != 2:
                    break
                nxt = next((x for x in g.neighbors(cur) if x != prev), None)
                if nxt is None or _edge(cur, nxt) in visited:
                    break
                visited.add(_edge(cur, nxt))
                prev, cur = cur, nxt
            if len(path) >= 2:
                polylines.append(np.array(path, dtype=float))

    # Pure loops (components with every node degree 2) leave no special node.
    for comp in nx.connected_components(g):
        untraced = any(_edge(*e) not in visited for e in _component_edges(g, comp))
        if untraced and all(g.degree(n) == 2 for n in comp):
            loop = _trace_loop(g, comp, visited, _edge)
            if loop is not None and len(loop) >= 2:
                polylines.append(np.array(loop, dtype=float))
    return polylines, warnings


def _component_edges(g, comp):
    sub = g.subgraph(comp)
    return list(sub.edges())


def _trace_loop(g, comp, visited, edge_fn):
    start = next(iter(comp))
    nb = next((x for x in g.neighbors(start) if edge_fn(start, x) not in visited), None)
    if nb is None:
        return None
    path = [start]
    prev, cur = start, nb
    visited.add(edge_fn(prev, cur))
    while cur != start:
        path.append(cur)
        nxt = next((x for x in g.neighbors(cur) if x != prev), None)
        if nxt is None:
            break
        visited.add(edge_fn(cur, nxt))
        prev, cur = cur, nxt
    path.append(start)
    return path


def classify_river_mouths(polylines, land_mask: np.ndarray) -> list[int]:
    """For each polyline, which end (0=first, 1=last) is the sea/lake mouth.

    The mouth is the end nearest to water. ``land_mask`` is True on land, so the
    distance transform of the land mask gives each cell's distance to the sea.
    """
    from scipy import ndimage

    dist_to_sea = ndimage.distance_transform_edt(land_mask)
    h, w = land_mask.shape
    out: list[int] = []
    for poly in polylines:
        if len(poly) < 2:
            out.append(1)
            continue
        d = []
        for end in (poly[0], poly[-1]):
            r = int(np.clip(round(end[0]), 0, h - 1))
            c = int(np.clip(round(end[1]), 0, w - 1))
            d.append(dist_to_sea[r, c])
        out.append(0 if d[0] <= d[1] else 1)
    return out


# --------------------------------------------------------------------------
# painted overlays
# --------------------------------------------------------------------------

_DEFAULT_PALETTE = {
    "mountain": (210, 40, 40),   # red
    "hill": (230, 140, 30),      # orange
    "plateau": (230, 220, 40),   # yellow
}


def read_class_overlay(overlay_path, palette: dict | None = None,
                       tolerance: float = 70.0) -> dict[str, np.ndarray]:
    """Read a painted terrain-class overlay into boolean masks per class.

    Pixels within ``tolerance`` (RGB Euclidean) of a palette color join that
    class; transparent/blank pixels are plains (no class). The classic overlay
    is red = high mountains, orange = hills, yellow = plateau.
    """
    rgb, alpha = _read_rgb(overlay_path)
    palette = palette or _DEFAULT_PALETTE
    rgbf = rgb.astype(np.float32)
    painted = np.ones(rgb.shape[:2], dtype=bool)
    if alpha is not None:
        painted = alpha > 128
    masks: dict[str, np.ndarray] = {}
    for name, color in palette.items():
        dist = np.linalg.norm(rgbf - np.array(color, np.float32), axis=2)
        masks[name] = (dist < tolerance) & painted
    return masks


def read_adjustment_layer(path, feather_px: float = 6.0) -> np.ndarray:
    """Read a mid-gray adjustment layer into a signed delta field in ~[-1, 1].

    128 is neutral, lighter raises, darker lowers; the result is Gaussian
    feathered so edits blend rather than step.
    """
    from scipy.ndimage import gaussian_filter

    rgb, _ = _read_rgb(path)
    gray = rgb.mean(axis=2).astype(np.float32)
    delta = (gray - 128.0) / 127.0
    if feather_px > 0:
        delta = gaussian_filter(delta, feather_px)
    return delta


# --------------------------------------------------------------------------
# labels (optional OCR)
# --------------------------------------------------------------------------

def ocr_labels(img: np.ndarray) -> list[dict]:
    """Lift place names + pixel positions from the artwork via OCR.

    Requires the optional OCR extra (``pytesseract`` or ``paddleocr``); raises a
    clear error if neither is installed. Returns dicts ``{name, x, y}``.
    """
    rgb = _to_rgb(img)
    try:
        import pytesseract
    except ImportError:
        pytesseract = None
    if pytesseract is not None:
        data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
        out = []
        for i, text in enumerate(data["text"]):
            t = (text or "").strip()
            if t and int(data.get("conf", [0])[i] or 0) > 30:
                out.append({"name": t,
                            "x": int(data["left"][i] + data["width"][i] / 2),
                            "y": int(data["top"][i] + data["height"][i] / 2)})
        return out

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError(
            "OCR needs the optional 'ocr' extra (pytesseract or paddleocr); "
            "neither is installed. Install one, or type labels into features.csv "
            "manually.") from exc
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    out = []
    for line in ocr.ocr(rgb, cls=True)[0] or []:
        box, (text, conf) = line
        if text and conf > 0.3:
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            out.append({"name": text.strip(),
                        "x": int(sum(xs) / len(xs)), "y": int(sum(ys) / len(ys))})
    return out
