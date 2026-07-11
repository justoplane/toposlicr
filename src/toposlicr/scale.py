"""Scale, exaggeration & layer-band math (Phase 0).

This is the analytical heart of the pipeline and is deliberately dependency-free
and pure so it can be unit-tested in isolation and reused unchanged by the CLI
today and a web front-end later.

Core relationships (see plan Section 2)::

    S = model_width_mm / (real_width_m * 1000)             # physical scale, S = 1/N
    interval_real_m = ply_thickness_mm / (S * E * 1000)    # elevation per ply layer

Note the ``* 1000`` in the interval formula: the plan states it as
``ply_thickness_mm / (S * E)``, but that mixes millimeters (thickness) with
meters (interval). One ply layer's *model* height equals the *real* interval
scaled down and exaggerated::

    ply_thickness_mm = interval_real_m * 1000 * S * E

so recovering meters from millimeters needs the extra factor of 1000. Sanity
check with the plan's own example — 40 km at 500 mm (S = 1:80000), 3 mm ply,
E = 1 → interval = 240 m, and 240 m x S = 3 mm exactly.

The user pins any one of {exaggeration E, contour interval, layer count} and we
solve for the rest. ``S`` itself is fixed the moment the model width and the
real-world extent are known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

# Interval values (meters) considered "nice" to round to when the tool is free
# to pick the contour interval. Ordered ascending.
_NICE_INTERVALS_M = (
    5, 10, 20, 25, 50, 100, 125, 150, 200, 250, 500, 1000, 2000, 2500, 5000,
)

# Above this many layers a build gets impractical to cut, glue and align.
DEFAULT_MAX_LAYERS = 25


class SolveMode(StrEnum):
    """Which pair of values the user pinned; the third is derived."""

    EXAGGERATION = "exaggeration"  # pin E -> derive interval (snapped) -> actual E
    LAYER_COUNT = "layer_count"    # pin layer count -> derive interval -> E
    INTERVAL = "interval"          # pin interval -> derive E


@dataclass(frozen=True)
class ScaleResult:
    """The fully-solved scale/exaggeration/banding report."""

    mode: SolveMode
    scale_denominator: float          # the N in "1:N"
    real_width_m: float
    model_width_mm: float
    ply_thickness_mm: float
    interval_m: float                 # elevation covered by one ply layer
    exaggeration: float               # actual E after any interval snapping
    requested_exaggeration: float | None = None
    layer_count: int | None = None    # None when elevation range not supplied
    base_elev_m: float | None = None
    max_elev_m: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def scale_ratio(self) -> float:
        """S = 1 / N."""
        return 1.0 / self.scale_denominator

    def scale_label(self) -> str:
        return f"1:{round(self.scale_denominator):,}"


def snap_interval(interval_m: float) -> float:
    """Snap a raw interval to the nearest cartographically 'nice' value."""
    if interval_m <= 0:
        raise ValueError("interval must be positive")
    # Nearest on a log scale so 60 -> 50 rather than being pulled toward 100.
    best = min(_NICE_INTERVALS_M, key=lambda v: abs(math.log(v) - math.log(interval_m)))
    return float(best)


def scale_denominator(model_width_mm: float, real_width_m: float) -> float:
    """N in 1:N, from the model width and the real-world extent it represents."""
    if model_width_mm <= 0:
        raise ValueError("model_width_mm must be positive")
    if real_width_m <= 0:
        raise ValueError("real_width_m must be positive")
    return (real_width_m * 1000.0) / model_width_mm


# The single home of the mm↔m unit factor. See the module docstring.
_MM_PER_M = 1000.0


def interval_from_exaggeration(ply_thickness_mm: float, scale_ratio: float, e: float) -> float:
    """Real-world elevation (m) that one ply layer represents at exaggeration ``e``."""
    return ply_thickness_mm / (scale_ratio * e * _MM_PER_M)


def exaggeration_from_interval(
    ply_thickness_mm: float, scale_ratio: float, interval_m: float
) -> float:
    """Vertical exaggeration implied by a chosen contour interval (m)."""
    return ply_thickness_mm / (scale_ratio * interval_m * _MM_PER_M)


def _layer_count(base_elev_m: float, max_elev_m: float, interval_m: float) -> int:
    span = max_elev_m - base_elev_m
    if span <= 0:
        raise ValueError("max_elev_m must be greater than base_elev_m")
    # Number of stacked bands whose thresholds fit within the span.
    return max(1, int(span // interval_m))


def solve_scale(
    *,
    model_width_mm: float,
    real_width_m: float,
    ply_thickness_mm: float,
    exaggeration: float | None = None,
    interval_m: float | None = None,
    layer_count: int | None = None,
    base_elev_m: float | None = None,
    max_elev_m: float | None = None,
    snap: bool = True,
    max_layers: int = DEFAULT_MAX_LAYERS,
    min_top_area_frac: float = 0.0,
) -> ScaleResult:
    """Solve the scale/exaggeration/banding relationship.

    Exactly one of ``exaggeration``, ``interval_m`` or ``layer_count`` selects
    the solve mode. ``base_elev_m``/``max_elev_m`` are optional at Phase 0 (the
    DEM that supplies them arrives in Phase 1); when both are given the layer
    count and practicality warnings are reported.
    """
    pinned = [
        ("exaggeration", exaggeration),
        ("interval_m", interval_m),
        ("layer_count", layer_count),
    ]
    provided = [name for name, val in pinned if val is not None]
    if len(provided) != 1:
        raise ValueError(
            "provide exactly one of exaggeration / interval_m / layer_count; "
            f"got {provided or 'none'}"
        )

    if ply_thickness_mm <= 0:
        raise ValueError("ply_thickness_mm must be positive")

    n = scale_denominator(model_width_mm, real_width_m)
    s = 1.0 / n
    warnings: list[str] = []

    if layer_count is not None:
        mode = SolveMode.LAYER_COUNT
        if base_elev_m is None or max_elev_m is None:
            raise ValueError("layer_count mode requires base_elev_m and max_elev_m")
        if layer_count < 1:
            raise ValueError("layer_count must be >= 1")
        raw_interval = (max_elev_m - base_elev_m) / layer_count
        interval = snap_interval(raw_interval) if snap else raw_interval
        e = exaggeration_from_interval(ply_thickness_mm, s, interval)
        requested_e = None
    elif interval_m is not None:
        mode = SolveMode.INTERVAL
        if interval_m <= 0:
            raise ValueError("interval_m must be positive")
        interval = float(interval_m)  # honor the user's exact interval, no snap
        e = exaggeration_from_interval(ply_thickness_mm, s, interval)
        requested_e = None
    else:
        mode = SolveMode.EXAGGERATION
        assert exaggeration is not None
        if exaggeration <= 0:
            raise ValueError("exaggeration must be positive")
        requested_e = float(exaggeration)
        raw_interval = interval_from_exaggeration(ply_thickness_mm, s, exaggeration)
        interval = snap_interval(raw_interval) if snap else raw_interval
        # Snapping the interval changes the true exaggeration; report the actual.
        e = exaggeration_from_interval(ply_thickness_mm, s, interval)

    count: int | None = None
    if base_elev_m is not None and max_elev_m is not None:
        count = _layer_count(base_elev_m, max_elev_m, interval)
        if count > max_layers:
            warnings.append(
                f"{count} layers exceeds the practical limit of {max_layers}; "
                "consider a larger interval, lower exaggeration, or thicker stock."
            )
        if count < 2:
            warnings.append(
                f"only {count} layer(s) — the interval is larger than the terrain's "
                "elevation range; lower the interval or raise exaggeration."
            )

    if mode is SolveMode.EXAGGERATION and requested_e is not None:
        drift = abs(e - requested_e) / requested_e
        if drift > 0.15:
            warnings.append(
                f"snapped interval shifts exaggeration from requested {requested_e:.2f} "
                f"to {e:.2f} (>{drift * 100:.0f}%); pin interval_m for an exact E."
            )

    return ScaleResult(
        mode=mode,
        scale_denominator=n,
        real_width_m=real_width_m,
        model_width_mm=model_width_mm,
        ply_thickness_mm=ply_thickness_mm,
        interval_m=interval,
        exaggeration=e,
        requested_exaggeration=requested_e,
        layer_count=count,
        base_elev_m=base_elev_m,
        max_elev_m=max_elev_m,
        warnings=warnings,
    )
