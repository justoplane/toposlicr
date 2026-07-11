"""toposlicr — automated laser-cut topographic map pipeline.

The package is organized as a core library of stateless, JSON-serializable stage
functions with a thin CLI wrapper (``toposlicr.cli``), so the same core can back
a future web front-end. Phase 0 ships the geographic/scale math and the config
schema; later phases add DEM acquisition, contouring, symbology, panelization,
nesting and SVG/guide output.
"""

from __future__ import annotations

__version__ = "0.0.1"

from .config import Config, ConfigError, load_config, parse_config
from .geo import BBox
from .scale import ScaleResult, SolveMode, solve_scale

__all__ = [
    "__version__",
    "BBox",
    "Config",
    "ConfigError",
    "load_config",
    "parse_config",
    "ScaleResult",
    "SolveMode",
    "solve_scale",
]
