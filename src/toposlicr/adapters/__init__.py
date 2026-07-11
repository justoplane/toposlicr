"""Source adapters — thin standalone scripts that produce terrain bundles.

Each adapter converts a fictional source (extractable game terrain, a 3D mesh,
2D artwork, an Azgaar export) into a ``*.terrainbundle`` directory that the core
pipeline consumes like any DEM. Game-format churn, reverse-engineering, and
heavy optional dependencies stay here, out of the pipeline core.
"""
