"""Web GUI for toposlicr — a thin FastAPI front-end over the core library."""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
