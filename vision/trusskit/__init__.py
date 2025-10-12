"""TrussKit package facade.

Exports submodules lazily so both:
  - from trusskit import skeleton2d
  - from trusskit.pipeline import run
work without touching heavy deps until used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Public API (modules + types)
__all__ = [
    "RunConfig",
    "pipeline",
    "skeleton2d",
    "plane",
    "project2d",
    "nodes_edges",
    "graph_build",
    "graph_traverse",
    "mesh",
    "viewer",
    "io",
    "transforms",
    "regions",
    "normals",
    "overlays",
]

# Light types re-export (no heavy deps here)
try:
    from .config import RunConfig  # type: ignore
except Exception:
    # Keep package import resilient if config is missing.
    class RunConfig:  # type: ignore
        pass


def __getattr__(name: str) -> Any:
    """Lazy import submodules on first attribute access."""
    if name in __all__ and name != "RunConfig":
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
