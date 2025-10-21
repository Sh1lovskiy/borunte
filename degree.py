# degree.py (backward-compatibility shim)
"""DEPRECATED: Use borunte.core.geometry.angles instead.

This file provides backward compatibility. It will be removed in a future version.
"""

from warnings import warn

# Re-export all functions from new location
from borunte.core.geometry.angles import *  # noqa: F401, F403

warn(
    "degree.py is deprecated; use 'from borunte.core.geometry.angles import ...' instead",
    DeprecationWarning,
    stacklevel=2,
)
