# all.py (backward-compatibility shim)
"""DEPRECATED: This orchestration script has been removed.

Public exports are available via 'from borunte import ...'.
For CLI orchestration, see borunte.cli modules.

This file provides minimal backward compatibility. It will be removed in a future version.
"""

from warnings import warn

warn(
    "all.py is deprecated and its functionality has been distributed to borunte modules",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export common symbols from borunte package
from borunte import *  # noqa: F401, F403
