# new5.py (backward-compatibility shim)
"""DEPRECATED: Use borunte.cli.capture_runner instead.

This file provides backward compatibility. It will be removed in a future version.
"""

from warnings import warn

from borunte.cli.capture_runner import main

if __name__ == "__main__":
    warn(
        "new5.py is deprecated; use 'uv run -m borunte.cli.capture_runner' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
