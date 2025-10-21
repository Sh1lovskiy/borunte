# merge.py (backward-compatibility shim)
"""DEPRECATED: Use borunte.vision.merge instead.

This file provides backward compatibility. It will be removed in a future version.
"""

from warnings import warn

from borunte.vision.merge import main

if __name__ == "__main__":
    warn(
        "merge.py is deprecated; use 'uv run -m borunte.vision.merge' instead",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
