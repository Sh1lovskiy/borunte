# utils/format.py
"""Formatting helpers for NumPy outputs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt


@contextmanager
def numpy_print_options(*, precision: int = 4, suppress: bool = True) -> Iterator[None]:
    original = np.get_printoptions()
    np.set_printoptions(precision=precision, suppress=suppress)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def format_matrix(arr: npt.NDArray[np.float64], precision: int = 6) -> str:
    """Format a NumPy array as a clean multi-line string.

    Args:
        arr: NumPy array to format
        precision: Number of decimal places

    Returns:
        Formatted string representation
    """
    with numpy_print_options(precision=precision, suppress=True):
        return str(arr)


__all__ = ["numpy_print_options", "format_matrix"]
