# utils/format.py
"""Formatting helpers for NumPy outputs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def numpy_print_options(*, precision: int = 4, suppress: bool = True) -> Iterator[None]:
    original = np.get_printoptions()
    np.set_printoptions(precision=precision, suppress=suppress)
    try:
        yield
    finally:
        np.set_printoptions(**original)


__all__ = ["numpy_print_options"]
