# utils/progress.py
"""Shared tqdm wrapper to ensure consistent progress indicators."""

from __future__ import annotations

from typing import Iterable, Iterator, TypeVar, overload

from tqdm.auto import tqdm

T = TypeVar("T")


@overload
def track(iterable: Iterable[T], *, description: str | None = None, total: int | None = None) -> Iterator[T]:
    ...


def track(iterable: Iterable[T], *, description: str | None = None, total: int | None = None) -> Iterator[T]:
    yield from tqdm(
        iterable,
        desc=description,
        total=total,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )


__all__ = ["track"]
