# utils/__init__.py
"""Shared helper modules used across the project.

The :mod:`utils` package contains lightweight helpers for logging, CLI
dispatching and simple file I/O. These utilities are used by most other
packages and avoid additional dependencies.
"""

from .logger import Logger, LoggerType


__all__ = [
    "Logger",
    "LoggerType",
]
