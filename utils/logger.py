# utils/logger.py
"""Project logging utilities with a consistent loguru configuration."""

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar, cast

from loguru import logger as _logger
from loguru._logger import Logger as LoguruLogger

try:  # pragma: no cover - optional config import
    from config import get_settings
except Exception:  # noqa: BLE001
    get_settings = None  # type: ignore

T = TypeVar("T")

_DEFAULT_LEVEL = os.environ.get("BORUNTE_LOG_LEVEL", "INFO")


@dataclass(slots=True)
class _LogOptions:
    level: str = _DEFAULT_LEVEL
    json: bool = False
    bar_format: str = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


_OPTIONS = _LogOptions()
_CONFIGURED = False
_LOG_FILE: Optional[Path] = None


def _resolve_log_dir() -> Path:
    if get_settings is not None:
        try:
            return get_settings().paths.logs_root
        except Exception:  # pragma: no cover - fallback
            pass
    return Path.cwd() / "logs"


def _configure_logger(level: str | None = None, json: bool | None = None) -> None:
    global _CONFIGURED, _LOG_FILE
    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = log_dir / f"borunte_{timestamp}.log"
    _logger.remove()
    _logger.add(
        sink=file_path,
        level=level or _OPTIONS.level,
        serialize=json if json is not None else _OPTIONS.json,
        format="{time} | {level} | {extra[module]} | {message}",
    )
    _logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level or _OPTIONS.level,
        serialize=False,
        format="{time:HH:mm:ss} | {level} | {extra[module]} | {message}",
    )
    _CONFIGURED = True
    _LOG_FILE = file_path


def get_logger(name: str | None = None) -> LoguruLogger:
    if not _CONFIGURED:
        _configure_logger()
    frame = inspect.currentframe()
    module_name = name
    if module_name is None and frame is not None:
        caller_frame = frame.f_back
        if caller_frame is not None:
            module = inspect.getmodule(caller_frame)
            if module is not None and module.__name__ != "__main__":
                module_name = module.__name__
    bound = _logger.bind(module=module_name or "unknown")
    if not hasattr(bound, "warn"):
        setattr(bound, "warn", bound.warning)
    return bound


def configure(level: str | None = None, json: bool | None = None) -> None:
    _configure_logger(level=level, json=json)


@contextmanager
def logging_context(*, level: str | None = None, json: bool | None = None) -> Iterator[LoguruLogger]:
    configure(level=level, json=json)
    logger = get_logger()
    yield logger


def iter_progress(iterable: Iterable[T], *, description: str | None = None, total: int | None = None) -> Iterable[T]:
    from utils.progress import track

    return cast(Iterable[T], track(iterable, description=description, total=total))


__all__ = [
    "LoguruLogger",
    "configure",
    "get_logger",
    "iter_progress",
    "logging_context",
]
