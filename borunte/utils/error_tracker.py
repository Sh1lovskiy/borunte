# utils/error_tracker.py
"""Centralised error tracking utility for pipelines and services."""

from __future__ import annotations

import atexit
import signal
import sys
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType
from typing import ClassVar

from borunte.utils.logger import get_logger

CleanupFn = Callable[[], None]


@dataclass(slots=True)
class ErrorTracker:
    """Collect exceptions and contextual information during execution."""

    context: str = "ErrorTracker"
    errors: dict[str, list[str]] = field(default_factory=dict)

    # ────────────── collection API ──────────────
    def record(self, key: str, message: str) -> None:
        logger = get_logger(self.context)
        logger.error(f"{key}: {message}")
        self.errors.setdefault(key, []).append(message)

    def summary(self) -> dict[str, list[str]]:
        logger = get_logger(self.context)
        if not self.errors:
            logger.info("No errors recorded")
            return {}
        for key, messages in self.errors.items():
            logger.warning(f"Encountered {len(messages)} issues for {key}")
        return dict(self.errors)

    # ────────────── runtime plumbing ──────────────
    _cleanups: ClassVar[list[CleanupFn]] = []
    _installed: ClassVar[bool] = False
    _cleanup_running: ClassVar[bool] = False

    @classmethod
    def register_cleanup(cls, fn: CleanupFn) -> None:
        cls._cleanups.append(fn)

    @classmethod
    def _run_cleanups(cls) -> None:
        # Защита от рекурсивных вызовов
        if cls._cleanup_running:
            return
        cls._cleanup_running = True

        logger = get_logger("ErrorTracker")
        for fn in cls._cleanups[:]:
            try:
                fn()
            except Exception as exc:
                logger.warning(f"cleanup failed: {exc}")

        cls._cleanup_running = False

    @classmethod
    def install_excepthook(cls) -> None:
        if cls._installed:
            return
        cls._installed = True
        logger = get_logger("ErrorTracker")

        def _hook(exctype, value, tb):
            logger.error(
                f"Uncaught exception: {''.join(traceback.format_exception(exctype, value, tb))}"
            )
            cls._run_cleanups()
            sys.__excepthook__(exctype, value, tb)

        sys.excepthook = _hook
        atexit.register(cls._run_cleanups)
        logger.tag("ET", "excepthook installed")

    @classmethod
    def report(
        cls,
        exc: BaseException,
        *,
        key: str = "exception",
        context: str = "ErrorTracker",
    ) -> None:
        """
        Back-compat helper used as ErrorTracker.report(exc).
        Logs error and stores message into a transient tracker.
        """
        logger = get_logger(context)
        logger.error(f"{key}: {exc}")
        try:
            tracker = cls(context=context)
            tracker.record(key, f"{type(exc).__name__}: {exc}")
        except Exception:
            pass

    @classmethod
    def install_signal_handlers(cls) -> None:
        logger = get_logger("ErrorTracker")

        signal_count = [0]  # Mutable counter для отслеживания повторных сигналов

        def _handler(signum: int, frame: FrameType | None) -> None:
            signal_count[0] += 1

            # При первом сигнале пытаемся graceful cleanup
            if signal_count[0] == 1:
                logger.warning(f"signal {signum} received, running cleanups")
                cls._run_cleanups()
                sys.exit(128 + signum)

            # При повторных сигналах (если cleanup завис) - жесткий выход
            elif signal_count[0] == 2:
                logger.warning(f"second signal {signum}, forcing exit")
                sys.exit(128 + signum)

            # Третий и далее - немедленный _exit без cleanup
            else:
                logger.warning(f"multiple signals ({signal_count[0]}), hard exit")
                import os

                os._exit(128 + signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except Exception:
                pass
        logger.tag("ET", "signal handlers installed")

    @classmethod
    def install_keyboard_listener(cls, key: str = "esc") -> None:
        logger = get_logger("ErrorTracker")
        logger.tag("ET", f"keyboard listener (noop) = {key}")


# ────────────── context manager API ──────────────


@contextmanager
def error_scope(name: str = "scope"):
    """Wrap a block to log exceptions and keep going (if desired)."""
    logger = get_logger("ErrorScope")
    try:
        yield
    except Exception:
        logger.error(f"Traceback:\n{traceback.format_exc()}")


__all__ = ["ErrorTracker", "error_scope"]
