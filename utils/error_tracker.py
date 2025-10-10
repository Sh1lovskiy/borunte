# utils/error_tracker.py
"""Centralized fatal and non-fatal error tracking with safe shutdown."""

from __future__ import annotations

import signal
import sys
import traceback
from contextlib import contextmanager
from typing import Any, Callable, List, Optional

from utils.logger import Logger


class CameraError(Exception):
    """Base class for camera related errors."""


class CameraConnectionError(CameraError):
    """Raised when the camera device cannot be opened."""


class ErrorTracker:
    """
    Global exception, signal, and keyboard handler.
    - Logs uncaught exceptions with traceback.
    - Runs all registered cleanup callbacks exactly once.
    - Provides fatal() to abort safely.
    """

    logger = Logger.get_logger()
    _installed = False
    _orig_hook: Optional[Callable[..., None]] = None
    _cleanup_funcs: List[Callable[[], None]] = []
    _ran_cleanup = False

    _keyboard_listener: Any = None
    _terminal_echo: Any = None

    # ---------- Cleanup registration ----------
    @classmethod
    def register_cleanup(cls, func: Callable[[], None]) -> None:
        """Register a cleanup function executed on shutdown."""
        cls._cleanup_funcs.append(func)

    @classmethod
    def _run_cleanup(cls) -> None:
        """Execute all cleanup callbacks once."""
        if cls._ran_cleanup:
            return
        cls._ran_cleanup = True
        for func in cls._cleanup_funcs:
            try:
                func()
            except Exception as e:
                cls.logger.tag("CLEANUP", f"callback failed: {e}", level="warning")
        cls.stop_keyboard_listener()

    # ---------- Hooks and signals ----------
    @classmethod
    def install_excepthook(cls) -> None:
        """Log unhandled exceptions via project logger."""
        if cls._installed:
            return
        cls._orig_hook = sys.excepthook

        def _hook(exc_type, exc, tb) -> None:
            msg = "".join(traceback.format_exception(exc_type, exc, tb))
            cls.logger.tag("HOOK", f"unhandled exception\n{msg}", level="error")
            cls._run_cleanup()
            if cls._orig_hook:
                try:
                    cls._orig_hook(exc_type, exc, tb)
                except Exception:
                    pass

        sys.excepthook = _hook
        cls._installed = True
        cls.logger.tag("HOOK", "global exception hook installed", level="debug")

    @classmethod
    def install_signal_handlers(cls) -> None:
        """Shutdown gracefully on SIGINT and SIGTERM."""

        def _handler(signum, _frame) -> None:
            cls.logger.tag("SIG", f"received signal {signum}", level="info")
            cls._run_cleanup()
            raise SystemExit(1)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        cls.logger.tag("SIG", "handlers installed", level="debug")

    # ---------- Keyboard stop (escape or control C) ----------
    @classmethod
    def install_keyboard_listener(cls, stop_key: str = "esc") -> None:
        """Start background listener that exits on the stop key or control C."""
        if cls._keyboard_listener is not None:
            return
        try:
            from utils.keyboard import GlobalKeyListener, TerminalEchoSuppressor
        except Exception as e:
            cls.logger.tag("KBD", f"listener unavailable: {e}", level="warning")
            return

        def _on_stop() -> None:
            cls.logger.tag("KBD", f"stop key {stop_key} pressed", level="info")
            cls._run_cleanup()
            raise SystemExit(1)

        hotkeys = {f"<{stop_key}>": _on_stop, "<ctrl>+c": _on_stop}
        try:
            cls._keyboard_listener = GlobalKeyListener(hotkeys)
            cls._keyboard_listener.start()
            cls._terminal_echo = TerminalEchoSuppressor()
            cls._terminal_echo.start()
            cls.logger.tag(
                "KBD", f"listener installed for {list(hotkeys)}", level="debug"
            )
        except Exception as e:
            cls.logger.tag("KBD", f"listener failed: {e}", level="warning")

    @classmethod
    def stop_keyboard_listener(cls) -> None:
        """Stop keyboard listener and restore terminal echo."""
        if cls._keyboard_listener is not None:
            try:
                cls._keyboard_listener.stop()
            finally:
                cls._keyboard_listener = None
                cls.logger.tag("KBD", "listener stopped", level="debug")
        if cls._terminal_echo is not None:
            try:
                cls._terminal_echo.stop()
            finally:
                cls._terminal_echo = None
                cls.logger.tag("KBD", "terminal echo restored", level="debug")

    # ---------- Reporting and fatal ----------
    @classmethod
    def report(cls, exc: Exception) -> None:
        """Log exception with full traceback without re-raising."""
        tb = exc.__traceback__
        if tb:
            formatted = "".join(traceback.format_exception(type(exc), exc, tb))
        else:
            stack = "".join(traceback.format_stack())
            formatted = (
                f"{type(exc).__name__}: {exc}\n"
                f"Traceback (most recent call last):\n{stack}"
            )
        cls.logger.tag("ERROR", formatted, level="error")

    @classmethod
    def fatal(cls, message: str, *, code: int = 1) -> None:
        """Log fatal error, run cleanup, then exit."""
        cls.logger.tag("FATAL", message, level="error")
        cls._run_cleanup()
        raise SystemExit(code)


@contextmanager
def error_scope():
    """
    Context manager that ensures cleanup on any exception.
    Usage:
        with error_scope():
            work
    """
    try:
        yield
    except SystemExit:
        raise
    except Exception as e:
        ErrorTracker.report(e)
        ErrorTracker._run_cleanup()
        raise SystemExit(1)
