# utils/logger.py
"""Logging helpers built on top of loguru with RM helpers."""

from __future__ import annotations

import inspect
import os
import pathlib
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, TypeVar, cast, Sequence

from loguru import logger as _logger
from loguru._logger import Logger as LoguruLogger
from tqdm.auto import tqdm

from utils.settings import logging as LOGCFG

LoggerType = LoguruLogger
T = TypeVar("T")

_is_configured = False
_log_dir = LOGCFG.log_dir
_log_file = None


def _fmt_xyzuvw_thousand(vals: Sequence[int]) -> str:
    x, y, z, u, v, w = [int(v) / 1000.0 for v in vals[:6]]
    return f"X={x:.3f} Y={y:.3f} Z={z:.3f} " f"U={u:.3f} V={v:.3f} W={w:.3f}"


def _fmt_world(world: Sequence[float]) -> str:
    x, y, z, u, v, w = [float(v) for v in world[:6]]
    return f"X={x:.3f} Y={y:.3f} Z={z:.3f} " f"U={u:.3f} V={v:.3f} W={w:.3f}"


class Logger:
    """Project-wide logger wrapper using loguru and global config."""

    @staticmethod
    def _configure(level: str, json_format: bool) -> None:
        """Configure log sinks on first use."""
        global _is_configured, _log_file
        _logger.remove()
        os.makedirs(_log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _log_file = _log_dir / f"{ts}.log.json"
        _logger.add(
            sys.stdout,
            level=level,
            serialize=False,
            format=LOGCFG.log_format,
        )
        _logger.add(
            _log_file,
            level=level,
            serialize=json_format,
            format=LOGCFG.log_file_format,
        )
        _is_configured = True

    @staticmethod
    def get_logger(
        name: str | None = None, level: str = None, json_format: bool = None
    ) -> LoguruLogger:
        """
        Return a configured loguru logger bound to module name.

        If ``name`` is None, it's auto-detected from the caller file,
        e.g. "borunte.grid" for src/borunte/grid.py.
        """
        global _is_configured
        if not _is_configured:
            Logger._configure(
                level or LOGCFG.level,
                json_format if json_format is not None else LOGCFG.json,
            )

        # auto-detect caller if name is None
        if name is None:
            frame = inspect.stack()[1]
            module = inspect.getmodule(frame[0])
            if module and module.__name__ != "__main__":
                name = module.__name__
            else:
                # fallback: relative path without extension
                path = pathlib.Path(frame.filename)
                name = ".".join(path.with_suffix("").parts[-1:])  # e.g. borunte.grid

        # add a short filename for formatting
        frame = inspect.stack()[1]
        origin_file = pathlib.Path(frame.filename).stem

        logger = _logger.bind(module=name, origin=origin_file)
        # attach helpers on the bound logger (dynamic monkey-patch)
        if not hasattr(logger, "tag"):

            def _tag(tag: str, msg: str, level: str = "info") -> None:
                """
                Unified tagged logging: supports level in
                {"debug", "info", "warning", "error", "critical"}.
                Example: logger.tag("GRID", "initialized", level="debug")
                """
                text = f"[{tag}] {msg}"
                if hasattr(logger, level):
                    getattr(logger, level)(text)
                else:
                    logger.info(text)

            setattr(logger, "tag", _tag)
        # backward compatibility: some modules call .warn()
        if not hasattr(logger, "warn"):
            setattr(logger, "warn", logger.warning)

        return logger

    # ---------- Progress / configuration ----------
    @staticmethod
    def progress(
        iterable: Iterable[T],
        desc: str | None = None,
        total: int | None = None,
    ) -> Iterable[T]:
        """Return a tqdm iterator with unified style."""
        return cast(
            Iterable[T],
            tqdm(
                iterable,
                desc=desc,
                total=total,
                leave=False,
                bar_format=LOGCFG.progress_bar_format,
            ),
        )

    @staticmethod
    def configure_root_logger(level: str = "WARNING") -> None:
        """Configure the root logger for third-party libraries."""
        global _is_configured
        _logger.remove()
        _logger.add(sys.stdout, level=level)
        _is_configured = True

    @staticmethod
    def configure(
        level: str = None, log_dir: str | Path = None, json_format: bool = None
    ) -> None:
        """Manually configure the logger with given settings."""
        global _log_dir
        _log_dir = Path(log_dir) if log_dir is not None else LOGCFG.log_dir
        Logger._configure(
            level or LOGCFG.level,
            json_format if json_format is not None else LOGCFG.json,
        )

    # ---------- RemoteMonitor helpers (STATE/ALARM/ADDR) ----------
    @staticmethod
    def rm_state(
        logger: LoguruLogger,
        *,
        cur_mode: int,
        is_moving: int,
        cur_alarm: int,
        world: Sequence[float] | None = None,
        tag: str = "STATE",
    ) -> None:
        """Log condensed robot state line (mode/move/alarm/world)."""
        if world is not None:
            txt = _fmt_world(world)
            logger.info(
                f"[{tag}] curMode={cur_mode} isMoving={is_moving} "
                f"curAlarm={cur_alarm}  WORLD {txt}"
            )
        else:
            logger.info(
                f"[{tag}] curMode={cur_mode} isMoving={is_moving} "
                f"curAlarm={cur_alarm}"
            )

    @staticmethod
    def rm_alarm(
        logger: LoguruLogger,
        *,
        code: int | str,
        brief: str | None = None,
        tag: str = "ALARM",
    ) -> None:
        """Log alarm code with optional brief description."""
        if brief:
            logger.warning(f"[{tag}] {code} — {brief}")
        else:
            logger.warning(f"[{tag}] {code}")

    @staticmethod
    def rm_addrs(
        logger: LoguruLogger,
        *,
        base: int,
        values: Sequence[int],
        tag: str = "ADDR",
        decoded: bool = True,
    ) -> None:
        """Log raw Addr-<base..> and optional decoded XYZUVW."""
        rng = f"{base}..{base+len(values)-1}"
        logger.info(
            f"[{tag}] RAW {rng}: "
            + " ".join(f"{base+i}:{values[i]:d}" for i in range(len(values)))
        )
        if decoded and len(values) >= 6:
            logger.info(f"[{tag}] DECODED XYZUVW: " + _fmt_xyzuvw_thousand(values))

    @staticmethod
    def rm_step_target(logger: LoguruLogger, pose: Sequence[float]) -> None:
        """Log compact target pose for a step."""
        p = tuple(round(float(x), 3) for x in pose[:6])
        logger.info(f"[STEP] target {p}")

    @staticmethod
    def rm_check_tolerance(
        logger: LoguruLogger,
        *,
        actual: Sequence[float],
        target: Sequence[float],
        pos_tol: float,
        ang_tol: float,
    ) -> None:
        """Log tolerance check result with diffs."""
        ax = [float(x) for x in actual[:6]]
        tx = [float(x) for x in target[:6]]
        dx = [tx[i] - ax[i] for i in range(6)]
        logger.info(
            "[] diffs "
            f"dX={dx[0]:.3f} dY={dx[1]:.3f} dZ={dx[2]:.3f} "
            f"dU={dx[3]:.3f} dV={dx[4]:.3f} dW={dx[5]:.3f}"
        )
        ok = (
            abs(dx[0]) <= pos_tol
            and abs(dx[1]) <= pos_tol
            and abs(dx[2]) <= pos_tol
            and abs(dx[3]) <= ang_tol
            and abs(dx[4]) <= ang_tol
            and abs(dx[5]) <= ang_tol
        )
        if ok:
            logger.info("[CHECK] within tolerance")
        else:
            logger.warning("[CHECK] outside tolerance")


class CaptureStderrToLogger:
    """Context manager: redirects C/C++ stderr (fd=2) to the provided logger."""

    def __init__(self, logger: LoguruLogger):
        self.logger = logger
        self.pipe_read = None
        self.pipe_write = None
        self.thread = None
        self._old_stderr_fd = None

    def _reader(self):
        with os.fdopen(self.pipe_read, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                if line:
                    self.logger.warning(f"[STDERR] {line}")

    def __enter__(self):
        self._old_stderr_fd = os.dup(2)
        self.pipe_read, self.pipe_write = os.pipe()
        os.dup2(self.pipe_write, 2)
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.flush()
        os.dup2(self._old_stderr_fd, 2)
        os.close(self.pipe_write)
        os.close(self._old_stderr_fd)
        if self.thread:
            self.thread.join(timeout=0.2)


class SuppressO3DInfo:
    """Context: suppresses Open3D INFO/console help messages."""

    def __enter__(self):
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self.devnull)
        os.close(self._old_stdout)
        os.close(self._old_stderr)
