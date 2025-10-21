# utils/logger.py
"""Single-source Loguru setup: safe console/file sinks, no external placeholders."""

from __future__ import annotations
import inspect
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, TypeVar, cast, TextIO

from loguru._logger import Logger as LoguruLogger
from loguru import logger as _root_logger

try:
    from config import get_settings  # optional
except Exception:
    get_settings = None  # type: ignore

T = TypeVar("T")


# ---------- options ----------
@dataclass(slots=True)
class _LogOptions:
    level: str = os.environ.get("BORUNTE_LOG_LEVEL", "INFO")
    json: bool = False  # file serialize (off by default)


_OPTIONS = _LogOptions()
_CONFIGURED = False
_LOG_FILE: Optional[Path] = None


# ---------- safe extras ----------
class _DotSafe:
    """Absorb ANY nested attribute/item access and render empty string."""

    def __getattr__(self, _name: str):  # config.net -> _DotSafe
        return self

    def __getitem__(self, _key):  # config["net"] -> _DotSafe
        return self

    def __str__(self):  # str(config.net.host) -> ""
        return ""

    def __format__(self, _spec):  # f"{config.net:>10}" -> ""
        return ""


def _resolve_log_dir() -> Path:
    if get_settings is not None:
        try:
            return get_settings().paths.logs_root
        except Exception:
            pass
    return Path.cwd() / "logs"


# ---------- sinks as callables (никаких format_map) ----------
def _console_sink(msg) -> None:
    r = msg.record
    module = r["extra"].get("module", r.get("name", "unknown"))
    # один лог == одна строка
    print(f"{r['time']:%H:%M:%S} | {r['level'].name: <3.3} | {module} | {r['message']}")


def _make_file_sink(fh: TextIO):
    def _file_sink(msg) -> None:
        r = msg.record
        module = r["extra"].get("module", r.get("name", "unknown"))
        fh.write(
            f"{r['time'].isoformat()} | {r['level'].name} | {module} | {r['message']}\n"
        )
        fh.flush()

    return _file_sink


def _configure_logger(level: str | None = None, json: bool | None = None) -> None:
    global _CONFIGURED, _LOG_FILE

    # 1) сбросить ВСЕ существующие обработчики, чтобы не ловить чужие format-строки
    _root_logger.remove()

    # 2) патч: гарантируем нужные поля и безопасный config
    def _inject_extras(record):
        record["extra"].setdefault("module", record.get("name", "unknown"))
        record["extra"].setdefault("config", _DotSafe())
        # дублируем и на верхний уровень на всякий случай (если кто-то добавит ещё sink)
        record.setdefault("module", record["extra"]["module"])
        record.setdefault("config", record["extra"]["config"])
        return record

    logger = _root_logger.patch(_inject_extras)

    # 3) подготовить файл
    log_dir = _resolve_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = log_dir / f"borunte_{timestamp}.log"
    _LOG_FILE = file_path
    fh = file_path.open("a", encoding="utf-8")

    # 4) добавить только два sink-а, БЕЗ format-строк
    logger.add(_console_sink, level=(level or _OPTIONS.level), catch=True)
    logger.add(_make_file_sink(fh), level=(level or _OPTIONS.level), catch=True)

    # 5) сохранить ссылку, отметить как сконфигурированный
    globals()["_LOGGER"] = logger
    globals()["_CONFIGURED"] = True


def get_logger(name: str | None = None) -> LoguruLogger:
    if not _CONFIGURED:
        _configure_logger()

    # модульное имя определяем автоматически
    frame = inspect.currentframe()
    module_name = name
    if module_name is None and frame is not None:
        caller_frame = frame.f_back
        if caller_frame is not None:
            module = inspect.getmodule(caller_frame)
            if module is not None and module.__name__ != "__main__":
                module_name = module.__name__

    # критично: НЕ биндим config (оставляем _DotSafe из patch)
    bound = globals()["_LOGGER"].bind(module=module_name or "unknown")

    # совместимость: warn()
    if not hasattr(bound, "warn"):
        setattr(bound, "warn", bound.warning)

    # удобная метка
    def _tag(label: str, msg: str | None = None, *args, level: str = "info") -> None:
        prefix = f"[{label}] "
        text = prefix + (msg or "")
        method = getattr(bound, level, bound.info)
        if args:
            try:
                text = text.format(*args)
            except Exception:
                pass
        method(text)

    setattr(bound, "tag", _tag)
    return bound


def configure(level: str | None = None, json: bool | None = None) -> None:
    _configure_logger(level=level, json=json)


@contextmanager
def logging_context(
    *, level: str | None = None, json: bool | None = None
) -> Iterator[LoguruLogger]:
    configure(level=level, json=json)
    logger = get_logger()
    yield logger


# прогресс
def iter_progress(
    iterable: Iterable[T],
    *,
    description: str | None = None,
    total: int | None = None,
) -> Iterable[T]:
    from utils.progress import track

    return cast(Iterable[T], track(iterable, description=description, total=total))


# шима для старых вызовов
class Logger:
    @staticmethod
    def progress(it, desc: str, total: int | None = None):
        return iter_progress(it, description=desc, total=total)

    @staticmethod
    def rm_alarm(log: LoguruLogger, *, code: str | int | None) -> None:
        log.tag("RM", f"alarm code={code}")

    @staticmethod
    def get_logger(name: str | None = None):
        return get_logger(name)

    @staticmethod
    def rm_state(
        log: LoguruLogger,
        *,
        cur_mode=None,
        is_moving=None,
        cur_alarm=None,
        world: tuple[float, float, float, float, float, float] | None = None,
        tag: str = "STATE",
    ) -> None:
        log.tag(
            tag, f"mode={cur_mode} moving={is_moving} alarm={cur_alarm} world={world}"
        )

    @staticmethod
    def rm_addrs(log: LoguruLogger, *, base: int, values: list[int]) -> None:
        log.tag("ADDR", f"base={base} values={values}")

    @staticmethod
    def rm_step_target(log: LoguruLogger, pose: Sequence[float]) -> None:
        xyz = tuple(round(float(v), 3) for v in pose[:3])
        log.tag("STEP", f"target={xyz}")

    @staticmethod
    def rm_check_tolerance(
        log: LoguruLogger,
        *,
        actual: Sequence[float],
        target: Sequence[float],
        pos_tol: float,
        ang_tol: float,
    ) -> None:
        ax, ay, az, au, av, aw = actual
        tx, ty, tz, tu, tv, tw = target
        dp = (abs(ax - tx), abs(ay - ty), abs(az - tz))
        da = (abs(au - tu), abs(av - tv), abs(aw - tw))
        okp = all(d <= pos_tol for d in dp)
        oka = all(d <= ang_tol for d in da)
        log.tag(
            "TOL",
            f"pos={tuple(round(x,3) for x in dp)}<= {pos_tol}  "
            f"ang={tuple(round(x,3) for x in da)}<= {ang_tol}  "
            f"ok={okp and oka}",
        )


__all__ = ["get_logger", "configure", "logging_context", "iter_progress", "Logger"]
