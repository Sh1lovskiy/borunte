# utils/settings.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoggingCfg:
    level: str = "INFO"
    json: bool = True
    log_dir: Path = Path(".logs")
    log_format: str = (
        "<green>{time:MM-DD HH:mm:ss}</green>"
        "[<level>{level:.3}</level>]"
        "[<cyan>{extra[origin]:.16}</cyan>:<cyan>{line:<3}</cyan>]"
        "<level>{message}</level>"
    )
    log_file_format: str = "{time:YYYY-MM-DD HH:mm:ss}[{level}][{file}:{line}]{message}"
    progress_bar_format: str = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )


logging = LoggingCfg()
