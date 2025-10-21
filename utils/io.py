# utils/io.py
"""File IO helpers with atomic JSON/YAML writes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger

LOGGER = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_write(path: Path, content: str) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    LOGGER.debug("Wrote file {} atomically", path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    payload = json.dumps(data, indent=indent, sort_keys=True)
    _atomic_write(path, payload)


def atomic_write_yaml(path: Path, data: Any) -> None:
    payload = yaml.safe_dump(data, sort_keys=True)
    _atomic_write(path, payload)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


__all__ = [
    "atomic_write_json",
    "atomic_write_yaml",
    "ensure_directory",
    "load_json",
    "load_yaml",
]
