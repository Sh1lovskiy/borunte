# borunte/waypoints.py
"""Load waypoint poses from JSON or CSV files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Union

from utils.logger import Logger

from .config import BORUNTE_CONFIG, BorunteConfig

_log = Logger.get_logger()

Pose = List[float]
JsonVal = Union[dict, list, float, int, str]


def _as_float(value) -> float:
    if isinstance(value, bool):
        raise TypeError("bool is not a valid numeric pose value")
    return float(value)


def _norm_from_dict(data: dict) -> Pose:
    def pick(*names: str) -> float:
        for name in names:
            if name in data:
                return _as_float(data[name])
            lower = name.lower()
            for key in data.keys():
                if key.lower() == lower:
                    return _as_float(data[key])
        raise KeyError(f"missing key {names!r}")

    x = pick("x")
    y = pick("y")
    z = pick("z")
    rx = pick("rx", "u")
    ry = pick("ry", "v")
    rz = pick("rz", "w")
    return [x, y, z, rx, ry, rz]


def _norm_from_seq(seq: Sequence) -> Pose:
    if len(seq) != 6:
        raise ValueError("sequence must have 6 elements [x,y,z,rx,ry,rz]")
    return [
        _as_float(seq[0]),
        _as_float(seq[1]),
        _as_float(seq[2]),
        _as_float(seq[3]),
        _as_float(seq[4]),
        _as_float(seq[5]),
    ]


def _extract_json_array(root: JsonVal) -> Iterable:
    if isinstance(root, list):
        return root
    if isinstance(root, dict):
        for key in ("points", "poses", "waypoints", "data"):
            if key in root and isinstance(root[key], list):
                return root[key]
        try:
            keys = sorted(root.keys(), key=lambda item: int(str(item)))
        except Exception:
            keys = list(root.keys())
        return [root[key] for key in keys]
    raise ValueError("unsupported JSON root; must be list or dict")


def _load_json(path: Path) -> List[Pose]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = _extract_json_array(data)
    poses: List[Pose] = []
    for index, item in enumerate(items):
        try:
            if isinstance(item, dict):
                pose = _norm_from_dict(item)
            elif isinstance(item, (list, tuple)):
                pose = _norm_from_seq(item)
            else:
                raise TypeError(f"item type {type(item).__name__}")
            poses.append(pose)
        except Exception as exc:
            _log.tag("WPT", f"skip item #{index}: {exc}", level="warning")
    if not poses:
        raise ValueError("no valid waypoints parsed from JSON")
    _log.tag("WPT", f"loaded {len(poses)} poses from JSON {path.name}")
    return poses


def _load_csv(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open("r", encoding="utf-8") as handle:
        sniffer = csv.Sniffer()
        sample = handle.read(2048)
        handle.seek(0)
        try:
            has_header = sniffer.has_header(sample)
        except Exception:
            has_header = False
        reader = csv.reader(handle)
        first = True
        for row_index, row in enumerate(reader):
            if not row:
                continue
            if first and has_header:
                first = False
                headers = [cell.strip().lower() for cell in row]
                indices = {
                    name: headers.index(name)
                    for name in ("x", "y", "z", "rx", "ry", "rz")
                    if name in headers
                }
                if len(indices) == 6:
                    for inner_index, inner_row in enumerate(reader, start=row_index + 1):
                        if not inner_row or all(not cell.strip() for cell in inner_row):
                            continue
                        try:
                            pose = [
                                _as_float(inner_row[indices["x"]]),
                                _as_float(inner_row[indices["y"]]),
                                _as_float(inner_row[indices["z"]]),
                                _as_float(inner_row[indices["rx"]]),
                                _as_float(inner_row[indices["ry"]]),
                                _as_float(inner_row[indices["rz"]]),
                            ]
                            poses.append(pose)
                        except Exception as exc:
                            _log.tag("WPT", f"skip csv row #{inner_index}: {exc}", level="warning")
                    break
                else:
                    _log.tag(
                        "WPT",
                        "csv header incomplete; fallback to positional",
                        level="warning",
                    )
                    continue
            try:
                pose = _norm_from_seq([cell.strip() for cell in row][:6])
                poses.append(pose)
            except Exception as exc:
                _log.tag("WPT", f"skip csv row #{row_index}: {exc}", level="warning")
    if not poses:
        raise ValueError("no valid waypoints parsed from CSV")
    _log.tag("WPT", f"loaded {len(poses)} poses from CSV {path.name}")
    return poses


def load_waypoints(path: Union[str, Path], fmt: str = "json") -> List[Pose]:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"waypoints file not found: {target}")

    use_fmt = fmt.lower()
    if use_fmt == "auto":
        ext = target.suffix.lower()
        if ext in (".json",):
            use_fmt = "json"
        elif ext in (".csv", ".tsv"):
            use_fmt = "csv"
        else:
            use_fmt = "json"

    _log.tag("WPT", f"loading {target.name} as {use_fmt}")

    if use_fmt == "json":
        poses = _load_json(target)
    elif use_fmt == "csv":
        poses = _load_csv(target)
    else:
        raise ValueError(f"unknown waypoints format {fmt}")
    return poses


def load_default_waypoints(config: BorunteConfig = BORUNTE_CONFIG) -> List[Pose]:
    cfg = config.waypoints
    if not cfg.file.exists():
        raise FileNotFoundError(f"default waypoints file missing: {cfg.file}")
    return load_waypoints(cfg.file, fmt=cfg.fmt)


__all__ = ["load_waypoints", "load_default_waypoints"]
