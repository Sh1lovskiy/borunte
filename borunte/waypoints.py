# borunte/waypoints.py
"""Flexible waypoint loader for grid/waypoint runs.

Supports:
- JSON:
  * root list: [{"x":..,"y":..,"z":..,"rx":..,"ry":..,"rz":..}, ...]
  * dict-of-dicts: {"0": {...}, "1": {...}, ...}
  * wrapped arrays: {"points":[...]} / {"poses":[...]} / {"waypoints":[...]} / {"data":[...]}
  * list of [x,y,z,rx,ry,rz] vectors
- CSV:
  * with or without header; order: x,y,z,rx,ry,rz

Returns a list[List[float]] of [x, y, z, rx, ry, rz].
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

from utils.logger import Logger

_log = Logger.get_logger("")

Pose = List[float]
JsonVal = Union[dict, list, float, int, str]


def _as_float(v) -> float:
    if isinstance(v, bool):
        raise TypeError("bool is not a valid numeric pose value")
    return float(v)


def _norm_from_dict(d: dict) -> Pose:
    """Normalize dict with keys x,y,z,rx,ry,rz (case-insensitive)."""

    def pick(*names: str) -> float:
        for n in names:
            if n in d:
                return _as_float(d[n])
            ln = n.lower()
            for k in d.keys():
                if k.lower() == ln:
                    return _as_float(d[k])
        raise KeyError(f"missing key {names!r}")

    x = pick("x")
    y = pick("y")
    z = pick("z")
    rx = pick("rx", "u")
    ry = pick("ry", "v")
    rz = pick("rz", "w")
    return [x, y, z, rx, ry, rz]


def _norm_from_seq(seq: Sequence) -> Pose:
    """Normalize [x,y,z,rx,ry,rz] sequence."""
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
    """Accept list, dict-of-dicts, or wrappers with 'points'/'poses'/'waypoints'/'data'."""
    if isinstance(root, list):
        return root
    if isinstance(root, dict):
        # wrappers
        for k in ("points", "poses", "waypoints", "data"):
            if k in root and isinstance(root[k], list):
                return root[k]
        # dict of dicts or dict of vectors
        # sort by numeric key if possible for stable order
        try:
            keys = sorted(root.keys(), key=lambda s: int(str(s)))
        except Exception:
            keys = list(root.keys())
        return [root[k] for k in keys]
    raise ValueError("unsupported JSON root; must be list or dict")


def _load_json(path: Path) -> List[Pose]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = _extract_json_array(data)
    out: List[Pose] = []
    for i, it in enumerate(items):
        try:
            if isinstance(it, dict):
                pose = _norm_from_dict(it)
            elif isinstance(it, (list, tuple)):
                pose = _norm_from_seq(it)
            else:
                raise TypeError(f"item type {type(it).__name__}")
            out.append(pose)
        except Exception as e:
            _log.tag("WPT", f"skip item #{i}: {e}", "warning")
    if not out:
        raise ValueError("no valid waypoints parsed from JSON")
    _log.tag("WPT", f"loaded {len(out)} poses from JSON {path.name}")
    return out


def _load_csv(path: Path) -> List[Pose]:
    rows: List[Pose] = []
    with path.open("r", encoding="utf-8") as fh:
        sniffer = csv.Sniffer()
        sample = fh.read(2048)
        fh.seek(0)
        has_header = False
        try:
            has_header = sniffer.has_header(sample)
        except Exception:
            has_header = False
        reader = csv.reader(fh)
        first = True
        for i, row in enumerate(reader):
            if not row:
                continue
            if first and has_header:
                first = False
                # try mapping by header names
                hdr = [c.strip().lower() for c in row]
                idx = {
                    name: hdr.index(name)
                    for name in ("x", "y", "z", "rx", "ry", "rz")
                    if name in hdr
                }
                if len(idx) == 6:
                    for j, r in enumerate(reader, start=i + 1):
                        if not r or all(not c.strip() for c in r):
                            continue
                        try:
                            pose = [
                                _as_float(r[idx["x"]]),
                                _as_float(r[idx["y"]]),
                                _as_float(r[idx["z"]]),
                                _as_float(r[idx["rx"]]),
                                _as_float(r[idx["ry"]]),
                                _as_float(r[idx["rz"]]),
                            ]
                            rows.append(pose)
                        except Exception as e:
                            _log.tag("WPT", f"skip csv row #{j}: {e}", "warning")
                    break
                else:
                    # fall back to positional
                    _log.tag(
                        "WPT",
                        "csv header incomplete; fallback to positional",
                        "warning",
                    )
                    continue  # next loop will read positional rows
            try:
                # positional x,y,z,rx,ry,rz
                pose = _norm_from_seq([c.strip() for c in row][:6])
                rows.append(pose)
            except Exception as e:
                _log.tag("WPT", f"skip csv row #{i}: {e}", "warning")
    if not rows:
        raise ValueError("no valid waypoints parsed from CSV")
    _log.tag("WPT", f"loaded {len(rows)} poses from CSV {path.name}")
    return rows


def load_waypoints(path: Union[str, Path], fmt: str = "json") -> List[Pose]:
    """
    Load waypoints from file. fmt: "json" | "csv" | "auto".
    If "auto", picks by extension.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"waypoints file not found: {p}")

    use_fmt = fmt.lower()
    if use_fmt == "auto":
        ext = p.suffix.lower()
        if ext in (".json",):
            use_fmt = "json"
        elif ext in (".csv", ".tsv"):
            use_fmt = "csv"
        else:
            use_fmt = "json"  # safe default

    _log.tag("WPT", f"loading {p.name} as {use_fmt}")

    if use_fmt == "json":
        poses = _load_json(p)
    elif use_fmt == "csv":
        poses = _load_csv(p)
    else:
        raise ValueError(f"unsupported fmt {fmt}")

    # final sanity
    valid: List[Pose] = []
    for i, pose in enumerate(poses):
        ok = all(isinstance(v, (int, float)) for v in pose) and len(pose) == 6
        if not ok:
            _log.tag("WPT", f"invalid pose at #{i}, skip", "warning")
            continue
        valid.append([float(v) for v in pose])

    if not valid:
        raise ValueError("no valid waypoints after validation")
    _log.tag("WPT", f"ready {len(valid)} poses")
    return valid
