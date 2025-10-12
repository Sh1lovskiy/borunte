from __future__ import annotations
import json, math, re
from pathlib import Path
from typing import Any
import numpy as np
from utils.logger import Logger

LOG = Logger.get_logger("poses")


def _euler_xyz(rx: float, ry: float, rz: float, deg: bool) -> np.ndarray:
    if deg:
        rx, ry, rz = map(math.radians, (rx, ry, rz))
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _to_4x4(R: np.ndarray, t_m: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t_m
    return M


def _detect_deg(rx: float, ry: float, rz: float) -> bool:
    return max(abs(rx), abs(ry), abs(rz)) > 3.2


def _mm_to_m(t: np.ndarray, meta_units: str) -> np.ndarray:
    return t / 1000.0 if meta_units.startswith("mill") else t


_RX_WORLD = re.compile(
    r"(?:world)?\s*"
    r"x\s*[:=]\s*(?P<X>[-+]?[\d.]+)\s*[, ]\s*"
    r"y\s*[:=]\s*(?P<Y>[-+]?[\d.]+)\s*[, ]\s*"
    r"z\s*[:=]\s*(?P<Z>[-+]?[\d.]+)\s*[, ]\s*"
    r"(?:u\s*[:=]\s*(?P<U>[-+]?[\d.]+)\s*[, ]\s*)?"
    r"(?:v\s*[:=]\s*(?P<V>[-+]?[\d.]+)\s*[, ]\s*)?"
    r"(?:w\s*[:=]\s*(?P<W>[-+]?[\d.]+))?",
    re.IGNORECASE,
)


def _try_entry(item: Any, meta_units: str, meta_order: str) -> np.ndarray | None:
    # 4x4 flat
    try:
        arr = np.array(item, dtype=float)
        if arr.size == 16:
            return arr.reshape(4, 4)
    except Exception:
        pass

    # string "WORLD X=.. Y=.. Z=.. U=.. V=.. W=.."
    if isinstance(item, str):
        m = _RX_WORLD.search(item)
        if m:
            tx = float(m.group("X"))
            ty = float(m.group("Y"))
            tz = float(m.group("Z"))
            t = _mm_to_m(np.array([tx, ty, tz], float), meta_units)
            U = m.group("U")
            V = m.group("V")
            W = m.group("W")
            if U is not None and V is not None and W is not None:
                rx, ry, rz = float(U), float(V), float(W)
                R = _euler_xyz(rx, ry, rz, deg=True)
            else:
                R = np.eye(3)
            return _to_4x4(R, t)

    # list [tx,ty,tz, rx,ry,rz]
    if isinstance(item, (list, tuple)) and len(item) == 6:
        tx, ty, tz, rx, ry, rz = map(float, item)
        t = _mm_to_m(np.array([tx, ty, tz], float), meta_units)
        if meta_order != "xyz":
            return None
        R = _euler_xyz(rx, ry, rz, deg=_detect_deg(rx, ry, rz))
        return _to_4x4(R, t)

    if not isinstance(item, dict):
        return None

    # T / R+t
    if "T" in item:
        arr = np.array(item["T"], float)
        if arr.size == 16:
            return arr.reshape(4, 4)

    if "R" in item and "t" in item:
        R = np.array(item["R"], float).reshape(3, 3)
        t = _mm_to_m(np.array(item["t"], float).reshape(3), meta_units)
        return _to_4x4(R, t)

    # t + euler
    if "t" in item and ("r" in item or "euler" in item):
        t = _mm_to_m(np.array(item["t"], float).reshape(3), meta_units)
        r = np.array(item.get("r", item.get("euler")), float).reshape(3)
        if meta_order != "xyz":
            return None
        R = _euler_xyz(*r, deg=_detect_deg(*r))
        return _to_4x4(R, t)

    # X,Y,Z,U,V,W (deg, mm) in any case
    keys = set(k.lower() for k in item.keys())
    if {"x", "y", "z"}.issubset(keys):
        tx = float(item.get("X", item.get("x")))
        ty = float(item.get("Y", item.get("y")))
        tz = float(item.get("Z", item.get("z")))
        t = _mm_to_m(np.array([tx, ty, tz], float), meta_units)
        if {"u", "v", "w"}.issubset(keys):
            rx = float(item.get("U", item.get("u")))
            ry = float(item.get("V", item.get("v")))
            rz = float(item.get("W", item.get("w")))
            R = _euler_xyz(rx, ry, rz, deg=True)
        else:
            R = np.eye(3)
        return _to_4x4(R, t)

    # tx..tz + rx..rz (deg or rad)
    if {"tx", "ty", "tz", "rx", "ry", "rz"}.issubset(keys):
        t = _mm_to_m(np.array([item["tx"], item["ty"], item["tz"]], float), meta_units)
        r = np.array([item["rx"], item["ry"], item["rz"]], float)
        R = _euler_xyz(*r, deg=_detect_deg(*r))
        return _to_4x4(R, t)

    # nested aliases
    for key in ("matrix", "pose", "M", "T_camera_base", "T_cam_base"):
        if key in item:
            arr = np.array(item[key], float)
            if arr.size == 16:
                return arr.reshape(4, 4)
            if arr.size == 6:
                tx, ty, tz, rx, ry, rz = map(float, arr)
                t = _mm_to_m(np.array([tx, ty, tz], float), meta_units)
                R = _euler_xyz(rx, ry, rz, deg=_detect_deg(rx, ry, rz))
                return _to_4x4(R, t)
    return None


def load_poses_generic(poses_json: Path) -> list[np.ndarray]:
    data = json.loads(poses_json.read_text(encoding="utf-8"))
    units = "millimeters"
    order = "xyz"
    if isinstance(data, dict):
        units = str(data.get("units", units)).lower()
        order = str(data.get("order", order)).lower()
        seq = data.get("poses", None)
        if isinstance(seq, list):
            items = seq
        else:
            items = [
                v
                for k, v in sorted(data.items())
                if k not in ("units", "order", "poses")
            ]
    else:
        items = data if isinstance(data, list) else [data]

    out: list[np.ndarray] = []
    bad = 0
    for idx, item in enumerate(items):
        M = _try_entry(item, units, order)
        if M is None:
            bad += 1
            LOG.warning(f"[poses] unrecognized at idx={idx}; identity used")
            M = np.eye(4, dtype=float)
        out.append(M)
    if bad:
        LOG.warning(f"[poses] {bad} unrecognized entries")
    return out
