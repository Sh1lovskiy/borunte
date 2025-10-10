# borunte/wire.py
"""Low-level JSON wire protocol for RemoteMonitor."""

from __future__ import annotations

import json
import socket
import time
from threading import Lock
from typing import Sequence, List, Tuple, Dict, Any

from .config import TIMEOUT_S
from utils.logger import Logger

_log = Logger.get_logger()

_pack_id = 0
_pack_lock = Lock()


def _next_pack_id() -> str:
    """Return next monotonically increasing packID as string."""
    global _pack_id
    with _pack_lock:
        _pack_id = (_pack_id + 1) % 10_000_000
        return str(_pack_id)


def _recv_json(sock: socket.socket, timeout: float) -> Dict[str, Any]:
    """Receive a single JSON object delimited by a newline."""
    end = time.time() + float(timeout)
    buf = bytearray()
    while True:
        remain = end - time.time()
        if remain <= 0:
            raise TimeoutError("Receive timeout")
        sock.settimeout(remain)
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Peer closed connection")
        buf.extend(chunk)
        text = buf.decode("utf-8", errors="ignore")

        # Try to parse the last complete non-empty line first.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for ln in reversed(lines):
            try:
                return json.loads(ln)
            except Exception:
                continue

        # Fallback attempt on the whole buffer if there is a single JSON object without newline.
        try:
            return json.loads(text)
        except Exception:
            continue


def _send(sock: socket.socket, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Send one JSON object and await a single JSON reply."""
    payload = dict(obj)
    payload.setdefault("dsID", "www.hc-system.com.RemoteMonitor")
    payload.setdefault("packID", _next_pack_id())
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)
    return _recv_json(sock, TIMEOUT_S)


def q(sock: socket.socket, addrs: Sequence[str]) -> List[str]:
    """Query named fields and return their string values."""
    rep = _send(sock, {"reqType": "query", "queryAddr": list(addrs)})
    return list(map(str, rep.get("queryData", [])))


def cmd(sock: socket.socket, *cmdData: str) -> Tuple[bool, str, Dict[str, Any]]:
    """Send a command and return (ok, message, raw_reply)."""
    rep = _send(sock, {"reqType": "command", "cmdData": list(cmdData)})
    cr = rep.get("cmdReply", [])
    ok = False
    msg = ""
    if len(cr) == 1:
        ok = cr[0] == "ok"
    elif len(cr) >= 2:
        ok = cr[1] == "ok"
        if not ok and len(cr) >= 3:
            msg = str(cr[2])
    if not ok and msg:
        _log.tag("WIRE", f"command {' '.join(cmdData)} message={msg}", level="warning")
    return ok, msg, rep


def query_addrs(sock: socket.socket, base: int, length: int) -> List[int]:
    """Read a consecutive block of Addr-n values."""
    keys = [f"Addr-{base + i}" for i in range(length)]
    rep = _send(sock, {"reqType": "addressing", "addrList": keys})
    raw = rep.get("dataList", [])
    out: List[int] = []
    for i in range(length):
        try:
            out.append(int(raw[i]))
        except Exception:
            out.append(0)
    return out


def _thousand(x: float) -> int:
    return int(round(float(x) * 1000.0))


def rewrite_800_805(
    sock: socket.socket, x: float, y: float, z: float, u: float, v: float, w: float
) -> Tuple[bool, str, Dict[str, Any]]:
    """Write XYZUVW scaled by one thousand into Addr-800 to Addr-805 via rewriteDataList."""
    vals = [_thousand(a) for a in (x, y, z, u, v, w)]
    return cmd(sock, "rewriteDataList", "800", "6", "0", *map(str, vals))
