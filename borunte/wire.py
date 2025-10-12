# borunte/wire.py
"""TCP client for the Borunte RemoteMonitor protocol."""

from __future__ import annotations

import json
import socket
import time
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.error_tracker import ErrorTracker
from utils.logger import Logger

from .config import BORUNTE_CONFIG, BorunteConfig

_log = Logger.get_logger()


class RobotClient:
    """Encapsulate the RemoteMonitor TCP connection and protocol."""

    def __init__(self, config: BorunteConfig = BORUNTE_CONFIG):
        self.config = config
        self._sock: Optional[socket.socket] = None
        self._pack_id = 0
        self._pack_lock = Lock()

    @property
    def socket(self) -> socket.socket:
        if self._sock is None:
            raise RuntimeError("RobotClient is not connected")
        return self._sock

    def connect(self) -> None:
        if self._sock is not None:
            return
        net = self.config.network
        for attempt in range(self.config.network.retry_attempts):
            try:
                _log.tag(
                    "NET",
                    f"connect host={net.host} port={net.port} timeout={net.timeout_s}",
                )
                sock = socket.create_connection(
                    (net.host, net.port), timeout=net.timeout_s
                )
                if net.keepalive:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                self._sock = sock
                _log.tag("NET", "connected")
                return
            except Exception as exc:
                ErrorTracker.report(exc)
                _log.tag("NET", f"connect failed attempt {attempt+1}: {exc}", level="warning")
                time.sleep(net.retry_delay_s)
        raise ConnectionError("RobotClient failed to connect")

    def close(self, graceful: bool = True) -> None:
        if self._sock is None:
            return
        sock = self._sock
        self._sock = None
        if graceful:
            try:
                sock.shutdown(socket.SHUT_WR)
            except Exception:
                pass
            try:
                sock.settimeout(0.5)
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
            except Exception:
                pass
        try:
            sock.close()
        finally:
            _log.tag("NET", "socket closed")

    def _next_pack_id(self) -> str:
        with self._pack_lock:
            self._pack_id = (self._pack_id + 1) % 10_000_000
            return str(self._pack_id)

    def _recv_json(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        sock = self.socket
        deadline = time.time() + (timeout or self.config.network.timeout_s)
        buf = bytearray()
        while True:
            remain = deadline - time.time()
            if remain <= 0:
                raise TimeoutError("receive timeout")
            sock.settimeout(remain)
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("peer closed connection")
            buf.extend(chunk)
            text = buf.decode("utf-8", errors="ignore")
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            for ln in reversed(lines):
                try:
                    rep = json.loads(ln)
                    _log.tag("WIRE", f"rx {ln[:256]}")
                    return rep
                except json.JSONDecodeError:
                    continue
            try:
                rep = json.loads(text)
                _log.tag("WIRE", f"rx {text[:256]}")
                return rep
            except json.JSONDecodeError:
                continue

    def _send(self, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        sock = self.socket
        data = dict(payload)
        data.setdefault("dsID", "www.hc-system.com.RemoteMonitor")
        data.setdefault("packID", self._next_pack_id())
        encoded = (json.dumps(data, ensure_ascii=False) + "\n").encode("utf-8")
        _log.tag("WIRE", f"tx {encoded[:256]}")
        sock.sendall(encoded)
        return self._recv_json(timeout)

    def send(self, payload: Dict[str, Any], *, timeout: Optional[float] = None) -> Dict[str, Any]:
        try:
            return self._send(payload, timeout=timeout)
        except TimeoutError as exc:
            ErrorTracker.report(exc)
            _log.tag("WIRE", f"timeout payload={payload}", level="warning")
            raise
        except Exception as exc:
            ErrorTracker.report(exc)
            _log.tag("WIRE", f"send failed payload={payload} error={exc}", level="error")
            raise

    def query(self, addrs: Sequence[str]) -> List[str]:
        rep = self.send({"reqType": "query", "queryAddr": list(addrs)})
        values = list(map(str, rep.get("queryData", [])))
        _log.tag("WIRE", f"query addrs={addrs} -> {values}")
        return values

    def command(self, *cmd_data: str) -> Tuple[bool, str, Dict[str, Any]]:
        rep = self.send({"reqType": "command", "cmdData": list(cmd_data)})
        reply = rep.get("cmdReply", [])
        ok = False
        msg = ""
        if len(reply) == 1:
            ok = reply[0] == "ok"
        elif len(reply) >= 2:
            ok = reply[1] == "ok"
            if not ok and len(reply) >= 3:
                msg = str(reply[2])
        if not ok and msg:
            _log.tag("WIRE", f"command {' '.join(cmd_data)} msg={msg}", level="warning")
        return ok, msg, rep

    def query_addresses(self, base: int, length: int) -> List[int]:
        keys = [f"Addr-{base + i}" for i in range(length)]
        rep = self.send({"reqType": "addressing", "addrList": keys})
        raw = rep.get("dataList", [])
        values: List[int] = []
        for idx in range(length):
            try:
                values.append(int(raw[idx]))
            except Exception:
                values.append(0)
        _log.tag("WIRE", f"query_addr base={base} len={length} -> {values}")
        return values

    def rewrite_pose(
        self, x: float, y: float, z: float, u: float, v: float, w: float
    ) -> Tuple[bool, str, Dict[str, Any]]:
        vals = [x, y, z, u, v, w]
        scaled = [int(round(float(val) * 1000.0)) for val in vals]
        cmd_args = ["rewriteDataList", "800", "6", "0", *map(str, scaled)]
        _log.tag("WIRE", f"rewrite_pose vals={vals} scaled={scaled}")
        return self.command(*cmd_args)


def create_client(config: BorunteConfig = BORUNTE_CONFIG) -> RobotClient:
    client = RobotClient(config=config)
    client.connect()
    return client


__all__ = ["RobotClient", "create_client"]
