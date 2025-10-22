# borunte/client.py
"""Borunte RemoteMonitor JSON TCP client.

Provides JSON TCP communication with Borunte robot controller via RemoteMonitor protocol.
Handles pose reading/writing to registers 800-805 with position and angle scaling.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

from loguru import logger

from borunte.config import (
    ROBOT_ANG_SCALE,
    ROBOT_POS_SCALE,
    ROBOT_READ_RETRIES,
    ROBOT_REGISTER_BASE_ADDR,
)


class RobotClient:
    """TCP JSON-клиент протокола RemoteMonitor."""

    def __init__(self, host: str, port: int = 9760, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._log = logger.bind(module="borunte.client")
        self._pack = 0
        self._lock = threading.Lock()

    # ────────────── базовая связь ──────────────

    def connect(self) -> bool:
        if self._sock:
            self.disconnect()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(self.timeout)
            s.connect((self.host, self.port))
            self._sock = s
            self._log.info(f"connected to {self.host}:{self.port}")
            return True
        except Exception as exc:
            self._log.error(f"connect failed: {exc!r}")
            self._sock = None
            return False

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._log.info("disconnected")

    def is_connected(self) -> bool:
        return self._sock is not None

    def _next_pack(self) -> str:
        with self._lock:
            self._pack = (self._pack + 1) % 1000000
            return str(self._pack)

    def _recv_json(
        self, timeout: float | None = None, allow_interrupt: bool = False
    ) -> dict[str, Any] | None:
        if not self._sock:
            return None
        old = self._sock.gettimeout()
        if timeout is not None:
            self._sock.settimeout(timeout)
        try:
            buf = bytearray()
            # сервер присылает один JSON; иногда с '\n'
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
                buf.extend(chunk)
                try:
                    return json.loads(buf.decode("utf-8"))
                except json.JSONDecodeError:
                    # ждём ещё
                    continue
        except TimeoutError:
            if not allow_interrupt:
                self._log.warning(f"timeout after {self._sock.gettimeout()}s")
            return None
        except Exception as exc:
            self._log.error(f"recv error: {exc!r}")
            return None
        finally:
            if timeout is not None and self._sock:
                self._sock.settimeout(old)
        return None

    def _send_recv(
        self, payload: dict[str, Any], timeout: float | None = None
    ) -> dict[str, Any] | None:
        if not self._sock:
            self._log.warning("not connected")
            return None
        obj = dict(payload)
        obj.setdefault("dsID", "www.hc-system.com.RemoteMonitor")
        obj.setdefault("packID", self._next_pack())
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._sock.sendall(data)
        except Exception as exc:
            self._log.error(f"send error: {exc!r}")
            return None
        return self._recv_json(timeout=timeout)

    # ────────────── примитивы протокола ──────────────

    def query(self, keys: list[str], timeout: float | None = None) -> list[str] | None:
        rep = self._send_recv({"reqType": "query", "queryAddr": keys}, timeout=timeout)
        if not rep:
            return None
        data = rep.get("queryData")
        return data if isinstance(data, list) else None

    def command(
        self, cmd: str, *args: str, timeout: float | None = None
    ) -> tuple[bool, str, dict[str, Any] | None]:
        rep = self._send_recv({"reqType": "command", "cmdData": [cmd, *args]}, timeout=timeout)
        if not rep:
            return False, "no reply", None
        cr = rep.get("cmdReply", [])
        if not cr:
            return False, "empty cmdReply", rep
        # форматы встречаются разные, нормализуем
        if len(cr) == 1:
            status = cr[0]
            return (status == "ok"), status, rep
        status = cr[1]
        msg = cr[2] if len(cr) > 2 else ""
        return (status == "ok"), msg, rep

    # ────────────── чтение/запись позы ──────────────

    def _read_readDataList(self, start: int, count: int) -> list[int] | None:
        ok, msg, rep = self.command("readDataList", str(start), str(count), "0", timeout=3.0)
        if not ok or not rep:
            return None
        data = rep.get("cmdReply", [])
        # сервер может возвращать значения строками после заголовка
        vals: list[int] = []
        for v in data[3:]:
            try:
                vals.append(int(v))
            except Exception:
                return None
        return vals if len(vals) == count else None

    def _read_query_addrs(self, start: int, count: int) -> list[int] | None:
        keys = [f"Addr-{start + i}" for i in range(count)]
        vals = self.query(keys, timeout=3.0)
        if not vals or len(vals) != count:
            return None
        try:
            return [int(v) for v in vals]
        except Exception:
            return None

    def _read_addressing(self, start: int, count: int) -> list[int] | None:
        # запасной путь: индивидуально по одному адресу через query()
        try:
            out: list[int] = []
            for i in range(count):
                vals = self.query([f"Addr-{start + i}"], timeout=2.0)
                if not vals or len(vals) != 1:
                    return None
                out.append(int(vals[0]))
            return out
        except Exception:
            return None

    def read_registers(
        self, start: int = ROBOT_REGISTER_BASE_ADDR, count: int = 6
    ) -> list[int] | None:
        for _ in range(ROBOT_READ_RETRIES):
            vals = (
                self._read_readDataList(start, count)
                or self._read_query_addrs(start, count)
                or self._read_addressing(start, count)
            )
            if vals and len(vals) == count:
                self._log.info(
                    f"registers {start}-{start + count - 1}: [{', '.join(str(v) for v in vals)}]"
                )
                return vals
            time.sleep(0.05)
        self._log.warning(f"failed to read registers {start}-{start + count - 1}")
        return None

    def rewrite_pose(
        self, x: float, y: float, z: float, u: float, v: float, w: float
    ) -> tuple[bool, str, dict[str, Any] | None]:
        vals = [
            int(round(x * ROBOT_POS_SCALE)),
            int(round(y * ROBOT_POS_SCALE)),
            int(round(z * ROBOT_POS_SCALE)),
            int(round(u * ROBOT_ANG_SCALE)),
            int(round(v * ROBOT_ANG_SCALE)),
            int(round(w * ROBOT_ANG_SCALE)),
        ]
        rep = self._send_recv(
            {
                "reqType": "command",
                "cmdData": [
                    "rewriteDataList",
                    "800",
                    "6",
                    "0",
                    *[str(v) for v in vals],
                ],
            }
        )
        if not rep:
            return False, "no reply", None
        cr = rep.get("cmdReply", [])
        if not cr:
            return False, "empty cmdReply", rep
        if len(cr) == 1:
            status = cr[0]
            return (status == "ok"), status, rep
        status = cr[1]
        msg = cr[2] if len(cr) > 2 else ""
        return (status == "ok"), msg, rep

    def verify_pose_write(self, x: float, y: float, z: float, u: float, v: float, w: float) -> bool:
        expected = [
            int(round(x * ROBOT_POS_SCALE)),
            int(round(y * ROBOT_POS_SCALE)),
            int(round(z * ROBOT_POS_SCALE)),
            int(round(u * ROBOT_ANG_SCALE)),
            int(round(v * ROBOT_ANG_SCALE)),
            int(round(w * ROBOT_ANG_SCALE)),
        ]
        time.sleep(0.06)
        actual: list[int] | None = None
        for _ in range(ROBOT_READ_RETRIES):
            actual = self.read_registers(ROBOT_REGISTER_BASE_ADDR, 6)
            if actual and len(actual) == 6:
                break
            time.sleep(0.05)
        if not actual:
            self._log.warning("failed to read registers 800-805")
            return False
        match = all(abs(a - e) <= 1 for a, e in zip(actual, expected))
        if match:
            self._log.info("pose verified OK")
        else:
            self._log.warning(f"pose mismatch! expected=[{', '.join(str(v) for v in expected)}]")
        return match

    # ────────────── статусы ──────────────

    def heartbeat(self) -> tuple[bool, str]:
        rep = self._send_recv({"reqType": "heartbreak"})
        return (rep is not None, "ok" if rep else "no reply")

    def get_world_pose(self) -> tuple[float, ...] | None:
        vals = self.query(
            ["world-0", "world-1", "world-2", "world-3", "world-4", "world-5"],
            timeout=3.0,
        )
        if not vals or len(vals) < 6:
            return None
        try:
            return tuple(float(v) for v in vals[:6])
        except Exception:
            return None

    def get_mode(self) -> str | None:
        vals = self.query(["curMode"], timeout=3.0)
        return vals[0] if vals else None

    def is_moving(self) -> bool | None:
        vals = self.query(["isMoving"], timeout=3.0)
        if vals and vals[0] in ("0", "1"):
            return vals[0] == "1"
        return None

    def get_alarm(self) -> str | None:
        vals = self.query(["curAlarm"], timeout=3.0)
        return vals[0] if vals else None

    # ────────────── host/remote вспомогательное ──────────────

    def assert_host_session(self) -> None:
        """Реассерт удалённого управления/host (best-effort, не бросает)."""
        for c in (("setRemoteMode", "1"), ("connectHost", "1")):
            try:
                ok, msg, _ = self.command(*c)
                if not ok and msg:
                    self._log.debug(f"HOST {c} -> {msg}")
            except Exception:
                pass

    # ────────────── движение ──────────────

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        u: float,
        v: float,
        w: float,
        verify: bool = True,
    ) -> bool:
        self._log.info(f"move_to_pose: x={x:.1f} y={y:.1f} z={z:.1f} u={u:.3f} v={v:.3f} w={w:.3f}")
        ok, msg, _ = self.rewrite_pose(x, y, z, u, v, w)
        if not ok:
            self._log.error(f"rewrite_pose failed: {msg}")
            return False
        self._log.info("pose written to registers 800-805")

        if verify and not self.verify_pose_write(x, y, z, u, v, w):
            self._log.error("pose verification failed")
            return False

        time.sleep(0.1)

        ok, msg, _ = self.command("actionSingleCycle", timeout=3.0)
        if not ok:
            self._log.error(f"actionSingleCycle failed: {msg}")
            return False
        self._log.info("entered Single Loop mode")

        time.sleep(0.1)

        ok, msg, _ = self.command("startButton", timeout=3.0)
        if not ok:
            self._log.error(f"startButton failed: {msg}")
            return False
        self._log.info("motion started")
        return True
