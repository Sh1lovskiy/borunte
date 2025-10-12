# new5.py-

"""
new4.py — Borunte RemoteMonitor runner with uniform 3D grid and robust I/O.

- Writes XYZUVW (x1000) into Addr-800..805 via command "rewriteDataList".
- For each point: write -> Single Cycle -> set GSPD -> start -> monitor.
- Readback verifies via: readDataList -> query(Addr-*) -> addressing.
- Heartbeat + host reassert to avoid "connect host fail" mid-run.
- Grid is uniform in 3D: XxYxZ cells sized by box aspect; ordered Y->X->Z.
- English logs, <=55 lines per function, <=80 chars per line.
"""

from __future__ import annotations

import logging
import json
import signal
import socket
import threading
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # fallback if tqdm is not installed

# -------------------------- USER CONFIG ---------------------------------

IP = "192.168.4.4"
PORT = 9760

TO = 3.0
POLL = 0.2
ALARM_POLL = 1.0
HEARTBEAT_PERIOD = 5.0
HOST_KEEPALIVE = 9.0  # reassert host/remote before this many seconds

WAIT_MODE = 2.5
WAIT_START = 6.0
WAIT_MOVE = 180.0
WAIT_STOP = 5.0

SPEED_PERCENT = 45.0
POS_TOL = 2.0
ANG_TOL = 2.0

WORKSPACE_M: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
    (40.0, 440.0),  # X
    (747.0, 1250.0),  # Y
    (470.0, 770.0),  # Z
)

TCP_DOWN_UVW = (180.0, 0.0, -60.0)
DEV_MAX_DEG = 20.0

LOG_LEVEL = logging.INFO

# -------------------------- GLOBAL STATE --------------------------------

_shutdown = False
_pack_id = 0
_pack_lock = threading.Lock()


# -------------------------- LOGGING -------------------------------------


def setup_logging() -> None:
    """Configure root logger."""
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=LOG_LEVEL, format=fmt)


# -------------------------- SIGNALS -------------------------------------


def _sig_handler(signum, frame) -> None:
    """Set shutdown flag; do not raise inside signal handler."""
    global _shutdown
    _shutdown = True
    logging.warning("[INT ] shutdown flag set")


# -------------------------- JSON WIRE -----------------------------------


def _next_pack_id() -> str:
    """Return monotonic packID as a string."""
    global _pack_id
    with _pack_lock:
        _pack_id = (_pack_id + 1) % 10_000_000
        return f"{_pack_id}"


def _recv_json(
    sock: socket.socket, timeout: float, allow_interrupt: bool = False
) -> dict:
    """Robust JSON receive with line scanning and timeout."""
    end = time.time() + timeout
    sock.settimeout(max(0.1, timeout))
    buf = bytearray()
    while True:
        if _shutdown and not allow_interrupt:
            raise KeyboardInterrupt
        remain = end - time.time()
        if remain <= 0:
            raise TimeoutError("recv timeout")
        sock.settimeout(remain)
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed")
        buf.extend(chunk)
        text = buf.decode("utf-8", "ignore").strip()
        if "\n" in text:
            for line in reversed(text.split("\n")):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except Exception:
                    continue
        try:
            return json.loads(text)
        except Exception:
            continue


def _send(sock: socket.socket, obj: dict, allow_interrupt: bool = False) -> dict:
    """Send a JSON request and return parsed reply."""
    if _shutdown and not allow_interrupt:
        raise KeyboardInterrupt
    payload = dict(obj)
    payload.setdefault("dsID", "www.hc-system.com.RemoteMonitor")
    payload.setdefault("packID", _next_pack_id())
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)
    rep = _recv_json(sock, TO, allow_interrupt=allow_interrupt)
    logging.debug(f"[WIRE] {obj.get('reqType')} -> {rep}")
    return rep


# -------------------------- API THIN LAYER -------------------------------


def q(sock: socket.socket, addrs: Sequence[str]) -> List[str]:
    """Generic query by address keys."""
    rep = _send(sock, {"reqType": "query", "queryAddr": list(addrs)})
    lst = [str(x) for x in rep.get("queryData", [])]
    return lst


def cmd(
    sock: socket.socket, *cmd_data: str, allow_interrupt: bool = False
) -> Tuple[bool, str, dict]:
    """Command wrapper returning (ok, msg, full_reply)."""
    rep = _send(
        sock,
        {"reqType": "command", "cmdData": list(cmd_data)},
        allow_interrupt=allow_interrupt,
    )
    cr = rep.get("cmdReply", [])
    ok, msg = False, ""
    if len(cr) == 1:
        ok = cr[0] == "ok"
    elif len(cr) >= 2:
        ok = cr[1] == "ok"
        if not ok and len(cr) >= 3:
            msg = str(cr[2])
    logging.debug(f"[CMD ] {cmd_data} -> ok={ok} msg='{msg}'")
    return ok, msg, rep


# -------------------------- STATUS --------------------------------------


def query_mode_move_alarm(sock) -> Tuple[int, int, int]:
    """Return (curMode, isMoving, curAlarm)."""
    cm, mv, al = q(sock, ["curMode", "isMoving", "curAlarm"])
    return int(cm), int(mv), int(al)


def query_world(sock) -> Tuple[float, float, float, float, float, float]:
    """Return world XYZUVW floats."""
    vals = q(sock, [f"world-{i}" for i in range(6)])
    return tuple(float(v) for v in vals[:6])  # type: ignore


def _read_addrs_by_mode(sock, base=800, length=6) -> Optional[List[int]]:
    """Try readDataList or query or addressing; return ints or None."""
    # 1) readDataList
    ok, msg, rep = cmd(sock, "readDataList", f"{base}", f"{length}")
    if ok:
        cr = rep.get("cmdReply", [])
        vals = cr[2 : 2 + length]
        try:
            return [int(v) for v in vals]
        except Exception:
            logging.debug("[ADDR] readDataList parse failed")
    # 2) query(Addr-*)
    keys = [f"Addr-{base + i}" for i in range(length)]
    try:
        vals_s = q(sock, keys)
        return [int(v) for v in vals_s]
    except Exception:
        logging.debug("[ADDR] query path failed")
    # 3) addressing
    try:
        rep = _send(sock, {"reqType": "addressing", "addrList": keys})
        raw = rep.get("dataList", [])
        out = []
        for i in range(length):
            try:
                out.append(int(raw[i]))
            except Exception:
                out.append(0)
        return out
    except Exception:
        logging.debug("[ADDR] addressing path failed")
    return None


def read_addrs_retry(sock, base=800, length=6, tries=3, gap=0.15) -> List[int]:
    """Read with retries; log method and values."""
    for i in range(tries):
        vals = _read_addrs_by_mode(sock, base, length)
        if vals is not None and any(v != 0 for v in vals):
            logging.info(f"[ADDR] {base}..{base+length-1} = {vals}")
            return vals
        time.sleep(gap)
    logging.warning(f"[ADDR] zero/None after retries for {base}..{base+length-1}")
    return [0] * length


# -------------------------- WRITE ---------------------------------------


def _thousand(x: float) -> int:
    """Scale mm/deg to integer x1000."""
    return int(round(x * 1000.0))


def rewrite_800_805(sock, x, y, z, u, v, w) -> Tuple[bool, str, dict]:
    """Write six values into Addr-800..805 via command signature."""
    vals = [_thousand(a) for a in (x, y, z, u, v, w)]
    args = ("rewriteDataList", "800", "6", "0", *map(str, vals))
    ok, msg, rep = cmd(sock, *args)
    logging.info(f"[WRITE] 800..805 ← {vals} ok={ok} msg='{msg}'")
    return ok, msg, rep


# -------------------------- CONTROL / SESSION ---------------------------


def start_button(sock):
    return cmd(sock, "startButton")


def single_cycle(sock):
    return cmd(sock, "actionSingleCycle")


def clear_alarm(sock):
    return cmd(sock, "clearAlarm")


def clear_alarm_continue(sock):
    return cmd(sock, "clearAlarmContinue")


def stop_button(sock, *, allow_interrupt=False):
    return cmd(sock, "stopButton", allow_interrupt=allow_interrupt)


def pause_action(sock, *, allow_interrupt=False):
    return cmd(sock, "actionPause", allow_interrupt=allow_interrupt)


def stop_action(sock, *, allow_interrupt=False):
    return cmd(sock, "actionStop", allow_interrupt=allow_interrupt)


def modify_gspd(sock, p):
    val = int(round(max(0.0, min(100.0, float(p))) * 10))
    return cmd(sock, "modifyGSPD", f"{val}")


def assert_host_session(sock) -> None:
    """Reassert remote/host to avoid mid-run disconnects."""
    cmds = [("setRemoteMode", "1"), ("connectHost", "1")]
    for c in cmds:
        ok, msg, _ = cmd(sock, *c, allow_interrupt=True)
        logging.debug(f"[HOST] {c} ok={ok} msg='{msg}'")


class Heartbeat:
    """Background heartbeat thread."""

    def __init__(self, sock: socket.socket, period: float) -> None:
        self.sock = sock
        self.period = period
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._t.start()

    def stop(self, final_ping: bool = True) -> None:
        if final_ping:
            try:
                _send(self.sock, {"reqType": "heartbreak"}, allow_interrupt=True)
            except Exception:
                pass
        self._stop.set()
        self._t.join(timeout=1.0)

    def _loop(self) -> None:
        last_host = 0.0
        while not self._stop.is_set():
            if _shutdown:
                break
            try:
                _send(self.sock, {"reqType": "heartbreak"})
            except Exception:
                logging.error("[HB  ] heartbeat send failed")
                break
            now = time.time()
            if now - last_host > HOST_KEEPALIVE:
                try:
                    assert_host_session(self.sock)
                except Exception:
                    pass
                last_host = now
            self._stop.wait(self.period)


# -------------------------- MOTION STEP ---------------------------------


def _pre_start_snapshot(sock) -> None:
    """Log mode/move/alarm and WORLD before motion."""
    cm, mv, al = query_mode_move_alarm(sock)
    w = query_world(sock)
    logging.info(
        f"[PRE ] mode={cm} moving={mv} alarm={al} "
        f"WORLD X={w[0]:.3f} Y={w[1]:.3f} Z={w[2]:.3f} "
        f"U={w[3]:.3f} V={w[4]:.3f} W={w[5]:.3f}"
    )


def _wait_mode_ready(sock) -> None:
    """Wait Single Cycle latch (curMode==9) with timeout."""
    t0 = time.time()
    while time.time() - t0 < WAIT_MODE:
        if _shutdown:
            raise KeyboardInterrupt
        cm, _, _ = query_mode_move_alarm(sock)
        if cm == 9:
            return
        time.sleep(0.1)
    logging.warning("[MODE] Single Cycle did not latch")


def _wait_started(sock) -> bool:
    """Wait robot starts moving."""
    t0 = time.time()
    last_alarm = 0
    while time.time() - t0 < WAIT_START:
        if _shutdown:
            return False
        cm, mv, al = query_mode_move_alarm(sock)
        if al and al != last_alarm:
            logging.warning(f"[ALRM] {al}")
            last_alarm = al
        if mv == 1:
            w = query_world(sock)
            logging.info(
                f"[STATE] start mode={cm} alarm={al} "
                f"WORLD X={w[0]:.3f} Y={w[1]:.3f} Z={w[2]:.3f} "
                f"U={w[3]:.3f} V={w[4]:.3f} W={w[5]:.3f}"
            )
            return True
        time.sleep(POLL)
    logging.warning("[START] motion did not start")
    return False


def _monitor_motion(sock) -> None:
    """Monitor until motion stops or timeout."""
    t0 = time.time()
    last_alarm = 0
    last_log = 0.0
    while True:
        if _shutdown:
            return
        time.sleep(POLL)
        cm, mv, al = query_mode_move_alarm(sock)
        if al and (al != last_alarm or time.time() - last_log >= ALARM_POLL):
            logging.warning(f"[ALRM] {al}")
            last_alarm = al
            last_log = time.time()
        if time.time() - last_log >= 1.0:
            w = query_world(sock)
            logging.info(
                f"[STATE] move mode={cm} alarm={al} "
                f"WORLD X={w[0]:.3f} Y={w[1]:.3f} Z={w[2]:.3f} "
                f"U={w[3]:.3f} V={w[4]:.3f} W={w[5]:.3f}"
            )
            last_log = time.time()
        if mv == 0:
            break
        if time.time() - t0 > WAIT_MOVE:
            logging.warning("[MOVE] timeout, stopping")
            stop_button(sock)
            break


def write_and_run_point(sock, pose) -> bool:
    """Full step: write->mode->speed->start->monitor with verification."""
    logging.info(f"[STEP] target {tuple(round(x, 3) for x in pose)}")

    before = read_addrs_retry(sock, 800, 6)
    logging.info(f"[ADDR] before {before}")

    ok, msg, _ = rewrite_800_805(sock, *pose)
    time.sleep(0.12)
    after = read_addrs_retry(sock, 800, 6)
    logging.info(f"[ADDR] after  {after}")

    if not ok and all(v == 0 for v in after):
        logging.error(f"[WRITE] failed: {msg}")
        return False

    ok, msg, _ = single_cycle(sock)
    logging.info(f"[MODE] singleCycle ok={ok} msg='{msg}'")
    _wait_mode_ready(sock)

    ok, msg, _ = modify_gspd(sock, SPEED_PERCENT)
    logging.info(f"[GSPD] ok={ok} msg='{msg}'")

    _pre_start_snapshot(sock)
    ok, msg, _ = start_button(sock)
    logging.info(f"[RUN ] ok={ok} msg='{msg}'")

    if not _wait_started(sock):
        if _shutdown:
            return False
        clear_alarm(sock)
        clear_alarm_continue(sock)
        return False

    _monitor_motion(sock)

    w = query_world(sock)
    dx, dy, dz = pose[0] - w[0], pose[1] - w[1], pose[2] - w[2]
    du, dv, dw = pose[3] - w[3], pose[4] - w[4], pose[5] - w[5]
    logging.info(
        f"[DONE] WORLD X={w[0]:.3f} Y={w[1]:.3f} Z={w[2]:.3f} "
        f"U={w[3]:.3f} V={w[4]:.3f} W={w[5]:.3f}"
    )
    logging.info(
        f"[DIFF] dX={dx:.3f} dY={dy:.3f} dZ={dz:.3f} "
        f"dU={du:.3f} dV={dv:.3f} dW={dw:.3f}"
    )

    in_tol = (
        abs(dx) <= POS_TOL
        and abs(dy) <= POS_TOL
        and abs(dz) <= POS_TOL
        and abs(du) <= ANG_TOL
        and abs(dv) <= ANG_TOL
        and abs(dw) <= ANG_TOL
    )
    logging.info(f"[CHK ] within_tolerance={in_tol}")
    return in_tol


# -------------------------- GRID BUILDER --------------------------------


def _counts_for_total(ws, total: int) -> Tuple[int, int, int]:
    """Pick (nx,ny,nz) so nx*ny*nz >= total and respects box aspect."""
    lx = abs(ws[0][1] - ws[0][0])
    ly = abs(ws[1][1] - ws[1][0])
    lz = abs(ws[2][1] - ws[2][0])
    vol = max(1e-6, lx * ly * lz)
    r = max(1, round(total ** (1 / 3)))
    # scale counts by edge lengths to approximate cubic voxels
    sx = max(lx, 1e-6)
    sy = max(ly, 1e-6)
    sz = max(lz, 1e-6)
    base = (sx * sy * sz) ** (1 / 3)
    nx = max(1, round(r * sx / base))
    ny = max(1, round(r * sy / base))
    nz = max(1, round(r * sz / base))
    while nx * ny * nz < total:
        # grow the axis with coarsest spacing
        dens = np.array([nx / sx, ny / sy, nz / sz])
        k = int(np.argmin(dens))
        if k == 0:
            nx += 1
        elif k == 1:
            ny += 1
        else:
            nz += 1
    logging.info(f"[GRID] counts nx={nx} ny={ny} nz={nz}")
    return nx, ny, nz


def _linspace_any(a: float, b: float, n: int) -> np.ndarray:
    """Support descending ranges too."""
    return np.linspace(a, b, num=n)


def _jitter_down(rng: np.random.Generator, dev: float) -> Tuple[float, float, float]:
    """Return deviations for 1..3 axes around down orientation."""
    axes = [0, 1, 2]
    rng.shuffle(axes)
    k = int(rng.integers(1, 4))
    sel = set(axes[:k])
    du = float(rng.uniform(-dev, dev)) if 0 in sel else 0.0
    dv = float(rng.uniform(-dev, dev)) if 1 in sel else 0.0
    dw = float(rng.uniform(-dev, dev)) if 2 in sel else 0.0
    return du, dv, dw


def build_grid_for_count(
    ws: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    total: int,
    rx_base: float = TCP_DOWN_UVW[0],
    ry_base: float = TCP_DOWN_UVW[1],
    rz_base: float = TCP_DOWN_UVW[2],
    seed: int = 42,
    dev_max_deg: float = DEV_MAX_DEG,
) -> List[List[float]]:
    """
    Return an ordered scan list of [X,Y,Z,U,V,W] uniformly across the box.
    Ordering: outer Y, then X, inner Z. Z steps cover full range uniformly.
    """
    rng = np.random.default_rng(int(seed))
    nx, ny, nz = _counts_for_total(ws, total)

    Xs = _linspace_any(ws[0][0], ws[0][1], nx)
    Ys = _linspace_any(ws[1][0], ws[1][1], ny)
    Zs = _linspace_any(ws[2][0], ws[2][1], nz)

    poses: List[List[float]] = []
    for y in Ys:
        for x in Xs:
            for z in Zs:
                du, dv, dw = _jitter_down(rng, dev_max_deg)
                u = rx_base + du
                v = ry_base + dv
                w = rz_base + dw
                poses.append(
                    [float(x), float(y), float(z), float(u), float(v), float(w)]
                )

    if len(poses) > total:
        poses = poses[:total]
    elif len(poses) < total:
        poses += poses[-1:] * (total - len(poses))

    logging.info(f"[GRID] built {len(poses)} poses")
    return poses


# -------------------------- DISCONNECT ----------------------------------


def _host_disconnect_ritual(sock, hb: Optional[Heartbeat]) -> None:
    """Graceful release to avoid ERR9 on exit."""
    try:
        try:
            cm, mv, _ = query_mode_move_alarm(sock)
        except Exception:
            mv = 1
        if mv:
            pause_action(sock, allow_interrupt=True)
            time.sleep(0.1)
        stop_button(sock, allow_interrupt=True)
        stop_action(sock, allow_interrupt=True)

        t0 = time.time()
        while not _shutdown and time.time() - t0 < WAIT_STOP:
            try:
                cm, mv, _ = query_mode_move_alarm(sock)
                if cm == 3 and mv == 0:
                    break
            except Exception:
                break
            time.sleep(0.2)

        if hb:
            hb.stop(final_ping=True)

        # tell controller host is leaving, then short pause
        for c in [
            ("setRemoteMode", "0"),
            ("connectHost", "0"),
            ("endHeartbreak",),
            ("exitRemoteMonitor",),
            ("disconnectHost",),
            ("closeConnectHost",),
            ("hostExit",),
            ("logout",),
        ]:
            try:
                ok, msg, _ = cmd(sock, *c, allow_interrupt=True)
                if not ok and msg:
                    logging.debug(f"[DISC] {c} -> {msg}")
            except Exception:
                pass
        time.sleep(0.8)
    except Exception:
        pass


def graceful_socket_close(s: socket.socket) -> None:
    """Close TCP without RST."""
    try:
        s.shutdown(socket.SHUT_WR)
        s.settimeout(0.5)
        try:
            while True:
                if not s.recv(4096):
                    break
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            s.close()
        except Exception:
            pass


# -------------------------- MAIN ----------------------------------------


def main() -> None:
    setup_logging()
    logging.info(f"[NET ] connecting {IP}:{PORT}")
    signal.signal(signal.SIGINT, _sig_handler)
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    s: Optional[socket.socket] = None
    hb: Optional[Heartbeat] = None
    try:
        s = socket.create_connection((IP, PORT), timeout=TO)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        logging.info("[NET ] connected")

        hb = Heartbeat(s, HEARTBEAT_PERIOD)
        hb.start()
        assert_host_session(s)

        try:
            cm, mv, al = query_mode_move_alarm(s)
        except TimeoutError:
            logging.info("[NET ] first query timeout, one retry")
            time.sleep(0.2)
            cm, mv, al = query_mode_move_alarm(s)
        w = query_world(s)
        logging.info(f"[INIT] mode={cm} moving={mv} alarm={al}")
        logging.info(
            f"[INIT] WORLD X={w[0]:.3f} Y={w[1]:.3f} Z={w[2]:.3f} "
            f"U={w[3]:.3f} V={w[4]:.3f} W={w[5]:.3f}"
        )

        poses = build_grid_for_count(
            WORKSPACE_M,
            total=36,
            rx_base=TCP_DOWN_UVW[0],
            ry_base=TCP_DOWN_UVW[1],
            rz_base=TCP_DOWN_UVW[2],
            seed=7,
            dev_max_deg=DEV_MAX_DEG,
        )

        it = range(1, len(poses) + 1)
        if tqdm:
            it = tqdm(it, desc="Grid", unit="pt")

        last_host_assert = 0.0
        for idx in it:
            if _shutdown:
                break
            p = poses[idx - 1]
            now = time.time()
            if now - last_host_assert > HOST_KEEPALIVE:
                assert_host_session(s)
                last_host_assert = now
            logging.info(f"[GRID] {idx}/{len(poses)}")
            ok = write_and_run_point(s, p)
            if not ok and not _shutdown:
                clear_alarm(s)
                clear_alarm_continue(s)

        _host_disconnect_ritual(s, hb)
        logging.info("[FINAL] released to pendant")

    except KeyboardInterrupt:
        logging.warning("[INT ] KeyboardInterrupt")
        try:
            if s:
                pause_action(s, allow_interrupt=True)
                stop_button(s, allow_interrupt=True)
                stop_action(s, allow_interrupt=True)
                _host_disconnect_ritual(s, hb)
        finally:
            pass
    except Exception as e:
        logging.error(f"[ERROR] {e}")
        try:
            if s:
                _host_disconnect_ritual(s, hb)
        finally:
            pass
        raise
    finally:
        if s:
            graceful_socket_close(s)
        logging.info("[FINAL] done")


if __name__ == "__main__":
    main()
