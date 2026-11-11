"""Base-station side of fleetlink (API.md §8).

``Receiver(transport, on_update)`` mirrors every robot's graph (``GraphState``)
and mesh tiles, tracks link statistics, detects C2 sequence gaps and epoch
changes (→ rate-limited resync request) and sends C0 requests with retry until
acked.  A C2 gap holds the later deltas back and asks the agent to ``resend``
the missing range from its ring buffer; after ``resend_attempts`` tries (or when
the agent says the range is gone) it falls back to a rate-limited resync.
Once a second per robot it sends the C0 ``fb`` op (goodput, loss per class from
seq gaps, RTT) that drives the agent's auto mode.  Requests are never put
while the receiver's lock is held: they go to an outbox flushed afterwards.
``on_update(robot, kind, data)`` kinds:

* ``snapshot`` - ``{"epoch", "nodes", "edges"}`` after a snapshot completed
* ``delta``    - ``GraphDelta.encode()`` dict + ``{"seq"}``
* ``tiles``    - ``{"changed": [keys], "removed": [keys]}``
* ``state``    - the robot's C1 state dict
* ``link``     - same as :meth:`Receiver.link` (once per second per robot)
* ``grid``     - ``{hash, frame, width, height, resolution, t}`` (data via :meth:`Receiver.grid`)
* ``paths``    - the paths dict ``{smooth, mlp, target, frame, t, hash}``
* ``video``    - one C5 video frame ``{camera, codec, seq, t, key, width, height,
  fps, data, ...}`` plus ``gap`` (True when frames of that stream were lost
  in between; a delta after a gap cannot be decoded)
* ``lidar``    - ``{frame, n, stamp, seq}`` of a new C3 lidar frame (the frame
  itself via :meth:`Receiver.lidar`)

Live video is turned on / kept alive / off with :meth:`Receiver.video`
(C0 ``video`` op); :meth:`Receiver.video_stats` has per-stream receive rates.

The lidar stream is turned on / off with :meth:`Receiver.set_lidar` (C0
``lidar`` op).  Like ``store_send`` the wanted state is remembered and re-asked
whenever the robot's state disagrees (an agent restart starts with it off);
unlike it, nobody reading the frames (:meth:`Receiver.lidar`, i.e. the console's
``GET .../lidar`` poll) for ``lidar_idle_s`` turns it off again, so a closed
browser tab does not leave a robot streaming its lidar over the radio.  While on
it is refreshed every ``lidar_refresh_s``; the agent turns it off by itself
after ``lidar_ttl_s`` without a request (a base that went away).

Camera frames are pulled on demand with :meth:`Receiver.camera` (blocking) or
:meth:`Receiver.query_camera` (callback), over the transport's C4 request/reply.

Time comes from ``clock`` (default ``time.time``); without a background thread
(``threaded=False``) call :meth:`tick` yourself.
"""

from __future__ import annotations

import collections
import itertools
import secrets
import threading
import time
from typing import Callable

import msgpack

from fleetlink import codec
from fleetlink.codec import GraphDelta, GraphState
from fleetlink.transport import CLASSES, Transport

STREAM_CLASS = {
    "ctl/ack": "C0",
    "state": "C1",
    "dsg/delta": "C2",
    "mesh/tile": "C3",
    "mesh/manifest": "C3",
    "mesh/values": "C3",
    "grid/occupancy": "C3",
    "paths": "C1",
    "store/frame": "C6",
    "lidar/points": "C3",
}
RESEND_STREAMS = ("grid", "paths")
# classes whose budget / rate the base reports and may set (C0 is unmetered)
REPORTED_CLASSES = ("C1", "C2", "C3", "C4", "C5", "C6")
VIDEO_PREFIX = "video/"


def stream_class(stream: str) -> str | None:
    if stream.startswith(VIDEO_PREFIX):
        return "C5"
    return STREAM_CLASS.get(stream)


class _VideoRx:
    """Receive-side stats of one video stream."""

    def __init__(self):
        self.frames = 0
        self.keys = 0
        self.bytes = 0
        self.gaps = 0
        self.lost = 0
        self.win_frames = 0
        self.win_bytes = 0
        self.fps = 0.0
        self.kbps = 0.0
        self.last: dict = {}
        self.last_rx: float | None = None

    def stats(self, now: float) -> dict:
        return {
            "frames": self.frames,
            "keys": self.keys,
            "bytes": self.bytes,
            "gaps": self.gaps,
            "lost": self.lost,
            "fps": round(self.fps, 2),
            "kbps": round(self.kbps, 1),
            "age_s": None if self.last_rx is None else now - self.last_rx,
            **self.last,
        }


class _Robot:
    def __init__(self, name: str):
        self.name = name
        self.epoch: str | None = None
        self.store_send_t = -1e9  # last ``store_send`` re-sent to match the wanted state
        self.seqs: dict[str, int] = {}
        self.graph = GraphState()
        self.tiles: dict[str, dict] = {}
        self.manifest: dict[str, str] = {}
        self.mesh_lod: dict | None = None  # the robot's LOD (manifest ``lod``); None: full, old robot
        # odometric mesh mode: deforms the odometric tiles with mesh/values
        self.deform = None
        self.state: dict = {}
        self.last_seen: float | None = None
        self.last_state_rx: float | None = None
        self.latency: float | None = None
        self.rtt: float | None = None
        # robot clock - base clock, from pings (NTP-style; the lowest-RTT
        # recent sample is the least distorted by queueing)
        self.offset_samples: collections.deque = collections.deque(maxlen=20)
        self.clock_offset: float | None = None
        self.kf_req_t: dict[str, float] = {}
        self.bytes = dict.fromkeys(CLASSES, 0)
        self.rates = dict.fromkeys(CLASSES, 0.0)
        # base -> robot bytes (C0 ctl/req + camera query payloads) and its rate
        self.tx_bytes = 0
        self.tx_rate = 0.0
        self.tx_win = 0
        # the current link session: since the first message after a silence of
        # ``connected_timeout`` (or an epoch change), bytes counted from there
        self.connected_since: float | None = None
        self.sessions = 0
        self.session_rx0 = 0
        self.session_tx0 = 0
        self.win = dict.fromkeys(CLASSES, 0)
        self.win_t: float | None = None
        self.gaps = 0
        self.drops = 0
        self.resyncs = 0  # completed snapshots
        self.resync_requests = 0
        self.epochs_seen = 0
        self.retired: set[str] = set()
        self.last_resync_req = -1e9
        self.in_snapshot = False
        self.tile_req_t: dict[str, float] = {}
        self.t_ping = -1e9
        self.grid: dict | None = None
        self.paths: dict | None = None
        self.lidar: dict | None = None  # latest lidar frame (contract dict)
        self.lidar_t = -1e9  # last ``lidar`` request sent (reconcile rate limit)
        self.lidar_read_t = -1e9  # last read of the frames (idle turns it off)
        self.mismatch_since: dict[str, float] = {}
        self.resend_req_t: dict[str, float] = {}
        self.video: dict[str, _VideoRx] = {}
        # per-class loss accounting from seq gaps (window -> last-window fraction)
        self.win_msgs = dict.fromkeys(CLASSES, 0)
        self.win_lost = dict.fromkeys(CLASSES, 0)
        self.lost = dict.fromkeys(CLASSES, 0)
        self.loss = dict.fromkeys(CLASSES, 0.0)
        self.t_fb = -1e9
        # C2 in-order delivery: next expected seq, held-back later messages,
        # the outstanding resend {from, to, tries, t, rid}
        self.c2_next: int | None = None
        self.c2_hold: dict[int, dict] = {}
        self.c2_gap: dict | None = None
        self.c2_need_resync = False
        self.c2_tail_since: tuple | None = None  # (since, c2_next then)
        self.resend_requests = 0
        self.resend_escalations = 0


class Receiver:
    def __init__(
        self,
        transport: Transport,
        on_update: Callable[[str, str, object], None] | None = None,
        *,
        clock: Callable[[], float] = time.time,
        threaded: bool = True,
        tick_period: float = 0.1,
        retry_s: float = 1.0,
        max_retries: int = 30,
        resync_min_interval: float = 5.0,
        ping_period: float = 1.0,
        tile_rerequest_s: float = 10.0,
        connected_timeout: float = 3.0,
        resend_grace_s: float = 10.0,
        resend_attempts: int = 3,
        resend_timeout_s: float = 2.0,
        hold_max: int = 1024,
        fb_period: float = 1.0,
        store_root: str | None = None,
        store_mirror=None,
        mesh_lod: str | None = "decimated",
        lidar_idle_s: float | None = 15.0,
        lidar_refresh_s: float = 20.0,
    ):
        self.transport = transport
        self.on_update = on_update or (lambda r, k, d: None)
        self.clock = clock
        self.threaded = threaded
        self.tick_period = tick_period
        self.retry_s = retry_s
        self.max_retries = max_retries
        self.resync_min_interval = resync_min_interval
        self.ping_period = ping_period
        self.tile_rerequest_s = tile_rerequest_s
        self.connected_timeout = connected_timeout
        self.resend_grace_s = resend_grace_s
        self.resend_attempts = resend_attempts
        self.resend_timeout_s = resend_timeout_s
        self.hold_max = hold_max
        self.fb_period = fb_period
        # mesh LOD asked of every robot when its epoch starts (None: never ask,
        # robots send full tiles); per-robot choices from ``set_mesh_lod`` win
        self.default_mesh_lod = mesh_lod
        self._mesh_lod_wanted: dict[str, dict] = {}
        # C6 image-store mirror on/off asked per robot (``set_store_send``);
        # a robot absent here keeps what it was launched with (off by default)
        self._store_send_wanted: dict[str, bool] = {}
        # lidar stream on/off asked per robot (``set_lidar``); off when unread
        # for ``lidar_idle_s`` (None: never)
        self._lidar_wanted: dict[str, bool] = {}
        self.lidar_idle_s = lidar_idle_s
        # while on, ``lidar`` is re-sent this often: the agent stops by itself
        # when no request came for its ``lidar_ttl_s`` (a base that went away)
        self.lidar_refresh_s = lidar_refresh_s
        self._outbox: list[tuple[str, dict]] = []
        self._robots: dict[str, _Robot] = {}
        self._pending: dict[str, dict] = {}  # req id -> {robot, req, t, tries}
        self._ids = itertools.count()
        self._session = secrets.token_hex(3)
        self.started = clock()  # byte totals count from here
        self._req_seq = itertools.count()
        self._lock = threading.RLock()
        self._sub = None
        self._thread: threading.Thread | None = None
        self._running = False
        self.failed_requests: list[dict] = []
        # C6 live image-store mirror (fleetlink.store_mirror.StoreMirror)
        self.store = store_mirror
        if self.store is None and store_root:
            from fleetlink.store_mirror import StoreMirror

            self.store = StoreMirror(store_root, clock=clock)

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._sub = self.transport.subscribe("adt4/**", self._on_msg)
        if self.threaded:
            self._running = True
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="fleetlink-rx"
            )
            self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self.store is not None:
            try:
                self.store.close()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        unsub = getattr(self.transport, "unsubscribe", None)
        if unsub is not None and self._sub is not None:
            unsub(self._sub)
        self._sub = None

    def _loop(self) -> None:
        while self._running:
            try:
                self.tick()
            except Exception:
                pass
            time.sleep(self.tick_period)

    # -------------------------------------------------------------- queries

    def robots(self) -> list[str]:
        with self._lock:
            return sorted(self._robots)

    def graph(self, robot: str) -> GraphState:
        with self._lock:
            return self._robot(robot).graph

    def tiles(self, robot: str) -> dict[str, dict]:
        """The robot's mesh tiles for display: as received, or (odometric mesh
        mode) deformed with the latest control-point poses."""
        with self._lock:
            r = self._robot(robot)
            tiles, view = dict(r.tiles), r.deform
        return view.view(tiles) if view is not None else tiles

    def raw_tiles(self, robot: str) -> dict[str, dict]:
        """Tiles exactly as received (odometric ones undeformed)."""
        with self._lock:
            return dict(self._robot(robot).tiles)

    def state(self, robot: str) -> dict:
        with self._lock:
            return dict(self._robot(robot).state)

    def grid(self, robot: str) -> dict | None:
        """Latest occupancy grid wire dict (``data`` = zlib int8), or None."""
        with self._lock:
            r = self._robots.get(robot)
            return dict(r.grid) if r is not None and r.grid is not None else None

    def paths(self, robot: str) -> dict | None:
        with self._lock:
            r = self._robots.get(robot)
            return dict(r.paths) if r is not None and r.paths is not None else None

    def lidar(self, robot: str) -> dict | None:
        """Latest lidar frame ``{frame, origin, scale, n, xyz, intensity, stamp,
        seq, ...}`` (``xyz``: interleaved int16 LE in the sensor frame, units of
        ``scale`` m), or None.  Counts as a read for the idle turn-off."""
        with self._lock:
            r = self._robots.get(robot)
            if r is None:
                return None
            r.lidar_read_t = self.clock()
            return dict(r.lidar) if r.lidar is not None else None

    def cameras(self, robot: str) -> list[str]:
        """Camera names the robot advertises in its C1 state."""
        with self._lock:
            r = self._robots.get(robot)
            return list((r.state or {}).get("cameras") or []) if r is not None else []

    def query_camera(
        self,
        robot: str,
        name: str,
        callback: Callable[[dict | None], None],
        width: int = 640,
        quality: int = 70,
        timeout: float = 5.0,
    ) -> None:
        """Ask the robot for its newest ``name`` frame.  ``callback(reply)`` with
        ``{format, data, width, height, stamp, age_s, ...}`` or ``{error, ...}``;
        ``callback(None)`` on timeout."""

        def done(data):
            if data is None:
                callback(None)
                return
            try:
                rep = msgpack.unpackb(data, raw=False)
            except Exception:  # noqa: BLE001
                rep = {"error": "bad reply"}
            with self._lock:
                r = self._robot(robot)
                r.bytes["C4"] += len(data)
                r.win["C4"] += len(data)
            callback(rep)

        key = f"adt4/{robot}/media/camera/{name}"
        params = {"w": int(width), "q": int(quality)}
        with self._lock:  # ~ what goes on the wire: key + params
            self._count_tx(self._robot(robot), len(key) + len(msgpack.packb(params)))
        self.transport.query(key, params, done, timeout)

    def camera(self, robot, name, width=640, quality=70, timeout=5.0) -> dict | None:
        """Blocking :meth:`query_camera` (needs a transport that answers on its own
        threads, i.e. zenoh; with a virtual-clock link use ``query_camera``)."""
        ev = threading.Event()
        out = {}

        def cb(rep):
            out["rep"] = rep
            ev.set()

        self.query_camera(robot, name, cb, width, quality, timeout)
        ev.wait(timeout + 1.0)
        return out.get("rep")

    def link(self, robot: str, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self._lock:
            return self._link(self._robot(robot), now)

    def _link(self, r: _Robot, now: float) -> dict:
        stats = (r.state or {}).get("stats") or {}
        staleness = None if r.last_seen is None else now - r.last_seen
        return {
            "connected": staleness is not None and staleness < self.connected_timeout,
            "epoch": r.epoch,
            "last_seen": r.last_seen,
            "staleness_s": staleness,
            "state_age_s": None if r.last_state_rx is None else now - r.last_state_rx,
            "latency_ms": None if r.latency is None else 1000.0 * r.latency,
            "clock_offset_ms": (
                None if r.clock_offset is None else 1000.0 * r.clock_offset
            ),
            "rtt_ms": None if r.rtt is None else 1000.0 * r.rtt,
            "nodes": len(r.graph.records),
            "tiles": len(r.tiles),
            "mesh": self._mesh_status(r),
            "budgets": stats.get("budgets", {}),
            "rates": {c: r.rates[c] for c in REPORTED_CLASSES},
            "bytes": dict(r.bytes),
            **self._totals(r, now),
            "drops": r.drops,
            "gaps": r.gaps,
            "loss": {c: round(r.loss[c], 4) for c in CLASSES},
            "lost": dict(r.lost),
            "resend_requests": r.resend_requests,
            "resend_escalations": r.resend_escalations,
            "c2_held": len(r.c2_hold),
            "resyncs": r.resyncs,
            "resync_requests": r.resync_requests,
            "auto": stats.get("auto", False),
            "robot_queue": stats.get("queue", {}),
            # the robot's own per-class view: bytes/s put on the link and totals
            "robot_rates": stats.get("rate", {}),
            "robot_sent": stats.get("sent", {}),
            "store": self.store.stats(r.name) if self.store is not None else None,
            "store_send": self._store_send_status(r),
            # global frame <- scene-graph frame from the C1 state (None from
            # agents that predate it); ``state_age_s`` above ages the whole state
            "frame_align": (r.state or {}).get("frame_align"),
            # prior-map loop closures and map -> odom jumps (None: older agent)
            "reloc": (r.state or {}).get("reloc"),
        }

    def _totals(self, r: _Robot, now: float) -> dict:
        """Byte totals per direction: robot -> base (``rx``: every class, envelope
        bytes as received, i.e. after compression) and base -> robot (``tx``),
        since this receiver started and since the current link session began."""
        rx = sum(r.bytes.values())
        return {
            "rx_total_B": rx,
            "tx_total_B": r.tx_bytes,
            "rx_rate_Bps": sum(r.rates.values()),
            "tx_rate_Bps": r.tx_rate,
            "since_s": now - self.started,
            "connected_since": r.connected_since,
            "sessions": r.sessions,
            "rx_session_B": rx - r.session_rx0,
            "tx_session_B": r.tx_bytes - r.session_tx0,
        }

    @staticmethod
    def _count_tx(r: _Robot, n: int) -> None:
        r.tx_bytes += n
        r.tx_win += n

    @staticmethod
    def _new_session(r: _Robot, now: float) -> None:
        r.connected_since = now
        r.sessions += 1
        r.session_rx0 = sum(r.bytes.values())
        r.session_tx0 = r.tx_bytes

    def _robot(self, name: str) -> _Robot:
        r = self._robots.get(name)
        if r is None:
            r = self._robots[name] = _Robot(name)
        return r

    # ------------------------------------------------------------- requests

    def _send(
        self, robot: str, op: str, args: dict | None = None, retry: bool = True
    ) -> str:
        rid = f"{self._session}-{next(self._ids)}"
        req = {"id": rid, "op": op, "args": args or {}}
        with self._lock:
            if retry:
                self._pending[rid] = {
                    "robot": robot,
                    "req": req,
                    "t": self.clock(),
                    "tries": 1,
                }
        self._put_req(robot, req)
        return rid

    def _queue_req(self, robot: str, req: dict) -> None:
        """Lock held: the put happens in :meth:`_flush` once the lock is released."""
        self._outbox.append((robot, req))

    def _flush(self) -> None:
        with self._lock:
            out, self._outbox = self._outbox, []
        for robot, req in out:
            self._put_req(robot, req)

    def _put_req(self, robot: str, req: dict) -> None:
        env = codec.envelope("base", "ctl/req", next(self._req_seq), self._session, req)
        env["t"] = self.clock()
        data = codec.pack(env)
        with self._lock:
            self._count_tx(self._robot(robot), len(data))
        self.transport.put(f"adt4/{robot}/ctl/req", data, "C0")

    def set_budget(self, robot: str, cls: str, bps: float) -> str:
        if cls not in REPORTED_CLASSES:
            raise ValueError(f"unknown class {cls!r}")
        return self._send(robot, "budget", {cls: float(bps)})

    def video(
        self,
        robot: str,
        camera: str,
        enable: bool = True,
        retry: bool = True,
        **params,
    ) -> str:
        """C0 ``video {camera, enable, fps, kbps, width, codec}``: start / update /
        keep alive (repeat at least every 6 s) or stop (``enable=False``) a stream.
        Keepalives can go with ``retry=False``: the next one follows anyway."""
        args = {"camera": camera, "enable": bool(enable)}
        args.update({k: v for k, v in params.items() if v is not None})
        return self._send(robot, "video", args, retry=retry)

    def video_stats(self, robot: str, now: float | None = None) -> dict:
        """``{camera: {frames, keys, bytes, gaps, lost, fps, kbps, age_s, codec,
        width, height}}`` measured here, plus ``robot`` = the robot's own report
        for that stream from its C1 state (absent when it is not streaming)."""
        now = self.clock() if now is None else now
        with self._lock:
            r = self._robots.get(robot)
            if r is None:
                return {}
            reported = (r.state or {}).get("video") or {}
            out = {c: v.stats(now) for c, v in r.video.items()}
            for cam, st in reported.items():
                out.setdefault(cam, {})["robot"] = st
            return out

    def mesh_lod_wanted(self, robot: str) -> dict | None:
        """``{level, tiles}`` this base asks of ``robot`` (None: nothing asked)."""
        with self._lock:
            w = self._mesh_lod_wanted.get(robot)
            if w is not None:
                return {"level": w["level"], "tiles": dict(w["tiles"])}
            return None if self.default_mesh_lod is None else {"level": self.default_mesh_lod, "tiles": {}}

    def set_mesh_lod(self, robot: str, level: str, tiles: list | None = None) -> str:
        """C0 ``mesh_lod {level, tiles?}``: the robot re-sends its mesh tiles at
        ``level`` ("full" | "decimated"), only ``tiles`` when given (the rest
        keep the robot-wide level). Remembered and re-sent when the robot's
        epoch changes. An old robot answers "unknown op" and keeps full tiles."""
        if level not in codec.MESH_LODS:
            raise ValueError(f"unknown mesh LOD {level!r} (one of {codec.MESH_LODS})")
        with self._lock:
            cur = self._mesh_lod_wanted.get(robot) or {"level": self.default_mesh_lod or "full", "tiles": {}}
            if tiles:
                cur = {"level": cur["level"], "tiles": {**cur["tiles"], **{str(k): level for k in tiles}}}
            else:
                cur = {"level": level, "tiles": {}}
            self._mesh_lod_wanted[robot] = cur
        args = {"level": level}
        if tiles:
            args["tiles"] = [str(k) for k in tiles]
        return self._send(robot, "mesh_lod", args)

    def set_store_send(self, robot: str, enable: bool) -> str:
        """C0 ``store_send {enable}``: start / stop the robot's C6 image-store
        mirror. Remembered and re-asked whenever the robot's state disagrees
        (e.g. after an agent restart, which starts with it off)."""
        with self._lock:
            self._store_send_wanted[robot] = bool(enable)
            self._robot(robot).store_send_t = self.clock()
            for rid in [k for k, v in self._pending.items() if v["robot"] == robot and v["req"]["op"] == "store_send"]:
                del self._pending[rid]
        return self._send(robot, "store_send", {"enable": bool(enable)})

    def store_send_status(self, robot: str) -> dict:
        """``{robot, wanted}``: whether the robot sends C6 frames (None: no
        store, or an agent without the switch) and what this base asked."""
        with self._lock:
            r = self._robots.get(robot)
            return self._store_send_status(r) if r is not None else {"robot": None, "wanted": self._store_send_wanted.get(robot)}

    def _store_send_status(self, r: _Robot) -> dict:
        st = (r.state or {}).get("store") or {}
        return {"robot": st.get("send") if st.get("enabled") else None, "wanted": self._store_send_wanted.get(r.name)}

    def _store_send_reconcile(self, r: _Robot, st: dict | None, now: float) -> None:
        """Lock held: re-ask ``store_send`` when the robot's state disagrees."""
        want = self._store_send_wanted.get(r.name)
        if want is None or not st or not st.get("enabled") or "send" not in st:
            return
        if bool(st["send"]) == want or now - r.store_send_t < self.retry_s:
            return
        if any(v["robot"] == r.name and v["req"]["op"] == "store_send" for v in self._pending.values()):
            return
        r.store_send_t = now
        self._pending_send(r.name, "store_send", {"enable": want})

    def set_lidar(self, robot: str, enable: bool) -> str:
        """C0 ``lidar {enable}``: start / stop the robot's C3 lidar stream.
        Remembered and re-asked when the robot's state disagrees."""
        with self._lock:
            self._lidar_wanted[robot] = bool(enable)
            r = self._robot(robot)
            r.lidar_t = r.lidar_read_t = self.clock()
            if not enable:
                r.lidar = None
            for rid in [k for k, v in self._pending.items() if v["robot"] == robot and v["req"]["op"] == "lidar"]:
                del self._pending[rid]
        return self._send(robot, "lidar", {"enable": bool(enable)})

    def lidar_status(self, robot: str) -> dict:
        """``{available, robot, wanted}``: the robot has a lidar / streams it (None:
        unknown or no lidar state) and what this base asked."""
        with self._lock:
            r = self._robots.get(robot)
            st = ((r.state or {}).get("lidar") if r is not None else None) or {}
            return {
                "available": st.get("available"),
                "robot": st.get("enabled"),
                "wanted": self._lidar_wanted.get(robot),
            }

    def _lidar_reconcile(self, r: _Robot, st: dict | None, now: float) -> None:
        """Lock held: re-ask ``lidar`` when the robot's state disagrees."""
        want = self._lidar_wanted.get(r.name)
        if want is None or not st or not st.get("available") or "enabled" not in st:
            return
        if bool(st["enabled"]) == want or now - r.lidar_t < self.retry_s:
            return
        if any(v["robot"] == r.name and v["req"]["op"] == "lidar" for v in self._pending.values()):
            return
        r.lidar_t = now
        self._pending_send(r.name, "lidar", {"enable": want})

    def _lidar_idle(self, r: _Robot, now: float) -> None:
        """Lock held: nobody read the frames for ``lidar_idle_s``: turn it off
        (also a stream nobody here asked for, e.g. from before a console restart)."""
        want = self._lidar_wanted.get(r.name)
        streaming = bool(((r.state or {}).get("lidar") or {}).get("enabled"))
        if not self.lidar_idle_s or not (want or (want is None and streaming)):
            return
        if now - r.lidar_read_t < self.lidar_idle_s:
            if want and self.lidar_refresh_s and now - r.lidar_t >= self.lidar_refresh_s:
                r.lidar_t = now  # keepalive for the agent's lidar_ttl_s
                self._queue_req(r.name, {"id": f"{self._session}-l{next(self._ids)}",
                                         "op": "lidar", "args": {"enable": True}})  # fmt: skip
            return
        self._lidar_wanted[r.name] = False
        r.lidar = None
        r.lidar_t = now
        for rid in [k for k, v in self._pending.items() if v["robot"] == r.name and v["req"]["op"] == "lidar"]:
            del self._pending[rid]
        self._pending_send(r.name, "lidar", {"enable": False})

    def set_auto(self, robot: str, enabled: bool) -> str:
        return self._send(robot, "auto", {"enabled": bool(enabled)})

    def request_resync(self, robot: str) -> str:
        with self._lock:
            r = self._robot(robot)
            r.resync_requests += 1
            r.last_resync_req = self.clock()
        return self._send(robot, "resync")

    def pending_requests(self) -> list[dict]:
        with self._lock:
            return [
                dict(p["req"], robot=p["robot"], tries=p["tries"])
                for p in self._pending.values()
            ]

    def _maybe_resync(self, r: _Robot, now: float, force: bool = False) -> bool:
        """Request a resync unless rate-limited; True when one is sent or pending."""
        if any(
            p["req"]["op"] == "resync" and p["robot"] == r.name
            for p in self._pending.values()
        ):
            return True
        if force or now - r.last_resync_req >= self.resync_min_interval:
            r.resync_requests += 1
            r.last_resync_req = now
            self._pending_send(r.name, "resync", {})
            return True
        return False

    def _pending_send(self, robot: str, op: str, args: dict) -> str:
        """Lock held: register a retried request and queue it in the outbox."""
        rid = f"{self._session}-{next(self._ids)}"
        req = {"id": rid, "op": op, "args": args}
        self._pending[rid] = {"robot": robot, "req": req, "t": self.clock(), "tries": 1}
        self._queue_req(robot, req)
        return rid

    # ------------------------------------------------------------- receive

    def _on_msg(self, key: str, data: bytes) -> None:
        try:
            env = codec.unpack(data)
            robot, stream = env["robot"], env["stream"]
        except Exception:
            return
        cls = stream_class(stream)
        if cls is None:
            return  # e.g. our own ctl/req
        now = self.clock()
        with self._lock:
            events = self._receive(self._robot(robot), env, stream, cls, len(data), now)
        self._flush()
        for kind, d in events:
            self.on_update(robot, kind, d)

    def _receive(self, r: _Robot, env: dict, stream: str, cls: str, n, now) -> list:
        """Lock held: account and dispatch one message; returns the events."""
        if (
            r.last_seen is None
            or now - r.last_seen >= self.connected_timeout
            or (r.epoch is not None and env.get("epoch") not in (r.epoch, *r.retired))
        ):
            self._new_session(r, now)  # (re)connected after a silence / agent restart
        r.bytes[cls] += n
        r.win[cls] += n
        r.last_seen = now
        # one-way latency in the base clock (robot stamps corrected by the
        # estimated clock offset once a ping has measured it)
        lat = now - (float(env.get("t", now)) - (r.clock_offset or 0.0))
        r.latency = lat if r.latency is None else 0.8 * r.latency + 0.2 * lat
        if env["epoch"] in r.retired:
            return []  # late message from a previous agent incarnation
        if r.epoch != env["epoch"]:
            self._new_epoch(r, env["epoch"], now)
        seq = int(env["seq"])
        p = env["p"]
        if stream == "dsg/delta":
            return self._on_c2(r, seq, p, now)
        last = r.seqs.get(stream)
        if last is not None and seq <= last:
            return []  # duplicate / reordered
        if last is not None and seq > last + 1:
            r.drops += seq - last - 1
            r.win_lost[cls] += seq - last - 1
            r.lost[cls] += seq - last - 1
        r.win_msgs[cls] += 1
        r.seqs[stream] = seq
        events = []
        if cls == "C5":
            events.append(("video", self._on_video(r, stream, p, seq, last, now)))
        elif stream == "state":
            r.state = p
            r.last_state_rx = now
            events.append(("state", p))
            self._check_hashes(r, p, now)
            self._c2_check_tail(r, p, now)
            if self.store is not None:
                self._store_requests(self.store.on_state(r.name, p.get("store"), now))
            self._store_send_reconcile(r, p.get("store"), now)
            self._lidar_reconcile(r, p.get("lidar"), now)
        elif stream == "grid/occupancy":
            r.grid = p
            keys = ("hash", "frame", "width", "height", "resolution", "t")
            events.append(("grid", {k: p.get(k) for k in keys}))
        elif stream == "paths":
            r.paths = p
            events.append(("paths", p))
        elif stream == "lidar/points":
            if self._lidar_wanted.get(r.name) is False:
                return events  # turned off here: a late frame is not shown
            r.lidar = self._lidar_frame(p, seq)
            if r.lidar_read_t < 0:  # unasked stream: the idle time starts now
                r.lidar_read_t = now
            events.append(("lidar", {k: r.lidar.get(k) for k in ("frame", "n", "stamp", "seq")}))
        elif stream == "mesh/tile":
            r.tiles[p["key"]] = p
            r.tile_req_t.pop(p["key"], None)
            if p.get("odom") and r.deform is None:
                from fleetlink.mesh_deform_link import DeformView

                r.deform = DeformView()
            events.append(("tiles", {"changed": [p["key"]], "removed": []}))
        elif stream == "mesh/values":
            if r.deform is None:
                from fleetlink.mesh_deform_link import DeformView

                r.deform = DeformView()
            if r.deform.set_values(p) and r.tiles:
                # every vertex may move: all tiles change for the viewer
                events.append(("tiles", {"changed": sorted(r.tiles), "removed": []}))
        elif stream == "mesh/manifest":
            events.extend(self._on_manifest(r, p, now))
        elif stream == "ctl/ack":
            events.extend(self._on_ack(r, p, now))
        elif stream == "store/frame" and self.store is not None:
            self._store_requests(self.store.on_frame(r.name, p, now))
        return events

    @staticmethod
    def _lidar_frame(p: dict, seq: int) -> dict:
        """Wire lidar payload → the console's frame dict (xyz interleaved int16)."""
        from fleetlink.pointcloud_codec import xyz_int16

        return {
            "frame": p.get("frame"),
            "sensor_frame": p.get("sensor_frame"),
            "origin": p.get("origin"),
            "scale": p.get("scale"),
            "voxel": p.get("voxel"),
            "n": int(p.get("n") or 0),
            "xyz": xyz_int16(p),
            "intensity": p.get("intensity"),
            "stamp": p.get("stamp"),
            "seq": seq,
        }

    def _store_requests(self, reqs) -> None:
        """Lock held: send C6 ``store`` requests; a newer one replaces any still
        pending for that robot (a stale retry must not rewind the agent)."""
        for robot, op, args in reqs or ():
            for rid in [
                k
                for k, v in self._pending.items()
                if v["robot"] == robot and v["req"]["op"] == op
            ]:
                del self._pending[rid]
            self._pending_send(robot, op, args)

    def store_stats(self, robot: str | None = None) -> dict:
        """Live store mirror stats (all robots, or one); {} when disabled."""
        if self.store is None:
            return {}
        return self.store.stats(robot)

    # ------------------------------------------------- C2 in-order + resend

    def _on_c2(self, r: _Robot, seq: int, p: dict, now: float) -> list:
        """Apply C2 in seq order.  A message after a gap is held back and the gap
        is asked for again (``resend``); see :meth:`_c2_tick` for escalation."""
        if r.c2_next is None:
            r.c2_next = seq  # first C2 message of this epoch
        if seq < r.c2_next or seq in r.c2_hold:
            return []  # duplicate (e.g. a resend that crossed a late original)
        r.win_msgs["C2"] += 1
        r.seqs["dsg/delta"] = max(seq, r.seqs.get("dsg/delta", seq))
        if seq > r.c2_next:
            frontier = max(r.c2_next - 1, max(r.c2_hold, default=-1))
            if seq > frontier + 1:  # newly lost messages (count each once)
                lost = seq - frontier - 1
                r.gaps += 1
                r.drops += lost
                r.win_lost["C2"] += lost
                r.lost["C2"] += lost
            r.c2_hold[seq] = p
            if len(r.c2_hold) > self.hold_max:
                return self._c2_escalate(r, now)
            self._c2_check_gap(r, now)
            return []
        events = self._on_graph(r, p, seq)
        r.c2_next = seq + 1
        events.extend(self._c2_drain(r))
        self._c2_check_gap(r, now)
        return events

    def _c2_drain(self, r: _Robot) -> list:
        events = []
        while r.c2_next in r.c2_hold:
            seq = r.c2_next
            events.extend(self._on_graph(r, r.c2_hold.pop(seq), seq))
            r.c2_next = seq + 1
        return events

    def _c2_check_gap(self, r: _Robot, now: float) -> None:
        if not r.c2_hold:
            self._c2_clear_gap(r)
            return
        frm, to = r.c2_next, min(r.c2_hold) - 1
        gap = r.c2_gap
        if gap is not None and gap["from"] == frm and gap["to"] == to:
            return  # already asked
        tries = gap["tries"] if gap is not None and gap["from"] == frm else 0
        self._c2_request(r, frm, to, tries, now)

    def _c2_check_tail(self, r: _Robot, p: dict, now: float) -> None:
        """The state carries the last C2 seq the agent sent.  If we are behind
        for longer than ``resend_timeout_s`` with nothing held (the lost message
        was the last one, so no gap shows), ask for the tail."""
        last = (p.get("seqs") or {}).get("dsg")
        if (
            last is None
            or r.c2_next is None
            or int(last) < r.c2_next
            or r.c2_hold
            or r.c2_gap is not None
        ):
            r.c2_tail_since = None
            return
        if r.c2_tail_since is None or r.c2_tail_since[1] != r.c2_next:
            # C1 outruns C2: give it time to arrive (restarts while C2 progresses)
            r.c2_tail_since = (now, r.c2_next)
            return
        if now - r.c2_tail_since[0] >= self.resend_timeout_s:
            r.c2_tail_since = None
            lost = int(last) - r.c2_next + 1
            r.gaps += 1
            r.drops += lost
            r.win_lost["C2"] += lost
            r.lost["C2"] += lost
            self._c2_request(r, r.c2_next, int(last), 0, now)

    def _c2_request(self, r: _Robot, frm: int, to: int, tries: int, now) -> None:
        self._c2_clear_gap(r)
        rid = self._pending_send(
            r.name, "resend", {"stream": "dsg", "from_seq": frm, "to_seq": to}
        )
        r.resend_requests += 1
        r.c2_gap = {"from": frm, "to": to, "tries": tries + 1, "t": now, "rid": rid}

    def _c2_clear_gap(self, r: _Robot) -> None:
        if r.c2_gap is not None:
            self._pending.pop(r.c2_gap["rid"], None)
            r.c2_gap = None

    def _c2_escalate(self, r: _Robot, now: float) -> list:
        """Give up on the gap: apply what is held (as before resends existed) and
        fall back to a full resync (rate-limited; retried from tick if needed)."""
        self._c2_clear_gap(r)
        r.resend_escalations += 1
        events = []
        for seq in sorted(r.c2_hold):
            events.extend(self._on_graph(r, r.c2_hold.pop(seq), seq))
            r.c2_next = seq + 1
        r.c2_need_resync = not self._maybe_resync(r, now)
        return events

    def _c2_tick(self, r: _Robot, now: float) -> list:
        events = []
        gap = r.c2_gap
        if gap is not None:
            timeout = max(self.resend_timeout_s, 3.0 * (r.rtt or 0.0))
            if now - gap["t"] >= timeout:
                if gap["tries"] >= self.resend_attempts:
                    events = self._c2_escalate(r, now)
                else:
                    self._c2_request(r, gap["from"], gap["to"], gap["tries"], now)
        if r.c2_need_resync:
            r.c2_need_resync = not self._maybe_resync(r, now)
        return events

    def _on_video(self, r: _Robot, stream: str, p: dict, seq, last, now) -> dict:
        cam = stream[len(VIDEO_PREFIX) :]
        v = r.video.get(cam)
        if v is None:
            v = r.video[cam] = _VideoRx()
        n = len(p.get("data") or b"")
        gap = last is None or seq > last + 1
        if last is not None and seq > last + 1:
            v.gaps += 1
            v.lost += seq - last - 1
            if not p.get("key"):
                # frames were lost on the link: ask for a key frame now rather
                # than freezing until the next GOP boundary
                self._request_keyframe(r, cam, now)
        v.frames += 1
        v.keys += int(bool(p.get("key")))
        v.bytes += n
        v.win_frames += 1
        v.win_bytes += n
        v.last_rx = now
        v.last = {k: p.get(k) for k in ("codec", "width", "height")}
        v.last["target_fps"] = p.get("fps")
        # ``gap``: the first frame we see, or frames were lost before this one;
        # ``t_base``: the capture time in this (base) clock
        t_base = None
        if p.get("t") is not None:
            t_base = float(p["t"]) - (r.clock_offset or 0.0)
        return {**p, "camera": cam, "gap": gap, "t_base": t_base}

    def _request_keyframe(self, r: _Robot, camera: str, now: float) -> bool:
        """C0 ``video_keyframe`` (lock held), at most once per second per stream."""
        if now - r.kf_req_t.get(camera, -1e9) < 1.0:
            return False
        r.kf_req_t[camera] = now
        self._pending_send(r.name, "video_keyframe", {"camera": camera})
        return True

    def request_keyframe(self, robot: str, camera: str) -> bool:
        """Ask the robot for a key frame now (e.g. a browser's decoder failed)."""
        with self._lock:
            r = self._robots.get(robot)
            ok = r is not None and self._request_keyframe(r, camera, self.clock())
        self._flush()
        return ok

    def clock_offset(self, robot: str) -> float | None:
        """Estimated robot clock minus base clock (seconds), or None."""
        with self._lock:
            r = self._robots.get(robot)
            return None if r is None else r.clock_offset

    def _mesh_status(self, r: _Robot) -> dict:
        """What the console shows next to a robot's mesh: the LOD the robot
        reports (``lod`` None: an old robot, full tiles), the one asked, tiles
        still to arrive and the bytes held."""
        pending = sum(1 for k, h in r.manifest.items() if k not in r.tiles or r.tiles[k].get("hash") != h)
        held = sum(len(t.get("pos", b"")) + len(t.get("col", b"")) + len(t.get("idx", b"")) for t in r.tiles.values())
        w = self._mesh_lod_wanted.get(r.name)
        wanted = w["level"] if w else self.default_mesh_lod
        return {
            "lod": (r.mesh_lod or {}).get("level"),
            "lod_tiles": dict((r.mesh_lod or {}).get("tiles") or {}),
            "wanted": wanted,
            "tiles": len(r.manifest),
            "pending": pending,
            "bytes": held,
        }

    def _new_epoch(self, r: _Robot, epoch: str, now: float) -> None:
        if r.epoch is not None:
            r.retired.add(r.epoch)
        r.epoch = epoch
        r.epochs_seen += 1
        r.seqs.clear()
        r.graph = GraphState()
        r.tiles.clear()
        r.manifest = {}
        r.mesh_lod = None
        r.tile_req_t.clear()
        r.grid = None
        r.paths = None
        r.lidar = None
        r.mismatch_since.clear()
        r.resend_req_t.clear()
        r.video.clear()
        r.in_snapshot = False
        r.c2_next = None
        r.c2_hold.clear()
        r.c2_gap = None
        r.c2_need_resync = False
        r.c2_tail_since = None
        # requests for the previous incarnation are void
        for rid in [k for k, v in self._pending.items() if v["robot"] == r.name]:
            del self._pending[rid]
        self._maybe_resync(r, now, force=True)
        # a new robot process starts with full tiles: ask for this base's LOD
        w = self._mesh_lod_wanted.get(r.name)
        level = (w or {}).get("level") or self.default_mesh_lod
        if level is not None:
            self._pending_send(r.name, "mesh_lod", {"level": level})
            for lv in set((w or {}).get("tiles", {}).values()):
                keys = [k for k, v in w["tiles"].items() if v == lv]
                self._pending_send(r.name, "mesh_lod", {"level": lv, "tiles": keys})

    def _on_graph(self, r: _Robot, p: dict, seq: int) -> list:
        kind = p.get("kind", "delta")
        delta = GraphDelta.decode(p)
        if kind == "snapshot_begin":
            r.graph = GraphState()
            r.in_snapshot = True
        r.graph.apply(delta)
        if kind == "snapshot_end":
            r.in_snapshot = False
            r.resyncs += 1
            return [
                (
                    "snapshot",
                    {
                        "epoch": r.epoch,
                        "nodes": len(r.graph.records),
                        "edges": len(r.graph.edges),
                    },
                )
            ]
        if kind == "delta" and not delta.empty():
            return [
                ("delta", {**{k: v for k, v in p.items() if k != "kind"}, "seq": seq})
            ]
        return []

    def _check_hashes(self, r: _Robot, p: dict, now: float) -> None:
        """The state lists the hash of the last grid / paths the robot sent.  If ours
        differs for longer than ``resend_grace_s`` (the C3 / C1 message was dropped),
        ask the robot to send it again (C0 ``resend``)."""
        hashes = p.get("hashes") or {}
        stale = []
        for name in RESEND_STREAMS:
            theirs = hashes.get(name)
            have = getattr(r, name)
            ours = have.get("hash") if have else None
            if theirs is None or theirs == ours:
                r.mismatch_since.pop(name, None)
                continue
            since = r.mismatch_since.setdefault(name, now)
            if (
                now - since >= self.resend_grace_s
                and now - r.resend_req_t.get(name, -1e9) >= self.resend_grace_s
            ):
                stale.append(name)
        if stale:
            for name in stale:
                r.resend_req_t[name] = now
            self._pending_send(r.name, "resend", {"streams": stale})

    def _on_manifest(self, r: _Robot, p: dict, now: float) -> list:
        manifest = p.get("tiles") or {}
        r.manifest = manifest
        r.mesh_lod = p.get("lod") if isinstance(p.get("lod"), dict) else None
        removed = [k for k in r.tiles if k not in manifest]
        for k in removed:
            del r.tiles[k]
        missing = [
            k
            for k, h in manifest.items()
            if (k not in r.tiles or r.tiles[k]["hash"] != h)
            and now - r.tile_req_t.get(k, -1e9) >= self.tile_rerequest_s
        ]
        if missing:
            for k in missing:
                r.tile_req_t[k] = now
            self._pending_send(r.name, "tiles", {"keys": missing})
        return [("tiles", {"changed": [], "removed": removed})] if removed else []

    def _on_ack(self, r: _Robot, p: dict, now: float) -> list:
        pend = self._pending.pop(p.get("id"), None)
        res = p.get("result") or {}
        if (
            not p.get("ok", True)
            and pend is not None
            and r.c2_gap is not None
            and r.c2_gap["rid"] == p.get("id")
        ):
            # the agent no longer has (part of) the range: resend cannot help
            return self._c2_escalate(r, now)
        if res.get("t") is not None:  # ping ack (pings are fire-and-forget)
            rtt = now - float(res["t"])
            r.rtt = rtt if r.rtt is None else 0.8 * r.rtt + 0.2 * rtt
            if res.get("t_robot") is not None:
                off = float(res["t_robot"]) - (float(res["t"]) + now) / 2.0
                r.offset_samples.append((rtt, off))
                r.clock_offset = min(r.offset_samples)[1]
        if "budgets" in res and isinstance(r.state, dict):
            r.state.setdefault("stats", {})["budgets"] = {
                c: res["budgets"].get(c) for c in REPORTED_CLASSES
            }
        if "auto" in res and isinstance(r.state, dict):
            r.state.setdefault("stats", {})["auto"] = res["auto"]
        if pend is not None and pend["req"]["op"] == "lidar" and "enabled" in res:
            if isinstance(r.state, dict) and isinstance(r.state.get("lidar"), dict):
                r.state["lidar"]["enabled"] = bool(res["enabled"])  # shown before the next state
        if pend is not None and pend["req"]["op"] == "store_send" and "send" in res:
            if isinstance(r.state, dict) and isinstance(r.state.get("store"), dict):
                r.state["store"]["send"] = bool(res["send"])  # shown before the next state
        return []

    # ---------------------------------------------------------------- tick

    def tick(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        events = []
        with self._lock:
            for rid, pend in list(self._pending.items()):
                if now - pend["t"] < self.retry_s:
                    continue
                if pend["tries"] >= self.max_retries:
                    self.failed_requests.append(pend["req"])
                    del self._pending[rid]
                    continue
                pend["tries"] += 1
                pend["t"] = now
                self._queue_req(pend["robot"], pend["req"])
            for r in self._robots.values():
                events.extend((r.name, k, d) for k, d in self._c2_tick(r, now))
                self._lidar_idle(r, now)
                if r.win_t is None:
                    r.win_t = now
                elif now - r.win_t >= 1.0:
                    dt = now - r.win_t
                    r.tx_rate = 0.6 * r.tx_rate + 0.4 * (r.tx_win / dt)
                    r.tx_win = 0
                    for c in CLASSES:
                        r.rates[c] = 0.6 * r.rates[c] + 0.4 * (r.win[c] / dt)
                        r.win[c] = 0
                        seen = r.win_msgs[c] + r.win_lost[c]
                        r.loss[c] = r.win_lost[c] / seen if seen else 0.0
                        r.win_msgs[c] = r.win_lost[c] = 0
                    for v in r.video.values():
                        v.fps = 0.5 * v.fps + 0.5 * (v.win_frames / dt)
                        v.kbps = 0.5 * v.kbps + 0.5 * (v.win_bytes * 8 / 1000 / dt)
                        v.win_frames = v.win_bytes = 0
                    r.win_t = now
                    events.append((r.name, "link", self._link(r, now)))
                if r.epoch is None:
                    continue
                if now - r.t_ping >= self.ping_period:
                    r.t_ping = now
                    # ``rx`` only for agents that predate ``fb`` (they ignore fb)
                    rx = {c: r.rates[c] for c in ("C1", "C2", "C3")}
                    req = {
                        "id": f"{self._session}-p{next(self._ids)}",
                        "op": "ping",
                        "args": {"t": now, "rx": rx, "rtt": r.rtt},
                    }
                    self._queue_req(r.name, req)
                if self.fb_period and now - r.t_fb >= self.fb_period:
                    r.t_fb = now
                    self._queue_req(r.name, self._fb_req(r, now))
        self._flush()
        for robot, kind, d in events:
            self.on_update(robot, kind, d)

    def _fb_req(self, r: _Robot, now: float) -> dict:
        """C0 ``fb``: what arrived here over the last window, for the agent's AIMD
        (fire-and-forget, no ack)."""
        return {
            "id": f"{self._session}-f{next(self._ids)}",
            "op": "fb",
            "args": {
                "t": now,
                "rx": {c: round(r.rates[c], 1) for c in CLASSES},
                "loss": {c: round(r.loss[c], 4) for c in CLASSES},
                "rtt": r.rtt,
            },
        }
