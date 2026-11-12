"""Mission event log: one JSON object per line, append only.

A small, dependency-free writer for ``events/<source>.jsonl`` (see the ADT4
mission logging plan, section 3). Every line carries the envelope

    {"v", "t_wall", "t_mono", "t_ros", "machine", "robot", "run_id",
     "session_id", "mission_id", "goal_id", "plan_id", "source", "type",
     "seq", "payload"}

Writes are line buffered, fsync'd at most every ``fsync_s`` seconds, and the
file is rotated to ``<source>.<n>.jsonl`` once it reaches ``rotate_bytes``.
Payload keys that look like secrets are redacted.

Logging must never break execution: every public call swallows its errors,
reports the first one through ``warn`` and carries on (a failed open is
retried at most every ``retry_s`` seconds).

The same writer as spot_tools' ``spot_skills.event_log`` (each repo vendors it:
no shared dependency between the robot packages); ``EventLog.listeners`` get every
written record (the agent's live C7 relay).
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import threading
import time

SCHEMA_VERSION = 1
ROTATE_BYTES = 200 * 1024 * 1024
FSYNC_S = 2.0
REDACT_KEYS = (
    "api_key",
    "apikey",
    "password",
    "passwd",
    "token",
    "authorization",
    "secret",
)
REDACTED = "<redacted>"


def run_dir() -> str:
    """``$ADT4_RUN_DIR`` (the per-run directory set by run-adt4), or ""."""
    return os.environ.get("ADT4_RUN_DIR", "").strip()


def component_dir(component: str, fallback: str) -> str:
    """Where a component writes its files: ``$ADT4_RUN_DIR/<component>`` when
    run-adt4 set a run dir, else ``fallback`` (e.g. the node's output_dir)."""
    root = run_dir()
    return os.path.join(root, component) if root else fallback


def events_path(source: str, fallback_dir: str) -> str:
    """``$ADT4_RUN_DIR/events/<source>.jsonl``, else ``<fallback_dir>/events/...``."""
    root = run_dir() or fallback_dir
    return os.path.join(root, "events", f"{source}.jsonl")


def start_marker(clock=time.time) -> str:
    stamp = datetime.datetime.fromtimestamp(clock()).astimezone().isoformat()
    return f"# start {stamp} pid {os.getpid()}\n"


def open_append_log(path: str, header: str = "", clock=time.time) -> str:
    """Start a session in a plain text log without truncating earlier ones.

    Writes ``header`` only when the file is new or empty, then a
    ``# start <iso> pid <pid>`` marker. Returns the path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        if header and f.tell() == 0:
            f.write(header if header.endswith("\n") else header + "\n")
        f.write(start_marker(clock))
    return path


def redact(value):
    """Copy of ``value`` with secret-looking dict keys replaced."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).lower()
            if any(s in key for s in REDACT_KEYS):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def _default(obj):
    tolist = getattr(obj, "tolist", None)  # numpy arrays and scalars
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


class EventLog:
    def __init__(
        self,
        path: str,
        source: str,
        robot: str = "",
        machine: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        mission_id: str | None = None,
        ros_time=None,
        warn=print,
        fsync_s: float = FSYNC_S,
        rotate_bytes: int = ROTATE_BYTES,
        retry_s: float = 10.0,
        clock=time.time,
        mono=time.monotonic,
    ):
        env = os.environ
        self.path = path
        self.source = source
        self.robot = robot or env.get("ADT4_ROBOT_NAME", "")
        self.machine = machine if machine is not None else socket.gethostname()
        rd = run_dir()
        self.run_id = (
            run_id
            if run_id is not None
            else env.get("ADT4_RUN_ID", "")
            or (os.path.basename(rd.rstrip("/")) if rd else "")
        )
        self.session_id = (
            session_id if session_id is not None else env.get("ADT4_SESSION_ID", "")
        )
        self.mission_id = (
            mission_id if mission_id is not None else env.get("ADT4_MISSION_ID", "")
        )
        self.ros_time = ros_time
        self.warn = warn
        self.fsync_s = float(fsync_s)
        self.rotate_bytes = int(rotate_bytes)
        self.retry_s = float(retry_s)
        self.clock = clock
        self.mono = mono
        self.plan_id = ""
        self.goal_id = ""
        self._lock = threading.Lock()
        self._fh = None
        self._seq = 0
        self._last_sync = 0.0
        self._failed_at = None
        self._warned = False
        self.listeners: list = []  # fn(record) after each written event

    # ------------------------------------------------------------ internals
    def _fail(self, what, e):
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass
        self._fh = None
        self._failed_at = self.mono()
        if not self._warned:
            self._warned = True
            try:
                self.warn(
                    f"event log {self.path}: {what} failed (continuing without it): {e!r}"
                )
            except Exception:  # noqa: BLE001
                pass

    def _open(self):
        if self._fh is not None:
            return True
        if self._failed_at is not None and self.mono() - self._failed_at < self.retry_s:
            return False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._failed_at = None
        return True

    def _rotated_name(self):
        stem = self.path[:-6] if self.path.endswith(".jsonl") else self.path
        n = 1
        while os.path.exists(f"{stem}.{n}.jsonl"):
            n += 1
        return f"{stem}.{n}.jsonl"

    def _maybe_rotate(self):
        if self._fh is None or self._fh.tell() < self.rotate_bytes:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._fh = None
        os.replace(self.path, self._rotated_name())
        self._open()

    def _now_ros(self, t_wall):
        if self.ros_time is None:
            return t_wall
        try:
            return float(self.ros_time())
        except Exception:  # noqa: BLE001
            return t_wall

    # --------------------------------------------------------------- public
    def set_context(self, plan_id=None, goal_id=None, mission_id=None):
        """Defaults for the envelope ids of later events (None = unchanged)."""
        if plan_id is not None:
            self.plan_id = str(plan_id)
        if goal_id is not None:
            self.goal_id = str(goal_id)
        if mission_id is not None:
            self.mission_id = str(mission_id)

    def emit(self, etype: str, payload=None, plan_id=None, goal_id=None):
        """Append one event; returns True if it was written. Never raises."""
        try:
            with self._lock:
                if not self._open():
                    return False
                t_wall = self.clock()
                self._seq += 1
                record = {
                    "v": SCHEMA_VERSION,
                    "t_wall": t_wall,
                    "t_mono": self.mono(),
                    "t_ros": self._now_ros(t_wall),
                    "machine": self.machine,
                    "robot": self.robot,
                    "run_id": self.run_id,
                    "session_id": self.session_id,
                    "mission_id": self.mission_id,
                    "goal_id": self.goal_id if goal_id is None else str(goal_id),
                    "plan_id": self.plan_id if plan_id is None else str(plan_id),
                    "source": self.source,
                    "type": str(etype),
                    "seq": self._seq,
                    "payload": redact(payload or {}),
                }
                line = json.dumps(record, default=_default, separators=(",", ":"))
                try:
                    self._fh.write(line + "\n")
                    now = self.mono()
                    if now - self._last_sync >= self.fsync_s:
                        self._fh.flush()
                        os.fsync(self._fh.fileno())
                        self._last_sync = now
                    self._maybe_rotate()
                except Exception as e:  # noqa: BLE001
                    self._fail("write", e)
                    return False
            for fn in list(self.listeners):
                try:
                    fn(record)
                except Exception:  # noqa: BLE001 -- a listener never stops logging
                    pass
            return True
        except Exception as e:  # noqa: BLE001
            try:
                with self._lock:
                    self._fail("open", e)
            except Exception:  # noqa: BLE001
                pass
            return False

    def close(self):
        try:
            with self._lock:
                if self._fh is not None:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                    self._fh.close()
                    self._fh = None
        except Exception as e:  # noqa: BLE001
            self._fail("close", e)


