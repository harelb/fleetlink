"""Live image-store mirror over fleetlink (class C6, ADT4 console API.md §13).

Robot side, :class:`StoreTail`: tails the robot's image store (sqlite, WAL,
opened read-only and never indexed from here).  Hydra/Khronos do not write it
directly: the image store's tailer builds it from the keyframe zips, track logs
and event logs while Hydra runs, already
re-encoded (JPEG colour, 16-bit PNG depth), so no InputData zip / EXR ever goes
on the wire.  The tail mode is Hydra PR #221 and is not on Hydra ``develop``:
without it (:func:`tail_mode_available`) the agent turns the mirror off.  It selects frames (``keyframes`` = the agent nodes' own keyframes,
what SAM3 search uses, plus the frames of kept object observations; or ``all``,
which adds the other keyframes) in ``frame_id`` order
and encodes each with :func:`fleetlink.image_transfer.encode_frame` (``none`` |
``lossless`` | ``fast``).  The frame row carries the sensor calibration, so the
base can rebuild a valid store from any frame; a frame with kept object
observations also carries them with their track's node binding, so the base
shows object images during the run.

Only *settled* frames are sent: the tailer anchors a frame (which decides whether
it is an agent node's own keyframe) once the agent node after it exists, and keeps
undecided frames in meta ``indexer.r<robot>.pending_frames`` /
``provisional_frames``; the cursor never passes the lowest of those.  A kept
observation whose track is not bound to a node yet waits up to
``bind_wait_s`` for the binding (frontend events arrive within about a second).

Base side, :class:`StoreMirror`: writes every received frame into
``<root>/<robot>/<store_id>/image_store`` with
:func:`fleetlink.image_transfer.write_frame` (source frame / sensor ids
kept), on a writer thread.

Protocol (nothing lost or duplicated across robot, agent or console restarts):

* the agent's C1 ``state`` carries ``store = {id, active, after, sent, last,
  backlog, ...}``.  ``id`` names the robot's *store session*: the sqlite
  file's inode (run-adt4 clears the run dir, so a new run is a new file) plus,
  for a tailed store, the Hydra session (the ``wall_ns`` of the frontend event
  log's newest ``open``, from meta ``indexer.tail``), else the first frame's
  timestamp.  It changes when a new run starts or Hydra restarts;
* the base answers a state whose store it does not mirror yet, or whose
  ``active`` is false (agent restarted), with the C0 request ``store {id,
  after}``: ``after`` is the highest ``frame_id`` already in its mirror (None
  = empty).  The agent restarts its send cursor there;
* each C6 message ``adt4/<r>/store/frame`` is ``{id, frame_id, prev, blob}``
  where ``prev`` is the frame sent before it (None for the first after the
  cursor).  The base applies a frame only when ``prev`` equals its watermark;
  otherwise (a frame was lost) it re-sends ``store {id, after: watermark}``,
  and the agent rewinds.  A lost *last* frame shows as ``state.store.sent`` >
  watermark with nothing arriving for ``resend_timeout_s``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import queue
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

STORE_STREAM = "store/frame"
FRAME_SELECTIONS = ("keyframes", "all")
COMPRESSION_MODES = ("none", "lossless", "fast")


def _transfer():
    # lazy: image_transfer needs hydra_python.image_store, which the agent may lack
    from fleetlink import image_transfer

    return image_transfer


def store_id_of(
    store_dir: str | os.PathLike, first_ts: int | None, session: int | None = None
) -> str | None:
    """Identity of one store session: the sqlite file's inode plus the Hydra
    session of a tailed store (``indexer.tail.session``), else the timestamp of
    its first frame (a store written directly, e.g. by an older Hydra)."""
    db = pathlib.Path(store_dir) / "store.sqlite"
    try:
        st = os.stat(db)
    except OSError:
        return None
    key = f"s{int(session)}" if session is not None else first_ts
    if key is None:
        return None
    h = hashlib.sha1(f"{st.st_dev}:{st.st_ino}:{key}".encode())
    return h.hexdigest()[:12]


TAIL_MODE_MISSING = (
    "hydra_python.image_store has no tail (live) mode (Hydra PR #221, not on "
    "Hydra develop); live image-store mirroring is off"
)


def tail_mode_available() -> bool:
    """Whether ``hydra_python.image_store`` can follow a store while Hydra writes
    its logs (the tail mode). Without it no store exists during a run, so there is
    nothing to mirror."""
    try:
        from hydra_python.image_store.indexer import tail_state  # noqa: F401
    except ImportError:
        return False
    return True


def tail_session(store_dir: str | os.PathLike) -> tuple[bool, int | None]:
    """``(tailed, session)`` of a store from its tailer status (meta
    ``indexer.tail``); ``(False, None)`` for a store nobody tails."""
    try:
        from hydra_python.image_store.indexer import tail_state
    except ImportError:  # no tail mode (see tail_mode_available)
        return False, None
    state = tail_state(store_dir)
    if not state:
        return False, None
    # Hydra's frontend log opens with the first frontend update; the backend log only
    # with the first merge, minutes later, and indexer.tail.session (the newest open of
    # any log) changes then. The frontend log's session changes only when Hydra restarts.
    events = state.get("events") or {}
    front = [
        e.get("session")
        for e in events.values()
        if isinstance(e, dict)
        and e.get("source") == "frontend"
        and e.get("session") is not None
    ]
    s = max(front) if front else state.get("session")
    return True, None if s is None else int(s)


# ----------------------------------------------------------------------------
# robot side
# ----------------------------------------------------------------------------


class StoreTail:
    """Read-only tail of a robot image store; pure python, no ROS.

    :meth:`set_cursor` (from the base's ``store`` request) arms it;
    :meth:`take` returns the next frame ids to send (advancing the cursor);
    :meth:`encode` builds a frame's wire payload.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        frames: str = "keyframes",
        compression: str = "fast",
        clock: Callable[[], float] = time.time,
        reopen_s: float = 2.0,
        bind_wait_s: float = 30.0,
    ):
        if frames not in FRAME_SELECTIONS:
            raise ValueError(f"store_frames must be one of {FRAME_SELECTIONS}")
        if compression not in COMPRESSION_MODES:
            raise ValueError(f"store_compression must be one of {COMPRESSION_MODES}")
        self.path = pathlib.Path(os.path.expanduser(str(path)))
        self.frames = frames
        self.compression = compression
        self.clock = clock
        self.reopen_s = reopen_s
        self.bind_wait_s = bind_wait_s
        self._unbound_since: dict[int, float] = {}  # frame id -> first seen unbound
        self.held: int | None = None  # frame the cursor waits at (None = none)
        self.hold_reason: str | None = None
        self._store = None
        self._t_open = -1e9
        self.id: str | None = None
        self.error: str | None = None
        self.cursor: int | None = None  # last frame id handed out (None = start)
        self.after: int | None = None  # the base's watermark when armed
        self.active = False
        self.last_sent: int | None = None
        self.frames_sent = 0
        self.bytes_sent = 0
        self.encode_errors = 0
        self.rewinds = 0
        self._obs_sent: set[int] = set()  # kept obs_ids that went out with their frame
        self._late_pending: set[int] = set()  # late frames handed out, not encoded yet
        self.late_sent = 0
        self._avg_bytes = 60_000.0
        self._lock = threading.RLock()

    # -- store -----------------------------------------------------------

    def _open(self):
        """The reader (opened lazily: Hydra may create the store later)."""
        if self._store is not None:
            return self._store
        now = self.clock()
        if now - self._t_open < self.reopen_s:
            return None
        self._t_open = now
        if not (self.path / "store.sqlite").exists():
            self.error = f"no image store at {self.path}"
            return None
        try:
            from hydra_python.image_store import ImageStore

            # read only: the tailer is the store's one writer
            self._store = ImageStore.open(self.path, index=False)
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self._store = None
        return self._store

    def _first_ts(self, store) -> int | None:
        row = None
        for _db, r in store._query("SELECT MIN(timestamp_ns) FROM frames"):
            row = r
        return None if row is None or row[0] is None else int(row[0])

    def _store_id(self, store) -> str | None:
        tailed, session = tail_session(self.path)
        if tailed:  # a tailed store is ready once Hydra's first event log opened
            return store_id_of(self.path, None, session)
        return store_id_of(self.path, self._first_ts(store))

    def refresh(self) -> None:
        """Re-check the store identity (new run dir, or Hydra restarted)."""
        with self._lock:
            store = self._open()
            if store is None:
                return
            try:
                sid = self._store_id(store)
            except Exception as exc:  # noqa: BLE001  (store replaced underneath)
                self.error = f"{type(exc).__name__}: {exc}"
                self._close()
                return
            if sid is not None and self.id is not None and sid != self.id:
                self._close()  # new session: reopen the new file
                store = self._open()
                if store is None:
                    return
                sid = self._store_id(store)
            if sid != self.id:
                self.id = sid
                self.active = False  # the base must arm the new store
                self.cursor = self.after = self.last_sent = None
                self._obs_sent = set()

    def _close(self) -> None:
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001
                pass
        self._store = None
        self._t_open = -1e9

    # -- cursor ----------------------------------------------------------

    def set_cursor(self, store_id: str | None, after: int | None) -> dict:
        """The base's ``store`` request: resume after ``after`` (None = start)."""
        with self._lock:
            self.refresh()
            if self.id is None:
                raise ValueError(self.error or "store not available yet")
            if store_id != self.id:
                raise ValueError(f"store id {store_id!r} is not current ({self.id})")
            if self.active and after != self.cursor:
                self.rewinds += 1
            self.after = None if after is None else int(after)
            self.cursor = self.after
            # kept observations on frames the base already has are (re)sent as late
            # frames: unknown which of them a restarted base / agent saw
            self._obs_sent = set()
            self.active = True
            return {"id": self.id, "after": self.after, "backlog": self.backlog()}

    def pause(self) -> None:
        """Sending stopped: frames handed out but never sent go back, the next
        ``take`` continues after the last frame sent (or the base's watermark)."""
        with self._lock:
            self._late_pending.clear()
            if self.active:
                done = [f for f in (self.last_sent, self.after) if f is not None]
                self.cursor = max(done) if done else None

    def _select(self, after: int | None, limit: int | None):
        store = self._open()
        if store is None:
            return []
        return _transfer().select_frames(
            store, self.frames, after=after, limit=limit, include_kept_observations=True
        )

    def _unsettled(self, store) -> int | None:
        """Lowest frame id the indexer has not decided yet (anchor / keyframe role
        pending or provisional), None if all are settled."""
        lo = None
        rows = store._query(
            "SELECT value FROM meta WHERE key LIKE 'indexer.r%.pending_frames' "
            "OR key LIKE 'indexer.r%.provisional_frames'"
        )
        for _db, (value,) in rows:
            try:
                ids = json.loads(value)
            except ValueError:
                continue
            if ids:
                m = min(int(i) for i in ids)
                lo = m if lo is None else min(lo, m)
        return lo

    def _bound_or_given_up(self, store, frame_id: int, unbound: dict | None) -> bool:
        """A frame's kept observations all have a node, are known to have none
        (extraction failed, node deleted...) or waited ``bind_wait_s``."""
        t = _transfer()
        obs = t.frame_observations(store, frame_id)
        waiting = [o for o in obs if o["node_id"] is None]
        if not waiting:
            self._unbound_since.pop(frame_id, None)
            return True
        if unbound is None:
            unbound = {(r, k): why for r, k, why in store.unbound_tracks()}
        if all(
            unbound.get((o["robot_id"], o["track_id"]), "unbound") != "unbound"
            for o in waiting
        ):
            self._unbound_since.pop(frame_id, None)
            return True
        first = self._unbound_since.setdefault(frame_id, self.clock())
        if self.clock() - first >= self.bind_wait_s:
            self._unbound_since.pop(frame_id, None)
            return True
        return False

    def _late(self, store, n: int) -> list[tuple[int, str]]:
        """Frames at or below the cursor with a kept observation that did not go out
        with them (a track logged on a keyframe archive's frame after the cursor passed
        it, the final reconciliation, or all of them after an agent / base restart):
        sent out of the chain as ``late`` frames."""
        if self.cursor is None or n <= 0:
            return []
        rows = [
            r
            for _db, r in store._query(
                "SELECT obs_id, frame_id FROM observations WHERE kept != 0 AND frame_id <= ? ORDER BY obs_id",
                (self.cursor,),
            )
        ]
        out, seen = [], set()
        for obs_id, fid in rows:
            if obs_id in self._obs_sent or fid in seen or fid in self._late_pending:
                continue
            seen.add(fid)
            if self._bound_or_given_up(store, int(fid), None):
                out.append((int(fid), "late"))
                self._late_pending.add(int(fid))
                if len(out) >= n:
                    break
        return out

    def take(self, n: int) -> list[tuple[int, int | None | str]]:
        """Up to ``n`` next settled ``(frame_id, prev)`` after the cursor (advances it),
        plus late frames (``prev == "late"``) for kept observations that appeared on
        frames already passed."""
        with self._lock:
            if not self.active or n <= 0:
                return []
            try:
                late = self._late(self._open(), n)
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                late = []
            if late:
                return late
            try:
                store = self._open()
                frames = self._select(self.cursor, n)
                hold = self._unsettled(store) if frames else None
            except Exception as exc:  # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                return []
            out = []
            self.held = self.hold_reason = None
            for f in frames:
                fid = int(f.frame_id)
                if hold is not None and fid >= hold:
                    self.held, self.hold_reason = fid, "anchor"
                    break
                try:
                    ready = self._bound_or_given_up(store, fid, None)
                except Exception as exc:  # noqa: BLE001
                    self.error = f"{type(exc).__name__}: {exc}"
                    break
                if not ready:
                    self.held, self.hold_reason = fid, "binding"
                    break
                out.append((fid, self.cursor))
                self.cursor = fid
            return out

    def encode(self, frame_id: int, prev: int | None) -> dict | None:
        """Wire payload of one frame.  A frame that cannot be encoded (file gone,
        corrupt image) is sent as a skip (``blob`` None, ``error`` set) so the
        ``prev`` chain stays intact and the base moves its watermark past it."""
        with self._lock:
            store = self._open()
            if store is None:
                return None  # store vanished: nothing sent, the base re-arms later
            try:
                t = _transfer()
                obs = t.frame_observations(store, int(frame_id))
                blob = t.encode_frame(
                    store, int(frame_id), self.compression, observations=True
                )
                self._obs_sent.update(o["obs_id"] for o in obs)
                self._late_pending.discard(int(frame_id))
            except Exception as exc:  # noqa: BLE001
                self.encode_errors += 1
                self.error = f"encode {frame_id}: {type(exc).__name__}: {exc}"
                log.warning("store mirror: %s", self.error)
                return {"id": self.id, "frame_id": int(frame_id), "prev": prev,
                        "blob": None, "error": self.error}  # fmt: skip
            self._avg_bytes = 0.8 * self._avg_bytes + 0.2 * len(blob)
            if prev == "late":
                return {
                    "id": self.id,
                    "frame_id": int(frame_id),
                    "prev": None,
                    "late": True,
                    "blob": blob,
                }
            return {
                "id": self.id,
                "frame_id": int(frame_id),
                "prev": prev,
                "blob": blob,
            }

    def mark_sent(self, frame_id: int, nbytes: int, late: bool = False) -> None:
        with self._lock:
            if late:
                self.late_sent += 1
            else:
                self.last_sent = int(frame_id)
            self.frames_sent += 1
            self.bytes_sent += int(nbytes)

    @property
    def avg_bytes(self) -> float:
        return self._avg_bytes

    @staticmethod
    def _kept_sql(store) -> str:
        t = _transfer()
        t.register_sql_functions(store)
        return f"({t.OWN_KEYFRAME_SQL} OR frame_id IN (SELECT frame_id FROM observations WHERE kept != 0))"

    def backlog(self) -> int:
        """Selected frames after the cursor (not yet handed out)."""
        store = self._open()
        if store is None:
            return 0
        sql = "SELECT COUNT(*) FROM frames WHERE (rgb_path IS NOT NULL OR depth_path IS NOT NULL)"
        params: list = []
        if self.frames == "keyframes":
            sql += " AND " + self._kept_sql(store)
        if self.cursor is not None:
            sql += " AND frame_id > ?"
            params.append(self.cursor)
        try:
            return int(sum(r[0] for _db, r in store._query(sql, params)))
        except Exception:  # noqa: BLE001
            return 0

    def last_frame(self) -> int | None:
        store = self._open()
        if store is None:
            return None
        sql = "SELECT MAX(frame_id) FROM frames WHERE (rgb_path IS NOT NULL OR depth_path IS NOT NULL)"
        if self.frames == "keyframes":
            sql += " AND " + self._kept_sql(store)
        try:
            vals = [r[0] for _db, r in store._query(sql) if r[0] is not None]
        except Exception:  # noqa: BLE001
            return None
        return max(vals) if vals else None

    def state(self) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "active": self.active,
                "after": self.after,
                "sent": self.last_sent,
                "last": self.last_frame(),
                "backlog": self.backlog() if self.active else None,
                "frames_sent": self.frames_sent,
                "bytes_sent": self.bytes_sent,
                "avg_frame_bytes": round(self._avg_bytes),
                "frames": self.frames,
                "compression": self.compression,
                "rewinds": self.rewinds,
                "late_sent": self.late_sent,
                "held": self.held,
                "hold_reason": self.hold_reason,
                "error": self.error,
            }

    def close(self) -> None:
        with self._lock:
            self._close()


# ----------------------------------------------------------------------------
# base side
# ----------------------------------------------------------------------------


class _Mirror:
    def __init__(self, robot: str, store_id: str, path: pathlib.Path):
        self.robot = robot
        self.id = store_id
        self.path = path
        self.writer = None
        self.last: int | None = None  # watermark: highest frame id accepted
        self.frames = 0
        self.bytes = 0
        self.dups = 0
        self.late = 0
        self.skipped = 0
        self.gaps = 0
        self.errors = 0
        self.last_error: str | None = None
        self.t_req = -1e9
        self.req_after: object = "never"  # after of the last request sent
        self.t_rx = None
        self.robot_state: dict = {}


class StoreMirror:
    """Base-side writer of live store mirrors (one per robot store).

    ``on_state`` / ``on_frame`` return the C0 requests to send as
    ``[(robot, "store", args)]``; the receiver sends (and retries) them.
    Frames are written on a worker thread (``threaded=False``: inline, tests).
    """

    def __init__(
        self,
        root: str | os.PathLike,
        clock: Callable[[], float] = time.time,
        threaded: bool = True,
        request_interval_s: float = 2.0,
        resend_timeout_s: float = 5.0,
    ):
        self.root = pathlib.Path(os.path.expanduser(str(root)))
        self.clock = clock
        self.threaded = threaded
        self.request_interval_s = request_interval_s
        self.resend_timeout_s = resend_timeout_s
        self._mirrors: dict[str, _Mirror] = {}  # robot -> current mirror
        self._lock = threading.RLock()
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        if threaded:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="fleetlink-store-mirror"
            )
            self._thread.start()

    # -- mirrors ---------------------------------------------------------

    def path_for(self, robot: str, store_id: str) -> pathlib.Path:
        return self.root / robot / store_id / "image_store"

    def _mirror(self, robot: str, store_id: str) -> _Mirror:
        m = self._mirrors.get(robot)
        if m is not None and m.id == store_id:
            return m
        if m is not None:
            self._q.put(("close", m, None))
        m = _Mirror(robot, store_id, self.path_for(robot, store_id))
        m.last = self._existing_watermark(m.path)
        self._mirrors[robot] = m
        return m

    @staticmethod
    def _existing_watermark(path: pathlib.Path) -> int | None:
        db = path / "store.sqlite"
        if not db.exists():
            return None
        try:
            from hydra_python.image_store import ImageStore

            with ImageStore.open(path, index=False) as s:
                vals = [r[0] for _d, r in s._query("SELECT MAX(frame_id) FROM frames")]
            return vals[0] if vals and vals[0] is not None else None
        except Exception:  # noqa: BLE001
            log.exception("store mirror: cannot read %s", path)
            return None

    def _request(self, m: _Mirror, now: float, force: bool = False) -> list:
        if not force and now - m.t_req < self.request_interval_s:
            return []
        m.t_req = now
        m.req_after = m.last
        return [(m.robot, "store", {"id": m.id, "after": m.last})]

    # -- inputs (receiver thread) ----------------------------------------

    def on_state(self, robot: str, st: dict | None, now: float | None = None) -> list:
        """The ``store`` part of the agent's state."""
        if not st or not st.get("id"):
            return []
        now = self.clock() if now is None else now
        with self._lock:
            m = self._mirror(robot, str(st["id"]))
            m.robot_state = dict(st)
            if not st.get("active") or m.req_after == "never":
                # agent not armed (agent restart / new store), or armed by an
                # earlier console: tell it where this mirror stands
                return self._request(m, now)
            sent = st.get("sent")
            if (
                sent is not None
                and (m.last is None or int(sent) > m.last)
                and (m.t_rx is None or now - m.t_rx >= self.resend_timeout_s)
                and now - m.t_req >= self.resend_timeout_s
            ):
                m.gaps += 1  # the last frame(s) were lost
                return self._request(m, now, force=True)
            return []

    def on_frame(self, robot: str, p: dict, now: float | None = None) -> list:
        now = self.clock() if now is None else now
        with self._lock:
            m = self._mirrors.get(robot)
            if m is None or p.get("id") != m.id:
                return []  # not armed for this store (yet)
            m.t_rx = now
            fid = int(p["frame_id"])
            if p.get("late"):
                # a frame below the watermark that gained a kept object observation:
                # outside the prev chain (a lost one is re-sent after an agent restart)
                if m.last is None or fid > m.last or p.get("blob") is None:
                    return []
                m.late += 1
                m.bytes += len(p["blob"])
                item = ("write", m, p["blob"])
                if self.threaded:
                    self._q.put(item)
                else:
                    self._apply(*item)
                return []
            if m.last is not None and fid <= m.last:
                m.dups += 1
                return []
            if p.get("prev") != m.last:
                m.gaps += 1
                return self._request(m, now)
            m.last = fid
            if p.get("blob") is None:  # the robot could not encode it
                m.skipped += 1
                return []
            m.frames += 1
            m.bytes += len(p["blob"])
            item = ("write", m, p["blob"])
        if self.threaded:
            self._q.put(item)
        else:
            self._apply(*item)
        return []

    # -- writing ---------------------------------------------------------

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            self._apply(*item)

    def _apply(self, op: str, m: _Mirror, blob) -> None:
        if op == "flush":
            blob.set()
            return
        if op == "close":
            if m.writer is not None:
                try:
                    m.writer.close()
                except Exception:  # noqa: BLE001
                    pass
                m.writer = None
            return
        try:
            t = _transfer()
            if m.writer is None:
                from hydra_python.image_store import ImageStore

                m.path.mkdir(parents=True, exist_ok=True)
                m.writer = ImageStore.open_writer(m.path)
            t.write_frame(m.writer, t.decode_frame(blob))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                m.errors += 1
                m.last_error = f"{type(exc).__name__}: {exc}"
            log.warning("store mirror %s: %s", m.robot, m.last_error)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until queued frames are written (tests / shutdown)."""
        if not self.threaded:
            return
        done = threading.Event()
        self._q.put(("flush", None, done))
        done.wait(timeout)

    def stats(self, robot: str | None = None) -> dict:
        with self._lock:
            out = {}
            for r, m in self._mirrors.items():
                st = m.robot_state
                last = st.get("last")
                out[r] = {
                    "store_id": m.id,
                    "path": str(m.path),
                    "watermark": m.last,
                    "frames": m.frames,
                    "bytes": m.bytes,
                    "avg_frame_bytes": round(m.bytes / m.frames) if m.frames else None,
                    "backlog": st.get("backlog"),
                    "robot_last": last,
                    "duplicates": m.dups,
                    "late": m.late,
                    "skipped": m.skipped,
                    "gaps": m.gaps,
                    "write_errors": m.errors,
                    "last_error": m.last_error,
                    "frames_selection": st.get("frames"),
                    "compression": st.get("compression"),
                    "queued_writes": self._q.qsize(),
                }
            return out if robot is None else out.get(robot, {})

    def close(self) -> None:
        if self._thread is not None:
            self.flush()
            for m in list(self._mirrors.values()):
                self._q.put(("close", m, None))
            self._q.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None
        else:
            for m in list(self._mirrors.values()):
                self._apply("close", m, None)
