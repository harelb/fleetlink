"""The agent's mission event log and the live C7 event relay (pure, no ROS).

ADT4 mission logging plan §3 / §5 (phase B1).  :class:`AgentEvents` writes
``events/fleetlink_agent.jsonl`` in the robot's run dir (the envelope of
:mod:`fleetlink.eventlog`) from what the agent already holds:

* ``robot.pose`` at 1 Hz (``{x, y, z, yaw, frame}``);
* ``loc.snapshot`` every 10 s and on every mission / lease / plan change:
  ``map_T_odom`` (the ``frame_align`` transform), the body pose, ROMAN prior /
  self loop closures, best association, hydra_multi's accepted inter-robot LCs,
  ``reloc_ok`` (a prior-map match was seen);
* ``loc.jump`` for each ``map -> odom`` jump the RelocTracker records;
* ``guards`` on change (lease owner, estop, power, faults; battery in 5 % steps);
* phoenix: ``phoenix.progress`` (new plan, done, or 1 m closer, at most 1 Hz)
  and ``phoenix.guards`` on change;
* ``link.robot`` every 30 s (scheduler queues, rates, budgets);
* ``mission.ack`` when ``/global/mission/current`` (the console's latched
  mission, relayed down by fleetlink) names a new mission; the id then stamps
  every later event, and :meth:`AgentEvents.state` puts it into the C1 state so
  the base sees the robot acknowledge it.

**Live relay (C7).**  Selected event types (:data:`LIVE_PREFIXES`: goal / action
/ lease / loc / follow / phoenix / mission ...) from this log *and* from the other
``events/*.jsonl`` of the run (spot_executor's, tailed) go to the base as
``adt4/<robot>/events`` batches in class C7: the lowest priority, its own budget
(``live_bps``, 2 kB/s by default), best effort, held with the other bulk
classes behind a pending ``priority: high`` relay message.  One batch is queued
at a time; the waiting records sit in a bounded deque (oldest dropped first and
counted).  It feeds the live timeline only: the authoritative copy is collected
after the mission.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import deque
from typing import Callable

SOURCE = "fleetlink_agent"
STREAM = "events"
POSE_S = 1.0
LOC_S = 10.0
LINK_S = 30.0
PHOENIX_MIN_S = 1.0
PHOENIX_STEP_M = 1.0
TAIL_S = 1.0
TAIL_CHUNK = 64 * 1024
LIVE_BPS = 2000
LIVE_BATCH_BYTES = 1500
LIVE_QUEUE = 500
# event types relayed live (prefix match on ``type``)
LIVE_PREFIXES = (
    "goal.", "action.", "lease.", "loc.", "follow.", "phoenix.", "mission.", "sequence.",
    "plan.received", "plan.preempted", "grasp.episode", "guards", "teleop", "pause",
    "resume", "stop", "mlp.replan", "run.",
)  # fmt: skip
HYDRA_LC_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*valid", re.I)
GUARD_KEYS = (
    "estop_pressed", "estop_known", "powered_on", "power_known", "lease_owned",
    "lease_known", "lease_owner", "faults", "safety_stop", "manual_mode",
)  # fmt: skip


def yaw_of_pose(p: dict) -> float | None:
    try:
        qw, qx, qy, qz = (float(p[k]) for k in ("qw", "qx", "qy", "qz"))
    except (KeyError, TypeError, ValueError):
        return p.get("yaw") if isinstance(p, dict) else None
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def hydra_lcs(health: dict | None) -> tuple[int, int] | None:
    """``(accepted, total)`` from the hydra_multi row of the status table."""
    for n in (health or {}).get("nodes") or []:
        if not isinstance(n, dict):
            continue
        if "hydra_multi" not in f"{n.get('nickname') or ''} {n.get('node_name') or ''}":
            continue
        m = HYDRA_LC_RE.search(str(n.get("notes") or ""))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def parse_mission(data) -> dict:
    """``/global/mission/current``: JSON ``{mission_id, state, session_id}`` or a
    bare id; ``{}`` when unreadable."""
    if isinstance(data, dict):
        d = data
    else:
        text = str(data or "").strip()
        try:
            d = json.loads(text) if text.startswith("{") else {"mission_id": text, "state": "open" if text else "closed"}
        except ValueError:
            return {}
    if not isinstance(d, dict):
        return {}
    return {
        "mission_id": str(d.get("mission_id") or ""),
        "state": str(d.get("state") or ("open" if d.get("mission_id") else "closed")),
        "session_id": str(d.get("session_id") or ""),
    }


def is_live(etype: str, prefixes=LIVE_PREFIXES) -> bool:
    return any(etype == p or etype.startswith(p) for p in prefixes)


class _Tail:
    """New complete lines of one JSONL file (rotation: the file shrank)."""

    def __init__(self, path: str, from_end: bool):
        self.path = path
        self.offset = 0
        if from_end:
            try:
                self.offset = os.path.getsize(path)
            except OSError:
                self.offset = 0
        self.buf = b""

    def read(self, chunk: int = TAIL_CHUNK) -> list[bytes]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.offset:  # rotated / truncated
            self.offset, self.buf = 0, b""
        if size == self.offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            data = f.read(chunk)
        self.offset += len(data)
        data = self.buf + data
        lines = data.split(b"\n")
        self.buf = lines.pop()  # incomplete tail
        return [ln for ln in lines if ln.strip()]


class AgentEvents:
    def __init__(
        self,
        core,
        log=None,
        *,
        clock: Callable[[], float] = time.time,
        live: bool = True,
        live_bps: float | None = None,
        live_prefixes=LIVE_PREFIXES,
        tail_dir: str | None = None,
        live_queue: int = LIVE_QUEUE,
        run_info: dict | None = None,
    ):
        self.core = core
        self.log = log
        self.clock = clock
        self.live = bool(live)
        self.live_prefixes = tuple(live_prefixes)
        self.mission: dict = {"mission_id": "", "state": "closed", "session_id": "", "t_ack": None}
        self.run = dict(run_info or {})
        self._t = {"pose": -1e18, "loc": -1e18, "link": -1e18, "tail": -1e18, "phx": -1e18}
        self._loc_why: list[str] = []
        self._jumps_seen = 0
        self._guards_key = None
        self._phx_guards_key = None
        self._phx_last: dict | None = None
        self._plan_id = None
        self._lease_owner = None
        self._last_result = None
        self.queue: deque = deque()
        self.queue_max = int(live_queue)
        self.dropped = 0
        self.relayed = 0
        self.batches = 0
        self.emitted = 0
        self._own_path = getattr(log, "path", None)
        self.tail_dir = tail_dir
        self._tails: dict[str, _Tail] = {}
        self._tail_started = False
        if core is not None and live_bps is not None:
            core.sched.set_budget("C7", float(live_bps))
        if log is not None:
            log.listeners.append(self._on_record)

    # ------------------------------------------------------------ writing
    def emit(self, etype: str, payload=None, **ids) -> None:
        if self.log is None:
            return
        if self.log.emit(etype, payload or {}, **ids):
            self.emitted += 1

    def _on_record(self, rec: dict) -> None:
        if self.live and is_live(str(rec.get("type", "")), self.live_prefixes):
            self._enqueue(rec)

    def _enqueue(self, rec: dict) -> None:
        self.queue.append(rec)
        while len(self.queue) > self.queue_max:
            self.queue.popleft()
            self.dropped += 1

    # ------------------------------------------------------------ inputs
    def on_mission(self, data) -> bool:
        """The latched mission message; True when it named a new mission."""
        m = parse_mission(data)
        if not m:
            return False
        cur = self.mission
        if m["mission_id"] == cur["mission_id"] and m["state"] == cur["state"]:
            return False
        self.mission = {**m, "t_ack": self.clock()}
        if self.log is not None:
            self.log.set_context(mission_id=m["mission_id"])
            if m["session_id"]:
                self.log.session_id = m["session_id"]
        self.emit("mission.ack", {"mission_id": m["mission_id"], "state": m["state"],
                                  "session_id": m["session_id"], "previous": cur["mission_id"]})  # fmt: skip
        self.snapshot_soon("mission")
        return True

    def snapshot_soon(self, why: str) -> None:
        """A ``loc.snapshot`` at the next tick (goal / lease / mission events)."""
        if why not in self._loc_why:
            self._loc_why.append(why)

    # ------------------------------------------------------------ periodic
    def tick(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        core = self.core
        with core._lock:  # noqa: SLF001 — a consistent view of the agent state
            pose = dict(core.pose) if core.pose else None
            fa = dict(core.frame_align) if core.frame_align else None
            reloc = core.reloc
            guards = dict(core.guards or {})
            health = core.health
            plan = dict(core._plan) if getattr(core, "_plan", None) else None  # noqa: SLF001
            platform = core.platform
            reloc_state = reloc.state(now) if reloc is not None else None
            jumps = list(reloc.jumps) if reloc is not None else []
            n_jumps = reloc.n_jumps if reloc is not None else 0
            last_result = core.results[-1] if core.results else None
        if last_result is not None and last_result is not self._last_result:
            self._last_result = last_result
            self.snapshot_soon("action_result")
        if pose is not None and now - self._t["pose"] >= POSE_S:
            self._t["pose"] = now
            self.emit("robot.pose", {**{k: _r(pose.get(k), 3) for k in ("x", "y", "z")},
                                     "yaw": _r(yaw_of_pose(pose), 4), "frame": pose.get("frame")})  # fmt: skip
        if n_jumps > self._jumps_seen:
            new = jumps[-min(len(jumps), n_jumps - self._jumps_seen):] if jumps else []
            for j in new:
                self.emit("loc.jump", {"dxy": j.get("dxy"), "dyaw": j.get("dyaw"), "t_robot": j.get("t"),
                                       "n": n_jumps, "map_T_odom": (fa or {}).get("T")})  # fmt: skip
            self._jumps_seen = n_jumps
        self._guards(guards, platform)
        if plan is not None and plan.get("plan_id") != self._plan_id:
            self._plan_id = plan.get("plan_id")
            self.snapshot_soon("plan")
        if platform == "phoenix":
            self._phoenix(plan, pose, now)
        if self._loc_why or now - self._t["loc"] >= LOC_S:
            self._t["loc"] = now
            why = ",".join(self._loc_why) or "periodic"
            self._loc_why = []
            hm = hydra_lcs(health)
            rs = reloc_state or {}
            self.emit("loc.snapshot", {
                "why": why, "map_T_odom": (fa or {}).get("T"), "frame_align_ok": (fa or {}).get("ok"),
                "parent": (fa or {}).get("parent"), "child": (fa or {}).get("child"),
                "pose": None if pose is None else {k: _r(pose.get(k), 4) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")}
                | {"frame": pose.get("frame")},
                "prior_lcs": rs.get("prior_lcs"), "self_lcs": rs.get("self_lcs"),
                "best_assoc": rs.get("best_assoc"), "jumps": rs.get("jumps"),
                "hydra_accepted": hm[0] if hm else None, "hydra_total": hm[1] if hm else None,
                "reloc_ok": bool(rs.get("prior_lcs")) if reloc_state is not None else None,
                "last_lc_age_s": rs.get("last_lc_age_s"),
            })  # fmt: skip
        if now - self._t["link"] >= LINK_S:
            self._t["link"] = now
            s = core.sched
            self.emit("link.robot", {
                "queue": s.queue_stats(), "rate": {c: round(r, 1) for c, r in s.rate.items()},
                "budgets": dict(s.budgets), "base_rtt": core.ctl_rtt, "live": self.live_stats(),
            })  # fmt: skip
        if self.live and self.tail_dir and now - self._t["tail"] >= TAIL_S:
            self._t["tail"] = now
            self._tail()
        if self.live:
            self._send()

    def _guards(self, g: dict, platform: str) -> None:
        if not g:
            return
        if platform == "phoenix":
            key = json.dumps({k: g.get(k) for k in ("estop_pressed", "safety_stop", "manual_mode", "mcu")}
                             | {"battery": _step(g.get("battery_percent"), 5)}, sort_keys=True, default=str)  # fmt: skip
            if key != self._phx_guards_key:
                self._phx_guards_key = key
                self.emit("phoenix.guards", dict(g))
            return
        key = json.dumps({k: g.get(k) for k in GUARD_KEYS} | {"battery": _step(g.get("battery_percent"), 5)},
                         sort_keys=True, default=str)  # fmt: skip
        if key == self._guards_key:
            return
        self._guards_key = key
        owner = g.get("lease_owner")
        self.emit("guards", {k: g.get(k) for k in (*GUARD_KEYS, "battery_percent", "notes")})
        if owner != self._lease_owner:
            if self._lease_owner is not None:
                self.snapshot_soon("lease")
            self._lease_owner = owner

    def _phoenix(self, plan, pose, now) -> None:
        from fleetlink.agent import phoenix_progress

        p = phoenix_progress(plan, pose)
        if p is None:
            return
        last = self._phx_last
        changed = (
            last is None
            or p["plan_id"] != last.get("plan_id")
            or p["done"] != last.get("done")
            or (p["dist_to_goal_m"] is not None and last.get("dist_to_goal_m") is not None
                and abs(p["dist_to_goal_m"] - last["dist_to_goal_m"]) >= PHOENIX_STEP_M)
            or ((p["dist_to_goal_m"] is None) != (last.get("dist_to_goal_m") is None))
        )  # fmt: skip
        new_plan = last is not None and p["plan_id"] != last.get("plan_id")
        if new_plan or (changed and now - self._t["phx"] >= PHOENIX_MIN_S):
            self._t["phx"] = now
            self._phx_last = dict(p)
            self.emit("phoenix.progress", {"plan": p["plan_id"], "dist_left": p["dist_to_goal_m"],
                                           "done": p["done"], "stamp": p["stamp"]}, plan_id=p["plan_id"])  # fmt: skip

    # ------------------------------------------------------------ live relay
    def _tail(self) -> None:
        d = self.tail_dir
        try:
            names = sorted(n for n in os.listdir(d) if n.endswith(".jsonl"))
        except OSError:
            return
        first = not self._tail_started
        self._tail_started = True
        own = os.path.abspath(self._own_path) if self._own_path else None
        for n in names:
            path = os.path.join(d, n)
            if own and os.path.abspath(path) == own:
                continue
            if re.search(r"\.\d+\.jsonl$", n):
                continue  # rotated: its lines were read before it moved
            t = self._tails.get(path)
            if t is None:
                # files present when the agent started: only what comes next
                t = self._tails[path] = _Tail(path, from_end=first)
            for line in t.read():
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                etype = str(rec.get("type", ""))
                if etype.startswith(("lease.", "plan.received", "sequence.start")):
                    self.snapshot_soon(etype)  # goal / lease edges: where was it
                if is_live(etype, self.live_prefixes):
                    self._enqueue(rec)

    def _send(self) -> None:
        core = self.core
        if not self.queue or core.sched.queues["C7"]:
            return  # one batch at a time: the scheduler's C7 budget sets the pace
        batch, size = [], 0
        while self.queue:
            line = json.dumps(self.queue[0], default=str, separators=(",", ":"))
            if batch and size + len(line) > LIVE_BATCH_BYTES:
                break
            self.queue.popleft()
            batch.append(line)
            size += len(line) + 1
        n = len(batch)

        def payload(lines=batch):
            return {"lines": lines, "dropped": self.dropped}

        def sent(_n, k=n):
            self.relayed += k
            self.batches += 1

        core._queue(STREAM, f"{core.prefix}/{STREAM}", payload, size + 32, on_sent=sent)  # noqa: SLF001

    def live_stats(self) -> dict:
        return {"queued": len(self.queue), "relayed": self.relayed, "dropped": self.dropped,
                "batches": self.batches, "on": self.live}  # fmt: skip

    def state(self) -> dict:
        """C1 state keys: the mission this robot acknowledged, its run."""
        m = self.mission
        return {
            "mission": {"mission_id": m["mission_id"], "state": m["state"], "session_id": m["session_id"],
                        "t_ack": m["t_ack"]},
            "run": {k: self.run.get(k) for k in ("run_id", "session_id", "experiment", "machine") if k in self.run},
            "events": {"emitted": self.emitted, **self.live_stats()},
        }  # fmt: skip


def _r(v, nd):
    try:
        return None if v is None else round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _step(v, step):
    try:
        return None if v is None else int(float(v) // step)
    except (TypeError, ValueError):
        return None


def read_run_info(run_dir: str | None) -> dict:
    """``run_info.json`` of the run (run_id, session_id, experiment, machine)."""
    if not run_dir:
        return {}
    try:
        with open(os.path.join(run_dir, "run_info.json")) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    out = {k: d.get(k) for k in ("run_id", "session_id", "experiment", "machine") if d.get(k)}
    out.setdefault("run_id", os.path.basename(run_dir.rstrip("/")))
    if os.environ.get("ADT4_SESSION_ID"):
        out.setdefault("session_id", os.environ["ADT4_SESSION_ID"])
    return out
