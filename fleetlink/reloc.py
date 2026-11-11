"""Relocalization evidence for the C1 ``reloc`` state (pure, no ROS).

A ``map -> <robot>/odom`` transform alone does not mean a robot is relocalized:
hydra_multi publishes one from start-up, before ROMAN has matched anything
against the prior map.  :class:`RelocTracker` keeps what actually supports the
alignment:

* ROMAN loop closures between the live robot and another (prior) robot id
  (``roman_msgs/LoopClosure``: ``robot1_id``/``robot2_id``, ``num_associations``),
  counted, with the most recent ones listed;
* how ``frame_align`` (``map -> odom``) moved: every change above a small
  threshold is a *jump* (translation + yaw), plus the total since the first one.

Times are robot-clock seconds; :meth:`state` turns them into ages so the base
needs no clock-offset estimate (as ``frame_align.age_s`` does).
"""

from __future__ import annotations

import math
from collections import deque

# a map -> odom change below these is optimizer noise, not a jump
JUMP_MIN_M = 0.05
JUMP_MIN_DEG = 0.5
RECENT = 10


def yaw_of(T) -> float:
    """Yaw (rad) of ``[x, y, z, qw, qx, qy, qz]``."""
    _, _, _, qw, qx, qy, qz = T
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def _dyaw_deg(a: float, b: float) -> float:
    d = math.degrees(b - a)
    return (d + 180.0) % 360.0 - 180.0


def loop_closure_to_dict(msg) -> dict:
    """``roman_msgs/LoopClosure`` -> the fields the tracker uses."""
    return {
        "robot1": int(msg.robot1_id),
        "robot2": int(msg.robot2_id),
        "submap1": int(msg.submap1_id),
        "submap2": int(msg.submap2_id),
        "assoc": int(msg.num_associations),
    }


class RelocTracker:
    """Prior-map loop closures and ``map -> odom`` jumps of one robot."""

    def __init__(self, live_id: int = 0, recent: int = RECENT):
        self.live_id = int(live_id)
        self.prior_lcs = 0
        self.self_lcs = 0
        self.best_assoc = 0
        self.last_lc_t: float | None = None
        self.lcs: deque = deque(maxlen=recent)
        self.jumps: deque = deque(maxlen=recent)
        self.n_jumps = 0
        self._first_T = None
        self._last_T = None

    def on_loop_closure(self, lc: dict, t: float) -> None:
        r1, r2 = lc["robot1"], lc["robot2"]
        if r1 == r2:
            self.self_lcs += 1
            return
        if self.live_id not in (r1, r2):
            return  # between two other robots: says nothing about this one
        live_first = r1 == self.live_id
        rec = {
            "t": t,
            "prior": r2 if live_first else r1,
            "live_submap": lc["submap1"] if live_first else lc["submap2"],
            "prior_submap": lc["submap2"] if live_first else lc["submap1"],
            "assoc": lc["assoc"],
        }
        self.prior_lcs += 1
        self.best_assoc = max(self.best_assoc, rec["assoc"])
        self.last_lc_t = t
        self.lcs.append(rec)

    def on_align(self, T, t: float) -> None:
        """A ``map -> odom`` sample (``frame_align.T``); None/invalid is ignored."""
        if not T or len(T) != 7:
            return
        T = [float(v) for v in T]
        if self._first_T is None:
            self._first_T = self._last_T = T
            return
        dxy = math.hypot(T[0] - self._last_T[0], T[1] - self._last_T[1])
        dyaw = _dyaw_deg(yaw_of(self._last_T), yaw_of(T))
        if dxy < JUMP_MIN_M and abs(dyaw) < JUMP_MIN_DEG:
            return
        self.n_jumps += 1
        self.jumps.append({"t": t, "dxy": round(dxy, 3), "dyaw": round(dyaw, 2)})
        self._last_T = T

    def state(self, now: float) -> dict:
        def age(t):
            return None if t is None else round(max(0.0, now - t), 1)

        total = None
        if self._first_T is not None:
            a, b = self._first_T, self._last_T
            total = {
                "dxy": round(math.hypot(b[0] - a[0], b[1] - a[1]), 3),
                "dyaw": round(_dyaw_deg(yaw_of(a), yaw_of(b)), 2),
            }
        last_jump = self.jumps[-1] if self.jumps else None
        return {
            "prior_lcs": self.prior_lcs,
            "self_lcs": self.self_lcs,
            "best_assoc": self.best_assoc,
            "last_lc_age_s": age(self.last_lc_t),
            "lcs": [
                {**{k: v for k, v in r.items() if k != "t"}, "age_s": age(r["t"])}
                for r in reversed(self.lcs)
            ],
            "jumps": self.n_jumps,
            "last_jump": None
            if last_jump is None
            else {"dxy": last_jump["dxy"], "dyaw": last_jump["dyaw"], "age_s": age(last_jump["t"])},
            "recent_jumps": [
                {"dxy": j["dxy"], "dyaw": j["dyaw"], "age_s": age(j["t"])}
                for j in reversed(self.jumps)
            ],
            "total": total,
        }
