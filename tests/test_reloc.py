"""``reloc`` (prior-map loop closures, map -> odom jumps) in the C1 state."""

import math
from types import SimpleNamespace as NS

from helpers import RecordingTransport

from fleetlink.agent import AgentCore, frame_align_failed
from fleetlink.reloc import RelocTracker, loop_closure_to_dict
from fleetlink.transport import VirtualClock


def _yaw_T(x, y, deg):
    h = math.radians(deg) / 2
    return [x, y, 0.0, math.cos(h), 0.0, 0.0, math.sin(h)]


def _lc(r1, r2, s1=3, s2=7, assoc=5):
    return {"robot1": r1, "robot2": r2, "submap1": s1, "submap2": s2, "assoc": assoc}


def _align(T):
    return {"parent": "map", "child": "r/odom", "T": T, "stamp": 0.0, "ok": True, "reason": ""}


def test_loop_closure_to_dict_reads_the_roman_message():
    msg = NS(robot1_id=0, robot2_id=1, submap1_id=12, submap2_id=4, num_associations=9)
    assert loop_closure_to_dict(msg) == _lc(0, 1, 12, 4, 9)


def test_counts_only_closures_between_this_robot_and_another():
    tr = RelocTracker(live_id=0)
    tr.on_loop_closure(_lc(0, 0), 10.0)  # self: odometry drift, not relocalization
    tr.on_loop_closure(_lc(2, 3), 11.0)  # two other robots
    tr.on_loop_closure(_lc(0, 1, 12, 4, 9), 12.0)
    tr.on_loop_closure(_lc(1, 0, 5, 14, 6), 13.0)  # either order
    st = tr.state(20.0)
    assert (st["prior_lcs"], st["self_lcs"], st["best_assoc"]) == (2, 1, 9)
    assert st["last_lc_age_s"] == 7.0
    newest, older = st["lcs"]
    assert newest == {"prior": 1, "live_submap": 14, "prior_submap": 5, "assoc": 6, "age_s": 7.0}
    assert older["live_submap"] == 12 and older["prior_submap"] == 4


def test_recent_lists_are_bounded():
    tr = RelocTracker(recent=3)
    for i in range(5):
        tr.on_loop_closure(_lc(0, 1, i, i), float(i))
        tr.on_align(_yaw_T(float(i), 0.0, 0.0), float(i))
    st = tr.state(10.0)
    assert st["prior_lcs"] == 5 and [r["live_submap"] for r in st["lcs"]] == [4, 3, 2]
    assert st["jumps"] == 4 and len(st["recent_jumps"]) == 3


def test_align_jumps_and_total_ignore_noise():
    tr = RelocTracker()
    assert tr.state(0.0)["total"] is None and tr.state(0.0)["last_jump"] is None
    tr.on_align(_yaw_T(0.0, 0.0, 0.0), 1.0)  # first sample: the starting guess
    tr.on_align(_yaw_T(0.01, 0.0, 0.1), 2.0)  # optimizer noise
    assert tr.state(3.0)["jumps"] == 0
    tr.on_align(_yaw_T(3.0, 4.0, 179.0), 4.0)
    tr.on_align(_yaw_T(3.0, 4.0, -179.0), 6.0)  # 2 deg across the wrap, not 358
    st = tr.state(10.0)
    assert st["jumps"] == 2
    assert st["last_jump"] == {"dxy": 0.0, "dyaw": 2.0, "age_s": 4.0}
    assert st["recent_jumps"][1] == {"dxy": 5.0, "dyaw": 179.0, "age_s": 6.0}
    assert st["total"] == {"dxy": 5.0, "dyaw": -179.0}
    tr.on_align(None, 7.0)  # lookup failed: ignored
    assert tr.state(10.0)["jumps"] == 2


def _core():
    tr = RecordingTransport()
    clock = VirtualClock(1000.0)
    return tr, clock, AgentCore("r", tr, {}, clock)


def _states(tr):
    return [m["p"] for _, m, _ in tr.messages("adt4/r/state")]


def test_state_omits_reloc_until_enabled():
    tr, clock, core = _core()
    core.on_loop_closure(_lc(0, 1))  # no ROMAN subscription: ignored
    core.tick(clock.advance(0.5))
    assert "reloc" not in _states(tr)[-1]  # backward compatible payload


def test_state_carries_reloc_from_closures_and_alignment():
    tr, clock, core = _core()
    core.enable_reloc(0)
    core.on_frame_align(_align(_yaw_T(0.0, 0.0, 0.0)))
    clock.advance(1.0)
    core.on_loop_closure(_lc(0, 1, 2, 9, 12))
    clock.advance(1.0)
    core.on_frame_align(_align(_yaw_T(-18.0, -35.0, 178.0)))  # the closure moved it
    core.on_frame_align(frame_align_failed("map", "r/odom", "lost"))  # not a sample
    core.tick(clock.advance(0.5))
    rl = _states(tr)[-1]["reloc"]
    assert rl["prior_lcs"] == 1 and rl["best_assoc"] == 12
    assert rl["last_lc_age_s"] == 1.5
    assert rl["jumps"] == 1 and rl["last_jump"]["dyaw"] == 178.0
    assert rl["total"]["dxy"] == round(math.hypot(18.0, 35.0), 3)
