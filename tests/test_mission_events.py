"""Agent mission event log + live C7 event relay (ADT4 mission logging B1)."""

from __future__ import annotations

import json
import os
import statistics

from helpers import RecordingTransport

from fleetlink import codec
from fleetlink.agent import AgentCore
from fleetlink.eventlog import EventLog
from fleetlink.mission_events import AgentEvents, is_live, parse_mission, read_run_info
from fleetlink.receiver import Receiver
from fleetlink.reloc import RelocTracker
from fleetlink.scheduler import Item
from fleetlink.transport import CLASSES, PRIORITY_ORDER, LinkEmulator, VirtualClock

POSE = {"x": 1.0, "y": 2.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "frame": "r/map", "t": 0.0}  # fmt: skip
HEALTH = {"nodes": [{"nickname": "hydra_multi", "node_name": "/r/hydra_multi", "status": "nominal",
                     "notes": "3 / 7 valid interrobot loop closures"}]}  # fmt: skip


def _read(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _agent(tmp_path, live=True, tr=None, clock=None):
    tr = tr or RecordingTransport()
    clock = clock or VirtualClock(1000.0)
    core = AgentCore("r", tr, {}, clock)
    core.on_pose(dict(POSE))
    log = EventLog(str(tmp_path / "events" / "fleetlink_agent.jsonl"), "fleetlink_agent", robot="r",
                   machine="m", run_id="m_1", session_id="s1", clock=clock)  # fmt: skip
    ev = AgentEvents(core, log, clock=clock, live=live, tail_dir=str(tmp_path / "events"))
    core.events = ev
    return tr, clock, core, ev, log


def _run(clock, core, seconds, dt=0.05):
    for _ in range(int(round(seconds / dt))):
        core.tick(clock.advance(dt))


def test_c7_is_the_lowest_class():
    assert CLASSES[-1] == "C7" and PRIORITY_ORDER[-1] == "C7"
    assert AgentCore.STREAM_CLASS["events"] == "C7"


def test_parse_mission():
    assert parse_mission('{"mission_id": "m1", "state": "open", "session_id": "s"}') == {
        "mission_id": "m1", "state": "open", "session_id": "s"}  # fmt: skip
    assert parse_mission("m2")["mission_id"] == "m2"
    assert parse_mission("")["state"] == "closed"
    assert parse_mission("{bad") == {}
    assert is_live("lease.intervention") and is_live("action.start") and not is_live("robot.pose")


def test_pose_snapshot_jump_guards_and_ack(tmp_path):
    tr, clock, core, ev, log = _agent(tmp_path)
    core.enable_reloc(0)
    core.on_health(HEALTH)
    T0 = [0.0, 0.0, 0, 1, 0, 0, 0]
    core.on_frame_align({"parent": "map", "child": "r/map", "T": T0, "stamp": 0, "ok": True})
    core.on_loop_closure({"robot1": 0, "robot2": 1, "submap1": 3, "submap2": 9, "assoc": 12})
    core.on_guards({"lease_owner": "spot_executor", "lease_owned": True, "battery_percent": 80.0, "estop_pressed": False})
    _run(clock, core, 3.0)
    assert ev.on_mission('{"mission_id": "m1", "state": "open", "session_id": "s1"}')
    assert not ev.on_mission('{"mission_id": "m1", "state": "open", "session_id": "s1"}')  # same: no re-ack
    core.on_frame_align({"parent": "map", "child": "r/map", "T": [2.0, 0, 0, 1, 0, 0, 0], "stamp": 0, "ok": True})
    core.on_guards({"lease_owner": "tablet-operator", "lease_owned": False, "battery_percent": 79.0,
                    "estop_pressed": False})  # fmt: skip
    _run(clock, core, 11.0)
    log.close()
    recs = _read(tmp_path / "events" / "fleetlink_agent.jsonl")
    types = [r["type"] for r in recs]
    assert types.count("robot.pose") >= 12  # 1 Hz
    assert "mission.ack" in types and "loc.jump" in types
    ack = next(r for r in recs if r["type"] == "mission.ack")
    assert ack["payload"]["mission_id"] == "m1" and ack["mission_id"] == "m1"
    # every event after the ack carries the mission id
    after = recs[types.index("mission.ack"):]
    assert all(r["mission_id"] == "m1" for r in after)
    snaps = [r["payload"] for r in recs if r["type"] == "loc.snapshot"]
    assert snaps[0]["prior_lcs"] == 1 and snaps[0]["best_assoc"] == 12
    assert snaps[0]["hydra_accepted"] == 3 and snaps[0]["hydra_total"] == 7
    assert snaps[0]["reloc_ok"] is True and snaps[0]["map_T_odom"] == T0
    whys = [s["why"] for s in snaps]
    assert any("mission" in w for w in whys) and any("lease" in w for w in whys)
    jump = next(r for r in recs if r["type"] == "loc.jump")["payload"]
    assert jump["dxy"] == 2.0
    g = [r["payload"] for r in recs if r["type"] == "guards"]
    assert [x["lease_owner"] for x in g] == ["spot_executor", "tablet-operator"]  # battery 80 -> 79: no event
    # C1 state carries the acknowledged mission and the event counters
    st = codec.unpack(tr.messages("adt4/r/state")[-1][1]) if isinstance(tr.messages("adt4/r/state")[-1][1], bytes) \
        else tr.messages("adt4/r/state")[-1][1]
    assert st["p"]["mission"]["mission_id"] == "m1"
    assert st["p"]["events"]["emitted"] == len(recs)
    assert "link.robot" in types


def test_live_relay_tails_executor_log_and_respects_budget(tmp_path):
    tr, clock, core, ev, log = _agent(tmp_path)
    ex = EventLog(str(tmp_path / "events" / "spot_executor.jsonl"), "spot_executor", robot="r", clock=clock)
    _run(clock, core, 1.0)  # the tail starts (spot_executor.jsonl did not exist yet)
    for i in range(400):  # a burst: 400 action events (incompressible pads)
        ex.emit("action.start", {"i": i, "pad": os.urandom(60).hex()})
    ex.emit("follow.progress", {"frac": 0.5})
    ex.emit("robot.pose", {"x": 0})  # not relayed
    ex.close()
    _run(clock, core, 20.0)
    batches = [codec.unpack(d) for k, d, c in tr.sent if k == "adt4/r/events"]
    assert batches and all(c == "C7" for k, d, c in tr.sent if k == "adt4/r/events")
    lines = [json.loads(ln) for b in batches for ln in b["p"]["lines"]]
    types = {x["type"] for x in lines}
    assert "action.start" in types and "robot.pose" not in types
    sent = sum(len(d) for k, d, c in tr.sent if k == "adt4/r/events")
    # at most the C7 budget (2 kB/s) over the run plus one burst
    assert sent <= 2000 * 21 + 4000
    assert ev.live_stats()["queued"] > 0  # the rest waits (or is dropped), never floods
    ev.queue_max = 10
    for i in range(50):
        ev._enqueue({"type": "action.end", "i": i})
    assert len(ev.queue) == 10 and ev.dropped >= 40


def test_live_off_sends_nothing(tmp_path):
    tr, clock, core, ev, log = _agent(tmp_path, live=False)
    ev.on_mission("m9")
    _run(clock, core, 3.0)
    assert not [k for k, _, _ in tr.sent if k == "adt4/r/events"]
    assert any(r["type"] == "mission.ack" for r in _read(tmp_path / "events" / "fleetlink_agent.jsonl"))


def test_receiver_delivers_events(tmp_path):
    clk = VirtualClock(1000.0)
    link = LinkEmulator(clock=clk, bandwidth=200_000.0, latency=0.02)
    _, _, core, ev, log = _agent(tmp_path, tr=link.robot, clock=clk)
    got = []
    rx = Receiver(link.base, lambda r, k, d: got.append((r, k, d)) if k == "events" else None,
                  clock=clk, threaded=False)  # fmt: skip
    rx.start()
    ev.on_mission('{"mission_id": "m1", "state": "open"}')
    for _ in range(100):
        core.tick(clk.advance(0.05))
        link.advance()
        rx.tick()
    lines = [json.loads(ln) for r, k, d in got for ln in d["lines"]]
    assert any(x["type"] == "mission.ack" for x in lines)
    assert any(x["type"] == "loc.snapshot" for x in lines)
    assert rx.state("r")["mission"]["mission_id"] == "m1"


def test_run_info(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "m_20261111-000000_abcd"
    d.mkdir(parents=True)
    (d / "run_info.json").write_text(json.dumps({"run_id": "m_20261111-000000_abcd", "session_id": "s7",
                                                 "experiment": "e", "machine": "m", "git": {}}))  # fmt: skip
    monkeypatch.delenv("ADT4_SESSION_ID", raising=False)
    assert read_run_info(str(d)) == {"run_id": "m_20261111-000000_abcd", "session_id": "s7", "experiment": "e",
                                     "machine": "m"}  # fmt: skip
    assert read_run_info(str(tmp_path / "nope"))["run_id"] == "nope"


# ------------------------------------------------------------ fault harness
class _Fleet:
    """An agent on a saturated 150 kB/s radio (a 3.2 MB C3 mesh backlog, C2 at
    budget, C1 state at 4 Hz) with the base receiver asking ``ping`` / ``fb`` /
    ``set_budget`` over C0; optionally a flood of live mission events."""

    def __init__(self, tmp_path, events: bool, c7_budget=2000):
        self.clk = VirtualClock(1000.0)
        self.link = LinkEmulator(clock=self.clk, bandwidth=150_000.0, latency=0.05, queue_bytes=128 * 1024,
                                 backpressure=False, seed=3, reliable=dict.fromkeys(CLASSES, False))  # fmt: skip
        tmp_path.mkdir(parents=True, exist_ok=True)
        _, _, self.core, self.ev, self.log = _agent(tmp_path, live=events, tr=self.link.robot, clock=self.clk)
        self.core.sched.set_budget("C7", c7_budget)
        self.state_t = {}
        self.lat = []
        self.rx = Receiver(self.link.base, self._upd, clock=self.clk, threaded=False)
        self.rx.start()
        for i in range(40):
            self.core.sched.enqueue(Item("C3", f"adt4/r/mesh/tile/{i}", lambda: bytes(80_000), 80_000))
        self.events = events
        self.n = 0
        self.acks = []  # C0 request -> ack seconds
        self.open_reqs = {}

    def _upd(self, robot, kind, d):
        if kind == "state":
            self.lat.append(self.clk() - float(d["pose"]["t"]))

    def step(self, dt=0.05):
        self.clk.advance(dt)
        self.n += 1
        pose = dict(POSE, t=self.clk())
        self.core.on_pose(pose)
        if not self.core.sched.queues["C2"]:
            self.core.sched.enqueue(Item("C2", "adt4/r/dsg/delta/x", lambda: bytes(2500), 2500))
        if len(self.core.sched.queues["C3"]) < 40:
            self.core.sched.enqueue(Item("C3", "adt4/r/mesh/tile/x", lambda: bytes(80_000), 80_000))
        if self.events:  # far more events than C7 may carry
            for i in range(20):
                self.ev._enqueue({"type": "action.start", "i": i, "pad": "y" * 200})
        if self.n % 40 == 0:
            self.open_reqs[self.rx.set_budget("r", "C3", 100_000)] = self.clk()
        self.core.tick(self.clk())
        self.link.advance()
        self.rx.tick()
        pending = {p["id"] for p in self.rx.pending_requests()}
        for rid in [r for r in self.open_reqs if r not in pending]:
            self.acks.append(self.clk() - self.open_reqs.pop(rid))

    def run(self, seconds):
        for _ in range(int(seconds / 0.05)):
            self.step()


def test_fault_harness_c7_never_delays_c0_c1(tmp_path):
    quiet = _Fleet(tmp_path / "q", events=False)
    loud = _Fleet(tmp_path / "l", events=True, c7_budget=None)  # even unmetered C7
    capped = _Fleet(tmp_path / "c", events=True)
    for f in (quiet, loud, capped):
        f.run(60.0)
    for f in (quiet, loud, capped):
        assert len(f.lat) > 150
    q = statistics.quantiles(quiet.lat, n=20)[18]
    lo = statistics.quantiles(loud.lat, n=20)[18]
    ca = statistics.quantiles(capped.lat, n=20)[18]
    # strict priority: C7 can at most hold one fragment (8 kB at 150 kB/s ~ 55 ms)
    assert lo <= q + 0.06, (q, lo)
    assert ca <= q + 0.06, (q, ca)
    # C0: every request acked, as fast with the event flood as without
    for f in (quiet, loud, capped):
        assert len(f.acks) >= 25 and len(f.open_reqs) <= 1
    assert max(loud.acks) <= max(quiet.acks) + 0.06, (max(quiet.acks), max(loud.acks))
    assert max(capped.acks) <= max(quiet.acks) + 0.06
    # the capped relay stays within its budget
    c7 = capped.link.robot.tx.stats.put_bytes["C7"]
    assert 0 < c7 <= 2000 * 61 + 2000
    assert capped.ev.dropped > 0  # the flood is dropped at the robot, not queued on the radio
