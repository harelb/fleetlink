"""``priority: high`` relayed messages (a Spot manipulation approval request:
candidate JPEGs + depth PNG, a few hundred kB, newest wins) go ahead of the bulk
backlog: the C2 lane (zenoh DATA_HIGH) and the agent's scheduler holding C2-C6
while one is pending.  C0 / C1 keep going."""

import random

import pytest
from helpers import RecordingTransport

from fleetlink.relay import RelayConfig, RelayConfigError, RelayCore
from fleetlink.scheduler import Item, Scheduler
from fleetlink.transport import CLASSES, LinkEmulator, VirtualClock

REQ = "/robot1/spot_executor_node/manipulation_request"
BUDGETS = {
    "C1": 5000,
    "C2": 50000,
    "C3": 100000,
    "C4": 200000,
    "C5": 400000,
    "C6": 150000,
}


def cfg(priority=True):
    req = {"topic": "/{robot}/spot_executor_node/manipulation_request",
           "type": "nlu_interface_rviz/msg/ManipulationApprovalRequest",
           "dir": "up", "mode": "latest_reliable", "ttl_s": 120}  # fmt: skip
    req.update({"priority": "high"} if priority else {"cls": "C3"})
    return RelayConfig({"version": 1, "topics": [req]})


def payload(n, seed=0):
    return random.Random(seed).randbytes(n)  # images do not compress


class Fleet:
    """Robot agent scheduler + relay and the base relay over a saturated radio:
    150 kB/s, the scheduler's C3 queue holding 3.2 MB of mesh tiles, C2 deltas
    at budget and C1 state at 4 Hz (as on the robot test, 2026-11-11)."""

    def __init__(self, priority=True, base=True, bandwidth=150_000.0):
        self.clk = VirtualClock(1000.0)
        self.link = LinkEmulator(
            clock=self.clk, bandwidth=bandwidth, latency=0.05, loss=0.0,
            queue_bytes=128 * 1024, backpressure=False, seed=7,
            reliable=dict.fromkeys(CLASSES, False),
        )  # fmt: skip
        self.sched = Scheduler(self.link.robot, BUDGETS, self.clk)
        c = cfg(priority)
        self.robot = RelayCore("robot1", "robot", self.link.robot, c, clock=self.clk,
                               type_hash=lambda t: "H:" + t)  # fmt: skip
        self.got = []
        self.base = None
        if base:
            self.base = RelayCore(
                "robot1", "base", self.link.base, c, base="base1", clock=self.clk,
                on_message=lambda e, d, m: self.got.append((self.clk(), bytes(d))) or 1,
                type_hash=lambda t: "H:" + t,
            )  # fmt: skip
        if priority:
            self.sched.hold = self.robot.urgent
        self.state_lat = []  # C1 delivery latency (s)
        self.link.base.subscribe(
            "adt4/robot1/state",
            lambda k, d: self.state_lat.append(self.clk() - float(d.decode())),
        )
        for i in range(40):  # the mesh backlog: 40 tiles of 80 kB
            self.sched.enqueue(Item("C3", f"adt4/robot1/mesh/tile/{i}", lambda: bytes(80_000), 80_000))  # fmt: skip
        self.n = 0

    def step(self, dt=0.05):
        self.clk.advance(dt)
        self.n += 1
        if self.n % 5 == 0:  # C1 state, 4 Hz
            t = self.clk()
            self.sched.enqueue(Item("C1", "adt4/robot1/state", lambda t=t: str(t).encode(), 30, tag="state"))  # fmt: skip
        if not self.sched.queues["C2"]:  # graph deltas at budget
            self.sched.enqueue(Item("C2", "adt4/robot1/dsg/delta", lambda: bytes(2500), 2500))  # fmt: skip
        if len(self.sched.queues["C3"]) < 40:  # the mesh keeps coming
            self.sched.enqueue(Item("C3", "adt4/robot1/mesh/tile/x", lambda: bytes(80_000), 80_000))  # fmt: skip
        self.sched.drain()
        self.link.advance()
        self.robot.tick()
        if self.base is not None:
            self.base.tick()

    def run(self, seconds, until=None):
        for _ in range(int(round(seconds / 0.05))):
            self.step()
            if until is not None and until():
                return

    def deliver_time(self, data, budget_s):
        t0 = self.clk()
        self.robot.publish(REQ, data)
        self.run(budget_s, until=lambda: any(d == data for _, d in self.got))
        hit = [t for t, d in self.got if d == data]
        return hit[0] - t0 if hit else None


def test_config_priority_defaults_and_validation():
    e = cfg(True).resolve("robot1").topics[REQ]
    assert e.priority == "high" and e.cls == "C2"
    assert cfg(False).resolve("robot1").topics[REQ].priority == "normal"
    base = {"topic": "/a", "type": "p/msg/T", "dir": "up", "mode": "latest_reliable"}
    for bad in ({"priority": "high", "cls": "C3"}, {"priority": "urgent"},
                {"priority": "high", "mode": "latest"}):  # fmt: skip
        with pytest.raises(RelayConfigError):
            RelayConfig({"version": 1, "topics": [dict(base, **bad)]}).resolve("r")
    ok = RelayConfig({"version": 1, "topics": [dict(base, priority="high", cls="C1")]})
    assert ok.resolve("r").topics["/a"].cls == "C1"


def test_scheduler_hold_skips_bulk_but_not_c0_c1():
    tr = RecordingTransport()
    s = Scheduler(tr, {}, VirtualClock())
    for c in CLASSES:
        s.enqueue(Item(c, f"k/{c}", lambda: b"x", 1))
    s.hold = lambda now: True
    s.drain(0.0)
    assert sorted(c for _, _, c in tr.sent) == ["C0", "C1"]
    s.hold = lambda now: 1 / 0  # a broken hook never stops the link
    s.drain(1.0)
    assert sorted(c for _, _, c in tr.sent) == list(CLASSES)


def test_request_goes_on_the_wire_at_publish_and_holds_the_mesh():
    f = Fleet()
    f.run(3)  # heartbeats, backlog at budget
    c3 = f.sched.sent_bytes["C3"]
    put_c2 = f.link.robot.tx.stats.put_bytes["C2"]
    f.robot.publish(REQ, payload(300_000))
    # first window of parts put at publish time, before any scheduler tick
    assert f.robot.out[REQ].c["parts_sent"] == f.robot.chunk_window
    assert f.link.robot.tx.stats.put_bytes["C2"] > put_c2  # on the C2 lane
    assert f.robot.urgent()
    f.step()  # the next tick sends no bulk
    assert f.sched.sent_bytes["C3"] == c3 and f.sched.held_drains == 1


def test_request_beats_a_full_mesh_backlog_and_c1_keeps_flowing():
    # a radio slower than the bulk budgets (100 kB/s < C2 + C3): before, the
    # request shares DATA_LOW with the mesh and loses parts to its full queue
    # (~8.6 s here, ~85 s at 60 kB/s); now ~300 kB at the link rate (~3.1 s)
    old = Fleet(priority=False, bandwidth=100_000.0)
    old.run(5)
    t_old = old.deliver_time(payload(300_000), 60)
    new = Fleet(priority=True, bandwidth=100_000.0)
    new.run(5)
    n_state = len(new.state_lat)
    t_new = new.deliver_time(payload(300_000), 60)
    assert t_new is not None and t_new < 4.5
    assert t_old is None or t_old > 2 * t_new
    lat = new.state_lat[n_state:]
    assert lat and max(lat) < 0.5  # C1 state not starved meanwhile
    new.run(0.3)  # the ack comes back
    assert not new.robot.urgent()  # acked: the hold is released
    c3 = new.sched.sent_bytes["C3"]
    new.run(2)
    assert new.sched.sent_bytes["C3"] > c3  # mesh resumes
    print(f"\nrequest 300 kB behind 3.2 MB mesh: before {t_old} s, after {t_new:.2f} s")


def test_newer_request_supersedes_the_pending_one():
    f = Fleet()
    f.run(3)
    a, b = payload(300_000, 1), payload(300_000, 2)
    f.robot.publish(REQ, a)
    f.run(0.5)
    t = f.deliver_time(b, 30)
    assert t is not None
    f.run(2)
    assert [d for _, d in f.got] == [b]  # the older request never applied
    assert f.robot.out[REQ].c["superseded"] == 1
    assert not f.robot.out[REQ].pending


def test_hold_is_bounded_when_the_base_is_unreachable():
    f = Fleet(base=False)
    f.run(1)
    f.robot.publish(REQ, payload(100_000))
    f.run(5)
    assert f.robot.urgent()
    c3 = f.sched.sent_bytes["C3"]
    f.run(6)  # past priority_hold_s (10 s)
    assert not f.robot.urgent() and f.robot.out[REQ].pending
    assert f.sched.sent_bytes["C3"] > c3
