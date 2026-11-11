"""Application-level reliability with every class DROP (deterministic, emulated
link): C2 gap repair by ``resend``, escalation to resync, transport closes,
TCP head-of-line blocking and ``fb``-driven AIMD."""

import random

from helpers import (
    RecordingTransport,
    env_bytes,
    states_equal,
    synthetic_mesh,
    synthetic_records,
)

from fleetlink import codec
from fleetlink.agent import AgentCore, zenoh_version_warning
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator, VirtualClock

POSE = {"x": 0, "y": 0, "z": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0, "frame": "map"}
DT = 0.05


class Sim:
    def __init__(self, cfg=None, n=200, mesh=0, **link):
        self.clock = VirtualClock()
        self.link = LinkEmulator(self.clock, **{"latency": 0.1, **link})
        self.rx = Receiver(self.link.base, None, clock=self.clock, threaded=False)
        self.rx.start()
        self.agent = AgentCore("r", self.link.robot, cfg, self.clock)
        recs, edges, ls = synthetic_records(n, seed=2)
        self.agent.on_records(recs, edges, ls, synthetic_mesh(mesh) if mesh else None)
        self.agent.on_pose(dict(POSE, t=0.0))
        self.max_state_age = 0.0
        self.drop_c2 = set()  # C2 seqs to lose on the robot side (first copy only)

        put = self.link.robot.put

        def lossy_put(key, data, cls):
            if cls == "C2":
                seq = codec.unpack(data)["seq"]
                if seq in self.drop_c2:
                    self.drop_c2.discard(seq)
                    return
            put(key, data, cls)

        self.link.robot.put = lossy_put

    def run(self, seconds, track_from=None):
        for _ in range(int(round(seconds / DT))):
            t = self.clock.advance(DT)
            self.link.advance()
            self.agent.tick(t)
            self.rx.tick(t)
            r = self.rx._robots.get("r")
            if track_from is not None and t >= track_from and r and r.last_state_rx:
                self.max_state_age = max(self.max_state_age, t - r.last_state_rx)

    def graph_ok(self):
        if "r" not in self.rx.robots():
            return False
        return states_equal(self.rx.graph("r"), self.agent.graph_state())[0]

    def grow(self, n, seed):
        recs, edges, ls = synthetic_records(n, seed=seed)
        self.agent.on_records(recs, edges, ls)


def test_c2_gap_repaired_by_resend_without_resync():
    sim = Sim({"budgets": {"C2": 4000}})  # ~1 C2 message per second
    sim.drop_c2 = {3, 4}
    sim.run(40.0)
    assert sim.graph_ok()
    link = sim.rx.link("r")
    assert link["gaps"] == 1 and link["resend_requests"] >= 1
    assert link["resync_requests"] == 1 and link["resyncs"] == 1  # first contact only
    assert link["resend_escalations"] == 0
    assert sim.agent.c2_resent == 2


def test_resend_range_gone_escalates_to_resync():
    sim = Sim({"budgets": {"C2": 4000}, "c2_ring": 0})
    sim.drop_c2 = {3}
    sim.run(60.0)
    link = sim.rx.link("r")
    assert link["resend_escalations"] == 1 and sim.agent.c2_resend_gone == 1
    assert link["resync_requests"] == 2 and link["resyncs"] == 2
    assert sim.graph_ok()


def test_agent_resend_op_compatible_and_range_checked():
    tr = RecordingTransport()
    clock = VirtualClock()
    core = AgentCore("r", tr, {"c2_ring": 4}, clock)
    for i in range(6):  # six C2 messages: ring keeps seqs 2..5
        core.on_records(*synthetic_records(5 * (i + 1), seed=1)[:2])
        core._t_c2 = -1e9
        core.tick(clock.advance(1.0))

    def req(rid, args):
        p = {"id": rid, "op": "resend", "args": args}
        tr.inject("adt4/r/ctl/req", env_bytes("base", "ctl/req", 0, "S", p))
        core.tick(clock.advance(0.01))
        return next(
            m["p"] for _, m, _ in tr.messages("adt4/r/ctl/ack") if m["p"]["id"] == rid
        )

    before = [d for k, d, c in tr.sent if c == "C2"]
    ack = req("a", {"stream": "dsg", "from_seq": 3, "to_seq": 4})
    assert ack["ok"] and ack["result"]["dsg"] == {
        "from_seq": 3,
        "to_seq": 4,
        "resent": 2,
    }
    again = [d for k, d, c in tr.sent if c == "C2"][len(before) :]
    assert again[:2] == [before[3], before[4]]  # byte-identical, same seq
    ack = req("b", {"stream": "dsg", "from_seq": 0, "to_seq": 3})
    assert not ack["ok"] and "range gone" in ack["result"]["error"]
    ack = req("c", {"streams": ["grid"]})  # the legacy form still works
    assert ack["ok"] and ack["result"] == {"resend": ["grid"]}


def test_transport_close_recovers():
    sim = Sim({"budgets": {"C2": 8000}}, n=300, bandwidth=100_000)
    sim.run(10.0)
    sim.grow(400, seed=2)
    sim.link.close_transport(5.0)  # everything in flight and for 5 s is lost
    sim.run(5.0)
    assert sim.rx.link("r", sim.clock())["staleness_s"] > 4.0
    sim.run(2.0, track_from=sim.clock() + 1.0)  # C1 back within a second
    assert sim.max_state_age < 1.0
    sim.run(60.0)
    assert sim.graph_ok()
    assert sim.rx.link("r")["resend_requests"] >= 1


def test_tcp_hol_c1_staleness_bounded():
    """One FIFO for every class (TCP send buffer): C1 waits behind mesh tiles,
    but the budgets keep the FIFO short, so its staleness stays bounded."""
    cfg = {"budgets": {"C1": 5000, "C2": 50_000, "C3": 100_000}}
    sim = Sim(cfg, n=400, mesh=40_000, bandwidth=250_000, tcp_hol=True, loss=0.01)
    sim.run(60.0, track_from=2.0)
    assert sim.graph_ok()
    assert sim.max_state_age < 1.0
    up = sim.link.stats()["up"]
    assert up.wire_bytes["C3"] > 500_000  # the mesh really streamed alongside


def test_fb_aimd_converges_to_link_capacity():
    """The emulated sender is blind (``backpressure=False``, like zenoh DROP), the
    C3 budget starts at 4x the link: loss and goodput feedback bring it down,
    additive increase probes back up."""
    cap = 100_000.0
    cfg = {
        "auto": True,
        "budgets": {"C1": 5000, "C2": 5000, "C3": 4 * cap},
        "budget_min": {"C2": 5000, "C3": 5000},
        "budget_max": {"C2": 50_000, "C3": 4 * cap},
    }
    sim = Sim(cfg, n=20, bandwidth=cap, backpressure=False)
    sim.agent.sched.ai_step_frac = 0.05
    filler = iter(range(10**9))

    def keep_backlogged():
        sched = sim.agent.sched
        while len(sched.queues["C3"]) < 40:
            i = next(filler)
            idx = random.Random(i).randbytes(3000)  # incompressible
            tile = {"key": f"f{i}", "hash": "h", "pos": b"", "col": b"", "idx": idx}
            sim.agent._queue(
                "mesh/tile", f"adt4/r/mesh/tile/f{i}", lambda t=tile: t, 3100
            )

    budgets, delivered = [], 0
    for step in range(int(120 / DT)):
        keep_backlogged()
        sim.run(DT)
        if step * DT >= 60.0:
            budgets.append(sim.agent.sched.budgets["C3"])
    delivered = sim.link.stats()["up"].delivered_bytes["C3"]
    mean_budget = sum(budgets) / len(budgets)
    assert sim.agent.sched.decreases >= 3
    assert 0.5 * cap < mean_budget < 1.6 * cap, mean_budget
    assert delivered / 120.0 > 0.6 * cap  # goodput near the link rate
    assert sim.rx.link("r")["state_age_s"] < 1.0  # C1 unaffected


def test_zenoh_version_warning():
    assert zenoh_version_warning("1.8.0") is None
    assert "1.8" in zenoh_version_warning("1.9.1")
