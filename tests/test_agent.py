from helpers import RecordingTransport, env_bytes, synthetic_mesh, synthetic_records

from fleetlink.agent import AgentCore
from fleetlink.transport import VirtualClock


def _agent(**cfg):
    tr = RecordingTransport()
    clock = VirtualClock()
    core = AgentCore("r", tr, cfg, clock)
    recs, edges, ls = synthetic_records(30)
    core.on_records(recs, edges, ls, synthetic_mesh(100))
    core.on_pose(
        {
            "x": 0,
            "y": 0,
            "z": 0,
            "qw": 1,
            "qx": 0,
            "qy": 0,
            "qz": 0,
            "frame": "map",
            "t": 0,
        }
    )
    return tr, clock, core


def _req(tr, rid, op, args=None, seq=0):
    tr.inject(
        "adt4/r/ctl/req",
        env_bytes(
            "base", "ctl/req", seq, "S", {"id": rid, "op": op, "args": args or {}}
        ),
    )


def _acks(tr):
    return [m["p"] for k, m, c in tr.messages("adt4/r/ctl/ack")]


def _run(clock, core, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        core.tick(clock.advance(dt))


def test_streams_and_envelopes():
    tr, clock, core = _agent()
    _run(clock, core, 3.0)
    kinds = {(k.split("/", 2)[2], c) for k, _, c in tr.sent}
    assert ("state", "C1") in kinds and ("dsg/delta", "C2") in kinds
    assert any(k.startswith("mesh/tile/") and c == "C3" for k, c in kinds)
    msgs = tr.messages("adt4/r/state")
    assert [m["seq"] for _, m, _ in msgs] == list(range(len(msgs)))
    assert all(m["epoch"] == core.epoch and m["robot"] == "r" for _, m, _ in msgs)
    assert 11 <= len(msgs) <= 13  # 4 Hz
    st = msgs[-1][1]["p"]
    assert st["pose"]["frame"] == "map" and set(st["stats"]["budgets"]) == {
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
    }


def test_resync_snapshot_and_ack():
    tr, clock, core = _agent()
    _run(clock, core, 3.0)
    tr.sent.clear()
    _req(tr, "q1", "resync")
    _run(clock, core, 3.0)
    assert _acks(tr) == [{"id": "q1", "ok": True, "result": {"epoch": core.epoch}}]
    kinds = [m["p"]["kind"] for _, m, _ in tr.messages("adt4/r/dsg/delta")]
    assert kinds[0] == "snapshot_begin" and kinds[-1] == "snapshot_end"
    assert set(kinds[1:-1]) == {"snapshot_chunk"}
    n = sum(len(m["p"]["up"]["ids"]) for _, m, _ in tr.messages("adt4/r/dsg/delta"))
    assert n == 30
    # manifest is resent after the snapshot
    assert tr.messages("adt4/r/mesh/manifest")
    # an idempotent retry of the same id is re-acked but not re-executed
    tr.sent.clear()
    _req(tr, "q1", "resync", seq=1)
    _run(clock, core, 1.0)
    assert len(_acks(tr)) == 1
    assert not tr.messages("adt4/r/dsg/delta") or all(
        m["p"]["kind"] == "delta" for _, m, _ in tr.messages("adt4/r/dsg/delta")
    )


def test_budget_auto_ping_tiles_and_bad_ops():
    tr, clock, core = _agent()
    _run(clock, core, 1.0)
    _req(tr, "b", "budget", {"C3": 12345})
    _req(tr, "a", "auto", {"enabled": True})
    _req(tr, "p", "ping", {"t": 0.5, "rx": {"C2": 0, "C3": 0}})
    _req(tr, "x", "frobnicate")
    _run(clock, core, 0.1)
    acks = {a["id"]: a for a in _acks(tr)}
    assert acks["b"]["ok"] and acks["b"]["result"]["budgets"]["C3"] == 12345
    assert core.sched.budgets["C3"] == 12345
    assert acks["a"]["result"] == {"auto": True} and core.sched.auto
    assert acks["p"]["result"]["t"] == 0.5
    assert acks["x"]["ok"] is False
    # tiles op: forget those keys so they are re-sent
    _run(clock, core, 30.0)
    key = next(iter(core.sent_manifest))
    tr.sent.clear()
    _req(tr, "t", "tiles", {"keys": [key]})
    _run(clock, core, 2.0)
    assert any(k == f"adt4/r/mesh/tile/{key}" for k, _, _ in tr.sent)


def test_c2_respects_budget_and_sends_closest_first():
    tr, clock, core = _agent(budgets={"C2": 600})
    recs = {s: r for s, r in core._target[0].items()}
    _run(clock, core, 1.0)
    first = tr.messages("adt4/r/dsg/delta")[0][1]["p"]
    ids = first["up"]["ids"]
    assert 0 < len(ids) < len(recs)
    d = sorted(sum(v * v for v in r.pos) for r in recs.values())
    sent_d = sorted(sum(v * v for v in recs[s].pos) for s in ids)
    assert sent_d == d[: len(ids)]
    _run(clock, core, 60.0)
    total = sum(len(d) for k, d, c in tr.sent if c == "C2")
    assert total <= 600 * 61 * 1.1 + 600
    assert core.sent.diff(*core._target).empty()


def test_guards_to_dict_lease_owner():
    from types import SimpleNamespace

    from fleetlink.agent import guards_to_dict

    msg = SimpleNamespace(
        robot_name="hamilton", battery_percent=50.0, battery_known=True,
        estop_pressed=False, estop_known=True, powered_on=True, power_known=True,
        lease_owned=False, lease_known=True, faults=[], notes="Idle",
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
    )  # fmt: skip
    assert guards_to_dict(msg)["lease_owner"] == ""  # robot built before the field
    msg.lease_owner = "bosdyn.android.spotapp tablet"
    assert guards_to_dict(msg)["lease_owner"] == "bosdyn.android.spotapp tablet"
