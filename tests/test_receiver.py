from helpers import RecordingTransport, env_bytes, synthetic_records

from fleetlink import codec
from fleetlink.codec import GraphState
from fleetlink.receiver import Receiver
from fleetlink.transport import VirtualClock


def _rx():
    tr = RecordingTransport()
    clock = VirtualClock(100.0)
    events = []
    # mesh LOD requests at each epoch are covered in test_mesh_lod.py
    rx = Receiver(
        tr, lambda r, k, d: events.append((r, k, d)), clock=clock, threaded=False, mesh_lod=None
    )
    rx.start()
    return tr, clock, rx, events


def _reqs(tr, op=None):
    out = [codec.unpack(d)["p"] for k, d, c in tr.sent if k.endswith("/ctl/req")]
    return [r for r in out if op is None or r["op"] == op]


def _delta(records, edges=()):
    return {"kind": "delta", **GraphState().diff(records, set(edges)).encode()}


def _c2(rx, seq, payload, epoch="E1"):
    rx._on_msg("adt4/r/dsg/delta", env_bytes("r", "dsg/delta", seq, epoch, payload))


def _ack(rx, req, ok=True, result=None, seq=0):
    ack = {"id": req["id"], "ok": ok, "result": result or {}}
    rx._on_msg("adt4/r/ctl/ack", env_bytes("r", "ctl/ack", seq, "E1", ack))


def _first_contact(rx, tr, items):
    _c2(rx, 0, _delta(dict(items[:1])))
    assert len(_reqs(tr, "resync")) == 1  # first contact = epoch change
    _ack(rx, _reqs(tr, "resync")[0])


def test_gap_holds_back_and_resend_fills_it():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(6)
    items = list(recs.items())
    _first_contact(rx, tr, items)
    _c2(rx, 1, _delta(dict(items[1:2])))
    assert not _reqs(tr, "resend")  # contiguous
    _c2(rx, 4, _delta(dict(items[4:5])))  # 2 and 3 lost
    (req,) = _reqs(tr, "resend")
    assert req["args"] == {"stream": "dsg", "from_seq": 2, "to_seq": 3}
    assert set(rx.graph("r").records) == {s for s, _ in items[:2]}  # 4 held back
    link = rx.link("r")
    assert link["gaps"] == 1 and link["drops"] == 2 and link["c2_held"] == 1
    _ack(rx, req, result={"dsg": {"resent": 2}}, seq=1)
    _c2(rx, 2, _delta(dict(items[2:3])))
    _c2(rx, 3, _delta(dict(items[3:4])))
    _c2(rx, 3, _delta(dict(items[3:4])))  # duplicate: ignored
    assert set(rx.graph("r").records) == {s for s, _ in items[:5]}
    deltas = [d["seq"] for _, k, d in events if k == "delta"]
    assert deltas == [0, 1, 2, 3, 4]  # applied in order
    assert len(_reqs(tr, "resync")) == 1  # no resync needed
    assert rx.link("r")["c2_held"] == 0 and not rx.pending_requests()


def test_resend_retries_then_escalates_to_resync():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(6)
    items = list(recs.items())
    _first_contact(rx, tr, items)
    clock.advance(10.0)  # past the resync rate limit
    _c2(rx, 3, _delta(dict(items[3:4])))
    for _ in range(3):
        clock.advance(2.1)
        rx.tick()
    resends = _reqs(tr, "resend")
    # 3 attempts (first + 2 re-requests); C0 retries of the same ids may add more
    assert len({r["id"] for r in resends}) == 3
    assert len(_reqs(tr, "resync")) == 2  # the escalation
    link = rx.link("r")
    assert link["resend_escalations"] == 1 and link["c2_held"] == 0
    # what was held is applied (the resync's snapshot repairs the rest)
    assert items[3][0] in rx.graph("r").records


def test_range_gone_escalates_at_once():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(6)
    items = list(recs.items())
    _first_contact(rx, tr, items)
    clock.advance(10.0)
    _c2(rx, 2, _delta(dict(items[2:3])))
    (req,) = _reqs(tr, "resend")
    _ack(rx, req, ok=False, result={"error": "range gone: dsg 1-1"}, seq=1)
    assert len(_reqs(tr, "resync")) == 2
    assert rx.link("r")["resend_escalations"] == 1


def test_escalation_resync_waits_for_rate_limit():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(6)
    items = list(recs.items())
    _first_contact(rx, tr, items)  # resync requested at t=100
    _c2(rx, 2, _delta(dict(items[2:3])))
    (req,) = _reqs(tr, "resend")
    _ack(rx, req, ok=False, result={"error": "range gone"}, seq=1)
    assert len(_reqs(tr, "resync")) == 1  # rate-limited, not forgotten
    clock.advance(5.0)
    rx.tick()
    assert len(_reqs(tr, "resync")) == 2


def test_no_put_under_the_lock_and_loss_per_class():
    """Requests (resend, tiles, resync, ping, fb, retries) leave through the
    outbox after the receiver's lock is released."""
    clock = VirtualClock(100.0)

    class LockCheckingTransport(RecordingTransport):
        def put(self, key, data, cls):
            assert not rx._lock._is_owned(), f"put of {key} under the lock"
            super().put(key, data, cls)

    tr = LockCheckingTransport()
    rx = Receiver(tr, None, clock=clock, threaded=False)
    rx.start()
    recs, _, _ = synthetic_records(6)
    items = list(recs.items())
    _c2(rx, 0, _delta(dict(items[:1])))  # first contact → resync
    _c2(rx, 3, _delta(dict(items[3:4])))  # gap → resend
    rx._on_msg(
        "k", env_bytes("r", "mesh/manifest", 0, "E1", {"tiles": {"a": "1"}})
    )  # → tiles
    for seq in (0, 1, 2, 5, 6):  # 3 and 4 lost
        rx._on_msg("k", env_bytes("r", "state", seq, "E1", {"pose": None}))
    rx.request_keyframe("r", "rgb")
    for _ in range(3):
        clock.advance(1.1)
        rx.tick()  # retries, ping, fb
    ops = {r["op"] for r in _reqs(tr)}
    assert {"resync", "resend", "tiles", "ping", "fb", "video_keyframe"} <= ops
    link = rx.link("r")
    assert link["lost"]["C1"] == 2 and link["lost"]["C2"] == 2
    fb = _reqs(tr, "fb")[1]  # the first one goes before a window has closed
    assert set(fb["args"]) == {"t", "rx", "loss", "rtt"}
    assert fb["args"]["loss"]["C1"] == round(2 / 7, 4)


def test_epoch_change_resets_and_resyncs():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(4)
    rx._on_msg("adt4/r/dsg/delta", env_bytes("r", "dsg/delta", 0, "E1", _delta(recs)))
    tile = {"key": "0_0_0", "hash": "h", "pos": b"", "col": b"", "idx": b""}
    rx._on_msg("adt4/r/mesh/tile/0_0_0", env_bytes("r", "mesh/tile", 0, "E1", tile))
    assert rx.graph("r").records and rx.tiles("r")
    n = len(_reqs(tr, "resync"))
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 0, "E2", {"pose": None}))
    assert len(_reqs(tr, "resync")) == n + 1
    assert rx.link("r")["epoch"] == "E2"
    assert not rx.graph("r").records and not rx.tiles("r")
    # stale messages from the old incarnation are ignored, no flapping
    rx._on_msg("adt4/r/dsg/delta", env_bytes("r", "dsg/delta", 1, "E1", _delta(recs)))
    assert rx.link("r")["epoch"] == "E2" and not rx.graph("r").records
    # snapshot begin/chunk/end rebuilds the mirror
    rx._on_msg(
        "adt4/r/dsg/delta",
        env_bytes("r", "dsg/delta", 0, "E2", {**_delta({}), "kind": "snapshot_begin"}),
    )
    rx._on_msg(
        "adt4/r/dsg/delta",
        env_bytes(
            "r", "dsg/delta", 1, "E2", {**_delta(recs), "kind": "snapshot_chunk"}
        ),
    )
    rx._on_msg(
        "adt4/r/dsg/delta",
        env_bytes("r", "dsg/delta", 2, "E2", {**_delta({}), "kind": "snapshot_end"}),
    )
    assert set(rx.graph("r").records) == set(recs)
    assert rx.link("r")["resyncs"] == 1
    assert events[-1][1] == "snapshot"


def test_requests_retry_until_acked():
    tr, clock, rx, _ = _rx()
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 0, "E1", {}))
    tr.sent.clear()
    rid = rx.set_budget("r", "C3", 20_000)
    for _ in range(3):
        clock.advance(1.1)
        rx.tick()
    budget_reqs = _reqs(tr, "budget")
    assert len(budget_reqs) == 4 and {r["id"] for r in budget_reqs} == {rid}
    ack = {
        "id": rid,
        "ok": True,
        "result": {"budgets": {"C1": 5000, "C2": 50000, "C3": 20000}},
    }
    rx._on_msg("adt4/r/ctl/ack", env_bytes("r", "ctl/ack", 0, "E1", ack))
    clock.advance(5.0)
    rx.tick()
    assert len(_reqs(tr, "budget")) == 4
    assert rx.link("r")["budgets"]["C3"] == 20000
    assert [p["op"] for p in rx.pending_requests()] == [
        "resync"
    ]  # first-contact resync, never acked


def test_manifest_requests_missing_tiles_and_removes_stale():
    tr, clock, rx, events = _rx()
    t1 = {"key": "a", "hash": "1", "pos": b"", "col": b"", "idx": b""}
    t2 = {"key": "b", "hash": "2", "pos": b"", "col": b"", "idx": b""}
    rx._on_msg("k", env_bytes("r", "mesh/tile", 0, "E1", t1))
    rx._on_msg("k", env_bytes("r", "mesh/tile", 1, "E1", t2))
    rx._on_msg(
        "k", env_bytes("r", "mesh/manifest", 0, "E1", {"tiles": {"a": "1", "c": "3"}})
    )
    assert set(rx.tiles("r")) == {"a"}
    assert _reqs(tr, "tiles")[-1]["args"]["keys"] == ["c"]
    assert ("r", "tiles", {"changed": [], "removed": ["b"]}) in events


def test_byte_totals_per_direction_and_session():
    tr, clock, rx, events = _rx()
    recs, _, _ = synthetic_records(4)
    items = list(recs.items())
    _first_contact(rx, tr, items)
    link = rx.link("r")
    assert link["rx_total_B"] == sum(link["bytes"].values()) > 0
    tx_sent = sum(len(d) for k, d, c in tr.sent if k.endswith("/ctl/req"))
    assert link["tx_total_B"] == tx_sent > 0  # the resync request
    assert link["sessions"] == 1 and link["connected_since"] == 100.0
    assert link["rx_session_B"] == link["rx_total_B"]
    assert link["since_s"] == 0.0

    rx.tick()  # starts the rate window
    clock.advance(1.5)
    rx.tick()  # pings / fb go out: counted base -> robot
    assert rx.link("r")["tx_total_B"] == sum(
        len(d) for k, d, c in tr.sent if k.endswith("/ctl/req")
    )
    assert rx.link("r")["rx_rate_Bps"] > 0 and rx.link("r")["tx_rate_Bps"] > 0

    clock.advance(10.0)  # silence > connected_timeout, then the robot is back
    before = rx.link("r")
    _c2(rx, 1, _delta(dict(items[1:2])))
    after = rx.link("r")
    assert after["sessions"] == 2 and after["connected_since"] == clock()
    n = after["rx_total_B"] - before["rx_total_B"]
    assert n > 0 and after["rx_session_B"] == n
    assert after["tx_session_B"] == after["tx_total_B"] - before["tx_total_B"]
    assert after["since_s"] == clock() - 100.0

    # agent restart (new epoch) starts a new session too, counting its first message
    _c2(rx, 0, _delta(dict(items[2:3])), epoch="E2")
    restarted = rx.link("r")
    assert restarted["sessions"] == 3
    assert restarted["rx_session_B"] == restarted["rx_total_B"] - after["rx_total_B"]


def test_frame_align_passes_through_state_and_link():
    tr, clock, rx, events = _rx()
    fa = {
        "parent": "map",
        "child": "r/odom",
        "T": [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "stamp": 99.0,
        "ok": True,
        "reason": "",
        "age_s": 0.4,
    }
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 0, "E1", {"pose": None}))
    assert rx.link("r")["frame_align"] is None  # old agent: no field
    rx._on_msg(
        "adt4/r/state",
        env_bytes("r", "state", 1, "E1", {"pose": None, "frame_align": fa}),
    )
    assert rx.state("r")["frame_align"] == fa
    assert rx.link("r")["frame_align"] == fa
    assert events[-1][1] == "state" and events[-1][2]["frame_align"]["ok"]


def test_reloc_passes_through_state_and_link():
    tr, clock, rx, events = _rx()
    rl = {"prior_lcs": 3, "best_assoc": 9, "last_lc_age_s": 2.0, "lcs": [], "jumps": 1}
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 0, "E1", {"pose": None}))
    assert rx.link("r")["reloc"] is None  # old agent: no field
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 1, "E1", {"pose": None, "reloc": rl}))
    assert rx.state("r")["reloc"] == rl
    assert rx.link("r")["reloc"] == rl


def test_link_reports_the_robots_own_per_class_rates():
    tr, clock, rx, events = _rx()
    stats = {"rate": {"C3": 90_000.0}, "sent": {"C3": 1_234_567}, "queue": {"C3": {"n": 4, "bytes": 30_000}}}
    rx._on_msg("adt4/r/state", env_bytes("r", "state", 0, "E1", {"pose": None, "stats": stats}))
    link = rx.link("r")
    assert link["robot_rates"] == {"C3": 90_000.0}
    assert link["robot_sent"] == {"C3": 1_234_567}
    assert link["robot_queue"]["C3"]["bytes"] == 30_000
