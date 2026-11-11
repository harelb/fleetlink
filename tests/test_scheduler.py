import pytest
from helpers import RecordingTransport

from fleetlink.scheduler import Item, Scheduler
from fleetlink.transport import VirtualClock


def _item(cls, n, key=None, tag=None):
    data = bytes(n)
    return Item(cls, key or f"k/{cls}", lambda: data, n, tag=tag)


def test_strict_priority_order():
    tr = RecordingTransport()
    clock = VirtualClock()
    s = Scheduler(
        tr, {"C1": None, "C2": None, "C3": None}, clock, total_bps=1000, burst_s=1.0
    )
    for cls in ("C3", "C2", "C1", "C0", "C3", "C2", "C1", "C0"):
        s.enqueue(_item(cls, 300))
    s.drain(0.0)  # 1000 B of total tokens, debt allowed by one message
    assert [c for _, _, c in tr.sent] == ["C0", "C0", "C1", "C1"]
    tr.sent.clear()
    s.drain(2.0)
    assert [c for _, _, c in tr.sent] == ["C2", "C2", "C3", "C3"]


def test_latest_value_tag_replaces():
    tr = RecordingTransport()
    s = Scheduler(tr, {}, VirtualClock())
    s.enqueue(Item("C1", "state", lambda: b"old", 3, tag="state"))
    s.enqueue(Item("C1", "state", lambda: b"new", 3, tag="state"))
    s.drain(0.0)
    assert [d for _, d, _ in tr.sent] == [b"new"]


def _run(s, clock, seconds, hz=20.0, change=None):
    sent = {"C2": 0, "C3": 0}
    steps = int(seconds * hz)
    for i in range(steps):
        t = clock.advance(1.0 / hz)
        for cls in ("C2", "C3"):  # keep both classes permanently backlogged
            while len(s.queues[cls]) < 20:
                s.enqueue(_item(cls, 1200 + 300 * (len(s.queues[cls]) % 7)))
        if change and i == steps // 2:
            change()
        out = s.drain(t)
        for c in sent:
            sent[c] += out[c]
    return sent


def test_budget_adherence_30s():
    tr = RecordingTransport()
    clock = VirtualClock()
    s = Scheduler(tr, {"C1": 5000, "C2": 50_000, "C3": 100_000}, clock)
    s.drain(0.0)
    sent = _run(s, clock, 30.0)
    assert sent["C2"] / 30.0 == pytest.approx(50_000, rel=0.10)
    assert sent["C3"] / 30.0 == pytest.approx(100_000, rel=0.10)


def test_live_budget_change():
    tr = RecordingTransport()
    clock = VirtualClock()
    s = Scheduler(tr, {"C2": 50_000, "C3": 100_000}, clock)
    s.drain(0.0)
    _run(s, clock, 10.0)  # warm up at 100 kB/s
    s.set_budget("C3", 20_000)
    sent = _run(s, clock, 30.0)
    assert sent["C3"] / 30.0 == pytest.approx(20_000, rel=0.10)
    assert sent["C2"] / 30.0 == pytest.approx(50_000, rel=0.10)
    with pytest.raises(ValueError):
        s.set_budget("C9", 1)


def test_auto_aimd_stays_in_bounds():
    tr = RecordingTransport()
    clock = VirtualClock()
    s = Scheduler(
        tr,
        {"C2": 50_000, "C3": 100_000},
        clock,
        budget_min={"C2": 10_000, "C3": 10_000},
        budget_max={"C2": 80_000, "C3": 150_000},
        auto=True,
    )
    s.drain(0.0)
    _run(s, clock, 5.0)
    # receiver sees far less than we put → multiplicative decrease down to the floor
    for i in range(30):
        clock.advance(2.0)
        s.feedback({"C2": 0.0, "C3": 0.0})
    assert s.budgets["C2"] == 10_000 and s.budgets["C3"] == 10_000
    # receiver keeps up and we are backlogged → additive increase up to the ceiling
    _run(s, clock, 3.0)
    for i in range(40):
        clock.advance(1.0)
        s.feedback({"C2": 1e9, "C3": 1e9}, rtt=0.3, base_rtt=0.3)
    assert s.budgets["C2"] == 80_000 and s.budgets["C3"] == 150_000
    # RTT inflation alone also backs off
    clock.advance(5.0)
    s.feedback({"C2": 1e9, "C3": 1e9}, rtt=3.0, base_rtt=0.3)
    assert s.budgets["C3"] < 150_000
    s.auto = False
    before = dict(s.budgets)
    s.feedback({"C2": 0.0, "C3": 0.0})
    assert s.budgets == before
