"""``frame_align`` (global frame <- scene-graph frame) in the C1 state."""

from types import SimpleNamespace as NS

from helpers import RecordingTransport

from fleetlink.agent import (
    AgentCore,
    frame_align_failed,
    lookup_frame_align,
)
from fleetlink.transport import VirtualClock


def _tf(x, y, z, qw, qx, qy, qz, sec=100, nanosec=500_000_000):
    return NS(
        header=NS(stamp=NS(sec=sec, nanosec=nanosec)),
        transform=NS(
            translation=NS(x=x, y=y, z=z), rotation=NS(w=qw, x=qx, y=qy, z=qz)
        ),
    )


class FakeBuffer:
    """tf2 Buffer stand-in: ``{(parent, child): TransformStamped}``."""

    def __init__(self, tfs=None):
        self.tfs = dict(tfs or {})
        self.calls = []

    def lookup_transform(self, parent, child, _time):
        self.calls.append((parent, child))
        if (parent, child) not in self.tfs:
            raise LookupError(
                f'"{parent}" passed to lookupTransform argument target_frame does not exist'
            )
        return self.tfs[(parent, child)]


def test_lookup_ok_relocalized():
    buf = FakeBuffer({("map", "hamilton/odom"): _tf(0.53, 0.07, 0.7, 1, 0, 0, 0)})
    fa = lookup_frame_align(buf, "map", "hamilton/odom")
    assert fa["ok"] is True and fa["reason"] == ""
    assert (fa["parent"], fa["child"]) == ("map", "hamilton/odom")
    assert fa["T"] == [0.53, 0.07, 0.7, 1.0, 0.0, 0.0, 0.0]
    assert fa["stamp"] == 100.5
    assert buf.calls == [("map", "hamilton/odom")]  # parent <- child


def test_lookup_not_relocalized():
    fa = lookup_frame_align(FakeBuffer(), "map", "hamilton/odom")
    assert fa["ok"] is False and fa["T"] is None
    assert fa["reason"].startswith("not relocalized: LookupError")
    assert len(fa["reason"]) <= 200


def test_lookup_identity_no_buffer_and_unknown_frame():
    fa = lookup_frame_align(None, "map", "map")
    assert fa["ok"] and fa["T"] == [0, 0, 0, 1, 0, 0, 0] and fa["stamp"] == 0
    assert lookup_frame_align(None, "map", "r/odom")["reason"] == "tf2_ros unavailable"
    assert not lookup_frame_align(FakeBuffer(), "map", "")["ok"]


def _core():
    tr = RecordingTransport()
    clock = VirtualClock(1000.0)
    return tr, clock, AgentCore("r", tr, {}, clock)


def _states(tr):
    return [m["p"] for _, m, _ in tr.messages("adt4/r/state")]


def test_state_omits_frame_align_until_looked_up():
    tr, clock, core = _core()
    core.tick(clock.advance(0.5))
    assert "frame_align" not in _states(tr)[-1]  # backward compatible payload


def test_state_carries_frame_align_with_age():
    tr, clock, core = _core()
    buf = FakeBuffer({("map", "r/odom"): _tf(1, 2, 0, 0.7071068, 0, 0, 0.7071068, sec=998)})
    core.on_frame_align(lookup_frame_align(buf, "map", "r/odom"))
    core.tick(clock.advance(0.5))
    fa = _states(tr)[-1]["frame_align"]
    assert fa["ok"] and fa["child"] == "r/odom" and fa["T"][:3] == [1.0, 2.0, 0.0]
    assert abs(fa["age_s"] - (clock() - 998.5)) < 1e-6
    # the lookup starts failing (relocalization lost): ok false, reason, no T
    core.on_frame_align(frame_align_failed("map", "r/odom", "not relocalized: x"))
    core.tick(clock.advance(0.5))
    fa = _states(tr)[-1]["frame_align"]
    assert fa["ok"] is False and fa["T"] is None and fa["age_s"] is None
    assert fa["reason"] == "not relocalized: x"
