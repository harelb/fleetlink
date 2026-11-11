"""Large relayed messages (e.g. a manipulation approval request, a few hundred kB
of images) cross a slow radio in parts: zenoh drops a DROP message that does not
fit its 128 kB per-priority queue whole, every time it is resent."""

import random

import pytest

from fleetlink.relay import RelayConfig, RelayCore
from fleetlink.transport import CLASSES, LinkEmulator, VirtualClock

CFG = {
    "version": 1,
    "topics": [
        {"topic": "/{robot}/big", "type": "pkg/msg/Big", "dir": "up",
         "mode": "latest_reliable", "cls": "C3", "ttl_s": 120},
        {"topic": "/{robot}/events", "type": "pkg/msg/Event", "dir": "up",
         "mode": "reliable", "cls": "C3", "ttl_s": 120},
        {"topic": "/{robot}/answer", "type": "pkg/msg/Answer", "dir": "down",
         "mode": "reliable", "cls": "C0", "ttl_s": 30},
    ],
}  # fmt: skip


class Radio:
    """Robot and base relays over a 30 kB/s link that drops what does not fit
    the per-priority queue (as zenoh does; the sender cannot see the queue)."""

    def __init__(self, loss=0.0, bandwidth=30_000.0, seed=3, **kw):
        self.clk = VirtualClock(1000.0)
        self.link = LinkEmulator(
            clock=self.clk, bandwidth=bandwidth, latency=0.05, loss=loss,
            queue_bytes=128 * 1024, backpressure=False, seed=seed,
            reliable=dict.fromkeys(CLASSES, False),
        )  # fmt: skip
        cfg = RelayConfig(CFG)
        self.got = []
        self.events = []
        self.robot = RelayCore(
            "euclid", "robot", self.link.robot, cfg, platform="spot", clock=self.clk,
            on_event=lambda k, i: self.events.append((k, i)),
            type_hash=lambda t: "H:" + t, **kw,
        )  # fmt: skip
        self.base = RelayCore(
            "euclid", "base", self.link.base, cfg, base="willow", clock=self.clk,
            on_message=lambda e, d, m: self.got.append((e.name, bytes(d))) or 1,
            type_hash=lambda t: "H:" + t, **kw,
        )  # fmt: skip

    def run(self, seconds, dt=0.05):
        for _ in range(int(round(seconds / dt))):
            self.clk.advance(dt)
            self.link.advance()
            self.robot.tick()
            self.base.tick()

    def data(self, name):
        return [d for n, d in self.got if n == name]

    def acks(self, name):
        return [i for k, i in self.events if k == "ack" and i["topic"] == name]


def payload(n, seed=0):
    """Incompressible bytes (the relay's codec zstd-compresses: images do not shrink)."""
    return random.Random(seed).randbytes(n)


def test_a_large_message_crosses_a_slow_radio_whole_and_once():
    r = Radio()
    r.run(2)  # heartbeats: link up
    msg = payload(400_000)
    r.robot.publish("/euclid/big", msg)
    r.run(40)
    assert r.data("/euclid/big") == [msg]
    acks = r.acks("/euclid/big")
    assert len(acks) == 1 and acks[0]["ok"] and acks[0]["subs"] == 1
    st = r.robot.out["/euclid/big"]
    assert not st.pending and st.c["parts_sent"] >= 25  # 400 kB / 16 kB parts


def test_without_parts_the_same_message_never_arrives():
    """The failure the parts fix: one 400 kB message cannot fit the queue."""
    r = Radio(chunk_bytes=10_000_000)  # effectively unsplit
    r.run(2)
    r.robot.publish("/euclid/big", payload(400_000))
    r.run(40)
    assert r.data("/euclid/big") == []


@pytest.mark.parametrize("seed", [7, 11, 13])
def test_parts_survive_loss(seed):
    r = Radio(loss=0.2, seed=seed)
    r.run(2)
    msg = payload(300_000, seed=5)
    r.robot.publish("/euclid/big", msg)
    r.run(110)  # within the 120 s ttl
    assert r.data("/euclid/big") == [msg]
    assert r.robot.out["/euclid/big"].c["parts_sent"] > 19  # some parts were resent


def test_a_newer_large_message_supersedes_the_one_in_flight():
    r = Radio()
    r.run(2)
    old, new = payload(300_000, 1), payload(250_000, 2)
    r.robot.publish("/euclid/big", old)
    r.run(1.0)  # part of the old one is out
    r.robot.publish("/euclid/big", new)
    r.run(60)
    assert r.data("/euclid/big")[-1:] == [new]
    assert old not in r.data("/euclid/big")
    assert not r.base.inn["/euclid/big"].partial  # nothing left half-built


def test_reliable_keeps_order_across_large_and_small_messages():
    r = Radio(bandwidth=None)
    r.run(2)
    big, small = payload(100_000, 3), b"small"
    r.robot.publish("/euclid/events", big)
    r.robot.publish("/euclid/events", small)
    r.run(10)
    assert r.data("/euclid/events") == [big, small]


def test_small_messages_are_not_split():
    r = Radio(bandwidth=None)
    r.run(2)
    r.robot.publish("/euclid/big", b"x" * 1000)
    r.run(2)
    assert r.data("/euclid/big") == [b"x" * 1000]
    assert r.robot.out["/euclid/big"].c["parts_sent"] == 0


@pytest.mark.parametrize("size", [16 * 1024, 16 * 1024 + 1, 64 * 1024])
def test_part_boundaries(size):
    r = Radio(bandwidth=None)
    r.run(2)
    msg = payload(size, seed=size % 7)
    r.robot.publish("/euclid/big", msg)
    r.run(5)
    assert r.data("/euclid/big") == [msg]


def test_a_lost_final_ack_is_recovered():
    """Every part acked but the whole-message ack lost: the sender resends a part
    and the receiver, which already delivered it, acks again (no second delivery)."""
    r = Radio(bandwidth=None)
    r.run(2)
    real = r.base._ack
    dropped = []

    def lossy_ack(name, pid, ok=True, err=None, subs=None, now=0.0):
        if not dropped:
            dropped.append(pid)
            return  # the first whole-message ack never leaves
        real(name, pid, ok, err, subs, now)

    r.base._ack = lossy_ack
    msg = payload(100_000, 4)
    r.robot.publish("/euclid/big", msg)
    r.run(20)
    assert dropped and r.data("/euclid/big") == [msg]
    assert not r.robot.out["/euclid/big"].pending


def test_a_peer_restart_mid_transfer_starts_the_parts_over():
    r = Radio()
    r.run(2)
    msg = payload(200_000, 6)
    r.robot.publish("/euclid/big", msg)
    r.run(2)  # partly sent
    r.base = RelayCore(
        "euclid", "base", r.link.base, r.base.config, base="willow", clock=r.clk,
        on_message=lambda e, d, m: r.got.append((e.name, bytes(d))) or 1,
        type_hash=lambda t: "H:" + t,
    )  # fmt: skip
    r.run(60)
    # latest_reliable without replay_on_resync: the pending message is dropped on
    # the peer restart (no stale replays), so nothing half-built is delivered
    assert all(d == msg for d in r.data("/euclid/big"))
