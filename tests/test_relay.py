"""RelayCore (generic ROS relay, pure half) over the deterministic LinkEmulator."""

import os

import pytest

from fleetlink import codec
from fleetlink.relay import RelayConfig, RelayConfigError, RelayCore
from fleetlink.transport import CLASSES, LinkEmulator, VirtualClock

RELAY_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "relay.yaml")

CFG = {
    "version": 1,
    "topics": [
        {"topic": "/{robot}/plan", "type": "pkg/msg/Plan", "dir": "down",
         "mode": "latest_reliable", "cls": "C0", "ttl_s": 30},
        {"topic": "/{robot}/cmd", "type": "std_msgs/msg/Bool", "dir": "down",
         "mode": "reliable", "cls": "C0", "ttl_s": 10},
        {"topic": "/{robot}/result", "type": "pkg/msg/Result", "dir": "up",
         "mode": "reliable", "cls": "C0", "replay_on_resync": True},
        {"topic": "/{robot}/guards", "type": "pkg/msg/Guards", "dir": "up",
         "mode": "latest", "cls": "C1", "rate_hz": 2.0, "ttl_s": 10},
        {"topic": "/{robot}/static", "type": "pkg/msg/Static", "dir": "up",
         "mode": "latest_reliable", "cls": "C0", "replay_on_resync": True},
    ],
    "services": [
        {"service": "/{robot}/hold", "base_name": "/{base}/hold", "type": "pkg/srv/Hold",
         "dir": "up", "timeout_s": 5},
        {"service": "/{robot}/save", "type": "pkg/srv/Save", "dir": "down", "timeout_s": 3},
    ],
}  # fmt: skip


class Pair:
    """A robot and a base RelayCore joined by an emulated link (virtual clock)."""

    def __init__(self, cfg=None, loss=0.0, latency=0.05, robot_skew=0.0, seed=1,
                 robot_cfg=None, robot_type_hash=None, **kw):  # fmt: skip
        self.clk = VirtualClock(1000.0)
        self.link = LinkEmulator(
            clock=self.clk,
            bandwidth=None,
            latency=latency,
            loss=loss,
            seed=seed,
            reliable=dict.fromkeys(CLASSES, False),  # every loss reaches the app
        )
        self.cfg = RelayConfig(cfg or CFG)
        self.skew = robot_skew
        self.got = {"robot": [], "base": []}
        self.events = []
        self.kw = kw
        self.robot_cfg = RelayConfig(robot_cfg) if robot_cfg else self.cfg
        self.robot_type_hash = robot_type_hash
        self.robot = self.new_robot()
        self.base = self.new_base()

    def new_robot(self, **extra):
        return RelayCore(
            "euclid", "robot", self.link.robot, self.robot_cfg, platform="spot",
            clock=lambda: self.clk() + self.skew,
            on_message=lambda e, d, m: self.got["robot"].append((e.name, bytes(d))) or 1,
            type_hash=self.robot_type_hash or (lambda t: "H:" + t),
            **{**self.kw, **extra},
        )  # fmt: skip

    def new_base(self, **extra):
        return RelayCore(
            "euclid", "base", self.link.base, self.cfg, base="willow", clock=self.clk,
            on_message=lambda e, d, m: self.got["base"].append((e.name, bytes(d))),
            on_event=lambda k, i: self.events.append((k, i)),
            type_hash=lambda t: "H:" + t,
            **{**self.kw, **extra},
        )  # fmt: skip

    def run(self, seconds, dt=0.05):
        for _ in range(int(round(seconds / dt))):
            self.clk.advance(dt)
            self.link.advance()
            self.robot.tick()
            self.base.tick()

    def data(self, side, name):
        return [d for n, d in self.got[side] if n == name]


# ----------------------------------------------------------------- config


def test_real_relay_yaml_loads_and_filters_platforms():
    cfg = RelayConfig.load(RELAY_YAML)
    spot = cfg.resolve("euclid", "spot", base="willow")
    husky = cfg.resolve("gauss", "phoenix", base="willow")
    plan = "/euclid/omniplanner_node/compiled_plan_out"
    assert spot.topics[plan].type == "robot_executor_msgs/msg/ActionSequenceMsg"
    assert (
        husky.topics["/gauss/omniplanner_node/compiled_plan_out"].type
        == "nav_msgs/msg/Path"
    )
    assert "/euclid/spot_executor_node/pause" in spot.topics
    assert "/gauss/spot_executor_node/pause" not in husky.topics
    assert "/global/ros_system_monitor/table_in" in husky.topics  # no platform filter
    assert (
        "/euclid/omniplanner_node/llm_response" not in spot.topics
    )  # dead: not relayed
    hold = spot.services["/euclid/update_holding_state"]
    assert hold.base_name == "/willow/update_holding_state" and hold.dir == "up"
    assert "/gauss/update_holding_state" not in husky.services
    # Stop & save: hydra then roman, both robots; no bare /<robot>/shutdown
    for robot, res in (("euclid", spot), ("gauss", husky)):
        for srv in ("hydra/shutdown", "roman/shutdown"):
            s = res.services[f"/{robot}/{srv}"]
            assert s.dir == "down" and s.type == "std_srvs/srv/Empty"
        assert f"/{robot}/shutdown" not in res.services
    # roman pickles its map before answering: the relay must wait for it
    assert spot.services["/euclid/roman/shutdown"].timeout_s >= 120
    take = spot.services["/euclid/spot_executor_node/take_lease"]
    assert take.dir == "down" and take.type == "std_srvs/srv/Trigger"
    assert "/gauss/spot_executor_node/take_lease" not in husky.services
    assert "/euclid/spot_executor_node/runtime_guards" in spot.topics
    assert "/_tf/map/euclid/base_link" in spot.topics and "/_tf_static" in spot.topics
    assert len(cfg.hash) == 12


def test_real_relay_yaml_phoenix_rows():
    cfg = RelayConfig.load(RELAY_YAML)
    spot = cfg.resolve("euclid", "spot", base="willow")
    for platform in ("phoenix", "husky"):  # husky: the topology alias
        phx = cfg.resolve("gauss", platform, base="willow")
        # Stop / Resume: toggle_manual_mode, a C0 down service like take_lease
        mode = phx.services["/gauss/toggle_manual_mode"]
        assert mode.dir == "down" and mode.type == "std_srvs/srv/SetBool"
        assert mode.cls == "C0" and mode.ttl_s <= 10
        # teleop into twist_mux: same strict 0.5 s ttl as the Spot cmd_vel row
        tw = phx.topics["/gauss/twist_marker_server/cmd_vel"]
        spot_tw = spot.topics["/euclid/spot_executor_node/cmd_vel"]
        assert tw.type == "geometry_msgs/msg/TwistStamped" and tw.dir == "down"
        assert (tw.mode, tw.cls, tw.ttl_s, tw.ttl_strict, tw.replay_on_resync) == (
            spot_tw.mode, spot_tw.cls, 0.5, True, False
        )
        mm = phx.topics["/gauss/manual_mode"]
        assert mm.dir == "up" and mm.type == "std_msgs/msg/Bool" and mm.mode == "latest"
        assert "/gauss/spot_executor_node/cmd_vel" not in phx.topics
        assert "/gauss/spot_executor_node/runtime_guards" not in phx.topics
        assert "/gauss/spot_executor_node/stand" not in phx.services
        assert "/_tf/map/gauss/base_link" in phx.topics
    # and none of it on a Spot
    assert "/euclid/toggle_manual_mode" not in spot.services
    assert "/euclid/twist_marker_server/cmd_vel" not in spot.topics
    assert "/euclid/manual_mode" not in spot.topics


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({"topics": [{"topic": "/{robot}/x", "type": "a/msg/B", "dir": "up", "mode": "latest"},
                     {"topic": "/{robot}/x", "type": "a/msg/B", "dir": "down", "mode": "latest"}]},
         "both directions"),
        ({"topics": [{"topic": "/x", "type": "a/msg/B", "dir": "up", "mode": "eventually"}]},
         "unknown mode"),
        ({"topics": [{"topic": "/x", "type": "a/msg/B", "dir": "sideways", "mode": "latest"}]},
         "dir must be"),
        ({"topics": [{"topic": "/x", "type": "a/msg/B", "dir": "up", "mode": "latest", "cls": "C9"}]},
         "unknown class"),
        ({"topics": [{"topic": "/x", "type": "a/msg/B", "dir": "up", "mode": "latest", "ttl": 3}]},
         "unknown keys"),
        ({"services": [{"service": "/{robot}/s", "type": "a/srv/S", "dir": "up"},
                       {"service": "/{robot}/s", "type": "a/srv/S", "dir": "down"}]},
         "both directions"),
        ({"topics": [{"topic": "/x", "type": "a/msg/B", "dir": "up", "mode": "latest", "platform": ["spot"]},
                     {"topic": "/x", "type": "a/msg/C", "dir": "up", "mode": "latest"}]},
         "twice"),
    ],
)  # fmt: skip
def test_config_rejects(bad, msg):
    with pytest.raises(RelayConfigError, match=msg):
        RelayConfig({"version": 1, **bad})


def test_config_rejects_both_directions_after_templating():
    cfg = RelayConfig({"topics": [
        {"topic": "/{robot}/x", "type": "a/msg/B", "dir": "up", "mode": "latest"},
        {"topic": "/euclid/x", "type": "a/msg/B", "dir": "down", "mode": "latest"},
    ]})  # fmt: skip
    cfg.resolve("hamilton")  # fine for another robot
    with pytest.raises(RelayConfigError, match="twice"):
        cfg.resolve("euclid")


# ----------------------------------------------------------------- modes


def test_wire_keys_and_payload():
    pr = Pair()
    seen = []
    # what crosses the link, in each direction
    pr.link.robot.subscribe("adt4/**", lambda k, d: seen.append((k, codec.unpack(d))))
    pr.link.base.subscribe("adt4/**", lambda k, d: seen.append((k, codec.unpack(d))))
    pr.base.publish("/euclid/plan", b"PLAN")
    pr.run(0.5)
    k, env = next((k, e) for k, e in seen if e["stream"] == "ros/down")
    assert k == "adt4/euclid/ros/down/euclid/plan"
    p = env["p"]
    assert p["topic"] == "/euclid/plan" and p["type"] == "pkg/msg/Plan"
    assert p["th"] == "H:pkg/msg/Plan" and p["mode"] == "latest_reliable"
    assert p["data"] == b"PLAN" and p["tseq"] == 0 and p["id"]
    acks = [
        e for k, e in seen if k == "adt4/euclid/ros/ack/up" and e["p"]["op"] == "ack"
    ]
    assert acks and acks[0]["p"]["ok"] and acks[0]["p"]["subs"] == 1
    assert any(e["p"]["op"] == "hb" for k, e in seen if k == "adt4/euclid/ros/ack/up")
    assert pr.data("robot", "/euclid/plan") == [b"PLAN"]
    st = pr.base.status()
    assert st["topics"]["/euclid/plan"]["acked"] == 1
    assert st["topics"]["/euclid/plan"]["subs"] == 1
    assert (
        "ack",
        {
            "robot": "euclid",
            "topic": "/euclid/plan",
            "ok": True,
            "err": None,
            "subs": 1,
        },
    ) in pr.events


def test_reliable_in_order_exactly_once_under_loss():
    pr = Pair(loss=0.3, seed=7)
    pr.run(2.0)
    sent = [f"cmd{i}".encode() for i in range(20)]
    for i, m in enumerate(sent):
        pr.base.publish("/euclid/cmd", m)
        pr.run(0.1)
    pr.run(8.0)
    assert pr.data("robot", "/euclid/cmd") == sent
    st = pr.base.status()["topics"]["/euclid/cmd"]
    assert st["pending"] == 0 and st["resent"] > 0 and st["gave_up"] == 0
    assert pr.robot.status()["loss"] > 0.05  # measured from seq gaps


def test_reliable_up_with_loss_and_queue_cap():
    pr = Pair(loss=0.2, seed=3, queue_cap=5)
    pr.run(1.0)
    pr.link.close_transport(3.0)
    for i in range(8):  # cap 5: the three oldest are dropped
        pr.robot.publish("/euclid/result", f"r{i}".encode())
    pr.run(10.0)
    assert pr.data("base", "/euclid/result") == [f"r{i}".encode() for i in range(3, 8)]
    assert pr.robot.status()["topics"]["/euclid/result"]["overflow"] == 3


def test_reliable_hole_given_up_does_not_block_later_messages():
    """A message the sender gave up on must not hold back the ones after it,
    even when nothing else follows (the heartbeat carries ``lo``)."""
    pr = Pair(max_tries=1)
    pr.run(1.0)
    pr.link.close_transport(0.3)
    pr.base.publish("/euclid/cmd", b"lost")
    pr.run(0.5)
    pr.base.publish("/euclid/cmd", b"after")
    pr.run(0.5)
    assert pr.data("robot", "/euclid/cmd") == []  # held behind the hole
    pr.run(3.0)
    assert pr.data("robot", "/euclid/cmd") == [b"after"]
    assert pr.base.status()["topics"]["/euclid/cmd"]["gave_up"] == 1


def test_latest_reliable_newer_supersedes_pending_retry():
    pr = Pair()
    pr.run(1.0)
    pr.link.close_transport(2.0)
    for v in (b"v1", b"v2", b"v3"):
        pr.base.publish("/euclid/plan", v)
        pr.run(0.2)
    pr.run(4.0)
    assert pr.data("robot", "/euclid/plan") == [b"v3"]
    st = pr.base.status()["topics"]["/euclid/plan"]
    assert st["superseded"] == 2 and st["pending"] == 0


def test_latest_rate_limited_keeps_newest():
    pr = Pair()
    pr.run(1.0)
    for i in range(10):  # 10 values in 1 s at rate_hz 2
        pr.robot.publish("/euclid/guards", f"g{i}".encode())
        pr.run(0.1)
    pr.run(1.0)
    got = pr.data("base", "/euclid/guards")
    assert 2 <= len(got) <= 3
    assert got[-1] == b"g9"


def test_ttl_uses_heartbeat_clock_offset():
    # robot clock 100 s ahead: without the offset every command would look 100 s old
    pr = Pair(robot_skew=100.0)
    pr.run(3.0)
    assert pr.robot.clock_offset == pytest.approx(-100.0, abs=0.06)
    assert pr.base.clock_offset == pytest.approx(100.0, abs=0.06)
    pr.base.publish("/euclid/cmd", b"pause")
    pr.run(1.0)
    assert pr.data("robot", "/euclid/cmd") == [b"pause"]
    assert pr.robot.status()["rtt"] == pytest.approx(0.1, abs=0.06)


@pytest.mark.parametrize("skew", [45.0, -45.0])
def test_ttl_not_checked_before_clock_offset_known(skew):
    """Robot clock 45 s off and a plan (ttl 30) published at link-up, before any
    heartbeat: the receiver cannot correct the envelope ``t`` yet, so it must
    not reject the plan as expired (or as from the future); it is delivered."""
    pr = Pair(robot_skew=skew)
    pr.base.publish("/euclid/plan", b"P1")
    pr.base.publish("/euclid/cmd", b"pause")  # reliable, ttl 10
    pr.run(5.0)
    assert pr.data("robot", "/euclid/plan") == [b"P1"]
    assert pr.data("robot", "/euclid/cmd") == [b"pause"]
    st = pr.robot.status()["topics"]
    assert st["/euclid/plan"]["expired"] == 0 and st["/euclid/cmd"]["expired"] == 0
    assert pr.base.status()["topics"]["/euclid/plan"]["rejected"] == 0
    # once the offset is known, the TTL is enforced again (corrected)
    assert pr.robot.clock_offset == pytest.approx(-skew, abs=0.06)


def test_ttl_after_peer_restart_uses_heartbeat_offset():
    """After a peer restart the NTP offset is reset; until it is re-measured the
    one-way heartbeat estimate keeps a 45 s skew from expiring commands."""
    pr = Pair(robot_skew=45.0)
    pr.run(3.0)
    pr.base.close()
    pr.base = pr.new_base()
    pr.clk.advance(0.05)
    pr.link.advance()
    pr.base.tick()  # one heartbeat from the new base, no round trip yet
    pr.base.publish("/euclid/cmd", b"pause")
    pr.run(3.0)
    assert pr.data("robot", "/euclid/cmd") == [b"pause"]
    assert pr.robot.status()["topics"]["/euclid/cmd"]["expired"] == 0


def test_ttl_expired_command_never_delivered():
    pr = Pair()
    pr.run(1.0)
    pr.link.close_transport(15.0)  # longer than the cmd's ttl_s (10)
    pr.base.publish("/euclid/cmd", b"stale")
    pr.run(20.0)
    assert pr.data("robot", "/euclid/cmd") == []
    assert pr.base.status()["topics"]["/euclid/cmd"]["expired"] == 1


def test_ttl_rejected_on_receiver_with_negative_ack():
    # one-way 0.5 s, so a 0.3 s ttl always expires in flight
    cfg = {**CFG, "topics": [dict(CFG["topics"][1], ttl_s=0.3)]}
    pr = Pair(cfg=cfg, latency=0.5)
    pr.run(3.0)
    pr.base.publish("/euclid/cmd", b"late")
    pr.run(3.0)
    assert pr.data("robot", "/euclid/cmd") == []
    assert pr.robot.status()["topics"]["/euclid/cmd"]["expired"] >= 1


def test_no_replay_after_robot_restart():
    pr = Pair()
    pr.run(2.0)
    pr.base.publish("/euclid/plan", b"old plan")
    pr.run(1.0)
    assert pr.data("robot", "/euclid/plan") == [b"old plan"]
    # radio drops; the base queues a command; the robot reboots meanwhile
    pr.link.close_transport(5.0)
    pr.base.publish("/euclid/cmd", b"pause")
    pr.base.publish("/euclid/plan", b"new plan")
    pr.run(1.0)
    old_epoch = pr.robot.epoch
    pr.robot.close()
    pr.got["robot"].clear()
    pr.robot = pr.new_robot()
    pr.run(10.0)
    assert pr.robot.epoch != old_epoch
    assert pr.got["robot"] == []  # nothing re-run on the rebooted robot
    st = pr.base.status()
    assert st["peer_restarts"] == 1 and st["peer_epoch"] == pr.robot.epoch
    assert st["topics"]["/euclid/cmd"]["dropped_restart"] == 1
    assert st["topics"]["/euclid/plan"]["dropped_restart"] == 1
    # the link itself works for new commands
    pr.base.publish("/euclid/cmd", b"resume")
    pr.run(1.0)
    assert pr.data("robot", "/euclid/cmd") == [b"resume"]


def test_stale_epoch_retry_rejected_by_new_robot():
    """A retry addressed to the old robot epoch that arrives before the base saw
    the new epoch is rejected (and never applied)."""
    pr = Pair(
        hb_period=100.0
    )  # no heartbeats: the base learns the epoch from data only
    pr.base.tick()
    pr.robot.publish("/euclid/result", b"hello")
    pr.run(1.0)
    pr.link.close_transport(2.0)
    pr.base.publish("/euclid/cmd", b"pause")
    pr.robot.close()
    pr.robot = pr.new_robot(hb_period=100.0)
    pr.run(4.0)
    assert pr.data("robot", "/euclid/cmd") == []
    assert pr.robot.status()["topics"]["/euclid/cmd"]["stale_epoch"] >= 1
    # the nack itself reveals the new epoch: the base drops the command for good
    st = pr.base.status()["topics"]["/euclid/cmd"]
    assert st["dropped_restart"] == 1 and st["pending"] == 0


def test_replay_on_resync_entries_resent_to_new_base():
    """A restarted base gets the robot's state (latest_reliable ``static``) again,
    but not an action result the old base already acked and applied: re-sending
    that would publish it twice on the base graph."""
    pr = Pair()
    pr.run(1.0)
    pr.robot.publish("/euclid/result", b"done 3")
    pr.robot.publish("/euclid/guards", b"g")
    pr.robot.publish("/euclid/static", b"tf")
    pr.run(1.0)
    assert pr.data("base", "/euclid/result") == [b"done 3"]
    pr.base.close()
    pr.got["base"].clear()
    pr.base = pr.new_base()
    pr.run(3.0)
    assert pr.data("base", "/euclid/static") == [b"tf"]  # state: replayed
    assert pr.data("base", "/euclid/result") == []  # acked event: not again
    assert pr.data("base", "/euclid/guards") == []  # no replay_on_resync


def test_unacked_result_reaches_restarted_base_exactly_once():
    """A result still pending when the base restarts is re-addressed to the new
    base and delivered there once."""
    pr = Pair()
    pr.run(1.0)
    pr.link.close_transport(2.0)
    pr.robot.publish("/euclid/result", b"done 4")
    pr.run(1.0)
    pr.base.close()
    pr.base = pr.new_base()
    pr.run(5.0)
    assert pr.data("base", "/euclid/result") == [b"done 4"]
    st = pr.robot.status()["topics"]["/euclid/result"]
    assert st["pending"] == 0 and st["acked"] == 1


def test_stale_epoch_nack_does_not_drop_replayed_result():
    """Base bridge restart while a result is pending: the robot's retry, still
    addressed to the old base, reaches the new base before its first heartbeat.
    The new base nacks it ("stale epoch"); that nack is also how the robot learns
    of the restart and re-addresses the pending.  The nack must not then pop it:
    the result is delivered, exactly once."""
    pr = Pair()
    pr.run(3.0)
    pr.link.close_transport(3.0)  # radio out t=3..6: the old base never sees it
    pr.robot.publish("/euclid/result", b"R1")
    pr.run(1.5)
    o = pr.robot.out["/euclid/result"]
    assert [p.to for p in o.pending.values()] == [pr.base.epoch]
    pr.base.close()
    pr.got["base"].clear()
    pr.base = pr.new_base()  # its ticker (heartbeats) has not run yet
    for _ in range(60):  # link back at t=6; only the robot ticks
        pr.clk.advance(0.05)
        pr.link.advance()
        pr.robot.tick()
    assert pr.robot.peer_restarts == 1
    pr.run(10.0)
    assert pr.data("base", "/euclid/result") == [b"R1"]
    st = pr.robot.status()["topics"]["/euclid/result"]
    assert st["pending"] == 0 and st["stale_epoch"] >= 1 and st["rejected"] == 0


def test_type_hash_mismatch_rejected_and_counted():
    pr = Pair(robot_type_hash=lambda t: "OTHER:" + t)
    pr.run(1.0)
    pr.base.publish("/euclid/plan", b"PLAN")
    pr.run(2.0)
    assert pr.data("robot", "/euclid/plan") == []
    assert pr.robot.status()["topics"]["/euclid/plan"]["type_mismatch"] == 1
    st = pr.base.status()["topics"]["/euclid/plan"]
    assert st["rejected"] == 1 and st["err"] == "type mismatch" and st["pending"] == 0


def test_config_hash_mismatch_warns():
    other = {**CFG, "topics": CFG["topics"][:3]}
    pr = Pair(robot_cfg=other)
    pr.run(2.0)
    st = pr.base.status()
    assert st["cfg_match"] is False and st["peer_cfg_hash"] == RelayConfig(other).hash
    assert any(k == "cfg_mismatch" for k, _ in pr.events)


def test_link_status_connected_and_disconnected():
    pr = Pair()
    pr.run(2.0)
    st = pr.base.status()
    assert st["connected"] and st["rtt"] == pytest.approx(0.1, abs=0.05)
    assert st["platform"] == "spot" and st["cfg_match"] is True
    pr.link.close_transport(10.0)
    pr.run(5.0)
    assert not pr.base.status()["connected"]
    pr.run(7.0)
    assert pr.base.status()["connected"]


# ----------------------------------------------------------------- services


def test_service_round_trip_both_directions_under_loss():
    pr = Pair(loss=0.3, seed=11)
    calls = {"base": 0, "robot": 0}

    def serve(side):
        def handler(s, req, reply):
            calls[side] += 1
            reply(b"re:" + req + b"@" + s.name.encode() + b"/" + s.base_name.encode())

        return handler

    pr.base.on_service = serve("base")
    pr.robot.on_service = serve("robot")
    pr.run(1.0)
    up, down = [], []
    pr.robot.call("/euclid/hold", b"h", lambda d, e: up.append((d, e)))
    pr.base.call("/euclid/save", b"s", lambda d, e: down.append((d, e)))
    pr.run(4.0)
    assert up == [(b"re:h@/euclid/hold//willow/hold", None)]
    assert down == [(b"re:s@/euclid/save//euclid/save", None)]
    assert calls == {"base": 1, "robot": 1}  # retries deduped by id
    with pytest.raises(KeyError):
        pr.robot.call("/euclid/save", b"", lambda d, e: None)  # wrong direction


def test_service_timeout():
    pr = Pair()
    pr.base.on_service = lambda s, req, reply: None  # never answers
    pr.run(1.0)
    out = []
    pr.robot.call("/euclid/hold", b"h", lambda d, e: out.append((d, e)))
    pr.run(4.9)
    assert out == []
    pr.run(0.2)
    assert out == [(None, "timeout")]
    assert pr.robot.status()["services"]["/euclid/hold"]["timeouts"] == 1


def test_service_timeout_when_link_is_down():
    pr = Pair()
    pr.base.on_service = lambda s, req, reply: reply(b"ok")
    pr.run(1.0)
    pr.link.close_transport(30.0)
    out = []
    pr.base.call("/euclid/save", b"s", lambda d, e: out.append((d, e)))
    pr.run(3.5)
    assert out == [(None, "timeout")]


# ----------------------------------------------------------------- teleop


TELEOP = {
    "version": 1,
    "topics": [
        {"topic": "/{robot}/cmd_vel", "type": "geometry_msgs/msg/Twist", "dir": "down",
         "mode": "latest", "cls": "C0", "rate_hz": 10.0, "ttl_s": 0.5, "ttl_strict": True},
    ],
}  # fmt: skip


def test_relay_yaml_teleop_entries():
    cfg = RelayConfig.load(RELAY_YAML)
    spot = cfg.resolve("euclid", "spot", base="willow")
    husky = cfg.resolve("gauss", "phoenix", base="willow")
    vel = spot.topics["/euclid/spot_executor_node/cmd_vel"]
    assert vel.type == "geometry_msgs/msg/Twist" and vel.dir == "down"
    assert vel.mode == "latest" and vel.cls in ("C0", "C1")
    assert vel.ttl_s is not None and vel.ttl_s <= 0.5 and vel.ttl_strict
    assert vel.rate_hz and 5 <= vel.rate_hz <= 20
    assert not vel.replay_on_resync  # a restarted robot never replays motion
    assert "/gauss/spot_executor_node/cmd_vel" not in husky.topics
    for srv in ("stand", "sit"):
        s = spot.services[f"/euclid/spot_executor_node/{srv}"]
        assert s.dir == "down" and s.type == "std_srvs/srv/Trigger" and s.cls == "C0"
        assert s.ttl_s is not None and s.ttl_s <= 10 and s.timeout_s >= 10
        assert f"/gauss/spot_executor_node/{srv}" not in husky.services
    st = spot.topics["/euclid/spot_executor_node/teleop_state"]
    assert st.dir == "up" and st.type == "std_msgs/msg/String"


@pytest.mark.parametrize(
    "entry, msg",
    [
        ({"mode": "latest", "ttl_strict": True}, "needs ttl_s"),
        ({"mode": "reliable", "ttl_s": 1, "ttl_strict": True}, "only for mode latest"),
    ],
)
def test_ttl_strict_validation(entry, msg):
    t = {"topic": "/x", "type": "a/msg/B", "dir": "down", **entry}
    with pytest.raises(RelayConfigError, match=msg):
        RelayConfig({"version": 1, "topics": [t]})


def test_strict_ttl_drops_until_clock_offset_measured():
    """Before the first heartbeat round trip, a strict entry cannot tell how old a
    velocity command is: it is dropped, not applied."""
    pr = Pair(cfg=TELEOP, robot_skew=45.0)
    pr.base.publish("/euclid/cmd_vel", b"v0")  # at link-up, no heartbeat yet
    pr.run(0.2)
    assert pr.data("robot", "/euclid/cmd_vel") == []
    assert pr.robot.status()["topics"]["/euclid/cmd_vel"]["expired"] == 1
    pr.run(3.0)  # offset measured
    assert pr.robot.clock_offset is not None
    pr.base.publish("/euclid/cmd_vel", b"v1")
    pr.run(0.3)
    assert pr.data("robot", "/euclid/cmd_vel") == [b"v1"]


def test_strict_ttl_drops_radio_delayed_velocity():
    """One-way 0.7 s: every command is older than its 0.5 s ttl on arrival, so
    nothing moves the robot (the executor deadman then holds it still)."""
    pr = Pair(cfg=TELEOP, latency=0.7)
    pr.run(5.0)
    assert pr.robot.clock_offset is not None
    for i in range(10):
        pr.base.publish("/euclid/cmd_vel", b"v%d" % i)
        pr.run(0.1)
    pr.run(2.0)
    assert pr.data("robot", "/euclid/cmd_vel") == []
    assert pr.robot.status()["topics"]["/euclid/cmd_vel"]["expired"] >= 1


def test_velocity_stream_delivered_on_good_link():
    pr = Pair(cfg=TELEOP, latency=0.05)
    pr.run(3.0)
    for i in range(10):
        pr.base.publish("/euclid/cmd_vel", b"v%d" % i)
        pr.run(0.1)
    pr.run(0.3)
    got = pr.data("robot", "/euclid/cmd_vel")
    assert len(got) >= 8 and got[-1] == b"v9"


def test_stale_latest_value_dropped_on_sender():
    """A latest value that waits for its rate slot longer than its ttl is dropped
    on the sender instead of being sent late."""
    cfg = {"version": 1, "topics": [dict(TELEOP["topics"][0], rate_hz=0.5)]}
    pr = Pair(cfg=cfg)
    pr.run(3.0)
    pr.base.publish("/euclid/cmd_vel", b"go")
    pr.base.publish("/euclid/cmd_vel", b"late")  # waits ~2 s for the next slot
    pr.run(3.0)
    assert pr.data("robot", "/euclid/cmd_vel") == [b"go"]
    assert pr.base.status()["topics"]["/euclid/cmd_vel"]["expired"] == 1
