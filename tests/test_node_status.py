"""ros_system_monitor status (node_status.py): pure computation, no ROS."""

import threading

from helpers import RecordingTransport, env_bytes
from test_relay import Pair

from fleetlink.agent import AgentCore
from fleetlink.node_status import (
    BRIDGE_IDLE_NOTES,
    ERROR,
    NOMINAL,
    WARNING,
    agent_status,
    bridge_status,
    fmt_bytes,
    fmt_s,
)
from fleetlink.transport import VirtualClock

UP = {"connected": True, "rtt": 0.08, "loss": 0.0, "last_seen": 1.0, "age_s": 0.2}
DOWN = {"connected": False, "rtt": 0.08, "loss": 0.0, "last_seen": 1.0, "age_s": 7.0}
NEVER = {"connected": False, "rtt": None, "last_seen": None, "age_s": None}
EP = ["tcp/192.0.2.10:7450", "tcp/198.51.100.10:7450"]


def test_formatting():
    assert fmt_s(0.084) == "84ms"
    assert fmt_s(4.24) == "4.2s"
    assert fmt_s(180) == "3m"
    assert fmt_s(None) == "?"
    assert fmt_bytes(12) == "12B"
    assert fmt_bytes(12_345) == "12.3kB"
    assert fmt_bytes(2_500_000) == "2.5MB"


def test_agent_fleetlink_nominal():
    level, notes = agent_status(
        opened=True, endpoints=EP, relay=UP, mesh_source="odometric"
    )
    assert level == NOMINAL
    assert notes == (
        "fleetlink | base up rtt 80ms | tcp/192.0.2.10:7450 (+1 failover)"
        " | mesh odometric"
    )


def test_agent_fleetlink_link_down_or_degraded_warns():
    assert agent_status(opened=True, relay=DOWN)[0] == WARNING
    assert "base link down 7.0s" in agent_status(opened=True, relay=DOWN)[1]
    level, notes = agent_status(opened=True, relay=NEVER)
    assert (level, "no base heartbeat yet" in notes) == (WARNING, True)
    slow = {**UP, "rtt": 2.5, "loss": 0.12}
    level, notes = agent_status(opened=True, relay=slow)
    assert level == WARNING and "rtt 2.5s (high) loss 12%" in notes
    assert agent_status(opened=True, relay=slow, rtt_warn_s=5.0)[0] == NOMINAL
    level, notes = agent_status(opened=True, relay={**UP, "cfg_match": False})
    assert level == WARNING and "relay.yaml differs" in notes


def test_agent_router_mode_console_is_a_note():
    level, notes = agent_status(opened=True, endpoints=["tcp/127.0.0.1:7447"])
    assert level == NOMINAL
    assert notes == "router | no console yet | tcp/127.0.0.1:7447"
    level, notes = agent_status(
        opened=True, console_age_s=0.4, console_rtt_s=0.012, queued_bytes=4200
    )
    assert level == NOMINAL
    assert "console 400ms ago rtt 12ms" in notes and "queued 4.2kB" in notes


def test_agent_mesh_fallback_warns_store_error_does_not():
    level, notes = agent_status(opened=True, mesh_fallback="missed deltas")
    assert level == WARNING and "mesh dsg (fallback: missed deltas)" in notes
    level, notes = agent_status(opened=True, store_error="no hydra_python")
    assert level == NOMINAL and "store mirror off: no hydra_python" in notes


def test_agent_zenoh_not_open_is_error():
    level, notes = agent_status(opened=False, open_error="ZError: bad", endpoints=EP)
    assert level == ERROR and notes.startswith("zenoh session not open: ZError: bad")


def test_bridge_status():
    assert bridge_status({}) == (NOMINAL, "no robots yet")
    assert bridge_status({}, idle=True) == (NOMINAL, BRIDGE_IDLE_NOTES)
    level, notes = bridge_status({"euclid": UP, "hamilton": NEVER})
    assert (level, notes) == (NOMINAL, "up: euclid 80ms | waiting: hamilton")
    level, notes = bridge_status({"euclid": UP, "gauss": DOWN})
    assert level == WARNING
    assert notes == "up: euclid 80ms | DOWN: gauss silent 7.0s"
    level, notes = bridge_status({"euclid": {**UP, "cfg_match": False}})
    assert level == WARNING and "euclid relay.yaml differs" in notes
    assert bridge_status({}, opened=False, open_error="x")[0] == ERROR


def test_relay_heartbeats_drive_both_ends():
    """Real RelayCore.status() over the emulated link: up, then the robot stops."""
    pair = Pair()
    assert agent_status(opened=True, relay=pair.robot.status())[0] == WARNING
    pair.run(3.0)
    assert agent_status(opened=True, relay=pair.robot.status())[0] == NOMINAL
    level, notes = bridge_status({"robot1": pair.base.status()})
    assert level == NOMINAL and notes.startswith("up: robot1 ")
    for _ in range(100):  # the robot goes silent; only the base ticks
        pair.clk.advance(0.05)
        pair.link.advance()
        pair.base.tick()
    level, notes = bridge_status({"robot1": pair.base.status()})
    assert level == WARNING and "DOWN: robot1 silent" in notes


class _Opened(RecordingTransport):
    def __init__(self):
        super().__init__()
        self.opened = threading.Event()
        self.open_error = "no session"
        self.endpoints = ["tcp/127.0.0.1:7447"]


def test_agent_core_node_status_tracks_console_requests():
    tr = _Opened()
    clock = VirtualClock(100.0)
    core = AgentCore("r", tr, {"mesh_source": "odometric"}, clock)
    assert core.node_status()[0] == ERROR
    tr.opened.set()
    assert core.node_status() == (
        NOMINAL,
        "router | no console yet | tcp/127.0.0.1:7447 | mesh odometric",
    )
    args = {"t": 1.0, "rtt": 0.03}
    req = {"id": "1", "op": "ping", "args": args}
    tr.inject("adt4/r/ctl/req", env_bytes("base", "ctl/req", 0, "S", req))
    core.tick()
    clock.advance(2.0)
    level, notes = core.node_status()
    assert level == NOMINAL and "console 2.0s ago rtt 30ms" in notes
    core.fallback_to_dsg_mesh("no mesh deltas")
    assert core.node_status()[0] == WARNING
