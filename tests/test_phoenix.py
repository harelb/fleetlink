"""Phoenix (Clearpath Husky) support in the agent: platform in the C1 state,
guards from the platform topics, plan progress, body frame, and the on-demand
C3 lidar stream end to end (agent → LinkEmulator → Receiver)."""

import numpy as np
import yaml
from helpers import RecordingTransport, env_bytes

from fleetlink import pointcloud_codec as pc
from fleetlink.agent import (
    AgentCore,
    PhoenixGuards,
    default_body_frame,
    phoenix_progress,
    resolve_platform,
    trajectory_points,
)
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator, VirtualClock

POSE = {"x": 0.0, "y": 0.0, "z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "frame": "gauss/map", "t": 0.0}  # fmt: skip


def _agent(**cfg):
    tr = RecordingTransport()
    clock = VirtualClock()
    core = AgentCore("r", tr, cfg, clock)
    core.on_pose(dict(POSE))
    return tr, clock, core


def _run(clock, core, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        core.tick(clock.advance(dt))


def _state(tr):
    return tr.messages("adt4/r/state")[-1][1]["p"]


def _req(tr, rid, op, args=None, seq=0):
    tr.inject(
        "adt4/r/ctl/req",
        env_bytes("base", "ctl/req", seq, "S", {"id": rid, "op": op, "args": args or {}}),
    )


# --------------------------------------------------------------- platform / state


def test_spot_state_unchanged_but_names_its_platform():
    tr, clock, core = _agent()
    _run(clock, core, 1.0)
    st = _state(tr)
    assert st["platform"] == "spot"
    assert "phoenix_progress" not in st and "lidar" not in st
    assert not any("lidar" in k for k, _, _ in tr.sent)


def test_phoenix_state_has_progress_and_lidar_keys():
    tr, clock, core = _agent(platform="phoenix", lidar_topic="/r/ouster/points")
    _run(clock, core, 1.0)
    st = _state(tr)
    assert st["platform"] == "phoenix"
    assert st["phoenix_progress"] is None  # no plan yet
    assert st["lidar"]["available"] is True and st["lidar"]["enabled"] is False


def test_body_frame_and_platform_resolution(tmp_path):
    assert default_body_frame("gauss", "phoenix") == "gauss/base_link"
    assert default_body_frame("hamilton", "spot") == "hamilton/body"
    topo = tmp_path / "topology.yaml"
    topo.write_text(yaml.safe_dump({"machines": {
        "gauss": {"role": "robot", "platform_type": "husky"},
        "hamilton": {"role": "robot"},
    }}))  # fmt: skip
    assert resolve_platform("gauss", "", str(topo)) == "phoenix"
    assert resolve_platform("hamilton", "", str(topo)) == "spot"
    assert resolve_platform("nobody", "", str(topo)) == "spot"
    assert resolve_platform("hamilton", "Phoenix", str(topo)) == "phoenix"  # param wins
    assert resolve_platform("gauss", "", str(tmp_path / "missing.yaml")) == "spot"


# ----------------------------------------------------------------------- guards


def test_phoenix_guards_adapter():
    t = [100.0]
    g = PhoenixGuards(clock=lambda: t[0])
    d = g.as_dict()
    assert d == {"battery_percent": None, "estop_pressed": None, "safety_stop": None,
                 "manual_mode": None, "source": "phoenix", "t": None}  # fmt: skip
    assert g.battery(0.468)["battery_percent"] == 46.8  # BatteryState 0-1 → %
    assert g.battery(float("nan"))["battery_percent"] is None
    g.battery(0.5)
    g.estop(False)
    g.safety_stop(True)
    t[0] = 101.0
    d = g.manual_mode(True)
    assert d == {"battery_percent": 50.0, "estop_pressed": False, "safety_stop": True,
                 "manual_mode": True, "source": "phoenix", "t": 101.0}  # fmt: skip
    tr, clock, core = _agent(platform="phoenix")
    core.on_guards(d)
    _run(clock, core, 0.5)
    assert _state(tr)["guards"]["manual_mode"] is True


# --------------------------------------------------------------------- progress


def test_phoenix_progress_distance_done_and_frames():
    plan = {"plan_id": "5.000000007", "stamp": 5.0, "frame": "gauss/map",
            "end": [3.0, 4.0, 0.0], "left": True}  # fmt: skip
    p = phoenix_progress(plan, POSE)
    assert p == {"plan_id": "5.000000007", "stamp": 5.0, "dist_to_goal_m": 5.0, "done": False}
    near = dict(POSE, x=2.5, y=4.5, z=7.0)  # xy only
    assert phoenix_progress(plan, near)["done"] is True
    # never away from the end since the plan arrived: not done yet
    assert phoenix_progress(dict(plan, left=False), near)["done"] is False
    other = dict(POSE, frame="map")  # no TF between them: unknown, never "done"
    assert phoenix_progress(plan, other) == dict(p, dist_to_goal_m=None)
    assert phoenix_progress(plan, None)["dist_to_goal_m"] is None
    assert phoenix_progress(None, POSE) is None
    assert phoenix_progress(dict(plan, end=None), POSE)["done"] is False  # empty Path


def test_phoenix_progress_in_state():
    tr, clock, core = _agent(platform="phoenix")
    core.on_plan_path([[0, 0, 0], [1, 1, 0], [0.5, 0.2, 0]], "gauss/map", 1700000000, 42)
    _run(clock, core, 0.5)
    prog = _state(tr)["phoenix_progress"]
    assert prog["plan_id"] == "1700000000.000000042"
    assert prog["dist_to_goal_m"] == round(float(np.hypot(0.5, 0.2)), 3)
    assert prog["done"] is False  # a round trip is not done at dispatch
    core.on_pose(dict(POSE, x=20.0))
    _run(clock, core, 0.5)
    assert _state(tr)["phoenix_progress"]["done"] is False
    core.on_pose(dict(POSE, x=0.4, y=0.1))  # back at the end, having left it
    _run(clock, core, 0.5)
    assert _state(tr)["phoenix_progress"]["done"] is True


def test_phoenix_round_trip_done_only_after_leaving_the_goal():
    tr, clock, core = _agent(platform="phoenix")
    loop = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 0.2, 0]]
    core.on_plan_path(loop, "gauss/map", 10, 1)
    for x in (0.0, 0.5, 0.9):  # wiggling near the start (within 1 m of the end)
        core.on_pose(dict(POSE, x=x))
        _run(clock, core, 0.3)
        assert _state(tr)["phoenix_progress"]["done"] is False
    core.on_pose(dict(POSE, x=1.5))  # left the goal
    core.on_plan_path(loop, "gauss/map", 10, 1)  # the same plan again keeps that
    core.on_pose(dict(POSE, x=0.1))
    _run(clock, core, 0.3)
    assert _state(tr)["phoenix_progress"]["done"] is True
    # a new plan starts over
    core.on_plan_path(loop, "gauss/map", 11, 0)
    _run(clock, core, 0.3)
    p = _state(tr)["phoenix_progress"]
    assert p["plan_id"] == "11.000000000" and p["done"] is False


def test_phoenix_short_or_single_pose_path_done_at_once():
    for path in ([[0.3, 0.0, 0.0]], [[0, 0, 0], [0.4, 0.3, 0.0]]):
        tr, clock, core = _agent(platform="phoenix")
        core.on_plan_path(path, "gauss/map", 7, 0)
        _run(clock, core, 0.3)
        p = _state(tr)["phoenix_progress"]
        assert p["done"] is True and p["dist_to_goal_m"] < 1.0
    # a plan received while the robot is already far from the end: left at once
    tr, clock, core = _agent(platform="phoenix")
    core.on_pose(dict(POSE, x=5.0))
    core.on_plan_path([[5, 0, 0], [0, 0, 0]], "gauss/map", 8, 0)
    core.on_pose(dict(POSE, x=0.2))
    _run(clock, core, 0.3)
    assert _state(tr)["phoenix_progress"]["done"] is True


def test_trajectory_points_duck_typed():
    class P:  # geometry_msgs/Point-like
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Pose:
        def __init__(self, *xyz):
            self.position = P(*xyz)

    class Msg:
        def __init__(self, fields):
            self.__dict__.update(fields)
            self._f = fields

        def get_fields_and_field_types(self):
            return {k: "" for k in self._f}

    traj = Msg({"id": 3, "poses": [Pose(1, 2, 0), Pose(3, 4, 0)]})
    arr = Msg({"header": None, "trajectories": [traj]})
    assert trajectory_points(arr).tolist() == [[1, 2, 0], [3, 4, 0]]
    assert trajectory_points(Msg({"trajectories": []})).shape == (0, 3)


# ------------------------------------------------------------------------ lidar


def _frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-10, 10, (n, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(-0.5, 3.0, n)
    f = pc.encode_cloud(xyz)
    f.update(frame="r/map", sensor_frame="r/os_sensor",
             origin=[1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0], stamp=12.5)  # fmt: skip
    return f


def _pair(cfg=None, rx_kw=None):
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.01)
    cfg = {"platform": "phoenix", "lidar_topic": "/r/ouster/points", **(cfg or {})}
    core = AgentCore("r", link.robot, cfg, clock=clock)
    core.on_pose({**POSE, "frame": "r/map"})
    rx = Receiver(link.base, clock=clock, threaded=False, **(rx_kw or {}))
    rx.start()
    return clock, link, core, rx


def _go(clock, link, core, rx, seconds, feed=True, read=True, dt=0.05):
    for i in range(int(seconds / dt)):
        clock.advance(dt)
        if feed and i % 2 == 0:  # a 10 Hz lidar
            core.on_lidar(_frame(seed=i))
        core.tick()
        link.advance()
        rx.tick()
        if read:
            rx.lidar("r")


def test_lidar_off_by_default_then_on_at_1hz_on_c3():
    clock, link, core, rx = _pair()
    _go(clock, link, core, rx, 3.0)
    assert rx.lidar("r") is None and core.seq["lidar/points"] == 0
    assert rx.lidar_status("r") == {"available": True, "robot": False, "wanted": None}

    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 0.5)
    assert core.lidar_enabled is True
    msgs0 = core.seq["lidar/points"]
    _go(clock, link, core, rx, 5.0)
    n = core.seq["lidar/points"] - msgs0
    assert 4 <= n <= 6  # ~1 Hz
    f = rx.lidar("r")
    assert f["frame"] == "r/map" and f["origin"] == [1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    assert f["scale"] == 0.01 and f["stamp"] == 12.5 and isinstance(f["seq"], int)
    assert len(f["xyz"]) == 6 * f["n"] and f["intensity"] is None
    # the receiver hands out the interleaved int16 of the newest frame it got
    want = pc.decode_cloud({"xyz": f["xyz"], "n": f["n"], "scale": 0.01})[0]
    assert len(want) == f["n"] and np.abs(want).max() <= 15.0
    assert rx.lidar_status("r") == {"available": True, "robot": True, "wanted": True}
    assert rx.link("r")["bytes"]["C3"] > 0  # shows on the Network card as C3

    rx.set_lidar("r", False)
    _go(clock, link, core, rx, 1.0)
    sent = core.seq["lidar/points"]
    _go(clock, link, core, rx, 3.0)
    assert core.lidar_enabled is False and core.seq["lidar/points"] == sent
    assert rx.lidar("r") is None


def test_lidar_bandwidth_cap_slows_big_frames():
    clock, link, core, rx = _pair({"lidar_bps": 2_000})
    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 1.0)
    m0 = core.seq["lidar/points"]
    _go(clock, link, core, rx, 6.0)
    frame_b = core.lidar_sent_bytes
    assert frame_b > 2_000
    assert core.seq["lidar/points"] - m0 <= 6.0 * 2_000 / frame_b + 1


def test_lidar_turns_itself_off_when_nobody_reads():
    clock, link, core, rx = _pair(rx_kw={"lidar_idle_s": 3.0})
    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 2.0, read=False)
    assert core.lidar_enabled is True
    _go(clock, link, core, rx, 3.0, read=False)
    assert core.lidar_enabled is False
    assert rx.lidar_status("r")["wanted"] is False


def test_lidar_reasked_after_agent_restart():
    clock, link, core, rx = _pair()
    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 2.0)
    assert core.lidar_enabled
    core.close()
    core2 = AgentCore("r", link.robot, dict(core.cfg), clock=clock)  # starts off
    core2.on_pose({**POSE, "frame": "r/map"})
    _go(clock, link, core2, rx, 4.0)
    assert core2.lidar_enabled is True
    assert rx.lidar("r") is not None


def test_lidar_op_without_lidar_is_refused():
    tr, clock, core = _agent()
    _req(tr, "l1", "lidar", {"enable": True})
    _run(clock, core, 0.5)
    ack = [m["p"] for _, m, _ in tr.messages("adt4/r/ctl/ack")][-1]
    assert ack["ok"] is False and "no lidar" in ack["result"]["error"]
    assert core.lidar_enabled is False


def test_lidar_frame_kept_off_c3_mesh_requeue():
    # re-tiling the mesh clears queued C3 tiles but never the pending lidar frame
    tr, clock, core = _agent(platform="phoenix", lidar_topic="/r/ouster/points",
                             lidar=True, budgets={"C3": 1})  # fmt: skip
    core.on_lidar(_frame())
    _run(clock, core, 0.2)  # the first one spends the C3 burst
    core.on_lidar(_frame(seed=1))
    _run(clock, core, 1.1)
    assert any(it.tag == "lidar" for it in core.sched.queues["C3"])
    core._tiles_dirty = True
    core._c3_tick()
    assert any(it.tag == "lidar" for it in core.sched.queues["C3"])


def test_agent_lidar_ttl_without_base_refresh():
    tr, clock, core = _agent(platform="phoenix", lidar_topic="/r/ouster/points",
                             lidar_ttl_s=60.0)  # fmt: skip
    _req(tr, "e1", "lidar", {"enable": True})
    _run(clock, core, 30.0)
    _req(tr, "e2", "lidar", {"enable": True}, seq=1)  # a refresh restarts the ttl
    _run(clock, core, 50.0)
    assert core.lidar_enabled is True
    _run(clock, core, 15.0)  # 65 s since the last request
    assert core.lidar_enabled is False
    assert _state(tr)["lidar"]["enabled"] is False


def test_base_refresh_keeps_a_watched_stream_on():
    clock, link, core, rx = _pair({"lidar_ttl_s": 5.0}, {"lidar_refresh_s": 2.0})
    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 12.0)  # read every tick: refreshed past the ttl
    assert core.lidar_enabled is True
    assert rx.lidar("r") is not None


def test_bad_cloud_reports_an_error_but_keeps_the_lidar_available():
    clock, link, core, rx = _pair()
    rx.set_lidar("r", True)
    _go(clock, link, core, rx, 2.0, feed=False)
    core.lidar_error = "ValueError: bad cloud"  # what AgentNode._on_cloud sets
    _go(clock, link, core, rx, 1.5, feed=False)
    st = rx._robots["r"].state["lidar"]
    assert st["available"] is True and st["enabled"] is True
    assert st["error"] == "ValueError: bad cloud"
    assert rx.lidar_status("r")["available"] is True
    # the receiver still reconciles: the agent lost the request -> re-asked
    core._set_lidar(False)
    core.lidar_error = "ValueError: bad cloud"
    _go(clock, link, core, rx, 4.0, feed=False)
    assert core.lidar_enabled is True
    assert core.lidar_error is None  # a fresh enable clears it
    core.lidar_error = "ValueError: bad cloud"
    _go(clock, link, core, rx, 2.0)  # a good frame clears it too
    assert core.lidar_error is None and rx.lidar("r") is not None
    assert rx._robots["r"].state["lidar"]["error"] is None


def test_lidar_subscription_depth1_best_effort_own_group():
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

    from fleetlink.agent import AgentNode, lidar_qos

    q = lidar_qos()
    assert q.depth == 1 and q.history == HistoryPolicy.KEEP_LAST
    # best effort: compatible with the Ouster's RELIABLE KEEP_LAST 10 publisher
    assert q.reliability == ReliabilityPolicy.BEST_EFFORT
    assert q.durability == DurabilityPolicy.VOLATILE

    calls = []

    class Node:
        def create_subscription(self, *a, **k):
            calls.append((a, k))
            return object()

        def destroy_subscription(self, sub):
            calls.append(("destroy", sub))

    _, _, core = _agent(platform="phoenix", lidar_topic="/r/ouster/points")
    an = AgentNode.__new__(AgentNode)
    an.node, an.core, an._lidar_sub, an._lidar_t = Node(), core, None, -1e9
    an._cloud_type, an._lidar_group = "PointCloud2", MutuallyExclusiveCallbackGroup()
    an._update_lidar()
    assert calls == []  # off: not subscribed
    core._set_lidar(True)
    an._update_lidar()
    (args, kw), = calls
    assert args[1] == "/r/ouster/points" and args[3].depth == 1
    assert kw["callback_group"] is an._lidar_group
    # the time gate comes first: a cloud right after the last one is not touched
    an._lidar_t = __import__("time").time()
    an._on_cloud(None)  # would raise if it read the message
    core._set_lidar(False)
    an._update_lidar()
    assert calls[-1][0] == "destroy" and an._lidar_sub is None
