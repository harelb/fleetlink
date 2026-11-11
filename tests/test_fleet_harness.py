"""Fleetlink mode end to end on the real topology (``harness.Fleet``): a base ROS
router, the fleetlink router (``bin/fleetlink-router``), two robot ROS routers,
real ``fleetlink-agent`` (relay on) and ``fleetlink-bridge`` processes, fake
robot/base nodes, and a :class:`harness.FaultProxy` on each robot's radio link.

One fleet per module (~3 s to start); the tests run in file order and each
leaves the fleet healthy.  Knobs (environment):

* ``FLEETLINK_STALL_S`` (20): radio stall length; the plan's figure is 60.
* ``FLEETLINK_RESUME_S`` (3.0): bound on C1 state resuming after the stall.
* ``FLEETLINK_THROTTLE_BPS`` (30000): throttled radio rate, bytes/s.
* ``FLEETLINK_HARNESS_PORT`` (17470) / ``FLEETLINK_HARNESS_DOMAIN`` (94): first
  port (8 used) / first ROS domain (3 used).  Never 7447.
"""

import json
import os
import shutil
import time

import pytest

pytestmark = pytest.mark.integration

STALL_S = float(os.environ.get("FLEETLINK_STALL_S", 20))
RESUME_S = float(os.environ.get("FLEETLINK_RESUME_S", 3.0))
THROTTLE_BPS = float(os.environ.get("FLEETLINK_THROTTLE_BPS", 30000))
NON_DROPPABLE = "Unable to push non droppable"


@pytest.fixture(scope="module")
def fleet(tmp_path_factory):
    harness = pytest.importorskip("harness")
    if not os.path.exists(harness.RMW_ZENOHD):
        pytest.skip("rmw_zenohd not installed")
    for mod in ("rclpy", "zenoh", "tf2_ros", "robot_executor_msgs.msg",
                "heracles_ros_interfaces.srv", "hydra_msgs"):  # fmt: skip
        pytest.importorskip(mod)
    tmp = str(tmp_path_factory.mktemp("fleet"))
    f = harness.Fleet(tmp)
    try:
        f.start()
    except harness.PortBusy as exc:
        f.stop()
        pytest.skip(str(exc))
    except BaseException:
        dump = f.dump()
        f.stop()
        pytest.fail(f"fleet did not start:\n{dump}")
    yield f
    f.stop()
    keep = os.environ.get("FLEETLINK_HARNESS_KEEP_LOGS")
    if keep:
        shutil.copytree(tmp, os.path.join(keep, f"fleet-{f.tag}"), dirs_exist_ok=True)
    strays = harness.stray_pids(f.tag)
    assert not strays, f"harness processes left behind: {strays}"


def _no_non_droppable(fleet):
    bad = [l for l in fleet.logs().splitlines() if NON_DROPPABLE in l]
    assert not bad, "zenoh closed a transport (#1876):\n" + "\n".join(bad[:10])


def _liveliness_keys(endpoint: str) -> list[str]:
    import zenoh

    conf = zenoh.Config()
    conf.insert_json5("mode", '"client"')
    conf.insert_json5("connect/endpoints", json.dumps([endpoint]))
    conf.insert_json5("scouting/multicast/enabled", "false")
    s = zenoh.open(conf)
    try:
        keys = []
        for reply in s.liveliness().get("@ros2_lv/**", timeout=3.0):
            try:
                keys.append(str(reply.ok.key_expr))
            except Exception:
                pass
        return keys
    finally:
        s.close()


def test_graphs_are_isolated(fleet):
    """Each robot graph holds only its own nodes; the base graph only the bridge and
    base nodes.  Checked through the ROS graph API and, below ROS, through the
    liveliness tokens each ROS router knows (any domain)."""
    expect = {
        "base": {("fleetlink_bridge", "/"), ("harness_base", "/")},
        **{r: {("fleetlink_agent", f"/{r}"), (f"harness_robot_{r}", "/")} for r in fleet.robot_names},  # fmt: skip
    }
    helpers = {"base": fleet.base, **{r: fleet.robots[r]["helper"] for r in fleet.robot_names}}  # fmt: skip
    for graph, h in helpers.items():
        got = h.ask("nodes", "nodes")
        assert got is not None, f"{graph} helper did not answer"
        assert {tuple(n) for n in got["nodes"]} == expect[graph], (graph, got["nodes"])

    routers = {"base": fleet.ports["base_ros"], **{r: fleet.ports[f"{r}_ros"] for r in fleet.robot_names}}  # fmt: skip
    for graph, port in routers.items():
        keys = _liveliness_keys(f"tcp/127.0.0.1:{port}")
        assert keys, f"no liveliness tokens on the {graph} router"
        domains = {k.split("/")[1] for k in keys}
        assert domains == {str(fleet.domains[graph])}, (graph, domains)
        foreign = {name for other, names in expect.items() if other != graph for name, _ in names}  # fmt: skip
        own = {name for name, _ in expect[graph]}
        leaked = [k for k in keys if any(f"/{n}" in k for n in foreign - own)]
        assert not leaked, (graph, leaked[:5])


def test_relay_round_trips_per_robot(fleet):
    """Plan down -> action_result up, base TF lookup, robot -> base service, and
    /fleetlink/link_status, for every robot."""
    for r in fleet.robot_names:
        lat = fleet.plan_round_trip(r, f"rt-{r}", 10)
        assert lat is not None and lat < 3.0, (r, lat)

    tf = None
    for _ in range(50):
        tf = fleet.base.ask("tf", "tf")["tf"]
        if all(tf.get(r) is not None for r in fleet.robot_names):
            break
        time.sleep(0.1)
    assert tf == {r: pytest.approx(float(i + 1)) for i, r in enumerate(fleet.robot_names)}

    for r in fleet.robot_names:
        h = fleet.robots[r]["helper"]
        mb = fleet.base.mark()
        rep = h.ask("hold", "hold_reply", timeout=15, id=f"O({r})")
        assert rep is not None and rep["ok"], (r, rep)
        assert fleet.base.find(lambda e: e.get("ev") == "hold" and e.get("id") == f"O({r})", since=mb)  # fmt: skip

    m = fleet.base.mark()
    for r in fleet.robot_names:
        st = fleet.base.wait_for(
            lambda e, r=r: e.get("ev") == "link_status" and e.get("robot") == r, 5, since=m
        )
        assert st and st["connected"] and st["cfg_match"] is True and st["rtt"] is not None, st  # fmt: skip


def test_throttled_link_keeps_commands_flowing(fleet):
    """r1's radio throttled to THROTTLE_BPS and saturated by large C3 messages (a
    random ~64 kB occupancy grid every 2.5 s) while the base asks for a ~30 kB C4
    camera frame: the large messages arrive whole, plans, results and a service
    call still get through in seconds, C1 keeps coming, r2 does not notice."""
    px = fleet.robots["r1"]["proxy"]
    h1 = fleet.robots["r1"]["helper"]
    # random 0..100 cells compress to ~0.84 B: side 277 is a ~64 kB grid, 8 zenoh
    # fragments of 8 kB.  With zenoh's default tx queue (2 batches per priority) and
    # 50 ms fragment wait, a message this size was dropped whole on a saturated link
    # and its hash-mismatch resend was dropped again, forever.  ZENOH_DEFAULTS'
    # 16-batch data queues hold it.  Every 2.5 s is ~26 kB/s: the link stays full.
    h1.send(cmd="grid", side=277, period=2.5)
    # random rgb8 256x192 -> a ~29 kB JPEG reply (4 fragments), DATA priority
    h1.send(cmd="camera", w=256, h=192, period=1.0)
    px.throttle(THROTTLE_BPS)
    try:
        time.sleep(6.0)  # let every queue fill
        b0, t0 = px.bytes["up"], time.time()
        n0 = len(fleet.grid_rx["r1"])
        lat1 = [fleet.plan_round_trip("r1", f"thr-r1-{i}", 20) for i in range(3)]
        cam_dt, cam, cam_n = fleet.camera_query("r1", timeout=10.0, w=640, q=70)
        lat2 = [fleet.plan_round_trip("r2", f"thr-r2-{i}", 5) for i in range(3)]
        rep = h1.ask("hold", "hold_reply", timeout=15, id="O(thr)")
        wait(lambda: len(fleet.grid_rx["r1"]) > n0, max(0.0, t0 + 15.0 - time.time()))  # fmt: skip
        t1 = time.time()
        grids = [n for t, n in fleet.grid_rx["r1"][n0:]]
        rate = (px.bytes["up"] - b0) / (t1 - t0)
        gap1 = max_gap(fleet.state_times["r1"], t0, t1)
        gap2 = max_gap(fleet.state_times["r2"], t0, t1)
        print(f"\nthrottle {THROTTLE_BPS:.0f} B/s: up {rate:.0f} B/s, r1 plan rt {lat1}, "
              f"r2 plan rt {lat2}, hold {rep and rep.get('dt')}, C1 max gap r1 {gap1:.2f} s "
              f"r2 {gap2:.2f} s; camera {cam_n} B in {cam_dt}; grids {grids} in "
              f"{t1 - t0:.1f} s")  # fmt: skip
        errs = (h1.find(lambda e: e.get('ev') in ('error', 'warn')), h1.ask('grid_count', 'grid_count'))
        assert grids and min(grids) > 50_000, f"no large grid arrived: {grids} {errs}"
        assert cam is not None and cam.get("format") == "jpeg" and cam_n > 20_000, (cam_n, cam and cam.get("error"))  # fmt: skip
        assert cam_dt < 5.0, cam_dt
        assert rate > 0.7 * THROTTLE_BPS, f'link not saturated ({rate:.0f} B/s) {errs}'
        assert all(x is not None and x < 3.0 for x in lat1), lat1
        assert rep is not None and rep["ok"] and rep["dt"] < 4.0, rep
        assert gap1 < 3.0, gap1
        assert all(x is not None and x < 2.0 for x in lat2), lat2
        assert gap2 < 1.0, gap2
    finally:
        h1.send(cmd="grid", side=0)
        h1.send(cmd="camera", w=0, h=0)
        px.resume()
    assert fleet.plan_round_trip("r1", "thr-after", 15) is not None
    _no_non_droppable(fleet)


def wait(pred, timeout):
    from harness import wait as w

    return w(pred, timeout)


def max_gap(times, t0, t1):
    from harness import max_gap as mg

    return mg(times, t0, t1)


def test_stall_recovers_without_transport_close(fleet):
    """r1's radio stalls for STALL_S (sockets open, nothing moves: TCP head-of-line
    at its worst).  r2 keeps working throughout; no zenoh transport is closed for a
    non-droppable message (#1876); C1 state is back within RESUME_S of the end; a
    plan sent during the stall (inside its TTL) runs exactly once after it."""
    assert STALL_S >= 12, "FLEETLINK_STALL_S must be >= 12"
    px = fleet.robots["r1"]["proxy"]
    h1 = fleet.robots["r1"]["helper"]
    accepted0 = px.accepted
    mr = h1.mark()
    t0 = time.time()
    px.stall()
    try:
        time.sleep(STALL_S / 2)
        lat2 = fleet.plan_round_trip("r2", "stall-r2", 5)
        # the plan's TTL is 30 s: send it late enough to still be valid at the end
        time.sleep(max(0.0, t0 + STALL_S - 10.0 - time.time()))
        mb = fleet.base.mark()
        fleet.base.send(cmd="plan", robot="r1", plan_id="stall-r1")
        time.sleep(max(0.0, t0 + STALL_S - time.time()))
        assert not h1.find(lambda e: e.get("ev") == "plan", since=mr), "stall leaked"
    finally:
        t1 = time.time()
        px.resume()
    gap2 = max_gap(fleet.state_times["r2"], t0, t1)
    ok = fleet.base.wait_for(lambda e: e.get("ev") == "result" and e.get("plan_id") == "stall-r1", 20, since=mb)  # fmt: skip
    after = [t for t in fleet.state_times["r1"] if t > t1]
    resume = (after[0] - t1) if after else None
    time.sleep(2.0)
    runs = h1.find(lambda e: e.get("ev") == "plan" and e.get("plan_id") == "stall-r1", since=mr)  # fmt: skip
    print(f"\nstall {STALL_S:.0f} s: r1 C1 back after {resume} s, plan result "
          f"{ok and ok['rx'] - t1:.2f} s after; r2 plan rt {lat2}, r2 C1 max gap {gap2:.2f} s; "
          f"r1 link reconnects {px.accepted - accepted0} (zenoh lease expiry)")  # fmt: skip
    assert lat2 is not None and lat2 < 2.0, lat2
    assert gap2 < 1.0, gap2
    assert resume is not None and resume < RESUME_S, resume
    assert ok is not None, "plan sent during the stall never completed"
    assert len(runs) == 1, runs
    _no_non_droppable(fleet)


def test_fleetlink_router_restart_mid_plan(fleet):
    """Kill -9 the fleetlink router right after r1 received a plan (before its ack
    and result went up), restart it: the plan is not executed again, its result
    still reaches the base, and both robots work afterwards."""
    h1 = fleet.robots["r1"]["helper"]
    h1.send(cmd="exec_delay", s=2.0)
    try:
        mr, mb = h1.mark(), fleet.base.mark()
        fleet.base.send(cmd="plan", robot="r1", plan_id="restart-1")
        got = h1.wait_for(lambda e: e.get("ev") == "plan" and e.get("plan_id") == "restart-1", 10, since=mr)  # fmt: skip
        assert got is not None
        t0 = time.time()
        fleet.restart_fleetlink_router(down_s=3.0)
        res = fleet.base.wait_for(lambda e: e.get("ev") == "result" and e.get("plan_id") == "restart-1", 30, since=mb)  # fmt: skip
        assert res is not None, "result lost across the router restart\n" + fleet.dump()
        time.sleep(3.0)  # any retry of the plan would have landed by now
        runs = h1.find(lambda e: e.get("ev") == "plan" and e.get("plan_id") == "restart-1", since=mr)  # fmt: skip
        print(f"\nrouter restart: result {res['rx'] - t0:.2f} s after the kill, "
              f"plan received {len(runs)}x")  # fmt: skip
        assert len(runs) == 1, runs
    finally:
        h1.send(cmd="exec_delay", s=fleet.exec_delay)
    for r in fleet.robot_names:
        assert fleet.plan_round_trip(r, f"after-restart-{r}", 15) is not None, r
    _no_non_droppable(fleet)
