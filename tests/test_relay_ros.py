"""Generic ROS relay end to end: a robot ROS graph (domain 90) and a base ROS graph
(domain 91) that never see each other, joined only by fleetlink over a throwaway
zenoh router (never the live 7447 one).

Robot: ``AgentNode`` with ``relay_config`` (so ``RobotRelay``) + test publishers /
servers.  Base: ``BaseBridge`` + test publishers / servers.  Uses the real
``config/relay.yaml``.
"""

import json
import os
import threading
import time

import pytest

pytestmark = pytest.mark.integration

RELAY_YAML = os.path.join(os.path.dirname(__file__), "..", "config", "relay.yaml")
ROBOT_DOMAIN, BASE_DOMAIN = 90, 91
PROXY_DOMAIN = 92


def _wait(pred, timeout, step=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        if step is not None:
            step()
        time.sleep(0.1)
    return pred()


def _call(client, req, timeout=15.0):
    if not client.wait_for_service(timeout_sec=timeout):
        return None
    fut = client.call_async(req)
    done = threading.Event()
    fut.add_done_callback(lambda _f: done.set())
    done.wait(timeout)
    return fut.result() if fut.done() else None


@pytest.fixture
def ros_pair(zenoh_router, monkeypatch):
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("zenoh")
    pytest.importorskip("hydra_msgs")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("robot_executor_msgs.msg")
    pytest.importorskip("heracles_ros_interfaces.srv")
    pytest.importorskip("dcist_launch_system_msgs.srv")
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.signals import SignalHandlerOptions

    from fleetlink.agent import AgentNode
    from fleetlink.relay import RelayConfig
    from fleetlink.relay_ros import BaseBridge
    from fleetlink.transport import ZenohTransport

    ctxs, executors, threads, closers = [], [], [], []

    def context(domain):
        ctx = rclpy.Context()
        rclpy.init(
            context=ctx,
            domain_id=domain,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        ex = MultiThreadedExecutor(num_threads=6, context=ctx)
        ctxs.append(ctx)
        executors.append(ex)
        return ctx, ex

    rctx, rex = context(ROBOT_DOMAIN)
    bctx, bex = context(BASE_DOMAIN)
    t_robot = ZenohTransport(zenoh_router)
    t_base = ZenohTransport(zenoh_router)
    assert t_robot.opened.wait(10) and t_base.opened.wait(10)

    agent_node = rclpy.create_node(
        "fleetlink_agent",
        namespace="euclid",
        context=rctx,
        parameter_overrides=[
            Parameter("relay_config", value=RELAY_YAML),
            Parameter("relay_platform", value="spot"),
        ],
    )
    agent = AgentNode(transport=t_robot, node=agent_node)
    assert agent.relay is not None
    robot = rclpy.create_node("robot_test", context=rctx)
    bridge_node = rclpy.create_node("fleetlink_bridge", context=bctx)
    bridge = BaseBridge(
        bridge_node,
        t_base,
        RelayConfig.load(RELAY_YAML),
        {"euclid": "spot"},
        base="willow",
        discover=False,
    )
    base = rclpy.create_node("base_test", context=bctx)
    for ex, nodes in ((rex, (agent_node, robot)), (bex, (bridge_node, base))):
        for n in nodes:
            ex.add_node(n)
        th = threading.Thread(target=ex.spin, daemon=True)
        th.start()
        threads.append(th)
    closers += [agent.shutdown, bridge.shutdown]
    ns = type("RosPair", (), {})()
    ns.robot, ns.base, ns.agent, ns.bridge = robot, base, agent, bridge
    try:
        yield ns
    finally:
        for fn in closers:
            try:
                fn()
            except Exception:
                pass
        for ex in executors:
            ex.shutdown(timeout_sec=2)
        for n in (agent_node, robot, bridge_node, base):
            n.destroy_node()
        t_robot.close()
        t_base.close()
        for ctx in ctxs:
            ctx.try_shutdown()
        for th in threads:
            th.join(timeout=2)


def test_relay_topics_services_tf_and_link_status(ros_pair):
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from heracles_ros_interfaces.srv import UpdateHoldingState
    from rclpy.time import Time
    from robot_executor_msgs.msg import ActionResultMsg, ActionSequenceMsg
    from std_msgs.msg import String

    from dcist_launch_system_msgs.srv import SaveDsg

    robot, base, bridge = ros_pair.robot, ros_pair.base, ros_pair.bridge
    core = bridge.core("euclid")
    # clients of the real base servers exist up front (created on the executor /
    # before spinning, never lazily on zenoh's callback thread)
    assert "/willow/update_holding_state" in bridge._clients
    assert _wait(lambda: core.connected(), 15), "relay heartbeats never arrived"

    # graphs are isolated: the base cannot see the robot's agent node
    assert ("fleetlink_agent", "/euclid") not in base.get_node_names_and_namespaces()

    # -- down: ActionSequenceMsg (latest_reliable) --------------------------------
    plan_topic = "/euclid/omniplanner_node/compiled_plan_out"
    plans = []
    robot.create_subscription(ActionSequenceMsg, plan_topic, plans.append, 1)
    plan_pub = base.create_publisher(ActionSequenceMsg, plan_topic, 1)
    n = [0]

    def send_plan():
        n[0] += 1
        plan_pub.publish(ActionSequenceMsg(plan_id=f"p{n[0]}", robot_name="euclid"))

    assert _wait(lambda: plans, 20, step=send_plan), core.status()
    assert plans[-1].robot_name == "euclid" and plans[-1].plan_id.startswith("p")
    st = core.status()["topics"][plan_topic]
    assert st["acked"] >= 1 and st["subs"] >= 1  # robot-side subscriber reported back

    # -- up: action_result (reliable) ----------------------------------------------
    res_topic = "/euclid/spot_executor_node/action_result"
    results = []
    base.create_subscription(ActionResultMsg, res_topic, results.append, 10)
    res_pub = robot.create_publisher(ActionResultMsg, res_topic, 10)

    def send_result():
        res_pub.publish(ActionResultMsg(plan_id="p1", action_index=2, status="SUCCESS"))

    assert _wait(lambda: results, 20, step=send_result)
    assert results[0].plan_id == "p1" and results[0].action_index == 2

    # -- service, base calls robot: /euclid/dsg_saver/save_dsg ----------------------
    saved = []

    def save(req, resp):
        saved.append(req.save_path)
        resp.success = True
        return resp

    robot.create_service(SaveDsg, "/euclid/dsg_saver/save_dsg", save)
    cli = base.create_client(SaveDsg, "/euclid/dsg_saver/save_dsg")
    resp = _call(cli, SaveDsg.Request(save_path="/tmp/x.json", include_mesh=True), 30)
    assert resp is not None and resp.success and saved == ["/tmp/x.json"]

    # -- service, robot calls base: update_holding_state (served as /willow/...) ----
    held = []

    def hold(req, resp):
        held.append((req.is_holding, req.id))
        resp.success = True
        return resp

    base.create_service(UpdateHoldingState, "/willow/update_holding_state", hold)
    cli = robot.create_client(UpdateHoldingState, "/euclid/update_holding_state")
    resp = _call(cli, UpdateHoldingState.Request(is_holding=True, id="O(3)"), 30)
    assert resp is not None and resp.success and held == [(True, "O(3)")]

    # -- TF: composed map -> euclid/base_link + static euclid/* on the base --------
    tfb = tf2_ros.TransformBroadcaster(robot)
    stb = tf2_ros.StaticTransformBroadcaster(robot)

    def tf(parent, child, x):
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = parent, child
        t.header.stamp = robot.get_clock().now().to_msg()
        t.transform.translation.x = x
        t.transform.rotation.w = 1.0
        return t

    stb.sendTransform(
        [
            tf("euclid/odom", "euclid/base_link", 1.0),
            tf("euclid/base_link", "euclid/camera", 0.5),
        ]
    )
    robot.create_timer(0.1, lambda: tfb.sendTransform(tf("map", "euclid/odom", 2.0)))
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, base)

    def lookup(child):
        try:
            return buf.lookup_transform("map", child, Time()).transform.translation.x
        except Exception:
            return None

    assert _wait(lambda: lookup("euclid/camera") is not None, 20)
    assert lookup("euclid/base_link") == pytest.approx(3.0)
    assert lookup("euclid/camera") == pytest.approx(3.5)

    # -- fleetlink/link_status ------------------------------------------------------
    status = []
    base.create_subscription(
        String,
        "/fleetlink/link_status",
        lambda m: status.append(json.loads(m.data)),
        10,
    )
    assert _wait(lambda: status, 5)
    s = status[-1]
    assert s["robot"] == "euclid" and s["connected"] and s["cfg_match"] is True
    assert s["rtt"] is not None and s["topics"][plan_topic]["acked"] >= 1

    # after all that traffic the two ROS graphs still never saw each other
    names = [n for n, _ in base.get_node_names_and_namespaces()]
    assert "robot_test" not in names and "fleetlink_agent" not in names
    assert "base_test" in names


def test_proxy_calls_never_starve_the_agent(zenoh_router, monkeypatch):
    """A radio dropout while the robot calls a relayed (``dir: up``) service many
    times: at most MAX_PROXY_CALLS calls wait for the base, the rest fail at once,
    calls made once the link is known down fail at once, and the agent's own
    callbacks and C1 ticks keep running throughout."""
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("hydra_msgs")
    pytest.importorskip("heracles_ros_interfaces.srv")
    from heracles_ros_interfaces.srv import UpdateHoldingState
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.signals import SignalHandlerOptions

    from fleetlink import relay_ros
    from fleetlink.agent import AgentNode
    from fleetlink.relay import RelayConfig, RelayCore
    from fleetlink.transport import LinkEmulator

    link = LinkEmulator(clock=time.time, bandwidth=None, latency=0.01)
    puts = []
    put = link.robot.put
    link.robot.put = lambda k, d, c: (puts.append((time.time(), k)), put(k, d, c))[1]
    base = RelayCore("euclid", "base", link.base, RelayConfig.load(RELAY_YAML),
                     base="willow", on_service=lambda s, req, reply: None)  # fmt: skip
    running = [True]

    def pump():
        while running[0]:
            link.advance()
            base.tick()
            time.sleep(0.005)

    ctx = rclpy.Context()
    rclpy.init(context=ctx, domain_id=PROXY_DOMAIN, signal_handler_options=SignalHandlerOptions.NO)  # fmt: skip
    ex = relay_ros.make_executor(ctx)
    assert ex._executor._max_workers == relay_ros.EXECUTOR_THREADS
    agent_node = rclpy.create_node("fleetlink_agent", namespace="euclid", context=ctx,
        parameter_overrides=[Parameter("relay_config", value=RELAY_YAML),
                             Parameter("relay_platform", value="spot")])  # fmt: skip
    agent = AgentNode(transport=link.robot, node=agent_node)
    ticks = []
    agent_node.create_timer(0.05, lambda: ticks.append(time.time()))  # its own group
    caller = rclpy.create_node("caller", context=ctx)
    cex = SingleThreadedExecutor(context=ctx)
    cex.add_node(caller)
    ex.add_node(agent_node)
    threads = [threading.Thread(target=f, daemon=True) for f in (pump, ex.spin, cex.spin)]
    for th in threads:
        th.start()
    try:
        core = agent.relay.core
        assert _wait(core.connected, 10), "relay heartbeats never arrived"
        cli = caller.create_client(UpdateHoldingState, "/euclid/update_holding_state")
        assert cli.wait_for_service(timeout_sec=10)

        link.close_transport(60.0)  # the dropout; the core stays "connected" ~3 s
        n = relay_ros.EXECUTOR_THREADS + 1  # rmw_zenoh queues 10 requests per service
        t0 = time.time()
        done = []
        for i in range(n):
            fut = cli.call_async(UpdateHoldingState.Request(is_holding=True, id=f"O({i})"))
            fut.add_done_callback(lambda f: done.append((time.time(), f.result().success)))
        # all but MAX_PROXY_CALLS come back at once, with the default response
        assert _wait(lambda: len(done) >= n - relay_ros.MAX_PROXY_CALLS, 2.0), len(done)
        assert all(not ok for _, ok in done)
        assert len(done) == n - relay_ros.MAX_PROXY_CALLS  # the others still wait
        # the agent's own callbacks and C1 keep going while those wait
        time.sleep(4.0)
        t1 = time.time()
        assert len(done) == n - relay_ros.MAX_PROXY_CALLS
        win = [t for t in ticks if t0 <= t <= t1]
        assert len(win) >= 0.6 * (t1 - t0) / 0.05, len(win)
        states = [t for t, k in puts if k == "adt4/euclid/state" and t0 <= t <= t1]
        assert len(states) >= 0.6 * 4.0 * (t1 - t0), len(states)  # state_hz 4
        # the link is known down now: a new call fails at once
        assert not core.connected()
        refused = agent.relay.proxy_refused
        late = []
        t2 = time.time()
        f = cli.call_async(UpdateHoldingState.Request(is_holding=True, id="O(late)"))
        f.add_done_callback(lambda f: late.append(time.time() - t2))
        assert _wait(lambda: late, 1.0) and late[0] < 1.0
        assert agent.relay.proxy_refused == refused + 1
    finally:
        running[0] = False
        agent.shutdown()
        base.close()
        ex.shutdown(timeout_sec=2)
        cex.shutdown(timeout_sec=2)
        for n_ in (agent_node, caller):
            n_.destroy_node()
        ctx.try_shutdown()
