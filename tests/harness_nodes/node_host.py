#!/usr/bin/env python3
"""Fake robot-side / base-side ROS nodes for the multi-router fleet harness
(``tests/harness.py``).  One process per graph, driven by JSON lines on stdin and
reporting JSON lines on stdout (``{"ev": ...}``).

``--role robot --robot r1``: a fake executor (counts ``compiled_plan_out``
receipts per plan id and publishes one ``action_result`` per receipt after
``--exec-delay`` seconds), a ``map -> r1/base_link`` TF publisher, an
``update_holding_state`` client, an optional random occupancy-grid load and an
optional random camera image (the agent's ``rgb`` camera).

``--role base --robots r1,r2 --base willow``: plan publishers, ``action_result``
subscribers, a tf2 buffer, the ``/willow/update_holding_state`` server and a
``/fleetlink/link_status`` subscriber.

Everything that touches rclpy entities runs on the executor thread (a 50 Hz
timer drains the command queue): rmw_zenoh has had heap corruption publishing
from foreign threads.
"""

import argparse
import json
import queue
import sys
import threading
import time

_out_lock = threading.Lock()


def emit(**kw):
    kw.setdefault("t", time.time())
    with _out_lock:
        sys.stdout.write(json.dumps(kw, default=str) + "\n")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=("robot", "base"), required=True)
    ap.add_argument("--robot", default="")
    ap.add_argument("--robots", default="")
    ap.add_argument("--base", default="willow")
    ap.add_argument("--exec-delay", type=float, default=0.5)
    ap.add_argument("--tf-x", type=float, default=1.0)
    ap.add_argument("--harness-tag", default="")  # only to find strays
    args = ap.parse_args()

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    rclpy.init()
    name = f"harness_{args.role}" + (f"_{args.robot}" if args.robot else "")
    node = rclpy.create_node(name)
    cmds: queue.Queue = queue.Queue()
    handlers = {}

    def reliable(depth):
        return QoSProfile(
            depth=depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

    def nodes(_cmd):
        emit(ev="nodes", nodes=[[n, ns] for n, ns in node.get_node_names_and_namespaces()])

    handlers["nodes"] = nodes

    if args.role == "robot":
        _robot(node, args, handlers, reliable)
    else:
        _base(node, args, handlers, reliable)

    def drain():
        while True:
            try:
                cmd = cmds.get_nowait()
            except queue.Empty:
                return
            fn = handlers.get(cmd.get("cmd"))
            if fn is None:
                emit(ev="error", err=f"unknown cmd {cmd}")
                continue
            try:
                fn(cmd)
            except Exception as exc:  # noqa: BLE001
                emit(ev="error", err=f"{cmd.get('cmd')}: {type(exc).__name__}: {exc}")

    node.create_timer(0.02, drain)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    emit(ev="ready", node=name)
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "quit":
                break
            try:
                cmds.put(json.loads(line))
            except ValueError:
                emit(ev="error", err=f"bad json {line!r}")
    finally:
        ex.shutdown(timeout_sec=1)
        node.destroy_node()
        rclpy.try_shutdown()


def _robot(node, args, handlers, reliable):
    import numpy as np
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid

    r = args.robot
    receipts: dict[str, int] = {}
    try:
        from robot_executor_msgs.msg import ActionResultMsg, ActionSequenceMsg

        res_pub = node.create_publisher(
            ActionResultMsg, f"/{r}/spot_executor_node/action_result", reliable(10)
        )

        def on_plan(msg):
            receipts[msg.plan_id] = receipts.get(msg.plan_id, 0) + 1
            emit(ev="plan", robot=r, plan_id=msg.plan_id, n=receipts[msg.plan_id])

            def finish(pid=msg.plan_id):
                timer.cancel()
                res_pub.publish(
                    ActionResultMsg(plan_id=pid, action_index=0, status="SUCCESS")
                )
                emit(ev="result_sent", robot=r, plan_id=pid)

            timer = node.create_timer(args.exec_delay, finish)

        node.create_subscription(
            ActionSequenceMsg, f"/{r}/omniplanner_node/compiled_plan_out", on_plan, reliable(1)
        )
    except ImportError:
        emit(ev="warn", err="robot_executor_msgs unavailable")

    tfb = tf2_ros.TransformBroadcaster(node)

    def pub_tf():
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = "map", f"{r}/base_link"
        t.header.stamp = node.get_clock().now().to_msg()
        t.transform.translation.x = args.tf_x
        t.transform.rotation.w = 1.0
        tfb.sendTransform(t)

    node.create_timer(0.1, pub_tf)

    try:
        from heracles_ros_interfaces.srv import UpdateHoldingState

        hold_cli = node.create_client(UpdateHoldingState, f"/{r}/update_holding_state")

        def hold(cmd):
            t0 = time.time()
            fut = hold_cli.call_async(
                UpdateHoldingState.Request(is_holding=True, id=cmd.get("id", "O(1)"))
            )

            def done(f):
                try:
                    res = f.result()
                    emit(ev="hold_reply", robot=r, ok=bool(res.success), dt=time.time() - t0)
                except Exception as exc:  # noqa: BLE001
                    emit(ev="hold_reply", robot=r, ok=False, err=str(exc))

            fut.add_done_callback(done)

        handlers["hold"] = hold
    except ImportError:
        emit(ev="warn", err="heracles_ros_interfaces unavailable")

    grid = {"timer": None}
    grid_pub = node.create_publisher(OccupancyGrid, f"/{r}/hydra/tsdf/occupancy", reliable(1))
    rng = np.random.default_rng(0)

    def grid_load(cmd):
        if grid["timer"] is not None:
            grid["timer"].cancel()
            grid["timer"] = None
        side = int(cmd.get("side", 0))
        if side <= 0:
            return

        def pub():
            m = OccupancyGrid()
            m.header.frame_id = f"{r}/map"
            m.header.stamp = node.get_clock().now().to_msg()
            m.info.resolution = 0.1
            m.info.width = m.info.height = side
            m.info.origin.orientation.w = 1.0
            # random 0..100: barely compressible, so every grid is ~side^2 bytes
            m.data = rng.integers(0, 101, side * side, dtype=np.int8).tolist()
            grid_pub.publish(m)
            grid["n"] = grid.get("n", 0) + 1

        grid["timer"] = node.create_timer(float(cmd.get("period", 1.0)), pub)

    handlers["grid"] = grid_load

    from sensor_msgs.msg import Image

    cam = {"timer": None}
    cam_pub = node.create_publisher(Image, f"/{r}/{r}_zed/rgb/image_rect_color", reliable(1))

    def camera_load(cmd):
        """Publish a random rgb8 frame (barely compressible) every ``period`` s."""
        if cam["timer"] is not None:
            cam["timer"].cancel()
            cam["timer"] = None
        w, h = int(cmd.get("w", 0)), int(cmd.get("h", 0))
        if w <= 0 or h <= 0:
            return

        def pub():
            m = Image()
            m.header.frame_id = f"{r}/camera"
            m.header.stamp = node.get_clock().now().to_msg()
            m.width, m.height, m.encoding, m.step = w, h, "rgb8", 3 * w
            m.data = rng.integers(0, 256, w * h * 3, dtype=np.uint8).tobytes()
            cam_pub.publish(m)

        pub()
        cam["timer"] = node.create_timer(float(cmd.get("period", 1.0)), pub)

    handlers["camera"] = camera_load
    handlers["grid_count"] = lambda _c: emit(ev="grid_count", n=grid.get("n", 0), subs=grid_pub.get_subscription_count())
    handlers["exec_delay"] = lambda c: setattr(args, "exec_delay", float(c["s"]))
    handlers["receipts"] = lambda _c: emit(ev="receipts", robot=r, receipts=dict(receipts))


def _base(node, args, handlers, reliable):
    import tf2_ros
    from rclpy.time import Time
    from std_msgs.msg import String

    robots = [x for x in args.robots.split(",") if x]
    plan_pubs = {}
    try:
        from robot_executor_msgs.msg import ActionResultMsg, ActionSequenceMsg

        for r in robots:
            plan_pubs[r] = node.create_publisher(
                ActionSequenceMsg, f"/{r}/omniplanner_node/compiled_plan_out", reliable(1)
            )
            node.create_subscription(
                ActionResultMsg,
                f"/{r}/spot_executor_node/action_result",
                lambda m, r=r: emit(ev="result", robot=r, plan_id=m.plan_id, status=m.status),
                reliable(10),
            )

        def plan(cmd):
            r = cmd["robot"]
            pub = plan_pubs[r]
            emit(ev="plan_sent", robot=r, plan_id=cmd["plan_id"], subs=pub.get_subscription_count())
            pub.publish(ActionSequenceMsg(plan_id=cmd["plan_id"], robot_name=r))

        handlers["plan"] = plan
        handlers["plan_subs"] = lambda _c: emit(
            ev="plan_subs", subs={r: p.get_subscription_count() for r, p in plan_pubs.items()}
        )
    except ImportError:
        emit(ev="warn", err="robot_executor_msgs unavailable")

    buf = tf2_ros.Buffer()
    node._harness_tfl = tf2_ros.TransformListener(buf, node)

    def tf(_cmd):
        out = {}
        for r in robots:
            try:
                tr = buf.lookup_transform("map", f"{r}/base_link", Time())
                out[r] = tr.transform.translation.x
            except Exception:  # noqa: BLE001
                out[r] = None
        emit(ev="tf", tf=out)

    handlers["tf"] = tf

    try:
        from heracles_ros_interfaces.srv import UpdateHoldingState

        def serve(req, resp):
            emit(ev="hold", id=req.id, is_holding=bool(req.is_holding))
            resp.success = True
            return resp

        node.create_service(UpdateHoldingState, f"/{args.base}/update_holding_state", serve)
    except ImportError:
        emit(ev="warn", err="heracles_ros_interfaces unavailable")

    def on_status(m):
        try:
            s = json.loads(m.data)
        except ValueError:
            return
        emit(ev="link_status", robot=s.get("robot"), connected=s.get("connected"),
             rtt=s.get("rtt"), cfg_match=s.get("cfg_match"))  # fmt: skip

    node.create_subscription(String, "/fleetlink/link_status", on_status, 10)


if __name__ == "__main__":
    main()
