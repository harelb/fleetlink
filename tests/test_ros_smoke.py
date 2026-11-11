"""AgentNode (rclpy) → MemoryTransport → Receiver, fed by a real hydra_ros.DsgPublisher.

rmw_zenoh sessions are pointed at the throwaway router (never the live 7447 one)
and isolated with ROS_DOMAIN_ID=88.
"""

import threading
import time

import numpy as np
import pytest
from helpers import synthetic_mesh

from fleetlink import codec
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator

pytestmark = pytest.mark.integration


def _synthetic_graph(dsg):
    G = dsg.DynamicSceneGraph()
    for i in range(20):
        a = dsg.ObjectNodeAttributes()
        a.position = np.array([float(i), 0.5 * i, 0.0])
        a.semantic_label = i % 5
        a.name = f"obj{i}"
        G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", i), a)
    for i in range(19):
        G.insert_edge(dsg.NodeSymbol("O", i), dsg.NodeSymbol("O", i + 1))
    pos, col, faces = synthetic_mesh(200, extent=10)
    m = dsg.Mesh()
    m.set_vertices(np.vstack([pos.T, col.T / 255.0]).astype(np.float64))
    m.set_faces(faces.T.astype(np.int32))
    G.mesh = m
    return G


def test_agent_node_streams_ros_graph(zenoh_router, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "88")
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    dsg = pytest.importorskip("spark_dsg")
    hydra_ros = pytest.importorskip("hydra_ros")
    tf2_ros = pytest.importorskip("tf2_ros")
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node

    from fleetlink.agent import AgentNode

    rclpy.init()
    link = LinkEmulator(clock=time.time, bandwidth=None, latency=0.0)
    rx = Receiver(link.base, None, threaded=False)
    rx.start()
    pub_node = Node("dsg_smoke_pub", namespace="smoke")
    agent_node = Node("fleetlink_agent", namespace="smoke")
    agent = AgentNode(transport=link.robot, node=agent_node)
    G = _synthetic_graph(dsg)
    publisher = hydra_ros.DsgPublisher(pub_node, "hydra/backend/dsg")
    tfb = tf2_ros.StaticTransformBroadcaster(pub_node)
    tf = TransformStamped()
    tf.header.frame_id = "map"
    tf.child_frame_id = "smoke/body"
    tf.transform.translation.x = 3.0
    tf.transform.rotation.w = 1.0
    tfb.sendTransform(tf)
    ex = MultiThreadedExecutor()
    ex.add_node(pub_node)
    ex.add_node(agent_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    try:
        expected = codec.graph_records(G)
        deadline = time.time() + 30.0
        last_pub = 0.0
        ok = False
        while time.time() < deadline:
            if time.time() - last_pub > 1.0:
                publisher.publish(G)
                last_pub = time.time()
            link.advance()
            rx.tick()
            time.sleep(0.05)
            if "smoke" not in rx.robots():
                continue
            st = rx.state("smoke")
            if (
                set(rx.graph("smoke").records) == set(expected)
                and len(rx.graph("smoke").edges) == 19
                and rx.tiles("smoke")
                and (st.get("pose") or {}).get("x") == 3.0
            ):
                ok = True
                break
        assert ok, (rx.robots(), agent.graphs_received, agent.graphs_processed)
        assert rx.state("smoke")["pose"]["frame"] == "map"
        assert agent.graphs_processed >= 1
    finally:
        agent.shutdown()
        ex.shutdown()
        pub_node.destroy_node()
        agent_node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


def test_agent_node_grid_paths_cameras(zenoh_router, monkeypatch):
    """The ROS side of the RViz-replacement streams: OccupancyGrid, Path,
    MarkerArray and Image topics (relative to the namespace) reach AgentCore."""
    monkeypatch.setenv("ROS_DOMAIN_ID", "89")
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("hydra_msgs")
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from visualization_msgs.msg import Marker, MarkerArray

    from fleetlink import media
    from fleetlink.agent import AgentNode

    rclpy.init()
    link = LinkEmulator(clock=time.time, bandwidth=None, latency=0.0)
    pub_node = Node("rviz_smoke_pub", namespace="smoke2")
    agent_node = Node("fleetlink_agent", namespace="smoke2")
    agent = AgentNode(transport=link.robot, node=agent_node)
    grid_pub = pub_node.create_publisher(OccupancyGrid, "hydra/tsdf/occupancy", 1)
    path_pub = pub_node.create_publisher(
        Path, "spot_executor_node/smooth_path_publisher", 1
    )
    tgt_pub = pub_node.create_publisher(
        MarkerArray, "spot_executor_node/mlp_target_publisher", 1
    )
    img_pub = pub_node.create_publisher(
        Image, "smoke2_zed/rgb/image_rect_color", 1
    )  # <robot>_zed
    g = OccupancyGrid()
    g.header.frame_id = "smoke2/map"
    g.info.resolution = 0.25
    g.info.width, g.info.height = 4, 3
    g.info.origin.position.x = -1.0
    g.info.origin.orientation.w = 1.0
    g.data = [-1, 0, 100, 50] * 3
    path = Path()
    path.header.frame_id = "smoke2/map"
    for i in range(300):
        ps = PoseStamped()
        ps.pose.position.x = 0.1 * i
        path.poses.append(ps)
    ma = MarkerArray()
    mk = Marker()
    mk.header.frame_id = "smoke2/map"
    mk.pose.position.x, mk.pose.position.y = 4.0, 5.0
    ma.markers.append(mk)
    im = Image()
    im.encoding, im.width, im.height, im.step = "mono8", 8, 6, 8
    im.data = bytes(range(48))
    # relocalization: map -> the scene-graph frame (no DSG yet: the fallback
    # smoke2/map) goes out as ``frame_align``
    import tf2_ros
    from geometry_msgs.msg import TransformStamped

    tfb = tf2_ros.StaticTransformBroadcaster(pub_node)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "map", "smoke2/map"
    tf.transform.translation.x = 2.0
    tf.transform.rotation.w = 1.0
    tfb.sendTransform(tf)
    ex = MultiThreadedExecutor()
    ex.add_node(pub_node)
    ex.add_node(agent_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()
    core = agent.core
    try:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            for pub, msg in (
                (grid_pub, g),
                (path_pub, path),
                (tgt_pub, ma),
                (img_pub, im),
            ):
                pub.publish(msg)
            time.sleep(0.2)
            if (
                core._grid
                and core._paths["smooth"] is not None
                and core._paths["target"] is not None
                and "rgb" in core._frames
                and (core.frame_align or {}).get("ok")
            ):
                break
        assert core._grid and core._grid["width"] == 4
        enc = media.encode_grid(**{k: core._grid[k] for k in
                                   ("frame", "resolution", "origin", "width", "height", "data")})  # fmt: skip
        assert media.decode_grid(enc).ravel().tolist() == [-1, 0, 100, 50] * 3
        assert enc["origin"] == [-1.0, 0.0, 0.0, 0.0]
        assert len(core._paths["smooth"]) == 300 and core._paths["target"] == [
            4.0,
            5.0,
            0.0,
        ]
        assert core.camera_names == ["rgb", "overlay"]
        fa = core._state_payload()["frame_align"]
        assert (fa["parent"], fa["child"], fa["ok"]) == ("map", "smoke2/map", True)
        assert fa["T"] == [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        rep = media.camera_reply(core._frames["rgb"], 640, 70, time.time())
        assert rep["width"] == 8 and rep["height"] == 6
    finally:
        agent.shutdown()
        ex.shutdown()
        pub_node.destroy_node()
        agent_node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


def test_agent_node_lazy_camera_subscriptions(zenoh_router, monkeypatch):
    """Non-eager cameras are subscribed only while wanted, then released."""
    monkeypatch.setenv("ROS_DOMAIN_ID", "90")
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("hydra_msgs")
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import Image

    from fleetlink.agent import AgentNode

    rclpy.init()
    link = LinkEmulator(clock=time.time, bandwidth=None, latency=0.0)
    pub_node = Node("lazy_cam_pub", namespace="smoke3")
    agent_node = Node(
        "fleetlink_agent",
        namespace="smoke3",
        parameter_overrides=[
            Parameter("cameras", value=["rgb=rgb/image", "left=left/image"]),
            Parameter("camera_idle_s", value=2.0),
            Parameter("store_path", value=""),
        ],
    )
    agent = AgentNode(transport=link.robot, node=agent_node)
    left = pub_node.create_publisher(Image, "left/image", 1)
    rgb = pub_node.create_publisher(Image, "rgb/image", 1)

    def subs(pub, want, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if pub.get_subscription_count() == want:
                return True
            time.sleep(0.05)
        return False

    try:
        assert agent.core.store is None
        assert subs(rgb, 1) and subs(left, 0)
        agent.core.want_camera("left")
        agent._update_cameras()
        assert subs(left, 1)
        time.sleep(2.2)
        agent._update_cameras()
        assert subs(left, 0) and subs(rgb, 1)
    finally:
        agent.shutdown()
        pub_node.destroy_node()
        agent_node.destroy_node()
        rclpy.shutdown()
