"""AgentNode in odometric mesh mode, fed real ROS messages like Hydra's:
``hydra/frontend/full_mesh_update`` (kimera_pgmo_msgs/MeshDelta, transient
local), ``hydra/frontend/mesh_graph_incremental`` and ``hydra/backend/mesh_graph``
(pose_graph_tools_msgs/PoseGraph). rmw_zenoh is pointed at the throwaway
router and isolated with ROS_DOMAIN_ID=90."""

import threading
import time

import numpy as np
import pytest

from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator

pytestmark = pytest.mark.integration


def _delta_msg(MeshDelta, Vertex, Tri, pos, stamps, faces, seq):
    m = MeshDelta()
    m.header.frame_id = "smoke/odom"
    m.seq_number = seq
    m.prev_active_vertices = 0
    m.prev_active_faces = 0
    m.num_archived_vertices = len(pos)
    for p, s in zip(pos, stamps):
        v = Vertex()
        v.pos.x, v.pos.y, v.pos.z = (float(x) for x in p)
        v.has_color = True
        v.color.r = v.color.g = v.color.b = 0.5
        v.color.a = 1.0
        v.has_stamp = True
        v.stamp = int(s)
        m.vertex_updates.append(v)
    for f in faces:
        t = Tri()
        t.vertex_indices = [int(x) for x in f]
        m.face_archive_updates.append(t)
    return m


def _pose_graph(PoseGraph, PoseGraphNode, items):
    g = PoseGraph()
    for i, stamp, (x, y, z, qx, qy, qz, qw) in items:
        n = PoseGraphNode()
        n.key = int(i)
        n.header.stamp.sec, n.header.stamp.nanosec = int(stamp) // 1_000_000_000, int(stamp) % 1_000_000_000
        n.pose.position.x, n.pose.position.y, n.pose.position.z = float(x), float(y), float(z)
        n.pose.orientation.x, n.pose.orientation.y = float(qx), float(qy)
        n.pose.orientation.z, n.pose.orientation.w = float(qz), float(qw)
        g.nodes.append(n)
    return g


def test_agent_node_odometric_mesh(zenoh_router, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", "90")
    monkeypatch.setenv(
        "ZENOH_CONFIG_OVERRIDE",
        f'connect/endpoints=["{zenoh_router}"];scouting/multicast/enabled=false',
    )
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("hydra_msgs")
    km = pytest.importorskip("kimera_pgmo_msgs.msg")
    pg = pytest.importorskip("pose_graph_tools_msgs.msg")
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile

    from fleetlink.agent import AgentNode

    rclpy.init()
    link = LinkEmulator(clock=time.time, bandwidth=None, latency=0.0)
    rx = Receiver(link.base, None, threaded=False)
    rx.start()
    pub = Node("odom_mesh_pub", namespace="smoke")
    agent_node = Node(
        "fleetlink_agent",
        namespace="smoke",
        parameter_overrides=[Parameter("mesh_source", value="odometric"), Parameter("mesh_period_s", value=0.5)],
    )
    latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    delta_pub = pub.create_publisher(km.MeshDelta, "hydra/frontend/full_mesh_update", latched)
    inc_pub = pub.create_publisher(pg.PoseGraph, "hydra/frontend/mesh_graph_incremental", latched)
    graph_pub = pub.create_publisher(pg.PoseGraph, "hydra/backend/mesh_graph", 10)
    # a 20 x 20 grid of vertices over 10 m, stamped 0.1 s apart
    n = 20
    xs, ys = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 10, n))
    pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], 1)
    stamps = 1_000_000_000_000 + np.arange(n * n) * 100_000_000
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    # published BEFORE the agent exists: transient local must deliver it
    delta_pub.publish(_delta_msg(km.MeshDelta, km.Vertex, km.TriangleIndices, pos, stamps, faces, 0))
    ctrl = [(k, int(stamps[0] + k * 2_000_000_000), (2.5 * (k % 5), 2.5 * (k // 5), 0.0, 0, 0, 0, 1)) for k in range(25)]
    inc_pub.publish(_pose_graph(pg.PoseGraph, pg.PoseGraphNode, ctrl))
    agent = AgentNode(transport=link.robot, node=agent_node)
    ex = MultiThreadedExecutor()
    ex.add_node(pub)
    ex.add_node(agent_node)
    spin = threading.Thread(target=ex.spin, daemon=True)
    spin.start()

    def pump(until, seconds=20.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            link.advance()
            rx.tick()
            time.sleep(0.05)
            if until():
                return True
        return False

    try:
        assert agent.core.odom is not None
        got_odom = pump(lambda: "smoke" in rx.robots() and any(t.get("odom") for t in rx.raw_tiles("smoke").values()))
        assert got_odom, agent.core.stats()["mesh"]
        raw = rx.raw_tiles("smoke")
        assert any(k.endswith("~ctrl") for k in raw)  # the control points arrived
        before = {k: t["hash"] for k, t in rx.tiles("smoke").items()}
        # loop closure: every control point shifted 1 m in x
        moved = [(k, s, (p[0] + 1.0, p[1], p[2], 0, 0, 0, 1)) for k, s, p in ctrl]
        deadline = time.time() + 20
        ok = False
        while time.time() < deadline and not ok:
            graph_pub.publish(_pose_graph(pg.PoseGraph, pg.PoseGraphNode, moved))
            ok = pump(lambda: any(rx.tiles("smoke")[k]["hash"] != h for k, h in before.items() if k in rx.tiles("smoke")), 1.0)
        assert ok, "no deformed tiles after the mesh_graph update"
        shown = rx.tiles("smoke")
        t = next(t for t in shown.values() if t["nv"])
        p = np.frombuffer(t["pos"], np.int16).reshape(-1, 3) * t["scale"] + np.asarray(t["origin"])
        raw_t = rx.raw_tiles("smoke")[t["key"]]
        p0 = np.frombuffer(raw_t["pos"], np.int16).reshape(-1, 3) * raw_t["scale"] + np.asarray(raw_t["origin"])
        assert np.allclose(p[:, 0] - p0[:, 0], 1.0, atol=0.02)  # shifted like the control points
        assert agent.core.stats()["mesh"]["source"] == "odometric"
        # a gap in the delta sequence: the agent goes back to the DSG mesh
        delta_pub.publish(_delta_msg(km.MeshDelta, km.Vertex, km.TriangleIndices, pos, stamps, faces, 5))
        assert pump(lambda: agent.core.stats()["mesh"]["source"] == "dsg", 10.0)
        assert "gap" in agent.core.stats()["mesh"]["fallback"]
    finally:
        agent.shutdown()
        ex.shutdown()
        pub.destroy_node()
        agent_node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)
