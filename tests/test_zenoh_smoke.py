"""AgentCore + Receiver over real zenoh sessions through a throwaway router."""

import threading
import time

import pytest
from helpers import states_equal, synthetic_mesh, synthetic_records

from fleetlink import codec
from fleetlink.agent import AgentCore
from fleetlink.receiver import Receiver
from fleetlink.transport import ZenohTransport

pytestmark = pytest.mark.integration


def test_zenoh_end_to_end(zenoh_router):
    pytest.importorskip("zenoh")
    try:
        robot_tr = ZenohTransport(zenoh_router)
        base_tr = ZenohTransport(zenoh_router)
    except Exception as exc:
        pytest.skip(f"zenoh session failed: {exc}")
    rx = Receiver(base_tr, None, tick_period=0.05)
    rx.start()
    agent = AgentCore("zsmoke", robot_tr, {"budgets": {"C2": 20_000, "C3": 200_000}})
    recs, edges, ls = synthetic_records(300, seed=4)
    agent.on_records(recs, edges, ls, synthetic_mesh(2000, extent=40))
    agent.on_pose(
        {
            "x": 1.0,
            "y": 2.0,
            "z": 0.0,
            "qw": 1,
            "qx": 0,
            "qy": 0,
            "qz": 0,
            "frame": "map",
            "t": 0,
        }
    )
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            agent.tick()
            time.sleep(0.02)

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        deadline = time.time() + 30.0
        ok = False
        while time.time() < deadline:
            time.sleep(0.25)
            if "zsmoke" not in rx.robots():
                continue
            g_ok = states_equal(rx.graph("zsmoke"), agent.graph_state())[0]
            tiles = {k: t["hash"] for k, t in rx.tiles("zsmoke").items()}
            if g_ok and agent.tiles and tiles == codec.tile_manifest(agent.tiles):
                ok = True
                break
        assert ok, rx.link("zsmoke")
        # C0 round trip: budget change is applied on the robot and acked
        rx.set_budget("zsmoke", "C3", 33_000)
        deadline = time.time() + 5.0
        while time.time() < deadline and rx.pending_requests():
            time.sleep(0.05)
        assert agent.sched.budgets["C3"] == 33_000
        assert not rx.pending_requests()
        link = rx.link("zsmoke")
        assert (
            link["connected"] and link["epoch"] == agent.epoch and link["resyncs"] >= 1
        )
        assert rx.state("zsmoke")["pose"]["x"] == 1.0
    finally:
        stop.set()
        th.join(timeout=2)
        rx.stop()
        robot_tr.close()
        base_tr.close()


def test_zenoh_camera_query_grid_and_paths(zenoh_router):
    """C4 request/reply (declare_queryable + session.get) and the grid / paths
    streams over a real zenoh router."""
    pytest.importorskip("zenoh")
    import numpy as np

    from fleetlink import media

    try:
        robot_tr = ZenohTransport(zenoh_router)
        base_tr = ZenohTransport(zenoh_router)
    except Exception as exc:
        pytest.skip(f"zenoh session failed: {exc}")
    rx = Receiver(base_tr, None, tick_period=0.05)
    rx.start()
    agent = AgentCore("zcam", robot_tr, {"grid_period_s": 0.2})
    img = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    agent.on_camera_frame("rgb", "bgr8", 160, 120, 480, img.tobytes(), stamp=3.0)
    grid = np.zeros((20, 30), np.int8)
    grid[5, 5] = 100
    agent.on_grid("map", 0.2, [1, 2, 0, 0], 30, 20, grid.tobytes())
    agent.on_path("smooth", [[0, 0, 0], [1, 1, 0]], "map", 1.0)
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            agent.tick()
            time.sleep(0.02)

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        rep = None
        deadline = time.time() + 10.0
        while time.time() < deadline and not (rep and "data" in rep):
            rep = rx.camera("zcam", "rgb", width=80, quality=50, timeout=2.0)
        assert rep and rep["format"] == "jpeg" and rep["width"] == 80
        assert rep["data"][:2] == b"\xff\xd8" and rep["stamp"] == 3.0
        assert rx.camera("zcam", "nope", timeout=2.0)["error"] == "unknown camera"
        t0 = time.time()
        assert rx.camera("ghost", "rgb", timeout=0.5) is None
        assert time.time() - t0 < 2.0
        deadline = time.time() + 10.0
        while time.time() < deadline and not (rx.grid("zcam") and rx.paths("zcam")):
            time.sleep(0.05)
        assert np.array_equal(media.decode_grid(rx.grid("zcam")), grid)
        assert rx.paths("zcam")["smooth"] == [[0, 0, 0], [1, 1, 0]]
        assert "rgb" in rx.cameras("zcam")
    finally:
        stop.set()
        th.join(timeout=2)
        rx.stop()
        robot_tr.close()
        base_tr.close()


def test_zenoh_live_video_h264(zenoh_router):
    """Synthetic camera → AgentCore (real ffmpeg h264) → ZenohTransport (C5) →
    Receiver: the frames that arrive start with a key frame and decode."""
    pytest.importorskip("zenoh")
    from test_video import _has_encoder, decode_count, synth

    if not _has_encoder("libx264"):
        pytest.skip("ffmpeg without libx264")
    try:
        robot_tr = ZenohTransport(zenoh_router)
        base_tr = ZenohTransport(zenoh_router)
    except Exception as exc:
        pytest.skip(f"zenoh session failed: {exc}")
    frames = []
    lock = threading.Lock()

    def on_update(robot, kind, data):
        if kind == "video":
            with lock:
                frames.append(data)

    rx = Receiver(base_tr, on_update, tick_period=0.05)
    rx.start()
    agent = AgentCore("zvid", robot_tr, {})
    stop = threading.Event()

    def robot():
        i = 0
        while not stop.is_set():
            img = synth(i, 320, 240)
            agent.on_camera_frame("rgb", "rgb8", 320, 240, 960, img.tobytes(), i / 30)
            agent.tick()
            i += 1
            time.sleep(1 / 30)

    th = threading.Thread(target=robot, daemon=True)
    th.start()
    try:
        params = {"fps": 10, "kbps": 300, "width": 320, "codec": "h264"}
        t_end = time.time() + 4.0
        while time.time() < t_end:
            rx.video("zvid", "rgb", retry=False, **params)  # start + keepalives
            time.sleep(1.0)
        with lock:
            got = list(frames)
        assert len(got) >= 15, len(got)
        assert got[0]["key"] and got[0]["codec"] == "h264"
        assert got[0]["width"] == 320 and got[0]["height"] == 240
        assert [f["seq"] for f in got] == list(
            range(got[0]["seq"], got[0]["seq"] + len(got))
        )
        assert decode_count("h264", [f["data"] for f in got], 320, 240) == len(got)
        stats = rx.video_stats("zvid")["rgb"]
        assert stats["frames"] >= len(got) and stats["robot"]["codec"] == "h264"
        assert rx.link("zvid")["bytes"]["C5"] > 0
        rx.video("zvid", "rgb", enable=False)
        time.sleep(1.0)
        assert "rgb" not in agent.video.streams
    finally:
        stop.set()
        th.join(timeout=2)
        agent.close()
        rx.stop()
        robot_tr.close()
        base_tr.close()


def test_zenoh_open_without_router_then_connect_and_restart(late_router):
    """Agent and receiver sessions open while no router is up (the transport
    retries in the background), connect once it starts, and recover after the
    router is killed and restarted."""
    pytest.importorskip("zenoh")
    robot_tr = ZenohTransport(late_router.endpoint)  # nothing listens yet
    base_tr = ZenohTransport(f"tcp/127.0.0.1:1,{late_router.endpoint}")  # comma list
    assert not robot_tr.opened.is_set() and robot_tr.open_error
    rx = Receiver(base_tr, None, tick_period=0.05)
    rx.start()
    agent = AgentCore("zlate", robot_tr, {"budgets": {"C2": 20_000}})
    recs, edges, ls = synthetic_records(50, seed=5)
    agent.on_records(recs, edges, ls)
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            agent.tick()
            time.sleep(0.02)

    def wait_graph(timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if (
                "zlate" in rx.robots()
                and states_equal(rx.graph("zlate"), agent.graph_state())[0]
            ):
                return True
            time.sleep(0.1)
        return False

    th = threading.Thread(target=ticker, daemon=True)
    th.start()
    try:
        time.sleep(1.0)
        assert robot_tr.unopened_drops > 0  # puts were dropped, nothing crashed
        late_router.start()
        assert robot_tr.opened.wait(15.0) and base_tr.opened.wait(15.0)
        assert wait_graph(20.0)
        # router restart: the client sessions reconnect by themselves
        late_router.stop(kill=True)
        time.sleep(1.0)
        recs2, edges2, _ = synthetic_records(120, seed=6)
        agent.on_records(recs2, edges2, ls)
        late_router.start()
        assert wait_graph(30.0), rx.link("zlate")
        assert rx.link("zlate")["connected"]
    finally:
        stop.set()
        th.join(timeout=2)
        rx.stop()
        agent.close()
        robot_tr.close()
        base_tr.close()
