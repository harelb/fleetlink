"""Occupancy grid (C3), executor paths (C1) and on-demand camera frames (C4)."""

import numpy as np
import pytest
from helpers import RecordingTransport, env_bytes

from fleetlink import codec, media
from fleetlink.agent import AgentCore, marker_target, parse_cameras
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator, VirtualClock

# ----------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------


def _grid(w=40, h=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 0, 100], np.int8), size=(h, w))


def test_grid_round_trip():
    g = _grid()
    enc = media.encode_grid(
        "euclid/map", 0.1, [1.0, -2.0, 0.3, 0.5], 40, 30, g.tobytes(), 7.0
    )
    wire = codec.unpack(codec.pack(enc))
    out = media.decode_grid(wire)
    assert out.shape == (30, 40) and out.dtype == np.int8
    assert np.array_equal(out, g)
    assert wire["origin"] == [1.0, -2.0, 0.3, 0.5] and wire["frame"] == "euclid/map"
    assert len(wire["data"]) < g.size  # zlib
    with pytest.raises(ValueError):
        media.encode_grid("m", 0.1, [0, 0, 0, 0], 41, 30, g.tobytes())


def _agent(**cfg):
    tr = RecordingTransport()
    clock = VirtualClock()
    core = AgentCore("r", tr, cfg, clock)
    return tr, clock, core


def _run(clock, core, seconds, dt=0.05):
    for _ in range(int(round(seconds / dt))):
        core.tick(clock.advance(dt))


def test_grid_sent_only_when_changed():
    tr, clock, core = _agent(grid_period_s=1.0)
    g = _grid()
    core.on_grid("map", 0.1, [0, 0, 0, 0], 40, 30, g.tobytes())
    _run(clock, core, 3.0)
    core.on_grid("map", 0.1, [0, 0, 0, 0], 40, 30, g.tobytes())  # same content
    _run(clock, core, 3.0)
    msgs = tr.messages("adt4/r/grid/occupancy")
    assert len(msgs) == 1 and msgs[0][2] == "C3"
    g2 = g.copy()
    g2[0, 0] = 100 if g2[0, 0] != 100 else 0
    for _ in range(5):  # bursts inside one period collapse to the newest grid
        core.on_grid("map", 0.1, [0, 0, 0, 0], 40, 30, g2.tobytes())
    _run(clock, core, 3.0)
    msgs = tr.messages("adt4/r/grid/occupancy")
    assert len(msgs) == 2
    assert np.array_equal(media.decode_grid(msgs[1][1]["p"]), g2)
    state = tr.messages("adt4/r/state")[-1][1]["p"]
    assert state["hashes"]["grid"] == msgs[1][1]["p"]["hash"]


def test_grid_counts_against_c3_budget():
    tr, clock, core = _agent(grid_period_s=1.0, budgets={"C3": 2000})
    rng = np.random.default_rng(1)
    big = rng.integers(-1, 101, (100, 100)).astype(np.int8)
    core.on_grid("map", 0.05, [0, 0, 0, 0], 100, 100, big.tobytes())
    _run(clock, core, 0.5)
    sent = tr.messages("adt4/r/grid/occupancy")
    assert len(sent) == 1  # the bucket may go into debt by one message
    assert core.sched.sent_bytes["C3"] >= len(sent[0][1]["p"]["data"])
    core.on_grid("map", 0.05, [0, 0, 0, 0], 100, 100, (-big).tobytes())
    _run(clock, core, 2.0)
    assert len(tr.messages("adt4/r/grid/occupancy")) == 1  # waits for the budget
    _run(clock, core, 10.0)
    assert len(tr.messages("adt4/r/grid/occupancy")) == 2


def test_pending_grid_survives_mesh_requeue():
    from helpers import synthetic_mesh

    tr, clock, core = _agent(grid_period_s=1.0, budgets={"C3": 1})
    core.on_records({}, set(), {}, synthetic_mesh(400))
    core.sched.tokens["C3"] = 0  # nothing leaves: everything stays queued
    core.on_grid("map", 0.1, [0, 0, 0, 0], 40, 30, _grid().tobytes())
    core._grid_tick()
    core._tiles_dirty = True
    core._c3_tick()  # re-queues tiles; must not drop the pending grid
    assert any(it.tag == "grid" for it in core.sched.queues["C3"])


def test_paths_change_only_and_downsampled():
    tr, clock, core = _agent()
    line = np.stack([np.linspace(0, 50, 1000), np.zeros(1000), np.zeros(1000)], 1)
    core.on_path("smooth", line, "map", 1.0)
    core.on_path("mlp", line[:10], "map", 1.0)
    core.on_path("target", [5.0, 0.0, 0.0], "map")
    _run(clock, core, 2.0)
    msgs = tr.messages("adt4/r/paths")
    assert len(msgs) == 1 and msgs[0][2] == "C1"
    p = msgs[0][1]["p"]
    assert len(p["smooth"]) <= 150 and p["smooth"][0] == [0, 0, 0]
    assert p["smooth"][-1] == [50.0, 0.0, 0.0]
    assert (
        len(p["mlp"]) == 10 and p["target"] == [5.0, 0.0, 0.0] and p["frame"] == "map"
    )
    # re-published with a new stamp but the same geometry: nothing new
    core.on_path("smooth", line.copy(), "map", 2.0)
    _run(clock, core, 2.0)
    assert len(tr.messages("adt4/r/paths")) == 1
    core.on_path("target", None, "map")
    _run(clock, core, 2.0)
    msgs = tr.messages("adt4/r/paths")
    assert len(msgs) == 2 and msgs[1][1]["p"]["target"] is None


def test_receiver_stores_grid_paths_and_requests_resend():
    tr = RecordingTransport()
    clock = VirtualClock(100.0)
    events = []
    rx = Receiver(
        tr, lambda r, k, d: events.append((k, d)), clock=clock, threaded=False
    )
    rx.start()
    grid = media.encode_grid("map", 0.1, [0, 0, 0, 0], 40, 30, _grid().tobytes())
    paths = media.paths_payload([[0, 0, 0], [1, 0, 0]], [], None, "map", 1.0)
    tr.inject("adt4/x/grid/occupancy", env_bytes("x", "grid/occupancy", 0, "E", grid))
    tr.inject("adt4/x/paths", env_bytes("x", "paths", 0, "E", paths))
    assert rx.grid("x")["hash"] == grid["hash"] and rx.paths("x")["smooth"][1] == [
        1,
        0,
        0,
    ]
    kinds = [k for k, _ in events]
    assert "grid" in kinds and "paths" in kinds
    assert next(d for k, d in events if k == "grid")["width"] == 40
    # the robot says it sent a newer grid we never got: resend after the grace period
    st = {"hashes": {"grid": "deadbeef", "paths": paths["hash"]}, "cameras": ["rgb"]}
    for i in range(1, 13):
        tr.inject("adt4/x/state", env_bytes("x", "state", i, "E", st))
        clock.advance(1.0)
    reqs = [codec.unpack(d)["p"] for k, d, c in tr.sent if k == "adt4/x/ctl/req"]
    resend = [r for r in reqs if r["op"] == "resend"]
    assert len(resend) == 1 and resend[0]["args"] == {"streams": ["grid"]}
    assert rx.cameras("x") == ["rgb"]


def test_resend_op_clears_sent_hash():
    tr, clock, core = _agent(grid_period_s=1.0)
    core.on_grid("map", 0.1, [0, 0, 0, 0], 40, 30, _grid().tobytes())
    _run(clock, core, 2.0)
    tr.inject(
        "adt4/r/ctl/req",
        env_bytes(
            "base",
            "ctl/req",
            0,
            "S",
            {"id": "q", "op": "resend", "args": {"streams": ["grid"]}},
        ),
    )
    _run(clock, core, 2.0)
    assert len(tr.messages("adt4/r/grid/occupancy")) == 2


def test_marker_target_and_camera_param():
    class P:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class M:
        def __init__(self, action, pos, points=()):
            self.action = action
            self.pose = type("Pose", (), {"position": P(*pos)})()
            self.points = list(points)
            self.header = type("H", (), {"frame_id": "map"})()

    arr = type("A", (), {"markers": [M(3, (0, 0, 0)), M(0, (1, 2, 3))]})()
    assert marker_target(arr) == ([1, 2, 3], "map")
    assert marker_target(type("A", (), {"markers": [M(3, (0, 0, 0))]})())[0] is None
    assert parse_cameras(["rgb=a/b", "bad", "x= c "]) == {"rgb": "a/b", "x": "c"}


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------


def test_image_conversions():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[..., 0] = 200  # red
    bgr = rgb[..., ::-1].copy()
    out = media.image_to_array("bgr8", 6, 4, 18, bgr.tobytes())
    assert out.shape == (4, 6, 3) and out[0, 0].tolist() == [200, 0, 0]
    # row padding (step > width * channels)
    padded = np.zeros((4, 20), np.uint8)
    padded[:, :18] = rgb.reshape(4, 18)
    assert media.image_to_array("rgb8", 6, 4, 20, padded.tobytes())[0, 0].tolist() == [
        200,
        0,
        0,
    ]
    mono = media.image_to_array("mono8", 6, 4, 6, np.full(24, 7, np.uint8).tobytes())
    assert mono.shape == (4, 6) and mono[0, 0] == 7
    depth = np.arange(24, dtype=np.uint16).reshape(4, 6) * 100
    d8 = media.image_to_array("16UC1", 6, 4, 12, depth.tobytes())
    assert d8.dtype == np.uint8 and d8[0, 0] == 0 and d8[-1, -1] == 255  # 0 = invalid
    with pytest.raises(ValueError):
        media.image_to_array("yuv422", 6, 4, 12, bytes(48))


def test_png_fallback_decodes():
    cv2 = pytest.importorskip("cv2")
    img = np.random.default_rng(0).integers(0, 255, (20, 30, 3), dtype=np.uint8)
    data, fmt = media.encode_image(img, force_png=True)
    assert fmt == "png"
    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    assert np.array_equal(back, img)
    small = media.resize_max_width(img, 15)
    assert small.shape == (10, 15, 3)


# ----------------------------------------------------------------------------
# C4 camera request/reply over the emulated link
# ----------------------------------------------------------------------------


def _camera_setup(bandwidth=100_000.0, latency=0.1, **cfg):
    link = LinkEmulator(VirtualClock(), bandwidth=bandwidth, latency=latency, seed=3)
    agent = AgentCore("cam", link.robot, cfg, link.clock)
    img = np.random.default_rng(2).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    agent.on_camera_frame("rgb", "rgb8", 320, 240, 960, img.tobytes(), stamp=12.5)
    rx = Receiver(link.base, None, clock=link.clock, threaded=False)
    rx.start()
    return link, agent, rx


def _ask(link, rx, name, until=3.0, robot="cam", **kw):
    got = []
    rx.query_camera(robot, name, lambda rep: got.append((link.clock(), rep)), **kw)
    while link.clock() < until - 1e-9:
        link.clock.advance(0.01)
        link.advance()
    return got


def test_camera_query_over_link_respects_latency():
    link, agent, rx = _camera_setup()
    got = _ask(link, rx, "rgb", width=160, quality=60)
    assert len(got) == 1
    t, rep = got[0]
    assert (
        rep["format"] in ("jpeg", "png")
        and rep["width"] == 160
        and rep["height"] == 120
    )
    assert rep["stamp"] == 12.5 and rep["src_width"] == 320 and rep["camera"] == "rgb"
    # request + reply each cross the 100 ms link; the reply also pays serialisation
    assert t >= 0.2 + len(rep["data"]) / 100_000 - 0.011
    assert t < 0.2 + len(rep["data"]) / 100_000 + 0.1
    stats = link.stats()
    assert stats["up"].delivered_bytes["C4"] > len(rep["data"])
    assert rx.link("cam")["bytes"]["C4"] > len(rep["data"])


def test_camera_unknown_and_no_frame():
    link, agent, rx = _camera_setup()
    rep = _ask(link, rx, "thermal")[0][1]
    assert rep["error"] == "unknown camera" and "rgb" in rep["cameras"]
    rep = _ask(link, rx, "overlay", until=6.0)[0][1]
    assert rep["error"] == "no frame yet"


def test_camera_budget_refuses_when_spent():
    link, agent, rx = _camera_setup(budgets={"C4": 1000}, burst_s=1.0)
    first = _ask(link, rx, "rgb", width=320, quality=90)[0][1]
    assert "data" in first and len(first["data"]) > 1000  # one frame of debt allowed
    second = _ask(link, rx, "rgb", until=3.5)[0][1]
    assert second["error"] == "budget" and agent.media_refused == 1


def test_camera_query_times_out_cleanly():
    link, agent, rx = _camera_setup(bandwidth=2_000.0)  # a frame needs >> 1 s
    got = _ask(link, rx, "rgb", until=20.0, width=320, quality=90, timeout=1.0)
    assert len(got) == 1 and got[0][1] is None  # exactly one callback, the timeout
    assert got[0][0] == pytest.approx(1.0, abs=0.011)
    # nobody answers for an unknown robot either
    got = _ask(link, rx, "rgb", until=22.0, robot="ghost", timeout=1.0)
    assert len(got) == 1 and got[0][1] is None
