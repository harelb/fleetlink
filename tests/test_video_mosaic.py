"""Mosaic video (the console's "All cameras" grid, API.md §12): several cameras
tiled into one picture and one encoder, the tile layout in every frame, lazy
camera wants, the budget split and the cheap scaler."""

import subprocess
import time

import cv2
import numpy as np
import pytest
from helpers import RecordingTransport, env_bytes
from test_video import _has_encoder

from fleetlink import video
from fleetlink.agent import AgentCore
from fleetlink.transport import VirtualClock

SPOT = ["rgb", "overlay", "frontleft", "frontright", "left", "right", "back", "hand"]
COLORS = {
    "rgb": (250, 20, 20),
    "overlay": (20, 250, 20),
    "frontleft": (20, 20, 250),
    "frontright": (250, 250, 20),
    "left": (20, 250, 250),
    "right": (250, 20, 250),
    "back": (128, 128, 128),
    "hand": (250, 128, 20),
}


def solid(cam: str, w: int = 640, h: int = 480) -> dict:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = COLORS[cam]
    return {"encoding": "rgb8", "width": w, "height": h, "step": w * 3,
            "data": img.tobytes(), "stamp": 7.0, "rx_t": 99.5}  # fmt: skip


def test_grid_shapes():
    assert [video.mosaic_grid(n) for n in (1, 2, 4, 6, 8)] == [
        (1, 1), (2, 1), (2, 2), (3, 2), (4, 2)]  # fmt: skip
    cols, rows = video.mosaic_grid(12)
    assert cols * rows >= 12 and cols >= rows
    assert video.mosaic_cell(320, 6) == (320, 240, 3, 2)
    # many big tiles shrink to fit MOSAIC_MAX_PIXELS
    w, h, cols, rows = video.mosaic_cell(640, 12)
    assert w * cols * h * rows <= video.MOSAIC_MAX_PIXELS and w % 2 == h % 2 == 0
    assert video.fit(1280, 720, 320, 240) == (320, 180)
    assert video.fit(480, 640, 320, 240) == (180, 240)


def test_scale_integer_area_then_linear():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, 320:] = 200
    for w, h in ((320, 240), (240, 180), (213, 160), (640, 480)):
        out = video.scale(img, w, h)
        assert out.shape == (h, w, 3)
        assert out[h // 2, 2].tolist() == [0, 0, 0]
        assert out[h // 2, w - 3].tolist() == [200, 200, 200]


def _streams(frames: dict, budget=None):
    clock = VirtualClock(100.0)
    sent = []
    vs = video.VideoStreams(
        get_frame=frames.get,
        send=lambda cam, p: sent.append((cam, p)) or len(p["data"]),
        budget_fn=lambda: budget,
        clock=clock,
        cameras=lambda: list(SPOT),
    )
    return vs, clock, sent


def test_jpeg_mosaic_layout_and_pixels():
    frames = {c: solid(c) for c in SPOT[:5]}
    frames["frontleft"] = solid("frontleft", 1280, 720)  # wide: letterboxed
    cams = ["rgb", "frontleft", "left", "hand"]  # hand has no frame yet (lazy)
    vs, clock, sent = _streams(frames)
    res = vs.request(
        {"camera": "mosaic", "cameras": cams + ["rgb"], "width": 160, "fps": 5,
         "codec": "jpeg"}, clock())  # fmt: skip
    assert res["params"]["cameras"] == cams  # de-duplicated, order kept
    vs.tick(clock())
    assert len(sent) == 1
    cam, p = sent[0]
    assert cam == "mosaic" and p["codec"] == "jpeg" and p["key"]
    assert (p["width"], p["height"]) == (320, 240)  # 2x2 cells of 160x120
    tiles = {t[0]: t[1:] for t in p["tiles"]}
    assert list(tiles) == cams
    assert tiles["rgb"] == [0, 0, 160, 120, 7.0, 0.5]
    x, y, w, h, _, _ = tiles["frontleft"]
    assert (w, h) == (160, 90) and x == 160 and 0 < y < 30
    assert tiles["hand"][2:4] == [0, 0] and tiles["hand"][5] is None
    img = cv2.cvtColor(
        cv2.imdecode(np.frombuffer(p["data"], np.uint8), 1), cv2.COLOR_BGR2RGB
    )
    for c, (x, y, w, h, _, _) in tiles.items():
        if w:
            got = img[y + h // 2, x + w // 2].astype(int)
            assert np.abs(got - COLORS[c]).max() < 30, (c, got)
    assert img[230, 240].max() < 30  # hand's cell stays black
    st = vs.state()["mosaic"]
    assert st["cameras"] == cams and st["width"] == 320
    assert vs.cameras_in_use() == set(cams)
    # the frame arrives: next mosaic has it; unchanged cameras are not rescaled
    frames["hand"] = solid("hand")
    clock.advance(0.2)
    vs.tick(clock())
    tiles = {t[0]: t[1:] for t in sent[-1][1]["tiles"]}
    assert tiles["hand"][2:4] == [160, 120]
    vs.stop_all()


def test_mosaic_requests_validate_and_keepalive_keeps_list():
    vs, clock, _ = _streams({})
    with pytest.raises(ValueError, match="unknown camera"):
        vs.request({"camera": "mosaic", "cameras": ["rgb", "nope"]}, clock())
    with pytest.raises(ValueError, match="1.."):
        vs.request({"camera": "mosaic", "cameras": []}, clock())
    with pytest.raises(ValueError, match="list"):
        vs.request({"camera": "mosaic"}, clock())
    vs.request({"camera": "mosaic", "cameras": ["left", "right"], "codec": "jpeg"})
    res = vs.request({"camera": "mosaic", "codec": "jpeg"})  # keepalive, no list
    assert res["params"]["cameras"] == ["left", "right"]
    vs.request({"camera": "mosaic", "enable": False})
    assert vs.streams == {} and vs.cameras_in_use() == set()


def test_bitrate_split_in_proportion_to_requests():
    vs, clock, _ = _streams({}, budget=100_000.0)  # 0.8 * 800 = 640 kbit/s
    vs.request({"camera": "mosaic", "cameras": SPOT, "kbps": 800, "codec": "jpeg"})
    vs.request({"camera": "rgb", "kbps": 200, "codec": "jpeg"})
    assert vs.effective_kbps(vs.streams["mosaic"]) == 500  # 640 * 0.8 -> 512
    assert vs.effective_kbps(vs.streams["rgb"]) == 125  # 640 * 0.2 -> 128
    vs.stop_all()


def _req(tr, op_args, rid):
    req = {"id": rid, "op": "video", "args": op_args}
    tr.inject("adt4/r/ctl/req", env_bytes("base", "ctl/req", 0, "b", req))


def test_agent_wants_mosaic_cameras_and_sends_tiles():
    tr = RecordingTransport()
    clock = VirtualClock(100.0)
    core = AgentCore("r", tr, {"cameras": SPOT}, clock)
    assert core.wanted_cameras() == {"rgb", "overlay"}
    _req(tr, {"camera": "mosaic", "cameras": ["left", "back"], "width": 160,
              "fps": 5, "codec": "jpeg"}, "a")  # fmt: skip
    core.tick(clock.advance(0.05))
    ack = tr.messages("adt4/r/ctl/ack")[-1][1]["p"]
    assert ack["ok"], ack
    assert {"left", "back"} <= core.wanted_cameras()
    for c in ("left", "back"):  # the node subscribed; frames arrive
        f = solid(c, 64, 48)
        core.on_camera_frame(c, "rgb8", 64, 48, 64 * 3, f["data"], stamp=3.0)
    for _ in range(20):
        core.tick(clock.advance(0.05))
    fr = [m[1]["p"] for m in tr.messages("adt4/r/video/mosaic")]
    assert fr and [t[0] for t in fr[-1]["tiles"]] == ["left", "back"]
    assert fr[-1]["tiles"][0][3:5] == [160, 120]  # small sources fill the cell
    # keepalives keep the cameras wanted beyond camera_idle_s
    for i in range(10):
        _req(tr, {"camera": "mosaic", "codec": "jpeg", "width": 160}, f"k{i}")
        for _ in range(40):
            core.tick(clock.advance(0.05))
    assert {"left", "back"} <= core.wanted_cameras()
    core.close()


@pytest.mark.skipif(not _has_encoder("libx264"), reason="no libx264")
def test_h264_mosaic_decodes_with_the_right_tiles():
    frames = {c: solid(c) for c in SPOT[2:8]}  # the six Spot body cameras
    got = []
    vs = video.VideoStreams(
        get_frame=frames.get,
        send=lambda cam, p: got.append(p) or len(p["data"]),
        budget_fn=lambda: 400_000.0,
        cameras=lambda: list(SPOT),
    )
    vs.request({"camera": "mosaic", "cameras": SPOT[2:8], "width": 160, "fps": 10,
                "kbps": 400, "codec": "h264"})  # fmt: skip
    try:
        t_end = time.time() + 1.5
        while time.time() < t_end:
            vs.tick()
            time.sleep(0.01)
    finally:
        vs.stop_all()
    assert len(got) >= 8 and got[0]["key"]
    W, H = got[0]["width"], got[0]["height"]
    assert (W, H) == (480, 240)
    res = subprocess.run(
        [video.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "h264", "-i",
         "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=b"".join(p["data"] for p in got), capture_output=True, timeout=60,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr
    last = np.frombuffer(res.stdout, np.uint8)[-W * H * 3 :].reshape(H, W, 3)
    for cam, x, y, w, h, _stamp, _age in got[-1]["tiles"]:
        px = last[y + h // 2, x + w // 2].astype(int)
        assert np.abs(px - COLORS[cam]).max() < 40, (cam, px)
