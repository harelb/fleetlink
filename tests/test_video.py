"""Live camera video (C5, API.md §12): bitstream parsing, ffmpeg encoders,
keepalive expiry, the C5 drop policy, the JPEG fallback and the receiver."""

import os
import shutil
import struct
import subprocess
import time

import numpy as np
import pytest
from helpers import RecordingTransport, env_bytes

from fleetlink import codec, video
from fleetlink.agent import AgentCore
from fleetlink.receiver import Receiver
from fleetlink.transport import VirtualClock


def synth(i: int, w: int = 640, h: int = 360) -> np.ndarray:
    """A moving, textured RGB test frame (gradients + a box + mild noise)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 100 * np.sin((x + 7 * i) / 37.0) * np.cos((y - 3 * i) / 23.0)
    g = 128 + 100 * np.sin((x * 0.5 + y + 5 * i) / 29.0)
    b = (x * 255 / w + 4 * i) % 255
    img = np.stack([r, g, b], 2)
    cx, cy = int(w / 2 + w / 4 * np.sin(i / 8)), int(h / 2 + h / 5 * np.cos(i / 5))
    img[max(0, cy - h // 9) : cy + h // 9, max(0, cx - w // 10) : cx + w // 10] = (
        240,
        30,
        30,
    )
    img += np.random.default_rng(i).normal(0, 6, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _has_encoder(name: str) -> bool:
    if not os.path.exists(video.FFMPEG):
        return False
    out = subprocess.run(
        [video.FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True
    ).stdout
    return f" {name} " in out


needs_x264 = pytest.mark.skipif(not _has_encoder("libx264"), reason="no libx264")
needs_vpx = pytest.mark.skipif(not _has_encoder("libvpx"), reason="no libvpx")


def decode_count(codec_: str, frames: list[bytes], w: int, h: int) -> int:
    """Decode the frames with ffmpeg; returns the number of decoded pictures."""
    if codec_ == "h264":
        data, fmt = b"".join(frames), "h264"
    else:
        hdr = b"DKIF" + struct.pack(
            "<HH4sHHIIII", 0, 32, b"VP80", w, h, 5, 1, len(frames), 0
        )
        body = b"".join(struct.pack("<IQ", len(f), i) + f for i, f in enumerate(frames))
        data, fmt = hdr + body, "ivf"
    res = subprocess.run(
        [video.FFMPEG, "-hide_banner", "-loglevel", "error", "-f", fmt, "-i", "pipe:0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=data, capture_output=True, timeout=60,
    )  # fmt: skip
    assert res.returncode == 0, res.stderr.decode()
    return len(res.stdout) // (w * h * 3)


# ----------------------------------------------------------------------------
# parsers
# ----------------------------------------------------------------------------


def _au(*nal_types, payload=b"\x88\x84") -> bytes:
    return b"".join(
        b"\x00\x00\x00\x01" + bytes([0x60 | t]) + payload for t in nal_types
    )


def test_annexb_splitter_splits_on_aud_across_chunks():
    stream = _au(9, 7, 8, 5) + _au(9, 1) + _au(9, 1) + _au(9, 7, 8, 5)
    sp = video.AnnexBSplitter()
    out = []
    for i in range(0, len(stream), 5):  # arbitrary chunk boundaries
        out += sp.feed(stream[i : i + 5])
    assert [k for _, k in out] == [True, False, False]
    assert sp.has_complete_tail()
    out += sp.flush()
    assert [k for _, k in out] == [True, False, False, True]
    assert b"".join(au for au, _ in out) == stream
    assert video.h264_nal_types(out[1][0]) == [9, 1]
    assert sp.flush() == []


def test_ivf_parser_key_flags():
    frames = [b"\x10\x02\x00abc", b"\x11xyz", b"\x31\x00"]  # bit0 = 0 -> key
    hdr = b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"VP80", 64, 48, 5, 1, 3, 0)
    data = hdr + b"".join(
        struct.pack("<IQ", len(f), i) + f for i, f in enumerate(frames)
    )
    p = video.IvfParser()
    out = []
    for i in range(0, len(data), 7):
        out += p.feed(data[i : i + 7])
    assert [f for f, _ in out] == frames
    assert [k for _, k in out] == [True, False, False]
    assert p.header == {"fourcc": "VP80", "width": 64, "height": 48}


def test_out_size_even_and_never_upscaled():
    assert video.out_size(640, 360, 480) == (480, 270)
    assert video.out_size(640, 360, 320) == (320, 180)
    assert video.out_size(640, 360, 960) == (640, 360)
    w, h = video.out_size(1280, 721, 333)
    assert w % 2 == 0 and h % 2 == 0 and w == 332
    p = video.clamp_params({"fps": 100, "kbps": 1, "width": 641, "codec": "VP8"})
    assert p == {"fps": 15.0, "kbps": 50, "width": 640, "codec": "vp8"}
    with pytest.raises(ValueError):
        video.clamp_params({"codec": "av1"})


# ----------------------------------------------------------------------------
# real ffmpeg encoders
# ----------------------------------------------------------------------------


def _encode(codec_, fps, kbps, seconds=10.0, w=640, h=360):
    """Feed ``seconds`` of frames (virtual time: the encoder's rate control only
    sees the frame rate) and collect every encoded frame."""
    out = []
    enc = video.FfmpegEncoder(codec_, w, h, fps, kbps, lambda d, k: out.append((d, k)))
    n = int(round(seconds * fps))
    try:
        for i in range(n):
            enc.feed(synth(i, w, h))
            deadline = time.time() + 5.0
            while enc.frames_in <= i and time.time() < deadline:
                time.sleep(0.001)  # one frame at a time (feed() is newest-wins)
    finally:
        enc.close()
    assert enc.error is None
    return n, out


@needs_x264
@pytest.mark.parametrize("fps,kbps", [(5, 300), (10, 800)])
def test_h264_encoder_frames_keys_bitrate(fps, kbps):
    n, out = _encode("h264", fps, kbps)
    assert len(out) == n  # one access unit per input frame
    keys = [i for i, (_, k) in enumerate(out) if k]
    g = 2 * fps
    assert keys == list(range(0, n, g))  # IDR every 2 s
    for au, key in out:
        types = video.h264_nal_types(au)
        assert types[0] == 9  # AUD first
        assert (5 in types) == key
        if key:
            assert 7 in types and 8 in types  # SPS/PPS repeated on every IDR
    kbit = sum(len(d) for d, _ in out) * 8 / 1000 / 10.0
    assert abs(kbit - kbps) / kbps < 0.3, kbit
    assert decode_count("h264", [d for d, _ in out], 640, 360) == n


@needs_vpx
@pytest.mark.parametrize("fps,kbps", [(5, 300), (10, 800)])
def test_vp8_encoder_frames_keys_bitrate(fps, kbps):
    n, out = _encode("vp8", fps, kbps)
    assert len(out) == n
    keys = [i for i, (_, k) in enumerate(out) if k]
    assert keys[0] == 0 and len(keys) >= n // (2 * fps)
    kbit = sum(len(d) for d, _ in out) * 8 / 1000 / 10.0
    assert abs(kbit - kbps) / kbps < 0.3, kbit
    assert decode_count("vp8", [d for d, _ in out], 640, 360) == n


# ----------------------------------------------------------------------------
# agent: keepalive, budget, fallback
# ----------------------------------------------------------------------------


def _agent(**cfg):
    tr = RecordingTransport()
    clock = VirtualClock(100.0)
    core = AgentCore("r", tr, cfg, clock)
    img = synth(0, 160, 120)
    core.on_camera_frame("rgb", "rgb8", 160, 120, 480, img.tobytes(), stamp=5.0)
    return tr, clock, core


def _req(tr, op_args, rid):
    req = {"id": rid, "op": "video", "args": op_args}
    tr.inject("adt4/r/ctl/req", env_bytes("base", "ctl/req", 0, "b", req))


def _run(clock, core, seconds, dt=0.05):
    for _ in range(int(round(seconds / dt))):
        core.tick(clock.advance(dt))


def _frames(tr, cam="rgb"):
    return [m[1]["p"] for m in tr.messages(f"adt4/r/video/{cam}")]


def test_video_keepalive_expiry_and_stop():
    tr, clock, core = _agent()
    _req(tr, {"camera": "rgb", "fps": 5, "codec": "jpeg", "width": 80}, "a")
    _run(clock, core, 2.0)
    ack = tr.messages("adt4/r/ctl/ack")[-1][1]["p"]
    assert ack["ok"] and ack["result"]["params"]["codec"] == "jpeg"
    fr = _frames(tr)
    assert 9 <= len(fr) <= 11  # 5 fps
    assert all(f["key"] and f["codec"] == "jpeg" for f in fr)
    assert fr[0]["width"] == 80 and fr[0]["height"] == 60
    assert [f["seq"] for f in fr] == list(range(len(fr)))
    assert fr[0]["stamp"] == 5.0 and fr[0]["data"][:2] == b"\xff\xd8"
    assert all(cls == "C5" for _, _, cls in tr.messages("adt4/r/video/"))
    st = tr.messages("adt4/r/state")[-1][1]["p"]
    assert st["video"]["rgb"]["fps"] == 5 and st["video"]["rgb"]["frames"] >= 9
    # keepalives every 2 s keep it going well past 6 s
    for i in range(4):
        _req(tr, {"camera": "rgb", "fps": 5, "codec": "jpeg", "width": 80}, f"k{i}")
        _run(clock, core, 2.0)
    n = len(_frames(tr))
    assert n >= 45 and "rgb" in core.video.streams
    # silence: stops after 6 s
    _run(clock, core, 6.5)
    assert "rgb" not in core.video.streams
    n2 = len(_frames(tr))
    _run(clock, core, 3.0)
    assert len(_frames(tr)) == n2
    assert tr.messages("adt4/r/state")[-1][1]["p"]["video"] == {}
    # explicit stop
    _req(tr, {"camera": "rgb", "codec": "jpeg"}, "s1")
    _run(clock, core, 1.0)
    assert "rgb" in core.video.streams
    _req(tr, {"camera": "rgb", "enable": False}, "s2")
    _run(clock, core, 0.1)
    assert "rgb" not in core.video.streams
    # unknown camera is a failed ack, not a crash
    _req(tr, {"camera": "nope"}, "bad")
    _run(clock, core, 0.1)
    ack = tr.messages("adt4/r/ctl/ack")[-1][1]["p"]
    assert not ack["ok"] and "unknown camera" in ack["result"]["error"]


def test_c5_drop_policy_skips_deltas_until_keyframe():
    sent = []
    clock = VirtualClock(0.0)
    budget = {"C5": 1000.0}
    vs = video.VideoStreams(
        get_frame=lambda c: None,
        send=lambda cam, p: sent.append(p) or len(p["data"]),
        budget_fn=lambda: budget["C5"],
        clock=clock,
        cameras=lambda: ["rgb"],
        burst_s=1.0,
    )
    vs.request({"camera": "rgb", "codec": "jpeg"})
    st = vs.streams["rgb"]
    # pattern: K d d d d K d d ... each 400 B at 10 fps = 4000 B/s >> 1000 B/s budget
    pattern = [i % 5 == 0 for i in range(20)]
    for key in pattern:
        clock.advance(0.1)
        vs._on_encoded(st, b"x" * 400, key)
    sent_keys = [p["key"] for p in sent]
    # 1 s burst = 1000 B: K d d go out (debt by one), then deltas wait for a key
    assert sent_keys[:3] == [True, False, False]
    # map outputs to inputs by replaying the same bucket arithmetic, then check
    # that no delta was sent after a dropped frame (only a key may follow a gap)
    bucket = video.C5Bucket(lambda: 1000.0, 1.0)
    t, wait, chain = 0.0, False, []
    for i, key in enumerate(pattern):
        t += 0.1
        if wait and not key:
            continue
        if bucket.admit(400, key, t):
            wait = False
            chain.append(i)
        else:
            wait = True
    assert [pattern[i] for i in chain] == sent_keys
    for a, b in zip(chain, chain[1:]):
        assert b == a + 1 or pattern[b], f"delta {b} sent after a gap"
    assert st.dropped == len(pattern) - len(sent)
    # key frames get through even when the bucket is empty (borrowing a burst)
    assert sum(pattern) == sum(sent_keys)
    # with an unlimited budget everything goes
    budget["C5"] = None
    sent.clear()
    st.wait_key = False
    for key in pattern:
        vs._on_encoded(st, b"x" * 400, key)
    assert len(sent) == len(pattern)


def test_c5_budget_live_adjustable_via_budget_op():
    tr, clock, core = _agent()
    assert core.sched.budgets["C5"] == 400_000
    tr.inject(
        "adt4/r/ctl/req",
        env_bytes(
            "base",
            "ctl/req",
            0,
            "b",
            {"id": "b1", "op": "budget", "args": {"C5": 1234}},
        ),
    )
    _run(clock, core, 0.1)
    assert core.sched.budgets["C5"] == 1234
    assert core.video.bucket.budget_fn() == 1234
    ack = tr.messages("adt4/r/ctl/ack")[-1][1]["p"]
    assert ack["ok"] and ack["result"]["budgets"]["C5"] == 1234


def test_jpeg_fallback_when_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(video, "FFMPEG", "/nonexistent/ffmpeg")
    tr, clock, core = _agent()
    _req(tr, {"camera": "rgb", "fps": 2, "codec": "h264", "width": 160}, "a")
    _run(clock, core, 2.0)
    fr = _frames(tr)
    assert len(fr) >= 3 and all(f["codec"] == "jpeg" and f["key"] for f in fr)
    st = tr.messages("adt4/r/state")[-1][1]["p"]["video"]["rgb"]
    assert st["codec"] == "jpeg" and st["requested_codec"] == "h264"
    assert "ffmpeg not found" in st["error"]


def test_jpeg_fallback_when_ffmpeg_dies(tmp_path):
    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\necho 'boom: no encoder' >&2\nexit 3\n")
    fake.chmod(0o755)
    tr, clock, core = _agent(ffmpeg=str(fake))
    _req(tr, {"camera": "rgb", "fps": 5, "codec": "vp8"}, "a")
    deadline = time.time() + 10.0
    while time.time() < deadline and not any(f["codec"] == "jpeg" for f in _frames(tr)):
        _run(clock, core, 0.2)
        time.sleep(0.05)  # real time for the fake ffmpeg to exit
    st = core.video.state()["rgb"]
    assert st["codec"] == "jpeg" and "exited (3)" in st["error"]
    assert "boom" in st["error"]
    assert any(f["codec"] == "jpeg" for f in _frames(tr))


# ----------------------------------------------------------------------------
# receiver
# ----------------------------------------------------------------------------


def test_receiver_forwards_video_and_flags_gaps():
    tr = RecordingTransport()
    clock = VirtualClock(0.0)
    events = []
    rx = Receiver(
        tr, lambda r, k, d: events.append((r, k, d)), clock=clock, threaded=False
    )
    rx.start()

    def frame(seq, key):
        p = {"camera": "rgb", "codec": "h264", "seq": seq, "t": 1.0, "key": key,
             "width": 640, "height": 360, "fps": 5, "data": b"z" * 100}  # fmt: skip
        tr.inject("adt4/r/video/rgb", env_bytes("r", "video/rgb", seq, "e1", p))

    for seq, key in [(0, True), (1, False), (2, False), (5, False), (6, True)]:
        frame(seq, key)
        clock.advance(0.2)
    vids = [d for _, k, d in events if k == "video"]
    assert [v["seq"] for v in vids] == [0, 1, 2, 5, 6]
    assert [v["gap"] for v in vids] == [True, False, False, True, False]
    rx.tick(clock.advance(0.1))  # opens the rate window
    rx.tick(clock.advance(1.0))
    stats = rx.video_stats("r")["rgb"]
    assert stats["frames"] == 5 and stats["keys"] == 2 and stats["lost"] == 2
    assert stats["codec"] == "h264" and stats["fps"] > 0
    assert rx.link("r")["bytes"]["C5"] > 500
    rid = rx.video("r", "rgb", fps=10, kbps=None, codec="vp8", retry=False)
    req = codec.unpack(tr.sent[-1][1])["p"]
    assert req["id"] == rid and req["op"] == "video"
    assert req["args"] == {"camera": "rgb", "enable": True, "fps": 10, "codec": "vp8"}
    pending = [p for p in rx.pending_requests() if p["op"] == "video"]
    assert not pending  # keepalives are fire-and-forget
    rx.video("r", "rgb", enable=False)
    pending = [p for p in rx.pending_requests() if p["op"] == "video"]
    assert pending[0]["args"] == {"camera": "rgb", "enable": False}


@pytest.mark.skipif(not shutil.which("true"), reason="posix")
def test_agent_streams_h264_through_link(monkeypatch):
    """AgentCore → RecordingTransport with the real h264 encoder (real time)."""
    if not _has_encoder("libx264"):
        pytest.skip("no libx264")
    tr = RecordingTransport()
    core = AgentCore("r", tr, {}, time.time)
    img = synth(0, 320, 240)
    core.on_camera_frame("rgb", "rgb8", 320, 240, 960, img.tobytes())
    _req(tr, {"camera": "rgb", "fps": 10, "codec": "h264", "kbps": 200}, "a")
    t_end = time.time() + 2.5
    i = 0
    while time.time() < t_end:
        i += 1
        img = synth(i, 320, 240)
        core.on_camera_frame("rgb", "rgb8", 320, 240, 960, img.tobytes())
        core.tick()
        time.sleep(0.02)
    core.close()
    fr = _frames(tr)
    assert len(fr) >= 15 and fr[0]["key"] and fr[0]["codec"] == "h264"
    assert fr[0]["width"] == 320 and fr[0]["height"] == 240
    assert decode_count("h264", [f["data"] for f in fr], 320, 240) == len(fr)


@pytest.mark.skipif(not shutil.which("true"), reason="posix")
def test_force_keyframe_beats_the_gop(monkeypatch):
    """A forced key frame arrives within a few frames, not at the 2 s GOP edge."""
    if not _has_encoder("libx264"):
        pytest.skip("no libx264")
    got = []
    frame = {"encoding": "rgb8", "width": 320, "height": 240, "step": 960}
    i = [0]

    def get_frame(_cam):
        i[0] += 1
        return {**frame, "data": synth(i[0], 320, 240).tobytes(), "stamp": None}

    vs = video.VideoStreams(
        get_frame=get_frame,
        send=lambda cam, p: got.append((time.time(), p["key"])) or len(p["data"]),
        budget_fn=lambda: 1e9,
        cameras=lambda: ["rgb"],
    )
    vs.request({"camera": "rgb", "codec": "h264", "fps": 10, "kbps": 300})
    try:
        t_end = time.time() + 1.2  # past the first key frame, inside the 2 s GOP
        while time.time() < t_end:
            vs.tick()
            time.sleep(0.01)
        t_force = time.time()
        assert vs.force_keyframe("rgb")
        assert not vs.force_keyframe("rgb")  # rate-limited (0.5 s)
        t_end = time.time() + 0.6
        while time.time() < t_end:
            vs.tick()
            time.sleep(0.01)
    finally:
        vs.stop_all()
    after = [t for t, key in got if key and t > t_force]
    assert after, "no key frame after the forced restart"
    assert after[0] - t_force < 0.4  # natural next key would be ~0.8 s away


def test_video_keyframe_op_acks():
    tr, clock, core = _agent()
    _req(tr, {"camera": "rgb", "fps": 5, "codec": "jpeg", "width": 80}, "a")
    _run(clock, core, 1.0)
    req = {"id": "kf", "op": "video_keyframe", "args": {"camera": "rgb"}}
    tr.inject("adt4/r/ctl/req", env_bytes("base", "ctl/req", 1, "b", req))
    _run(clock, core, 0.1)
    ack = [m[1]["p"] for m in tr.messages("adt4/r/ctl/ack") if m[1]["p"]["id"] == "kf"]
    assert (
        ack and ack[0]["ok"] and ack[0]["result"] == {"camera": "rgb", "forced": True}
    )
    assert core.video.streams["rgb"].forced == 1


def test_receiver_requests_keyframe_on_delta_gap_and_corrects_clock():
    tr = RecordingTransport()
    clock = VirtualClock(1000.0)
    events = []
    rx = Receiver(
        tr, lambda r, k, d: events.append((k, d)), clock=clock, threaded=False
    )
    rx.start()

    def frame(seq, key, t):
        p = {"camera": "rgb", "codec": "h264", "seq": seq, "t": t, "key": key,
             "width": 640, "height": 360, "fps": 5, "data": b"z" * 100}  # fmt: skip
        env = env_bytes("r", "video/rgb", seq, "e1", p)
        tr.inject("adt4/r/video/rgb", env)

    def kf_requests():
        return [
            codec.unpack(d)["p"]
            for k, d, _ in tr.sent
            if k.endswith("ctl/req") and codec.unpack(d)["p"]["op"] == "video_keyframe"
        ]

    frame(0, True, 1000.0)
    frame(3, False, 1000.0)  # lost 1, 2 → ask for a key frame
    frame(5, False, 1000.0)  # second gap within 1 s: rate-limited
    assert [r["args"] for r in kf_requests()] == [{"camera": "rgb"}]
    clock.advance(1.5)
    frame(9, True, 1000.0)  # a gap ending on a key frame needs nothing
    assert len(kf_requests()) == 1

    # clock offset: robot clock runs 3 s ahead; the lowest-RTT ping wins
    r = rx._robots["r"]
    for t_send, rtt in [(1001.0, 0.4), (1002.0, 0.02), (1003.0, 0.3)]:
        clock.t = t_send + rtt
        robot_t = t_send + rtt / 2 + 3.0
        rx._on_ack(
            r, {"id": None, "result": {"t": t_send, "t_robot": robot_t}}, clock.t
        )
    assert abs(rx.clock_offset("r") - 3.0) < 1e-9
    assert abs(rx.link("r")["clock_offset_ms"] - 3000.0) < 1e-6
    # a frame captured 0.1 s ago in robot time maps to 0.1 s ago in base time
    clock.t = 1010.0
    frame(10, False, 1010.0 + 3.0 - 0.1)
    vid = [d for k, d in events if k == "video"][-1]
    assert abs(vid["t_base"] - (1010.0 - 0.1)) < 1e-9


def test_encoder_bitrate_capped_to_c5_budget():
    budget = {"C5": 400_000.0}
    vs = video.VideoStreams(
        get_frame=lambda c: None,
        send=lambda cam, p: 0,
        budget_fn=lambda: budget["C5"],
        cameras=lambda: ["rgb", "overlay"],
    )
    vs.request({"camera": "rgb", "codec": "h264", "kbps": 800})
    st = vs.streams["rgb"]
    assert vs.effective_kbps(st) == 800  # 0.8 * 3200 kbit/s leaves plenty
    budget["C5"] = 40_000.0  # 320 kbit/s → 256 → 250 (25 kbit/s steps)
    assert vs.effective_kbps(st) == 250
    vs.request({"camera": "overlay", "codec": "h264", "kbps": 800})
    assert vs.effective_kbps(st) == 125  # two streams share the budget
    budget["C5"] = 1_000.0
    assert vs.effective_kbps(st) == video.MIN_KBPS  # never below the floor
    budget["C5"] = None
    assert vs.effective_kbps(st) == 800  # unlimited
    vs.stop_all()
