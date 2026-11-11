"""Lazy camera subscriptions: only eager cameras (default rgb, overlay) are
subscribed all the time; others while a C4 query or C5 video wants them."""

import numpy as np
from helpers import RecordingTransport

from fleetlink import media
from fleetlink.agent import AgentCore
from fleetlink.transport import VirtualClock

SPOT = ["rgb", "overlay", "frontleft", "frontright", "left", "right", "back", "hand"]


def _core(**cfg):
    clock = VirtualClock()
    tr = RecordingTransport()
    core = AgentCore("r", tr, {"cameras": SPOT, **cfg}, clock)
    return tr, clock, core


def _frame(core, name):
    img = np.zeros((48, 64, 3), np.uint8).tobytes()
    core.on_camera_frame(name, "rgb8", 64, 48, 64 * 3, img)


def test_only_eager_cameras_are_wanted_at_start():
    _, _, core = _core()
    assert core.wanted_cameras() == {"rgb", "overlay"}
    _, _, core = _core(eager_cameras=[])
    assert core.wanted_cameras() == set()


def test_query_subscribes_then_idles_out():
    tr, clock, core = _core(camera_idle_s=15.0)
    rep = media.unpack_reply(tr.ask("adt4/r/media/camera/left"))
    assert rep["error"] == "no frame yet" and rep["subscribing"] is True
    assert "left" in core.wanted_cameras()
    _frame(core, "left")  # the node subscribed, a frame arrived
    rep = media.unpack_reply(tr.ask("adt4/r/media/camera/left"))
    assert "error" not in rep and rep["camera"] == "left"
    clock.advance(14.0)
    assert "left" in core.wanted_cameras()
    clock.advance(2.0)
    assert core.wanted_cameras() == {"rgb", "overlay"}
    core.drop_frame("left")  # what the node does when it unsubscribes
    rep = media.unpack_reply(tr.ask("adt4/r/media/camera/left"))
    assert rep["error"] == "no frame yet"


def test_unknown_camera_is_not_wanted():
    tr, _, core = _core()
    rep = media.unpack_reply(tr.ask("adt4/r/media/camera/nope"))
    assert rep["error"] == "unknown camera"
    assert "nope" not in core.wanted_cameras()


def test_video_keeps_a_camera_wanted_while_streaming():
    _, clock, core = _core(camera_idle_s=5.0)
    core._execute("video", {"camera": "back", "enable": True}, clock())
    assert "back" in core.wanted_cameras()
    clock.advance(4.0)  # keepalives arrive well within the video keepalive
    core._execute("video", {"camera": "back", "enable": True}, clock())
    clock.advance(4.0)
    assert "back" in core.wanted_cameras()
    core._execute("video", {"camera": "back", "enable": False}, clock())
    clock.advance(6.0)
    assert "back" not in core.wanted_cameras()
