"""C6 live image-store mirror: robot store -> fleetlink -> base store (ADT4 console API.md §13).

Exercises the real hydra_python.image_store writer and the fleetlink.image_transfer
codec (skipped without hydra_python.image_store): resume across agent and console
restarts with nothing lost or duplicated, gap repair on a lossy link, store
rotation (new Hydra session), and that the mirrored store loads with
agentic_navigation's SAM3 keyframe loader.
"""

import os
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("hydra_python.image_store")
pytest.importorskip("zstandard")
cv2 = pytest.importorskip("cv2")
from hydra_python.image_store import FrameKind, ImageStore, make_node_id  # noqa: E402

from fleetlink import codec, store_mirror  # noqa: E402
from fleetlink.agent import AgentCore  # noqa: E402
from fleetlink.receiver import Receiver  # noqa: E402
from fleetlink.store_mirror import StoreMirror, StoreTail  # noqa: E402
from fleetlink.transport import LinkEmulator, VirtualClock  # noqa: E402

# an agentic_navigation source checkout (its SAM3 keyframe loader); the test that
# needs it is skipped unless AGENTIC_NAV_SRC is set
AGENTIC_NAV_SRC = os.environ.get("AGENTIC_NAV_SRC", "")
AGENTIC_NAV = pathlib.Path(AGENTIC_NAV_SRC) if AGENTIC_NAV_SRC else None

W, H = 64, 48

# the agent mirrors only with the store's tail (live) mode, Hydra PR #221
needs_tail = pytest.mark.skipif(
    not store_mirror.tail_mode_available(), reason=store_mirror.TAIL_MODE_MISSING
)


class RobotStore:
    """A small robot-side store written the way Hydra lays it out."""

    def __init__(self, root: pathlib.Path, robot_id: int = 1):
        self.root = root
        self.src = root.parent / (root.name + "_src")
        self.src.mkdir(parents=True, exist_ok=True)
        self.w = ImageStore.open_writer(root, robot_id=robot_id)
        self.sensor = self.w.add_sensor(
            "camera", W, H, 50.0, 50.0, W / 2, H / 2, 0.001, np.eye(4)
        )
        self.ts = 1_000_000_000

    def add(self, n: int, kind=FrameKind.KEYFRAME, own: bool = True) -> list[int]:
        """``n`` frames; keyframes are anchored, each to its own agent node: ``own`` ones
        taken at the node (identity offset, ``keyframes`` mode sends them), the others
        0.3 m off it (only ``all`` mode sends them)."""
        ids = []
        rng = np.random.default_rng(self.ts)
        for _ in range(n):
            self.ts += 100_000_000
            rgb = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
            depth = rng.integers(500, 5000, (H, W), dtype=np.uint16)
            rp = self.src / f"{self.ts}_rgb.jpg"
            dp = self.src / f"{self.ts}_depth.png"
            cv2.imwrite(str(rp), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(dp), depth)
            pose = np.eye(4)
            pose[0, 3] = len(ids) * 0.5
            anchor = aTb = None
            if kind == FrameKind.KEYFRAME:
                anchor = make_node_id("a", self.ts // 100_000_000)
                aTb = np.eye(4)
                if not own:
                    aTb[0, 3] = 0.3
            ids.append(
                self.w.add_frame_ref(
                    self.sensor,
                    self.ts,
                    kind=kind,
                    world_T_body_odom=pose,
                    rgb=rp,
                    depth=dp,
                    link="copy",
                    anchor_node_id=anchor,
                    anchor_T_body=aTb,
                )  # fmt: skip
            )
        return ids

    def close(self):
        self.w.close()


def _pair(tmp_path, link_kw=None, store_cfg=None, clock=None, agent=None):
    clock = clock or VirtualClock()
    link = LinkEmulator(
        clock=clock, **(link_kw or {"bandwidth": None, "latency": 0.01})
    )
    cfg = {
        "store_path": str(tmp_path / "robot" / "image_store"),
        "store_poll_s": 0.2,
        "store_send": True,  # off by default; test_store_send_* cover the switch
        **(store_cfg or {}),
    }
    core = AgentCore("r", link.robot, cfg, clock=clock)
    rx = Receiver(
        link.base,
        clock=clock,
        threaded=False,
        store_mirror=StoreMirror(tmp_path / "live", clock=clock, threaded=False),
    )
    rx.start()
    return clock, link, core, rx


def _run(clock, link, core, rx, seconds, dt=0.05):
    for _ in range(int(seconds / dt)):
        clock.advance(dt)
        core.tick()
        link.advance()
        rx.tick()


def _mirror_ids(rx) -> list[int]:
    st = rx.store_stats("r")
    with ImageStore.open(st["path"]) as s:
        return [f.frame_id for f in s.frames()]


@needs_tail
def test_mirror_keyframes_and_resume_across_restarts(tmp_path):
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(5)
    robot.add(3, own=False)  # keyframes mode skips these
    clock, link, core, rx = _pair(tmp_path)
    _run(clock, link, core, rx, 6.0)
    assert _mirror_ids(rx) == kf
    st = rx.store_stats("r")
    assert st["frames"] == 5 and st["duplicates"] == 0 and st["backlog"] == 0
    assert st["path"].startswith(str(tmp_path / "live" / "r"))
    link_st = rx.link("r")
    assert link_st["store"]["watermark"] == kf[-1] and link_st["bytes"]["C6"] > 0
    assert core.stats()["budgets"]["C6"] == 150_000

    # agent restart (new epoch): resumes after the base watermark, no duplicates
    core.close()
    kf += robot.add(2)
    core2 = AgentCore("r", link.robot, core.cfg, clock=clock)
    _run(clock, link, core2, rx, 6.0)
    assert _mirror_ids(rx) == kf
    assert rx.store_stats("r")["duplicates"] == 0
    assert core2.store.frames_sent == 2

    # console restart: a new receiver reads its watermark from disk
    rx.stop()
    kf += robot.add(2)
    rx2 = Receiver(
        link.base, clock=clock, threaded=False,
        store_mirror=StoreMirror(tmp_path / "live", clock=clock, threaded=False),
    )  # fmt: skip
    rx2.start()
    _run(clock, link, core2, rx2, 6.0)
    assert _mirror_ids(rx2) == kf
    assert rx2.store_stats("r")["frames"] == 2  # only the new ones crossed
    robot.close()


@needs_tail
def test_mirror_store_loads_with_sam3_keyframe_loader(tmp_path):
    if AGENTIC_NAV is None:
        pytest.skip("set AGENTIC_NAV_SRC to an agentic_navigation checkout")
    if not (AGENTIC_NAV / "agentic_navigation").is_dir():
        pytest.skip("agentic_navigation checkout not found")
    sys.path.insert(0, str(AGENTIC_NAV))
    try:
        from agentic_navigation.sam3_frontend.keyframe_store import (
            load_keyframes_from_store,
        )
    except ImportError as exc:
        pytest.skip(f"agentic_navigation not importable: {exc}")
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(4)
    clock, link, core, rx = _pair(tmp_path)
    _run(clock, link, core, rx, 5.0)
    with ImageStore.open(rx.store_stats("r")["path"]) as store:
        kfs = load_keyframes_from_store(store)
        assert [k.timestamp_ns for k in kfs] == sorted(k.timestamp_ns for k in kfs)
        assert len(kfs) == len(kf)
        k = kfs[1]
        assert cv2.imread(k.rgb_path).shape[:2] == (H, W)
        # fast mode halves depth on the wire; the mirror restores full resolution
        depth = cv2.imread(k.depth_path, cv2.IMREAD_UNCHANGED)
        assert depth.shape == (H, W) and depth.dtype == np.uint16
        assert k.calib.width == W and k.calib.fx == 50.0
        assert np.isclose(k.world_T_body[0, 3], 0.5)
    robot.close()


@needs_tail
def test_mirror_repairs_gaps_on_a_lossy_link(tmp_path):
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(12)
    clock, link, core, rx = _pair(
        tmp_path,
        link_kw={
            "bandwidth": None, "latency": 0.02, "loss": 0.25, "seed": 3,
            # zenoh DROPs C6 on congestion: model it as lossy (C0 stays reliable)
            "reliable": {"C0": True, "C1": False, "C2": True, "C3": False,
                         "C4": True, "C5": False, "C6": False},
        },
    )  # fmt: skip
    _run(clock, link, core, rx, 60.0)
    assert _mirror_ids(rx) == kf
    st = rx.store_stats("r")
    assert st["gaps"] > 0 and st["frames"] == len(kf)
    robot.close()


@needs_tail
def test_mirror_respects_c6_budget_and_all_frames_mode(tmp_path):
    robot = RobotStore(tmp_path / "robot" / "image_store")
    ids = robot.add(3) + robot.add(3, own=False)
    clock, link, core, rx = _pair(
        tmp_path,
        store_cfg={"store_frames": "all", "store_compression": "lossless",
                   "budgets": {"C6": 20_000}},
    )  # fmt: skip
    _run(clock, link, core, rx, 2.0)
    sent_early = core.sched.sent_bytes["C6"]
    assert sent_early <= 20_000 * 3 + 30_000  # burst + one frame of debt
    _run(clock, link, core, rx, 30.0)
    assert sorted(_mirror_ids(rx)) == sorted(ids)
    robot.close()


@needs_tail
def test_store_rotation_starts_a_new_mirror(tmp_path):
    root = tmp_path / "robot" / "image_store"
    robot = RobotStore(root)
    first = robot.add(2)
    clock, link, core, rx = _pair(tmp_path)
    _run(clock, link, core, rx, 4.0)
    old = rx.store_stats("r")
    assert old["frames"] == 2
    # Hydra session restart: the store is moved aside and a new one starts
    robot.close()
    root.rename(tmp_path / "robot" / "session1")
    robot2 = RobotStore(root)
    robot2.ts = 5_000_000_000
    second = robot2.add(3)
    _run(clock, link, core, rx, 8.0)
    new = rx.store_stats("r")
    assert new["store_id"] != old["store_id"] and new["path"] != old["path"]
    assert _mirror_ids(rx) == second
    with ImageStore.open(old["path"]) as s:
        assert [f.frame_id for f in s.frames()] == first
    robot2.close()


def test_store_tail_without_store_and_disabled(tmp_path):
    tail = StoreTail(tmp_path / "nothing")
    tail.refresh()
    assert tail.id is None and "no image store" in tail.error
    with pytest.raises(ValueError):
        tail.set_cursor("x", None)
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.0)
    core = AgentCore("r", link.robot, {}, clock=clock)  # store_path unset
    assert core.store is None and core._store_state()["enabled"] is False
    with pytest.raises(ValueError):
        StoreTail(tmp_path, frames="bogus")


def test_receiver_set_budget_accepts_c6():
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.0)
    core = AgentCore("r", link.robot, {}, clock=clock)
    rx = Receiver(link.base, clock=clock, threaded=False)
    rx.start()
    rx.set_budget("r", "C6", 50_000)
    for _ in range(20):
        clock.advance(0.05)
        core.tick()
        link.advance()
        rx.tick()
    assert core.sched.budget("C6") == 50_000
    with pytest.raises(ValueError):
        rx.set_budget("r", "C9", 1)
    assert codec  # imported for parity with the other suites


@needs_tail
def test_threaded_mirror_writes_on_worker(tmp_path):
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(3)
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.01)
    core = AgentCore(
        "r", link.robot,
        {"store_path": str(tmp_path / "robot" / "image_store"), "store_poll_s": 0.2, "store_send": True},
        clock=clock,
    )  # fmt: skip
    rx = Receiver(
        link.base, clock=clock, threaded=False, store_root=str(tmp_path / "live")
    )
    rx.start()
    _run(clock, link, core, rx, 4.0)
    rx.store.flush()
    assert _mirror_ids(rx) == kf
    rx.stop()
    robot.close()


@needs_tail
def test_store_send_off_by_default_and_console_switch(tmp_path):
    """Nothing goes on C6 until the base asks; off again keeps the chain, so
    turning it back on continues with no loss and no duplicates."""
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(4)
    clock, link, core, rx = _pair(tmp_path, store_cfg={"store_send": False})
    _run(clock, link, core, rx, 4.0)
    assert core.store.frames_sent == 0 and rx.link("r")["bytes"].get("C6", 0) == 0
    st = rx.link("r")
    assert st["store_send"] == {"robot": False, "wanted": None}
    assert st["store"]["backlog"] == 4  # armed, waiting

    rx.set_store_send("r", True)
    _run(clock, link, core, rx, 4.0)
    assert _mirror_ids(rx) == kf
    assert rx.link("r")["store_send"] == {"robot": True, "wanted": True}

    # off with frames still queued (tiny budget): they go back, nothing lost
    rx.set_budget("r", "C6", 1_000)
    kf += robot.add(3)
    _run(clock, link, core, rx, 1.0)
    rx.set_store_send("r", False)
    _run(clock, link, core, rx, 2.0)
    assert core.store_send is False and not core.sched.queues["C6"]
    sent_off = core.store.frames_sent
    kf += robot.add(2)
    _run(clock, link, core, rx, 3.0)
    assert core.store.frames_sent == sent_off  # nothing sent while off

    rx.set_budget("r", "C6", 150_000)
    rx.set_store_send("r", True)
    _run(clock, link, core, rx, 6.0)
    assert _mirror_ids(rx) == kf
    assert rx.store_stats("r")["duplicates"] == 0
    rx.stop()
    robot.close()


@needs_tail
def test_store_send_reasked_after_agent_restart(tmp_path):
    robot = RobotStore(tmp_path / "robot" / "image_store")
    kf = robot.add(3)
    clock, link, core, rx = _pair(tmp_path, store_cfg={"store_send": False})
    rx.set_store_send("r", True)
    _run(clock, link, core, rx, 4.0)
    assert _mirror_ids(rx) == kf

    core.close()  # the new agent starts with it off again
    kf += robot.add(2)
    cfg = dict(core.cfg, store_send=False)
    core2 = AgentCore("r", link.robot, cfg, clock=clock)
    _run(clock, link, core2, rx, 6.0)
    assert core2.store_send is True
    assert _mirror_ids(rx) == kf
    rx.stop()
    robot.close()


def test_store_send_without_store_is_refused(tmp_path):
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.0)
    core = AgentCore("r", link.robot, {}, clock=clock)  # store_path unset
    rx = Receiver(link.base, clock=clock, threaded=False)
    rx.start()
    rx.set_store_send("r", True)
    for _ in range(40):
        clock.advance(0.05)
        core.tick()
        link.advance()
        rx.tick()
    assert core.store_send is False
    assert rx.link("r")["store_send"]["robot"] is None  # no store: no switch shown
    assert not rx.pending_requests()  # refused once, never re-asked


def test_no_tail_mode_turns_the_mirror_off(tmp_path, monkeypatch):
    """Hydra without the store's tail mode: store_path set, but no mirror, one reason."""
    monkeypatch.setattr(store_mirror, "tail_mode_available", lambda: False)
    clock = VirtualClock()
    link = LinkEmulator(clock=clock, bandwidth=None, latency=0.0)
    core = AgentCore("r", link.robot, {"store_path": str(tmp_path)}, clock=clock)
    assert core.store is None
    assert core.store_error == store_mirror.TAIL_MODE_MISSING
    assert core._store_state()["enabled"] is False
