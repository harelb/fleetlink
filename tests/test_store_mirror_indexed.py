"""C6 mirror + map pack on stores built by hydra_python.image_store's indexer / tailer.

Hydra and Khronos do not write the image store: they log keyframe zips, track logs
and event logs, and the store's tailer builds the store while Hydra runs
(FORMAT.md §6). The tail mode is Hydra PR #221 (not on Hydra develop); the tests that
need it skip without it. The mirror must only send frames the tailer has settled (anchor and
keyframe role decided), carry kept object observations with their node binding (object
images on the base during the run), key the store session on the tailer's Hydra
session, never ship InputData zips / EXR depth, and ``pack`` must index a run that has
only raw logs.

Uses the synthetic Hydra writers of Hydra's image-store tests (``raw_logs``,
``test_tail``, in ``hydra/python/tests/image_store``); skipped when they are not there.
"""

import json
import os
import pathlib
import sys

import numpy as np
import pytest

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
his = pytest.importorskip("hydra_python.image_store")
pytest.importorskip("zstandard")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("yaml")
from hydra_python.image_store import FrameKind, ImageStore, make_node_id  # noqa: E402
from test_store_mirror import RobotStore, _pair, _run, needs_tail  # noqa: E402

from fleetlink import image_pack, image_transfer  # noqa: E402
from fleetlink.store_mirror import StoreTail  # noqa: E402

# Hydra's image-store test helpers (synthetic Hydra logs). Skipping per test, not the
# module: a module-level skip under ROS's launch_testing pytest plugin ends the session.
HIS_TESTS = pathlib.Path(his.__file__).resolve().parents[3] / "tests" / "image_store"
R = TT = tail_mod = tail_state = None
if (HIS_TESTS / "raw_logs.py").is_file():
    sys.path.insert(0, str(HIS_TESTS))
    import raw_logs as R  # noqa: E402
if (HIS_TESTS / "test_tail.py").is_file():
    try:
        import test_tail as TT  # noqa: E402
        from hydra_python.image_store import tail as tail_mod  # noqa: E402
        from hydra_python.image_store.indexer import tail_state  # noqa: E402
    except ImportError:
        TT = None
needs_raw_logs = pytest.mark.skipif(R is None, reason=f"no raw_logs.py in {HIS_TESTS}")
needs_tail_tests = pytest.mark.skipif(
    TT is None, reason=f"no image-store tail mode / test_tail.py in {HIS_TESTS}"
)


def _robot_frames(store_dir) -> dict:
    with ImageStore.open(store_dir, index=False) as s:
        return {f.frame_id: f for f in s.frames()}


def _selected(store_dir) -> list[int]:
    with ImageStore.open(store_dir, index=False) as s:
        return [
            f.frame_id
            for f in image_transfer.select_frames(
                s, "keyframes", include_kept_observations=True
            )
        ]


@needs_tail_tests
@needs_tail
def test_mirror_follows_a_tailed_run_with_object_images(tmp_path):
    """Hydra's run written step by step, tailed, mirrored: the base gets exactly the
    keyframes + kept-observation frames, each only once settled, with the observations
    bound to their nodes (object images), and the session id stays stable."""
    run = tmp_path / "robot"
    ev, steps = TT.live_run(run)
    tailer = tail_mod.Tailer(run, settle_s=0.0)
    clock, link, core, rx = _pair(
        tmp_path, store_cfg={"store_path": str(run / "image_store")}
    )
    store_dir = run / "image_store"

    ids_seen = set()
    for i, step in enumerate(steps):
        step()
        tailer.poll()
        _run(clock, link, core, rx, 3.0)
        st = rx.store_stats("r")
        if st:
            ids_seen.add(st["store_id"])
        # every frame on the base is settled on the robot (not pending / provisional)
        if st.get("path") and pathlib.Path(st["path"], "store.sqlite").exists():
            robot = _robot_frames(store_dir)
            with ImageStore.open(st["path"], index=False) as m:
                for f in m.frames():
                    r = robot[f.frame_id]
                    assert (f.kind, f.anchor_node_id) == (r.kind, r.anchor_node_id), (
                        i,
                        f.frame_id,
                    )
        if i == 3:  # tracks bound at step 2, their frames settled once agent 1 exists:
            # object images reach the base during the run
            with ImageStore.open(st["path"], index=False) as m:
                obs = m.observations(R.O1, kept_only=True)
                assert obs and all(o.rgb_path.is_file() for o in obs)
                assert m.mask(obs[0]) is not None
    assert len(ids_seen) == 1  # one Hydra session: one mirror
    sess = tail_state(store_dir)["session"]
    assert sess is not None

    TT.write_save_files(run)
    assert tailer.poll()["final"]
    _run(clock, link, core, rx, 4.0)
    st = rx.store_stats("r")
    assert st["duplicates"] == 0 and st["write_errors"] == 0
    with ImageStore.open(st["path"], index=False) as m:
        mirrored = sorted(f.frame_id for f in m.frames())
        assert mirrored == sorted(_selected(store_dir))
        # images are JPEG / 16-bit PNG, never the zips' PNG-in-zip / EXR
        for f in m.frames():
            assert f.rgb_path is None or f.rgb_path.suffix == ".jpg"
            assert f.depth_path is None or f.depth_path.suffix == ".png"
        assert {o.track_id for o in m.observations(R.O1, kept_only=True)} >= {12}
    tail = core.store.state()
    assert tail["held"] is None and tail["backlog"] == 0


def test_tail_holds_pending_and_provisional_frames(tmp_path):
    root = tmp_path / "robot" / "image_store"
    robot = RobotStore(root)
    ids = robot.add(6)
    robot.w.set_meta("indexer.r1.pending_frames", json.dumps([ids[4]]))
    robot.w.set_meta("indexer.r1.provisional_frames", json.dumps([ids[2], ids[5]]))
    tail = StoreTail(root)
    tail.refresh()
    tail.set_cursor(tail.id, None)
    assert [f for f, _ in tail.take(10)] == ids[:2]
    assert tail.state()["held"] == ids[2] and tail.state()["hold_reason"] == "anchor"
    robot.w.set_meta("indexer.r1.provisional_frames", json.dumps([]))
    assert [f for f, _ in tail.take(10)] == ids[2:4]
    robot.w.set_meta("indexer.r1.pending_frames", json.dumps([]))
    assert [f for f, _ in tail.take(10)] == ids[4:]
    robot.close()


def test_tail_waits_for_the_binding_of_kept_observations(tmp_path):
    root = tmp_path / "robot" / "image_store"
    robot = RobotStore(root)
    (kf,) = robot.add(1)
    (fid,) = robot.add(1, FrameKind.OBSERVATION)
    (later,) = robot.add(1)
    mask = np.zeros((48, 64), bool)
    mask[10:20, 5:30] = True
    robot.w.add_observation(
        5, fid, mask=mask, bbox2d=(5, 10, 29, 19), score=0.7, kept=True
    )
    now = [100.0]
    tail = StoreTail(root, clock=lambda: now[0], bind_wait_s=30.0)
    tail.refresh()
    tail.set_cursor(tail.id, None)
    assert [f for f, _ in tail.take(10)] == [kf]  # track 5 not bound yet
    assert tail.state()["hold_reason"] == "binding"
    node = make_node_id("O", 3)
    robot.w.bind_track(5, node)
    assert [f for f, _ in tail.take(10)] == [fid, later]
    blob = tail.encode(fid, kf)["blob"]
    (o,) = image_transfer.decode_frame(blob)["meta"]["observations"]
    assert (
        o["node_id"] == node and o["track_id"] == 5 and o["bbox2d"] == [5, 10, 29, 19]
    )
    assert o["mask_rle"]  # the mask travels as RLE even when the store has a PNG

    # a binding that never comes: sent after bind_wait_s
    (fid2,) = robot.add(1, FrameKind.OBSERVATION)
    robot.w.add_observation(6, fid2, kept=True)
    assert tail.take(10) == []
    now[0] += 31.0
    assert [f for f, _ in tail.take(10)] == [fid2]
    robot.close()


@needs_tail
def test_session_id_follows_the_tailer(tmp_path):
    root = tmp_path / "robot" / "image_store"
    robot = RobotStore(root)
    robot.add(2)
    tail = StoreTail(root)
    tail.refresh()
    plain = tail.id  # not tailed: inode + first frame
    assert plain is not None
    state = {"live": True, "final": False, "heartbeat_wall": 0, "session": None}
    robot.w.set_meta("indexer.tail", json.dumps(state))
    tail.refresh()
    assert tail.id is None  # tailed, but Hydra's first event log is not open yet
    robot.w.set_meta("indexer.tail", json.dumps({**state, "session": 111}))
    tail.refresh()
    first = tail.id
    # the backend log opens later (first merge): the newest-open session changes, the
    # frontend log's does not -> same store session
    ev = {
        "frontend.jsonl": {"source": "frontend", "session": 111},
        "backend.jsonl": {"source": "backend", "session": 150},
    }
    robot.w.set_meta(
        "indexer.tail", json.dumps({**state, "session": 150, "events": ev})
    )
    tail.refresh()
    assert tail.id == first
    ev["frontend.jsonl"]["session"] = 222  # Hydra restarted: its frontend log reopened
    robot.w.set_meta(
        "indexer.tail", json.dumps({**state, "session": 222, "events": ev})
    )
    tail.refresh()
    assert first is not None and tail.id not in (None, first, plain)
    robot.close()


def test_encode_reencodes_exr_depth_and_refuses_archives(tmp_path):
    root = tmp_path / "store"
    w = ImageStore.open_writer(root, robot_id=0)
    sid = w.add_sensor("camera", 8, 6, 5.0, 5.0, 4.0, 3.0, 0.001, np.eye(4))
    rgb = tmp_path / "c.jpg"
    cv2.imwrite(str(rgb), np.full((6, 8, 3), 90, np.uint8))
    exr = tmp_path / "d.exr"
    cv2.imwrite(str(exr), np.full((6, 8), 1.5, np.float32))
    fid = w.add_frame_ref(
        sid, 10, kind=FrameKind.KEYFRAME, rgb=rgb, depth=exr, link="copy"
    )
    zipped = tmp_path / "k.zip"
    zipped.write_bytes(b"PK")
    bad = w.add_frame_ref(sid, 20, kind=FrameKind.KEYFRAME, rgb=zipped, link="copy")
    w.close()
    with ImageStore.open(root, index=False) as s:
        for mode in ("none", "lossless", "fast"):
            d = image_transfer.decode_frame(image_transfer.encode_frame(s, fid, mode))
            depth = image_transfer.full_res_depth(d)
            assert depth.dtype == np.uint16 and int(depth[0, 0]) == 1500, mode
        with pytest.raises(ValueError, match="archive"):
            image_transfer.encode_frame(s, bad, "none")


@needs_raw_logs
def test_pack_indexes_a_run_with_only_raw_logs(tmp_path):
    run = R.build_run(tmp_path / "run")
    assert not (run / "image_store").exists()
    out = tmp_path / "run.tar.zst"
    stats = image_pack.pack_store(run, out, frames="keyframes", compression="fast")
    assert (run / "image_store" / "store.sqlite").is_file() and stats["frames"] > 0
    dest = tmp_path / "dest"
    res = image_pack.unpack_store(out, dest)
    assert res["check"]["ok"], res["check"]
    with ImageStore.open(dest / "image_store", index=False) as s:
        assert s.observations(R.O1, kept_only=True)


@needs_tail
def test_kept_observation_on_a_passed_frame_is_sent_late(tmp_path):
    """A track logged on a keyframe archive's frame after the cursor passed it (the
    frame was not an agent's own keyframe, not selected then): the frame goes out of the chain as a
    late frame and the base gets the object image; an agent restart re-sends it."""
    root = tmp_path / "robot" / "image_store"
    robot = RobotStore(root)
    (sub,) = robot.add(1, own=False)
    kfs = robot.add(2)
    clock, link, core, rx = _pair(tmp_path, store_cfg={"store_path": str(root)})
    _run(clock, link, core, rx, 3.0)
    path = rx.store_stats("r")["path"]
    with ImageStore.open(path, index=False) as m:
        assert [f.frame_id for f in m.frames()] == kfs
    node = make_node_id("O", 9)
    robot.w.bind_track(4, node)
    robot.w.add_observation(4, sub, kept=True, score=0.5)
    _run(clock, link, core, rx, 3.0)
    st = rx.store_stats("r")
    assert st["late"] == 1 and st["watermark"] == kfs[-1] and st["gaps"] == 0
    with ImageStore.open(path, index=False) as m:
        (o,) = m.observations(node, kept_only=True)
        assert o.frame_id == sub and o.rgb_path.is_file()
    assert core.store.state()["late_sent"] == 1
    _run(clock, link, core, rx, 3.0)  # not sent again
    assert rx.store_stats("r")["late"] == 1
    robot.close()
