"""Base-side mesh deformation (fleetlink.deform) and the odometric mesh link
(fleetlink.mesh_deform_link): PGMO's interpolation, MeshDelta bookkeeping,
odometric tiles and the base's deformed view."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from fleetlink import codec, deform
from fleetlink.mesh_deform_link import ControlPoints, DeformView, OdomMesh, tile_odom_mesh, tile_vids

NS = 1_000_000_000


def _rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _quat_z(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])  # x y z w


def _grid_ctrl(n=6, step=2.5):
    xs = np.arange(n) * step
    ctrl = np.array([(x, y, 0.0) for x in xs for y in xs])
    stamp = (np.arange(len(ctrl)) * NS // 2).astype(np.uint64)  # 0.5 s apart
    return ctrl, stamp


def _poses(ctrl, R, t):
    q = _quat_z(np.arctan2(R[1, 0], R[0, 0]))
    pos = (R @ ctrl.T).T + t  # pose of each control point = correction applied to it
    return np.hstack([pos, np.tile(q, (len(ctrl), 1))])


def test_uniform_correction_is_a_rigid_motion():
    ctrl, cst = _grid_ctrl()
    R, t = _rot_z(0.1), np.array([1.0, -2.0, 0.5])
    pose = _poses(ctrl, R, t)
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 12, (500, 3)).astype(np.float32)
    st = rng.integers(0, int(cst.max()), 500).astype(np.uint64)
    out = deform.deform(pts, st, ctrl, cst, pose, np.ones(len(ctrl), bool), k=3, tol_s=1e6)
    want = (R @ pts.T.astype(np.float64)).T + t
    assert np.abs(out - want).max() < 1e-4


def _reference(pts, st, ctrl, cst, pose, valid, k, tol_s):
    """Direct, slow re-implementation of PGMO's stateful window + blend."""
    out = pts.astype(np.float64).copy()
    tol = int(tol_s * NS)
    hi = lo = count = 0
    R = deform.quat_to_rot(np.nan_to_num(pose[:, 3:]))
    for i in range(len(pts)):
        s = int(st[i])
        while hi < len(ctrl) and (int(cst[hi]) <= s + tol or count < k + 1):
            count += bool(valid[hi])
            hi += 1
        kk = k
        if count < k + 1:
            if count <= 1:
                continue
            k = kk = count - 1
        cand = [j for j in range(lo, hi) if valid[j]]
        d = np.linalg.norm(ctrl[cand] - pts[i], axis=1)
        o = np.argsort(d, kind="stable")[: kk + 2]
        nn, dd = np.asarray(cand)[o], d[o]
        dmax = dd[-1]
        const = dd[0] == dmax or dmax == 0
        w = np.ones(len(nn) - 1) if const else 1 - dd[:-1] / dmax
        q = (w[:, None] * pose[nn[:-1], 3:]).sum(0) / w.sum()
        q /= np.linalg.norm(q)
        t = (w[:, None] * (pose[nn[:-1], :3] - np.einsum("nij,nj->ni", R[nn[:-1]], ctrl[nn[:-1]]))).sum(0) / w.sum()
        out[i] = deform.quat_to_rot(q) @ pts[i] + t
        while lo < len(ctrl) and count > k + 1 and int(cst[lo]) < s - tol:
            if valid[lo]:
                count -= 1
            lo += 1
    return out


def test_matches_a_direct_reimplementation_with_windows_and_invalid_points():
    rng = np.random.default_rng(1)
    ctrl, cst = _grid_ctrl(8)
    # a smooth, non-rigid correction: rotation and shift grow along x
    pose = np.zeros((len(ctrl), 7))
    for j, c in enumerate(ctrl):
        a = 0.02 * c[0]
        pose[j, :3] = _rot_z(a) @ c + np.array([0.05 * c[0], 0.0, 0.01 * c[1]])
        pose[j, 3:] = _quat_z(a)
    valid = rng.random(len(ctrl)) > 0.15
    pose[~valid] = np.nan
    pts = rng.uniform(0, 17, (800, 3)).astype(np.float32)
    st = np.sort(rng.integers(0, int(cst.max()) + NS, 800)).astype(np.uint64)
    st[::7] = st[::7][::-1]  # some out-of-order stamps, like Hydra's
    got = deform.deform(pts, st, ctrl, cst, pose, valid, k=3, tol_s=3.0)
    want = _reference(pts, st, ctrl, cst, pose, valid, 3, 3.0)
    assert np.abs(got - want).max() < 1e-4


def test_indices_deform_only_those_vertices():
    ctrl, cst = _grid_ctrl()
    pose = _poses(ctrl, np.eye(3), np.array([0.0, 0.0, 1.0]))
    pts = np.zeros((10, 3), np.float32) + 3
    st = np.zeros(10, np.uint64)
    out = deform.deform(pts, st, ctrl, cst, pose, np.ones(len(ctrl), bool), indices=[5, 6])
    assert np.allclose(out[5:7, 2], 4.0) and np.allclose(out[:5, 2], 3.0)


def test_pose_packing_round_trip():
    ctrl, _ = _grid_ctrl()
    pose = _poses(ctrl, _rot_z(0.3), np.array([120.0, -80.0, 3.0]))  # > 32 m: int32 path
    valid = np.ones(len(ctrl), bool)
    valid[3] = False
    pose[3] = np.nan
    blob = deform.pack_poses(pose, valid)
    back, v2 = deform.unpack_poses(blob)
    assert (v2 == valid).all()
    assert np.abs(back[valid, :3] - pose[valid, :3]).max() <= 0.5e-3 + 1e-9
    # q and -q are the same rotation: compare rotations
    assert np.abs(deform.quat_to_rot(back[valid, 3:]) - deform.quat_to_rot(pose[valid, 3:])).max() < 1e-4
    assert len(blob) < 8 * 7 * len(ctrl)


def test_odom_mesh_applies_deltas_like_pgmo():
    m = OdomMesh()
    tri = lambda *f: np.array(f, np.int64).reshape(-1, 3)  # noqa: E731
    # delta 1: 4 vertices, none archived, 2 active faces
    m.apply({"prev_active_vertices": 0, "prev_active_faces": 0, "num_archived_vertices": 0, "prev_to_curr": {},
             "pos": np.arange(12, dtype=np.float32).reshape(4, 3), "col": np.zeros((4, 3), np.uint8),
             "stamp": np.arange(4, dtype=np.uint64), "faces": tri(0, 1, 2, 1, 2, 3), "archive_faces": tri()})  # fmt: skip
    assert len(m.pos) == 4 and len(m.faces()) == 2
    # delta 2: previous 4 were active; 2 now archived (sent first), active part has 3
    # vertices: previous active 2,3 became 0,1 of the new active part (prev_to_curr)
    pos2 = np.array([[0, 0, 0], [1, 1, 1], [20, 20, 20], [30, 30, 30], [40, 40, 40]], np.float32)
    m.apply({"prev_active_vertices": 4, "prev_active_faces": 2, "num_archived_vertices": 2,
             "prev_to_curr": {2: 0, 3: 1}, "pos": pos2, "col": np.zeros((5, 3), np.uint8),
             "stamp": np.arange(10, 15, dtype=np.uint64),
             "faces": tri(2, 3, 4), "archive_faces": tri(0, 1, 2)})  # fmt: skip
    assert len(m.pos) == 5 and m.archived_vertices == 2
    assert np.allclose(m.pos[2], [20, 20, 20])
    f = {tuple(x) for x in m.faces().tolist()}
    assert f == {(0, 1, 2), (2, 3, 4)}
    assert len(m.f_arch) == 1 and m.f_arch.tolist() == [[0, 1, 2]] or len(m.f_pend) == 1
    # delta 3: replaces the active part again (prev_active 3)
    m.apply({"prev_active_vertices": 3, "prev_active_faces": 1, "num_archived_vertices": 0, "prev_to_curr": {},
             "pos": np.ones((2, 3), np.float32), "col": np.zeros((2, 3), np.uint8),
             "stamp": np.zeros(2, np.uint64), "faces": tri(0, 1, 0), "archive_faces": tri()})  # fmt: skip
    assert len(m.pos) == 4


def _mesh(n=40, seed=2):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.linspace(0, 15, n), np.linspace(0, 15, n))
    pos = np.stack([xs.ravel(), ys.ravel(), 0.1 * rng.random(n * n)], 1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            faces += [(a, a + 1, a + n), (a + 1, a + n + 1, a + n)]
    stamp = (np.arange(n * n) * (NS // 20)).astype(np.uint64)  # 0.05 s per vertex
    col = (rng.random((n * n, 3)) * 255).astype(np.uint8)
    return pos, col, np.array(faces, np.int64), stamp


def test_odom_tiles_and_base_view_reproduce_the_deformation():
    pos, col, faces, stamp = _mesh()
    ctrl = ControlPoints()
    cpos, _ = _grid_ctrl(7)
    cst = np.linspace(0, int(stamp.max()), len(cpos)).astype(np.uint64)
    for i, (p, s) in enumerate(zip(cpos, cst)):
        ctrl.add_original(i, int(s), p)
    tiles = tile_odom_mesh(pos, col, faces, stamp, ctrl)
    assert all(t["odom"] for t in tiles.values())
    assert sum(len(np.frombuffer(t["ctrl"]["idx"], np.uint32)) for t in tiles.values()) == len(cpos)
    view = DeformView(k=3, tol_s=15.0)
    # no values yet: the view is the odometric mesh
    v0 = view.view(tiles)
    k0 = next(k for k, t in tiles.items() if t["nv"])
    assert v0[k0]["pos"] == tiles[k0]["pos"] and "vid" not in v0[k0]
    assert not any(k.endswith("~ctrl") for k in v0)  # control-point carriers are not drawn
    pose = np.zeros((len(cpos), 7))
    for j, c in enumerate(cpos):
        a = 0.01 * c[0]
        pose[j, :3] = _rot_z(a) @ c + np.array([0.0, 0.03 * c[0], 0.0])
        pose[j, 3:] = _quat_z(a)
    assert view.set_values({"data": deform.pack_poses(pose, np.ones(len(cpos), bool))})
    assert not view.set_values({"data": deform.pack_poses(pose, np.ones(len(cpos), bool))})  # same bytes
    out = view.view(tiles)
    # what the base shows == deforming the whole (tile-quantized) odometric mesh at once
    q, packed = [], []
    for k, t in tiles.items():
        if not t["nv"]:
            continue
        p0 = np.frombuffer(t["pos"], np.int16).reshape(-1, 3) * t["scale"] + np.asarray(t["origin"])
        vid = tile_vids(t)
        p1 = np.frombuffer(out[k]["pos"], np.int16).reshape(-1, 3) * t["scale"] + np.asarray(t["origin"])
        q.append((vid, p0, p1))
    vid = np.concatenate([v for v, _, _ in q])
    p0 = np.concatenate([a for _, a, _ in q]).astype(np.float32)
    p1 = np.concatenate([b for _, _, b in q])
    st = stamp[vid]
    p_back, v_back = deform.unpack_poses(deform.pack_poses(pose, np.ones(len(cpos), bool)))
    order = np.argsort(vid, kind="stable")
    want = deform.deform(p0, st // NS * NS + (st % NS) // 1_000_000 * 1_000_000, cpos, cst, p_back, v_back, 3, 15.0, indices=order)
    assert np.abs(p1 - want).max() <= codec.QUANT  # requantized at 1 cm
    assert np.abs(p1 - p0).max() > 0.1  # it really moved
    # a new values message changes every tile's hash for the viewer, not the raw tiles
    pose2 = pose.copy()
    pose2[:, 1] += 0.5
    view.set_values({"data": deform.pack_poses(pose2, np.ones(len(cpos), bool))})
    out2 = view.view(tiles)
    assert all(out2[k]["hash"] != out[k]["hash"] for k in out if tiles[k]["nv"])


PGMO_HARNESS = os.environ.get("PGMO_DEFORM_HARNESS", "")
LC_RUN = os.path.expanduser("~/adt4_output/imgstore_lc_on_20261017/hydra/backend")


@pytest.mark.skipif(not (PGMO_HARNESS and os.path.exists(LC_RUN)), reason="set PGMO_DEFORM_HARNESS (C++ ground truth)")
def test_matches_kimera_pgmo_on_a_real_loop_closure_run(tmp_path):
    """C++ ground truth: Kimera-PGMO's deformPoints on a real Hydra run with 77
    loop closures (see docs/MESH_DEFORMATION_DESIGN.md for the harness)."""
    g = deform.parse_dgrf(os.path.join(LC_RUN, "deformation_graph.dgrf"))["s"]
    with open(os.path.join(LC_RUN, "mesh.ply")) as f:
        n = 0
        for line in f:
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        a = np.loadtxt(f, max_rows=n)
    pts = a[:, :3].astype(np.float32)
    st = a[:, 7].astype(np.uint64) * np.uint64(NS) + a[:, 8].astype(np.uint64)
    g["ctrl"].astype(np.float64).tofile(tmp_path / "ctrl_pos.f64")
    g["ctrl_stamp"].tofile(tmp_path / "ctrl_stamp.u64")
    cv = np.zeros((len(g["ctrl"]), 8))
    cv[:, 0] = g["valid"]
    cv[:, 1:] = np.nan_to_num(g["pose"])
    cv.tofile(tmp_path / "ctrl_val.f64")
    pts.tofile(tmp_path / "vert_pos.f32")
    st.tofile(tmp_path / "vert_stamp.u64")
    subprocess.run([PGMO_HARNESS, str(tmp_path), "3", "15.0"], check=True, capture_output=True)
    ref = np.fromfile(tmp_path / "out_pos.f32", np.float32).reshape(-1, 3)
    got = deform.deform(pts, st, g["ctrl"], g["ctrl_stamp"], g["pose"], g["valid"], k=3, tol_s=15.0)
    e = np.linalg.norm(got.astype(np.float64) - ref, axis=1)
    assert np.mean(e < 1e-2) > 0.999 and e.max() < 0.05
