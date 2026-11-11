"""Kimera-PGMO mesh deformation, ported to numpy for the base station.

Hydra's backend moves mesh vertices after a loop closure with
``kimera_pgmo::deformation::deformPoints``: each vertex takes a weighted blend of
the rigid corrections of its nearest *control points* (the deformation graph's
mesh nodes, one per ~``d_graph_resolution`` m), searched only among control
points whose stamps lie within ``tol_t`` of the vertex's stamp. If the base
holds the odometric (undeformed) vertices and the control points, a loop
closure only needs the control points' optimized poses (a few hundred kB)
instead of every changed mesh tile (MBs).

This mirrors ``kimera_pgmo/src/mesh_deformation.cpp`` (``processPoints``
timed variant + ``interpDeformation``) step by step:

* vertices are visited in index order; control points enter the search window
  while their stamp <= vertex stamp + tol_t (or fewer than k + 1 are in it), and
  leave (oldest first) while more than k + 1 remain and their stamp < vertex
  stamp - tol_t; invalid control points (no optimized value) never count;
* the search returns k + 2 neighbours: the k + 1 nearest are blended, the
  farthest sets ``d_max``; weights are ``1 - d / d_max`` (all 1 when the
  nearest and farthest are equally far);
* rotations are blended as raw quaternion coefficients (x, y, z, w; no sign
  alignment, like Eigen's ``coeffs()``) then normalized; translations as
  ``t_j - R_j c_j``; the vertex becomes ``R v + t`` in float32;
* if the window holds fewer than k + 1 control points, k drops to (count - 1)
  for the rest of the call (PGMO mutates k), and with <= 1 the vertex is left
  unchanged.

The one known difference: PGMO's window is a PCL octree that counts and
removes *1 m voxels*, not points, so two control points in one voxel count
once and leave together. With control points ~2.5 m apart this rarely matters;
``tests/test_deform.py`` and the C++ comparison quantify it.
"""

from __future__ import annotations

import numpy as np

NS = 1_000_000_000


def parse_dgrf(path) -> dict:
    """A Kimera-PGMO ``deformation_graph.dgrf`` (text) → per prefix:
    ``{"ctrl": float64[N,3] original control points, "ctrl_stamp": uint64[N],
    "pose": float64[N,7] (x y z qx qy qz qw) optimized value or NaN,
    "valid": bool[N]}`` for mesh prefixes (VERTEX lines), plus ``"nodes"``:
    ``{prefix: {index: pose7}}`` for every NODE (agents included)."""
    verts: dict[str, list] = {}
    nodes: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "VERTEX":
                key = int(p[1])
                verts.setdefault(chr(key >> 56), []).append(
                    (key & ((1 << 56) - 1), int(p[2]), float(p[3]), float(p[4]), float(p[5]))
                )
            elif p[0] == "NODE":
                key = int(p[1])
                nodes.setdefault(chr(key >> 56), {})[key & ((1 << 56) - 1)] = [float(x) for x in p[2:9]]
    out: dict = {"nodes": nodes}
    for prefix, rows in verts.items():
        rows.sort()
        n = rows[-1][0] + 1
        ctrl = np.zeros((n, 3))
        stamp = np.zeros(n, np.uint64)
        for i, s, x, y, z in rows:
            ctrl[i] = (x, y, z)
            stamp[i] = s
        pose = np.full((n, 7), np.nan)
        for i, v in nodes.get(prefix, {}).items():
            if i < n:
                pose[i] = v
        out[prefix] = {"ctrl": ctrl, "ctrl_stamp": stamp, "pose": pose, "valid": ~np.isnan(pose[:, 0])}
    return out


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(…,4) x y z w (unit) → (…,3,3)."""
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def windows(vstamps, ctrl_stamp, valid, k: int, tol_ns: int, order):
    """PGMO's sliding control-point window per visited vertex.

    Returns ``(lo, hi, kk)`` int arrays aligned with ``order``: the window is the
    valid control points with index in ``[lo, hi)``; ``kk`` is PGMO's (possibly
    reduced) k for that vertex, or -1 when the vertex is left unchanged."""
    n_ctrl = len(ctrl_stamp)
    m = len(order)
    lo_out = np.empty(m, np.int64)
    hi_out = np.empty(m, np.int64)
    kk_out = np.empty(m, np.int64)
    cs = ctrl_stamp.astype(np.int64)
    vs = vstamps.astype(np.int64)
    hi = lo = 0
    count = 0  # valid control points in the window ("leaves")
    for j, ii in enumerate(order):
        stamp = vs[ii]
        while hi < n_ctrl and (cs[hi] <= stamp + tol_ns or count < k + 1):
            if valid[hi]:
                count += 1
            hi += 1
        kk = k
        if count < k + 1:
            if count > 1:
                k = count - 1  # PGMO lowers k for the rest of the call
                kk = k
            else:
                lo_out[j], hi_out[j], kk_out[j] = lo, hi, -1
                continue
        lo_out[j], hi_out[j], kk_out[j] = lo, hi, kk
        while lo < n_ctrl and count > k + 1 and cs[lo] < stamp - tol_ns:
            if not valid[lo]:
                lo += 1
                continue
            count -= 1
            lo += 1
    return lo_out, hi_out, kk_out


def deform(points, vstamps, ctrl, ctrl_stamp, pose, valid, k: int = 3, tol_s: float = 15.0, indices=None):
    """Deform ``points`` (float32[M,3], the odometric vertices) like PGMO.

    ``pose``: float64[N,7] optimized control-point values (x y z qx qy qz qw),
    ``valid``: which control points have one. ``indices``: the vertices to
    deform, in visiting order (default all, like ``deformPoints`` with
    ``start_idx == 0``); the others are returned unchanged."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float32)
    out = points.copy()
    order = np.arange(len(points)) if indices is None else np.asarray(indices, np.int64)
    if len(order) == 0 or valid.sum() < k:
        return out
    lo, hi, kk = windows(vstamps, ctrl_stamp, valid, k, int(round(tol_s * NS)), order)
    R = quat_to_rot(np.nan_to_num(pose[:, 3:7]))
    t_minus = np.nan_to_num(pose[:, :3]) - np.einsum("nij,nj->ni", R, ctrl)  # t_j - R_j c_j
    qc = np.nan_to_num(pose[:, 3:7])
    ctrl32 = ctrl.astype(np.float32)
    # consecutive vertices with the same window share one KD-tree
    key = np.stack([lo, hi, kk], 1)
    breaks = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
    for seg in np.split(np.arange(len(order)), breaks):
        a, b, kseg = lo[seg[0]], hi[seg[0]], kk[seg[0]]
        if kseg < 0:
            continue
        cand = np.arange(a, b)[valid[a:b]]
        nq = min(kseg + 2, len(cand))  # PGMO asks the octree for k + 2
        tree = cKDTree(ctrl32[cand])
        vi = order[seg]
        d, nn = tree.query(points[vi], k=nn_shape(nq))
        d = d.reshape(len(vi), nq)
        nn = cand[nn.reshape(len(vi), nq)]
        d_max = d[:, -1:]
        const = (d[:, :1] == d_max) | (d_max == 0)
        w = np.where(const, 1.0, 1.0 - d[:, :-1] / np.where(d_max == 0, 1, d_max))  # (G, nq-1)
        use = nn[:, :-1]
        ws = w.sum(1, keepdims=True)
        q = (w[..., None] * qc[use]).sum(1) / ws
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        t = (w[..., None] * t_minus[use]).sum(1) / ws
        Rq = quat_to_rot(q)
        out[vi] = (np.einsum("gij,gj->gi", Rq, points[vi].astype(np.float64)) + t).astype(np.float32)
    return out


def nn_shape(n: int):
    """cKDTree.query returns 1-D results for k=1; always ask for a list."""
    return list(range(1, n + 1))


# ------------------------------------------------------------ wire format
POS_QUANT = 1e-3  # m
ROT_QUANT = 1.0 / 32767.0


def pack_poses(pose: np.ndarray, valid: np.ndarray, ref: np.ndarray | None = None) -> bytes:
    """Optimized control-point poses for one loop closure → compact bytes.

    Translations as int32 mm *relative to the original control point* (``ref``,
    corrections are metres at most) — int16 when they fit — and unit quaternions
    as int16 (w made non-negative). zstd on top."""
    import zstandard

    n = len(pose)
    p = np.nan_to_num(pose)
    q = p[:, 3:7] * np.where(p[:, 6:7] < 0, -1.0, 1.0)
    dt = p[:, :3] - (ref if ref is not None else 0.0)
    dq = np.round(dt / POS_QUANT)
    wide = np.abs(dq).max(initial=0) > 32767
    tq = dq.astype(np.int32 if wide else np.int16)
    qq = np.round(q / ROT_QUANT).astype(np.int16)
    head = np.array([n, int(wide)], np.uint32).tobytes()
    raw = head + np.packbits(valid).tobytes() + tq.tobytes() + qq.tobytes()
    return zstandard.ZstdCompressor(level=3).compress(raw)


def unpack_poses(data: bytes, ref: np.ndarray | None = None):
    """Inverse of :func:`pack_poses` → (pose float64[N,7], valid bool[N])."""
    import zstandard

    raw = zstandard.ZstdDecompressor().decompress(data)
    n, wide = np.frombuffer(raw[:8], np.uint32)
    o = 8
    nb = (int(n) + 7) // 8
    valid = np.unpackbits(np.frombuffer(raw[o : o + nb], np.uint8))[: int(n)].astype(bool)
    o += nb
    tdt = np.int32 if wide else np.int16
    sz = int(n) * 3 * np.dtype(tdt).itemsize
    tq = np.frombuffer(raw[o : o + sz], tdt).reshape(-1, 3)
    o += sz
    qq = np.frombuffer(raw[o : o + int(n) * 8], np.int16).reshape(-1, 4)
    pose = np.empty((int(n), 7))
    pose[:, :3] = tq * POS_QUANT + (ref if ref is not None else 0.0)
    q = qq * ROT_QUANT
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    pose[:, 3:7] = q / np.where(norm == 0, 1.0, norm)  # invalid rows are all zero
    pose[~valid] = np.nan
    return pose, valid
