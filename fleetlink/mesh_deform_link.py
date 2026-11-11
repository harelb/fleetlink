"""Odometric mesh + base-side deformation ("mesh_source: odometric").

Default fleetlink sends the robot's *deformed* mesh (from the DSG) as hashed
tiles; a loop closure moves (nearly) every vertex, so (nearly) every tile is
sent again (MBs). In odometric mode:

robot (:class:`OdomMeshAgent`)
    rebuilds Hydra's **odometric** mesh from the frontend's
    ``~/full_mesh_update`` (``kimera_pgmo_msgs/MeshDelta``, frame odom; the
    stream Hydra's backend itself deforms), keeps the deformation-graph control
    points from ``~/mesh_graph_incremental`` (original position + stamp) and their
    optimized poses from the backend's ``~/mesh_graph``. Tiles are made from the
    odometric mesh, so they only change where the robot adds geometry, and each
    tile also carries its vertices' stamps and global indices and the control
    points that lie in it. The optimized poses travel as one small packed
    message (``mesh/values``, ~10 kB for 3,000 control points) whenever they
    change.

base (:class:`DeformView`)
    keeps the odometric tiles, gathers every vertex in global index order and
    runs PGMO's interpolation (:mod:`fleetlink.deform`) with the latest values,
    then hands out ordinary (deformed) tiles to the console.

Note: Hydra deforms already-archived vertices only once (its backend passes
``start_idx = last archived count`` to ``deformPoints``), so its own DSG mesh
keeps stale corrections in old areas after later loop closures. The base
deforms every vertex with the latest values, which agrees with the optimized
graph; the two meshes can therefore differ in old areas, by design.
"""

from __future__ import annotations

import hashlib
import threading

import numpy as np

from . import codec, deform

ODOM_TILE_VERSION = 1


# ------------------------------------------------------------------ robot side
class OdomMesh:
    """Kimera-PGMO ``MeshDelta::updateMesh`` applied to numpy arrays.

    Vertices: archived prefix + active suffix (the suffix is replaced by every
    delta). Faces are kept as three bands, fully archived / pending (archived
    but touching active vertices) / active, with the same band sizes as PGMO, so
    the next delta's truncation lines up (order inside a band does not matter for
    rendering)."""

    def __init__(self):
        self.pos = np.zeros((0, 3), np.float32)
        self.col = np.zeros((0, 3), np.uint8)
        self.stamp = np.zeros(0, np.uint64)
        self.f_arch = np.zeros((0, 3), np.int64)
        self.f_pend = np.zeros((0, 3), np.int64)
        self.f_act = np.zeros((0, 3), np.int64)
        self.archived_vertices = 0
        self.version = 0

    def apply(self, d: dict) -> None:
        """``d``: ``prev_active_vertices``, ``prev_active_faces``,
        ``num_archived_vertices``, ``prev_to_curr`` ({prev: curr}),
        ``pos`` f32[n,3], ``col`` u8[n,3], ``stamp`` u64[n], ``faces``
        (active, delta-local) and ``archive_faces`` (delta-local) int[...,3]."""
        nv = len(self.pos)
        vstart = nv - int(d["prev_active_vertices"]) if nv else 0
        if vstart < 0:
            raise ValueError("mesh delta does not fit (missed a delta?)")
        self.pos = np.concatenate([self.pos[:vstart], np.asarray(d["pos"], np.float32).reshape(-1, 3)])
        self.col = np.concatenate([self.col[:vstart], np.asarray(d["col"], np.uint8).reshape(-1, 3)])
        self.stamp = np.concatenate([self.stamp[:vstart], np.asarray(d["stamp"], np.uint64).reshape(-1)])
        threshold = vstart + int(d["num_archived_vertices"])

        all_f = np.concatenate([self.f_arch, self.f_pend, self.f_act])
        nf = len(all_f)
        fstart = nf - int(d["prev_active_faces"]) if nf else 0
        arch = self.f_arch
        pend = all_f[len(self.f_arch) : max(fstart, len(self.f_arch))]
        remap = d.get("prev_to_curr") or {}
        if len(pend) and remap:
            pend = _remap_faces(pend, vstart, remap)
        new_arch = np.asarray(d["archive_faces"], np.int64).reshape(-1, 3) + vstart
        cand = np.concatenate([pend, new_arch])
        below = np.all(cand < threshold, axis=1)
        self.f_arch = np.concatenate([arch, cand[below]])
        self.f_pend = cand[~below]
        self.f_act = np.asarray(d["faces"], np.int64).reshape(-1, 3) + vstart
        self.archived_vertices = threshold
        self.version += 1

    def faces(self) -> np.ndarray:
        f = np.concatenate([self.f_arch, self.f_pend, self.f_act])
        ok = np.all(f < len(self.pos), axis=1) & np.all(f >= 0, axis=1)
        return f[ok]


def _remap_faces(faces: np.ndarray, offset: int, remap: dict) -> np.ndarray:
    """PGMO ``remapFace``: indices >= offset go through ``remap`` (relative to
    offset); faces with an index the delta no longer has are dropped."""
    out = faces.copy()
    keep = np.ones(len(faces), bool)
    hi = faces >= offset
    if hi.any():
        lut = {int(k): int(v) for k, v in remap.items()}
        for (i, j) in zip(*np.nonzero(hi)):
            v = lut.get(int(faces[i, j]) - offset)
            if v is None:
                keep[i] = False
            else:
                out[i, j] = v + offset
    return out[keep]


def delta_from_msg(msg) -> dict:
    """``kimera_pgmo_msgs/MeshDelta`` → :meth:`OdomMesh.apply` input."""
    vs = msg.vertex_updates
    n = len(vs)
    pos = np.empty((n, 3), np.float32)
    col = np.zeros((n, 3), np.uint8)
    stamp = np.zeros(n, np.uint64)
    for i, v in enumerate(vs):
        pos[i] = (v.pos.x, v.pos.y, v.pos.z)
        if v.has_color:
            col[i] = (int(v.color.r * 255), int(v.color.g * 255), int(v.color.b * 255))
        if v.has_stamp:
            stamp[i] = v.stamp
    tri = lambda fs: np.array([list(f.vertex_indices) for f in fs], np.int64).reshape(-1, 3)  # noqa: E731
    return {
        "prev_active_vertices": msg.prev_active_vertices,
        "prev_active_faces": msg.prev_active_faces,
        "num_archived_vertices": msg.num_archived_vertices,
        "prev_to_curr": dict(zip(msg.previous_indices, msg.current_indices)),
        "pos": pos,
        "col": col,
        "stamp": stamp,
        "faces": tri(msg.face_updates),
        "archive_faces": tri(msg.face_archive_updates),
    }


class ControlPoints:
    """Deformation-graph mesh nodes: original position + stamp (from
    ``mesh_graph_incremental``), optimized pose (from the backend's
    ``mesh_graph``)."""

    def __init__(self):
        self.pos = np.zeros((0, 3))
        self.stamp = np.zeros(0, np.uint64)
        self.known = np.zeros(0, bool)
        self.pose = np.zeros((0, 7))
        self.valid = np.zeros(0, bool)
        self.version = 0  # bumps when the optimized poses change

    def _grow(self, n: int) -> None:
        if n <= len(self.pos):
            return
        add = n - len(self.pos)
        self.pos = np.concatenate([self.pos, np.zeros((add, 3))])
        self.stamp = np.concatenate([self.stamp, np.zeros(add, np.uint64)])
        self.known = np.concatenate([self.known, np.zeros(add, bool)])
        self.pose = np.concatenate([self.pose, np.full((add, 7), np.nan)])
        self.valid = np.concatenate([self.valid, np.zeros(add, bool)])

    def add_original(self, index: int, stamp: int, xyz) -> None:
        self._grow(index + 1)
        self.pos[index] = xyz
        self.stamp[index] = stamp
        self.known[index] = True

    def set_values(self, items) -> bool:
        """``items``: iterable of (index, stamp, pose7 x y z qx qy qz qw).
        Returns whether anything changed beyond the wire quantization."""
        items = list(items)
        if not items:
            return False
        self._grow(max(i for i, _, _ in items) + 1)
        before = self.pose.copy()
        for i, st, p in items:
            self.pose[i] = p
            self.valid[i] = True
            if not self.known[i] and st:
                self.stamp[i] = st
        t_moved = np.nan_to_num(np.abs(self.pose[:, :3] - before[:, :3]), nan=1.0).max(initial=0)
        q_moved = np.nan_to_num(np.abs(self.pose[:, 3:] - before[:, 3:]), nan=1.0).max(initial=0)
        changed = t_moved >= deform.POS_QUANT or q_moved >= deform.ROT_QUANT
        if changed:
            self.version += 1
        return changed


def nodes_from_pose_graph(msg):
    """``pose_graph_tools_msgs/PoseGraph`` → [(index, stamp_ns, pose7)]
    (the node stamp travels in ``header.stamp``)."""
    out = []
    for n in msg.nodes:
        p, q = n.pose.position, n.pose.orientation
        stamp = int(n.header.stamp.sec) * 1_000_000_000 + int(n.header.stamp.nanosec)
        out.append((int(n.key), stamp, (p.x, p.y, p.z, q.x, q.y, q.z, q.w)))
    return out


MAX_TILE_WIRE = 96 * 1024  # packed bytes; zenoh drops bigger messages whole on a busy link
MIN_TILE_SIZE = 1.0  # m: a tile is not split below this


def tile_odom_mesh(pos, col, faces, stamp, ctrl: ControlPoints | None, tile_size=codec.DEFAULT_TILE_SIZE,
                   max_wire=MAX_TILE_WIRE):  # fmt: skip
    """Odometric tiles: :func:`codec.tile_mesh` tiles plus, per tile, ``vid``
    (global vertex indices as uint32 deltas, sorted), ``stamp`` (u32 ms after
    ``t0``), ``t0`` (u64 ns) and ``odom: 1``. A tile that packs to more than
    ``max_wire`` bytes is split into half-size tiles (keys ``i_j_k@<size>``),
    recursively. Control points travel in small carrier tiles (``i_j_k~ctrl``,
    no faces) per ``tile_size`` cell. Every hash covers all fields."""
    tiles: dict = {}
    if len(faces):
        t0 = int(stamp.min()) if len(stamp) else 0
        _tile_into(tiles, pos, col, faces, stamp, t0, tile_size, "", max_wire)
    if ctrl is not None and ctrl.known.any():
        cidx = np.flatnonzero(ctrl.known)
        ckey = np.floor(ctrl.pos[cidx] / tile_size).astype(np.int64)
        groups: dict = {}
        for i, k in zip(cidx, ckey):
            groups.setdefault(tuple(int(x) for x in k), []).append(int(i))
        for k, idx in groups.items():
            ci = np.asarray(sorted(idx), np.int64)
            c = {"idx": ci.astype(np.uint32).tobytes(), "stamp": ctrl.stamp[ci].astype(np.uint64).tobytes(),
                 "pos": ctrl.pos[ci].astype(np.float32).tobytes()}  # fmt: skip
            key = codec.tile_key(k) + "~ctrl"
            tiles[key] = {
                "key": key, "origin": [float(v) * tile_size for v in k], "scale": codec.QUANT,
                "pos": b"", "col": b"", "idx": b"", "nv": 0, "nf": 0, "odom": ODOM_TILE_VERSION,
                "vid": b"", "t0": 0, "stamp": b"", "ctrl": c,
                "hash": hashlib.blake2b(c["idx"] + c["stamp"] + c["pos"], digest_size=8).hexdigest(),
            }  # fmt: skip
    return tiles


def _tile_into(out, pos, col, faces, stamp, t0, size, suffix, max_wire) -> None:
    tiles = codec.tile_mesh(pos, col, faces, size)
    if not tiles:
        return
    ijk = np.floor(pos[faces].mean(axis=1) / size).astype(np.int64)
    order = np.lexsort((ijk[:, 2], ijk[:, 1], ijk[:, 0]))
    breaks = np.flatnonzero(np.any(np.diff(ijk[order], axis=0) != 0, axis=1)) + 1
    for group in np.split(order, breaks):
        key = codec.tile_key(ijk[group[0]])
        t = tiles[key]
        used = np.unique(faces[group].reshape(-1))  # tile_mesh's vertex order
        t["vid"] = np.diff(used, prepend=0).astype(np.uint32).tobytes()
        t["t0"] = t0
        t["stamp"] = ((stamp[used].astype(np.int64) - t0) // 1_000_000).astype(np.uint32).tobytes()
        t["odom"] = ODOM_TILE_VERSION
        t["ctrl"] = {"idx": b"", "stamp": b"", "pos": b""}
        if size / 2 >= MIN_TILE_SIZE and len(codec.pack(t)) > max_wire:
            _tile_into(out, pos, col, faces[group], stamp, t0, size / 2, f"@{size / 2:g}", max_wire)
            continue
        h = hashlib.blake2b(digest_size=8)
        h.update(t["hash"].encode())
        h.update(t["vid"])
        h.update(t["stamp"])
        t["hash"] = h.hexdigest()
        t["key"] = key + suffix
        out[t["key"]] = t


def tile_vids(t: dict) -> np.ndarray:
    """A tile's global vertex indices (stored as deltas)."""
    return np.cumsum(np.frombuffer(t["vid"], np.uint32).astype(np.int64))


class OdomMeshAgent:
    """Robot side: odometric mesh + control points → (mesh arrays for the
    tiler, values payload)."""

    VALUES_RESEND_S = 30.0

    def __init__(self):
        self.mesh = OdomMesh()
        self.ctrl = ControlPoints()
        self._sent_values_version = -1
        self._t_values = -1e9
        self.version = 0  # mesh or control-point originals changed

    def on_delta(self, d: dict) -> None:
        self.mesh.apply(d)
        self.version += 1

    def on_ctrl_incremental(self, nodes) -> None:
        added = False
        for i, st, p in nodes:
            if i >= len(self.ctrl.known) or not self.ctrl.known[i]:
                self.ctrl.add_original(i, st, p[:3])
                added = True
        if added:
            self.version += 1

    def on_values(self, nodes) -> None:
        self.ctrl.set_values(nodes)

    def mesh_input(self):
        """What AgentCore's tiler gets: (pos, col, faces, stamp, ctrl)."""
        m = self.mesh
        if not len(m.pos):
            return None
        return (m.pos.astype(np.float64), m.col, m.faces(), m.stamp, self.ctrl)

    def values_due(self, now: float) -> bool:
        if not self.ctrl.valid.any():
            return False
        if self.ctrl.version != self._sent_values_version:
            return True
        return now - self._t_values >= self.VALUES_RESEND_S

    def values_payload(self, now: float) -> dict:
        self._sent_values_version = self.ctrl.version
        self._t_values = now
        # translations relative to the original control points (the base has them
        # from the carrier tiles): small corrections fit int16 (~10 kB for 3,000)
        ref = np.where(self.ctrl.known[:, None], self.ctrl.pos, 0.0)
        return {
            "version": self.ctrl.version,
            "n": int(len(self.ctrl.pose)),
            "rel": 1,
            "data": deform.pack_poses(self.ctrl.pose, self.ctrl.valid, ref=ref),
        }

    def force_values(self) -> None:
        """Re-send the values on the next tick (resync / new base session)."""
        self._sent_values_version = -1


def tile_odom(mesh_input, tile_size=codec.DEFAULT_TILE_SIZE):
    """AgentCore tiler for odometric mode."""
    pos, col, faces, stamp, ctrl = mesh_input
    return tile_odom_mesh(pos, col, faces, stamp, ctrl, tile_size)


# ------------------------------------------------------------------- base side
class DeformView:
    """Base side: odometric tiles + values → deformed tiles (ordinary format)."""

    K = 3  # Hydra pgmo num_interp_pts (dcist hydra.yaml)
    TOL_S = 15.0  # Hydra pgmo interp_horizon / time_horizon

    def __init__(self, k: int | None = None, tol_s: float | None = None):
        self.k = k or self.K
        self.tol_s = tol_s or self.TOL_S
        self._lock = threading.Lock()
        self.values = None  # (pose, valid, digest of the payload)
        self._data = None
        self._cache_key = None
        self._cache: dict = {}

    def set_values(self, p: dict) -> bool:
        """New optimized poses; True when they differ from the ones held (the
        payload bytes are compared: a restarted agent restarts its version)."""
        data = bytes(p["data"])
        with self._lock:
            if self.values is not None and self._data == data:
                return False
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            self._data = data
            # unpacked at view time against the control points held then
            self.values = (data, bool(p.get("rel")), digest)
            return True

    def view(self, tiles: dict) -> dict:
        """Deformed tiles for ``tiles`` (the odometric ones as received); tiles
        without ``odom`` pass through. Cached per (tile hashes, values version)."""
        odom = {k: t for k, t in tiles.items() if t.get("odom")}
        if not odom:
            return dict(tiles)
        with self._lock:
            values = self.values
        key = (tuple(sorted((k, t["hash"]) for k, t in odom.items())), values[2] if values else None)
        if key == self._cache_key:
            out = dict(self._cache)
        else:
            out = self._deform(odom, values)
            self._cache_key, self._cache = key, out
            out = dict(out)
        for k, t in tiles.items():
            if not t.get("odom"):
                out[k] = t
        return out

    def _deform(self, odom: dict, values) -> dict:
        parts, ctrl = [], {}
        for k, t in odom.items():
            c = t.get("ctrl") or {}
            if c.get("idx"):
                idx = np.frombuffer(c["idx"], np.uint32)
                st = np.frombuffer(c["stamp"], np.uint64)
                ps = np.frombuffer(c["pos"], np.float32).reshape(-1, 3)
                for i, s, p in zip(idx, st, ps):
                    ctrl[int(i)] = (int(s), p)
            if not t["nv"]:
                continue
            q = np.frombuffer(t["pos"], np.int16).reshape(-1, 3).astype(np.float64)
            pos = q * t["scale"] + np.asarray(t["origin"])
            vid = tile_vids(t)
            st = np.frombuffer(t["stamp"], np.uint32).astype(np.uint64) * np.uint64(1_000_000) + np.uint64(t["t0"])
            parts.append((k, pos, vid, st))
        out = {}
        moved = None
        if values is not None and ctrl and parts:
            data, rel, _digest = values
            n = max(ctrl) + 1
            cpos = np.zeros((n, 3))
            cst = np.zeros(n, np.uint64)
            known = np.zeros(n, bool)
            for i, (s, p) in ctrl.items():
                cpos[i], cst[i], known[i] = p, s, True
            n_vals = int(np.frombuffer(zstd_head(data), np.uint32)[0])
            if n_vals > n:
                cpos = np.concatenate([cpos, np.zeros((n_vals - n, 3))])
                cst = np.concatenate([cst, np.zeros(n_vals - n, np.uint64)])
                known = np.concatenate([known, np.zeros(n_vals - n, bool)])
                n = n_vals
            ref = cpos[:n_vals] if rel else None
            pose, valid = deform.unpack_poses(data, ref=ref)
            pz = np.full((n, 7), np.nan)
            pz[: len(pose)] = pose
            vz = np.zeros(n, bool)
            vz[: len(valid)] = valid
            vz &= known  # a value without its original control point cannot be used
            allpos = np.concatenate([p for _, p, _, _ in parts]).astype(np.float32)
            allvid = np.concatenate([v for _, _, v, _ in parts])
            allst = np.concatenate([s for _, _, _, s in parts])
            order = np.argsort(allvid, kind="stable")  # PGMO visits vertices in index order
            moved = deform.deform(allpos, allst, cpos, cst, pz, vz, self.k, self.tol_s, indices=order)
        o = 0
        for k, pos, _vid, _st in parts:
            t = odom[k]
            new = moved[o : o + len(pos)].astype(np.float64) if moved is not None else pos
            o += len(pos)
            out[k] = _requantize(t, new, values[2] if values is not None else None)
        return out


def zstd_head(data: bytes) -> bytes:
    """First 8 bytes of a :func:`deform.pack_poses` payload (count, width)."""
    import zstandard

    return zstandard.ZstdDecompressor().decompressobj().decompress(data)[:8]


def _requantize(t: dict, pos: np.ndarray, version) -> dict:
    origin = np.asarray(t["origin"])
    q = np.clip(np.round((pos - origin) / t["scale"]), -32768, 32767).astype(np.int16)
    h = hashlib.blake2b(digest_size=8)
    h.update(t["hash"].encode())
    h.update(str(version).encode())
    h.update(q.tobytes())
    return {
        "key": t["key"],
        "origin": t["origin"],
        "scale": t["scale"],
        "pos": q.tobytes(),
        "col": t["col"],
        "idx": t["idx"],
        "nv": t["nv"],
        "nf": t["nf"],
        "hash": h.hexdigest(),
    }
