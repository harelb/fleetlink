import dataclasses

import numpy as np
from helpers import states_equal, synthetic_mesh, synthetic_records

from fleetlink import codec
from fleetlink.codec import GraphDelta, GraphState, NodeRecord


def test_records_roundtrip():
    recs, _, _ = synthetic_records(30)
    dec = codec.decode_records(
        codec.unpack(codec.pack(codec.encode_records(list(recs.values()))))
    )
    assert [r.sym for r in dec] == list(recs)
    for r in dec:
        o = recs[r.sym]
        assert (r.layer, r.part, r.label, r.color, r.name, r.active) == (
            o.layer,
            o.part,
            o.label,
            o.color,
            o.name,
            o.active,
        )
        assert np.allclose(r.pos, o.pos, atol=1e-4)
        assert (r.bbox is None) == (o.bbox is None)
        if r.bbox:
            assert np.allclose(r.bbox, o.bbox, atol=1e-4)


def test_delta_and_snapshot_roundtrip():
    recs, edges, ls = synthetic_records(40)
    base = GraphState()
    delta = base.diff(recs, edges, ls)
    wire = codec.unpack(codec.pack(delta.encode()))
    mirror = GraphState()
    mirror.apply(GraphDelta.decode(wire))
    target = GraphState()
    target.records, target.edges, target.labelspaces = recs, edges, ls
    assert states_equal(mirror, target) == (True, "")
    assert mirror.labelspaces == ls

    snap = GraphState()
    snap.load_snapshot(codec.unpack(codec.pack(mirror.snapshot())))
    assert states_equal(snap, target) == (True, "")


def test_diff_apply_converges_over_evolution():
    rng = np.random.default_rng(3)
    recs, edges, ls = synthetic_records(60, seed=1)
    robot, mirror = GraphState(), GraphState()
    for step in range(15):
        # mutate: move some, delete some, add some, rewire edges
        recs = dict(recs)
        syms = list(recs)
        for s in rng.choice(syms, 5, replace=False):
            r = recs[s]
            recs[s] = dataclasses.replace(
                r, pos=tuple(np.asarray(r.pos) + rng.normal(0, 0.5, 3))
            )
        for s in rng.choice(syms, 3, replace=False):
            del recs[s]
        for i in range(4):
            sym = f"O({1000 + step * 10 + i})"
            recs[sym] = NodeRecord(sym, 2, 0, tuple(rng.uniform(-5, 5, 3)))
        live = list(recs)
        edges = {e for e in edges if e[0] in recs and e[1] in recs}
        edges |= {(live[i], live[i + 1]) for i in rng.integers(0, len(live) - 1, 5)}
        delta = robot.diff(recs, edges, ls)
        robot.apply(delta)
        mirror.apply(GraphDelta.decode(codec.unpack(codec.pack(delta.encode()))))
        target = GraphState()
        target.records, target.edges = recs, edges
        assert states_equal(mirror, target) == (True, ""), step
        assert robot.diff(recs, edges, ls).empty()


def test_split_delta_closest_first():
    recs = {
        f"O({i})": NodeRecord(f"O({i})", 2, 0, (float(i), 0.0, 0.0)) for i in range(10)
    }
    edges = {("O(0)", "O(1)"), ("O(1)", "O(9)"), ("O(8)", "O(9)")}
    delta = GraphState().diff(recs, edges)
    now, later = codec.split_delta(delta, 3, origin=(9.2, 0, 0))
    assert [r.sym for r in now.upserts] == ["O(9)", "O(8)", "O(7)"]
    assert now.edges_add == [("O(8)", "O(9)")]  # only edges between sent nodes go now
    assert set(later.edges_add) == {("O(0)", "O(1)"), ("O(1)", "O(9)")}
    mirror = GraphState()
    mirror.apply(now)
    mirror.apply(later)
    target = GraphState()
    target.records, target.edges = recs, edges
    assert states_equal(mirror, target) == (True, "")


def test_tile_quantization_error_below_1cm():
    pos, col, faces = synthetic_mesh(900, extent=37.3, seed=5)
    pos = pos + np.float32(0.00371)  # off-grid
    tiles = codec.tile_mesh(pos, col, faces, tile_size=8.0)
    assert len(tiles) > 4
    assert sum(t["nf"] for t in tiles.values()) == len(faces)
    err = 0.0
    tri_orig = pos[faces].mean(axis=1)
    for t in codec.unpack(codec.pack(tiles)).values():
        dpos, dcol, didx = codec.decode_tile(t)
        assert didx.max() < len(dpos)
        tri_dec = dpos[didx].mean(axis=1)
        # match each decoded triangle centroid to the nearest original centroid
        d = np.linalg.norm(tri_dec[:, None, :] - tri_orig[None, :, :], axis=2)
        err = max(err, float(d.min(axis=1).max()))
    assert err <= 0.01
    # vertex-level: every decoded vertex is within 1 cm of an original vertex
    t = next(iter(tiles.values()))
    dpos, _, _ = codec.decode_tile(t)
    d = np.linalg.norm(dpos[:, None, :] - pos[None, :, :], axis=2).min(axis=1)
    assert d.max() <= 0.01
    assert codec.tile_manifest(tiles) == {k: v["hash"] for k, v in tiles.items()}


def test_pack_is_thread_safe():
    # zstd (de)compressors are not thread-safe; the agent packs concurrently from
    # its tick, video and query threads.  A shared instance corrupted the heap
    # ("free(): corrupted unsorted chunks") under live video + mapping.
    import threading

    owners = {}

    def worker(i):
        owners[i] = (codec._zc(), codec._zd())  # keep alive: ids could be reused
        blob = {"i": i, "data": bytes([i % 256]) * 50_000}
        for _ in range(200):
            assert codec.unpack(codec.pack(blob)) == blob

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(c) for c, _ in owners.values()}) == 8  # one per thread
    assert len({id(d) for _, d in owners.values()}) == 8


def _place(sym, n=24, seed=0, **kw):
    rng = np.random.default_rng(seed)
    radii = tuple(round(float(v), 2) for v in rng.uniform(0.05, 4.0, n))
    states = tuple(int(v) for v in rng.integers(0, 4, n))
    return NodeRecord(sym, 3, 1, (1.0, 2.0, 0.5), radii=radii, states=states, **kw)


def test_boundary_roundtrip_and_old_peer_compat():
    plain = NodeRecord("O(1)", 2, 0, (0.0, 0.0, 0.0))
    places = [_place(f"t({i})", n=20 + i, seed=i) for i in range(3)]
    enc = codec.encode_records([plain, *places])
    dec = codec.decode_records(codec.unpack(codec.pack(enc)))
    assert dec[0].radii is None and dec[0].states is None
    for d, o in zip(dec[1:], places):
        assert d.states == o.states
        assert np.allclose(d.radii, o.radii, atol=0.006)
        assert d.key() == o.key()
    # graphs without boundaries keep the old wire keys exactly
    assert "nrad" not in codec.encode_records([plain])
    # an old payload (no boundary keys) still decodes
    old = {k: v for k, v in enc.items() if k not in ("nrad", "radii", "states")}
    assert all(r.radii is None for r in codec.decode_records(old))


def test_boundary_quantised_and_clamped():
    r = NodeRecord("t(1)", 3, 1, (0, 0, 0), radii=(0.123, -1.0, 1e6, float("nan")))
    (d,) = codec.decode_records(codec.encode_records([r]))
    assert d.radii == (0.12, 0.0, 655.35, 0.0)
    assert d.states == (0, 0, 0, 0)  # missing states pad as UNKNOWN


def test_boundary_delta_only_on_change():
    st = GraphState()
    recs = {"t(1)": _place("t(1)"), "t(2)": _place("t(2)", seed=2)}
    st.apply(st.diff(recs, set()))
    # sub-cm radius jitter: no resend
    jit = dataclasses.replace(recs["t(1)"], radii=tuple(v + 0.001 for v in recs["t(1)"].radii))
    assert st.diff({**recs, "t(1)": jit}, set()).empty()
    # a state flip: resend just that node, and the mirror converges
    flipped = list(recs["t(2)"].states)
    flipped[0] = (flipped[0] + 1) % 4
    new = {**recs, "t(2)": dataclasses.replace(recs["t(2)"], states=tuple(flipped))}
    delta = st.diff(new, set())
    assert [r.sym for r in delta.upserts] == ["t(2)"]
    mirror = GraphState()
    mirror.load_snapshot(codec.unpack(codec.pack(st.snapshot())))
    mirror.apply(GraphDelta.decode(codec.unpack(codec.pack(delta.encode()))))
    assert mirror.records["t(2)"].states == tuple(flipped)


def test_node_record_reads_trav_attributes():
    class State:
        def __init__(self, v):
            self.value = v

    class Attrs:
        position = (1.0, 2.0, 3.0)
        radii = [0.5, 1.004, 2.0]
        states = [State(1), State(2)]

    class Node:
        attributes = Attrs()
        layer = 3
        id = "t(7)"

    rec = codec.node_record(Node())
    assert rec.radii == (0.5, 1.0, 2.0)
    assert rec.states == (1, 2, 0)


def _tile_triangles(t):
    """A tile's triangles as a sorted array of quantised corner coordinates."""
    oq = np.round(np.asarray(t["origin"]) / t["scale"]).astype(np.int64)
    q = np.frombuffer(t["pos"], np.int16).reshape(-1, 3).astype(np.int64) + oq
    idx = np.frombuffer(t["idx"], np.uint32).reshape(-1, 3)
    tri = q[idx].reshape(len(idx), 9)
    return tri[np.lexsort(tri.T[::-1])]


def test_pack_tile_roundtrip_is_lossless():
    pos, col, faces = synthetic_mesh(900, extent=37.3, seed=5)
    tiles = codec.tile_mesh(pos, col, faces, tile_size=8.0)
    for t in codec.unpack(codec.pack(tiles)).values():
        p = codec.pack_tile(t)
        assert p["enc"] == codec.PACKED_TILE_ENCODING and p["ib"] == 2
        assert (p["nv"], p["nf"], p["hash"]) == (t["nv"], t["nf"], t["hash"])
        u = codec.unpack_tile(codec.unpack(codec.pack(p)))
        np.testing.assert_array_equal(_tile_triangles(u), _tile_triangles(t))
        # colours follow their vertices
        dp, dc, _ = codec.decode_tile(t)
        up, uc, _ = codec.decode_tile(u)
        ref = {tuple(np.round(a, 3)): tuple(c) for a, c in zip(dp, dc)}
        assert all(ref[tuple(np.round(a, 3))] == tuple(c) for a, c in zip(up, uc))


def test_pack_tile_wide_indices_and_extremes():
    # > 65536 vertices: 4-byte index planes; int16 extremes wrap correctly
    rng = np.random.default_rng(3)
    nv = 70000
    q = rng.integers(-32768, 32767, (nv, 3), dtype=np.int16)
    q[0] = (-32768, 32767, 0)
    q[1] = (32767, -32768, 0)
    idx = rng.permutation(np.tile(np.arange(nv), 3)).reshape(-1, 3).astype(np.uint32)
    t = {"key": "0_0_0", "origin": [8.0, -16.0, 0.0], "scale": 0.01, "pos": q.tobytes(),
         "col": rng.integers(0, 256, (nv, 3), dtype=np.uint8).tobytes(), "idx": idx.tobytes(),
         "nv": nv, "nf": len(idx), "hash": "x"}
    p = codec.pack_tile(t)
    assert p["ib"] == 4
    np.testing.assert_array_equal(_tile_triangles(codec.unpack_tile(p)), _tile_triangles(t))


def test_decimate_tiles_keeps_keys_and_has_no_cracks():
    pos, col, faces = synthetic_mesh(2500, extent=6.0, seed=2)  # 0.24 m spacing
    tiles = codec.tile_mesh(pos, col, faces, tile_size=8.0)
    dec = codec.decimate_tiles(tiles, 0.4)
    assert set(dec) <= set(tiles) and len(dec) >= len(tiles) - 1
    assert sum(t["nf"] for t in dec.values()) < sum(t["nf"] for t in tiles.values()) * 0.8  # bumpy: z noise 0.2 m
    # every shared border edge of the full mesh maps to identical vertices in
    # both tiles: collect each tile's quantised vertex set and check that
    # neighbouring tiles agree on all vertices near their common border
    verts = {}
    for k, t in dec.items():
        oq = np.round(np.asarray(t["origin"]) / t["scale"]).astype(np.int64)
        verts[k] = {tuple(v) for v in (np.frombuffer(t["pos"], np.int16).reshape(-1, 3).astype(np.int64) + oq)}
    shared = 0
    for a in dec:
        for b in dec:
            if a < b:
                shared += len(verts[a] & verts[b])
    assert shared > 20  # tiles share border vertices exactly
    # the coarse surface stays close to the original (vertex clustering error <= cell)
    for t in dec.values():
        dp, _, didx = codec.decode_tile(t)
        assert didx.max() < len(dp)
        d = np.linalg.norm(dp[:, None, :2] - pos[None, :, :2], axis=2).min(axis=1)
        assert d.max() < 0.4
    assert codec.decimate_tiles({}, 0.4) == {}
    assert codec.decimate_tiles(tiles, 0) == tiles
