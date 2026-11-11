"""Mesh level of detail: decimated tiles by default once a base asks, full tiles
on request (robot-wide or per tile), and old bases / robots keep full tiles."""

import numpy as np
from helpers import RecordingTransport, env_bytes, synthetic_mesh, synthetic_records

from fleetlink import codec
from fleetlink.agent import AgentCore
from fleetlink.receiver import Receiver
from fleetlink.transport import VirtualClock

from test_agent import _acks, _req, _run


def _dense():
    # ~0.1 m vertex spacing, like a Hydra mesh (the shared synthetic one is 3 m)
    return synthetic_mesh(n_quads=40_000, extent=10.0)


# ---------------------------------------------------------------- codec
def test_decimate_reduces_and_keeps_a_clean_mesh():
    pos, col, faces = _dense()
    dp, dc, df = codec.decimate_mesh(pos, col, faces, 0.4)
    assert 3 < len(faces) / len(df) < 40  # flat noisy grid; the Penn mesh: 8.7x
    assert dp.dtype == np.float32 and dc.dtype == np.uint8 and df.dtype == np.uint32
    assert df.max() < len(dp)  # every index valid, no unused vertex left
    assert len(np.unique(df)) == len(dp)
    assert (df[:, 0] != df[:, 1]).all() and (df[:, 1] != df[:, 2]).all() and (df[:, 0] != df[:, 2]).all()
    srt = np.sort(df, axis=1)
    assert len(np.unique(srt, axis=0)) == len(df)  # no duplicate triangles
    # the surface stays where it was: every new vertex is a mean of old ones in its cell
    assert np.abs(dp[:, :2]).max() <= 10.0 + 1e-3
    # tiling still covers the same area
    full, dec = codec.tile_mesh(pos, col, faces), codec.tile_mesh(dp, dc, df)
    assert set(dec) <= set(full) and len(dec) >= 0.9 * len(full)


def test_cells_never_straddle_tiles():
    """With the cell dividing the tile size, every cluster lies in one tile, so
    two tiles meeting at an edge use identical border vertices (no cracks)."""
    pos, col, faces = _dense()
    cell = 0.4
    q = np.floor(pos / cell).astype(np.int64)
    tile_of_vertex = np.floor(pos / codec.DEFAULT_TILE_SIZE).astype(np.int64)
    tile_of_cell = np.floor(q * cell / codec.DEFAULT_TILE_SIZE + 1e-9).astype(np.int64)
    assert (tile_of_vertex == tile_of_cell).all()


def test_decimate_edge_cases():
    pos, col, faces = _dense()
    assert codec.decimate_mesh(pos, col, faces[:0], 0.4)[2].shape == (0, 3)
    same = codec.decimate_mesh(pos, col, faces, 0)  # no cell: unchanged
    assert len(same[2]) == len(faces)
    tiny = codec.decimate_mesh(pos[:3] * 0.001, col[:3], np.array([[0, 1, 2]], np.uint32), 0.4)
    assert tiny[2].shape == (0, 3)  # a triangle inside one cell collapses


# ---------------------------------------------------------------- agent
def _agent():
    tr, clock = RecordingTransport(), VirtualClock()
    core = AgentCore("r", tr, {}, clock)
    recs, edges, ls = synthetic_records(10)
    core.on_records(recs, edges, ls, _dense())
    core.on_pose({"x": 0, "y": 0, "z": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0, "frame": "map", "t": 0})
    return tr, clock, core


def _tile_bytes(tr):
    return sum(len(d) for k, d, c in tr.sent if "/mesh/tile/" in k)


def _manifest(tr):
    return tr.messages("adt4/r/mesh/manifest")[-1][1]["p"]


def test_full_tiles_until_a_base_asks():
    """An old base never sends mesh_lod: it keeps getting the full tiles."""
    tr, clock, core = _agent()
    _run(clock, core, 30.0)
    assert core.mesh_lod is None and "lod" not in _manifest(tr)
    full = codec.tile_mesh(*_dense())
    assert {k: t["hash"] for k, t in core.tiles.items()} == {k: t["hash"] for k, t in full.items()}


def test_decimated_then_full_on_request():
    tr, clock, core = _agent()
    _req(tr, "d", "mesh_lod", {"level": "decimated"})
    _run(clock, core, 60.0)
    assert _acks(tr)[-1]["ok"] and core.mesh_lod == "decimated"
    dec_bytes = _tile_bytes(tr)
    m = _manifest(tr)
    assert m["lod"]["level"] == "decimated" and set(m["tiles"]) == set(core.tiles)
    assert all(t["lod"] == "decimated" for t in core.tiles.values())
    # full on request: every tile is re-sent (new hashes), more bytes
    tr.sent.clear()
    _req(tr, "f", "mesh_lod", {"level": "full"})
    _run(clock, core, 120.0)
    assert all(t["lod"] == "full" for t in core.tiles.values())
    assert _manifest(tr)["lod"]["level"] == "full"
    assert _tile_bytes(tr) > 3 * dec_bytes


def test_per_tile_full_on_top_of_decimated():
    tr, clock, core = _agent()
    _req(tr, "d", "mesh_lod", {"level": "decimated"})
    _run(clock, core, 60.0)
    key = sorted(core.tiles)[0]
    tr.sent.clear()
    _req(tr, "t", "mesh_lod", {"level": "full", "tiles": [key]})
    _run(clock, core, 30.0)
    assert core.tiles[key]["lod"] == "full"
    assert all(t["lod"] == "decimated" for k, t in core.tiles.items() if k != key)
    sent = {k.rsplit("/", 1)[1] for k, d, c in tr.sent if "/mesh/tile/" in k}
    assert sent == {key}  # only that tile changed
    assert _manifest(tr)["lod"]["tiles"] == {key: "full"}


def test_bad_level_is_refused():
    tr, clock, core = _agent()
    _req(tr, "x", "mesh_lod", {"level": "ultra"})
    _run(clock, core, 0.2)
    assert _acks(tr)[-1]["ok"] is False and core.mesh_lod is None


# ---------------------------------------------------------------- receiver
def _rx(**kw):
    tr, clock = RecordingTransport(), VirtualClock(100.0)
    rx = Receiver(tr, lambda r, k, d: None, clock=clock, threaded=False, **kw)
    rx.start()
    return tr, clock, rx


def _reqs(tr, op):
    out = [codec.unpack(d)["p"] for k, d, c in tr.sent if k.endswith("/ctl/req")]
    return [r for r in out if r["op"] == op]


def test_receiver_asks_new_robots_for_the_decimated_lod():
    tr, clock, rx = _rx()
    rx._on_msg("k", env_bytes("r", "mesh/manifest", 0, "E1", {"tiles": {}}))
    rx.tick(clock.advance(0.1))
    assert _reqs(tr, "mesh_lod")[0]["args"] == {"level": "decimated"}
    # the console's choice is remembered and re-sent to a restarted robot (new epoch)
    rx.set_mesh_lod("r", "full")
    rx._on_msg("k", env_bytes("r", "mesh/manifest", 0, "E2", {"tiles": {}}))
    rx.tick(clock.advance(0.1))
    assert _reqs(tr, "mesh_lod")[-1]["args"] == {"level": "full"}


def test_receiver_mesh_status_and_old_base_mode():
    tr, clock, rx = _rx()
    lod = {"level": "decimated", "tiles": {"a": "full"}, "cell": 0.4}
    rx._on_msg("k", env_bytes("r", "mesh/manifest", 0, "E1", {"tiles": {"a": "h1", "b": "h2"}, "lod": lod}))
    rx._on_msg("k", env_bytes("r", "mesh/tile", 0, "E1", {"key": "a", "hash": "h1", "pos": b"x" * 60, "col": b"c" * 30, "idx": b"i" * 120}))
    m = rx.link("r")["mesh"]
    assert m == {"lod": "decimated", "lod_tiles": {"a": "full"}, "wanted": "decimated", "tiles": 2, "pending": 1, "bytes": 210}
    # a base configured not to ask (mesh_lod=None) never sends the op
    tr2, clock2, rx2 = _rx(mesh_lod=None)
    rx2._on_msg("k", env_bytes("r", "mesh/manifest", 0, "E1", {"tiles": {}}))
    rx2.tick(clock2.advance(0.1))
    assert not _reqs(tr2, "mesh_lod") and rx2.link("r")["mesh"]["wanted"] is None


def test_odometric_mode_keeps_full_tiles_and_says_so():
    """Deformation needs every vertex's stamp: in odometric mode a decimated
    request is acknowledged, but the robot keeps sending full tiles and the
    manifest reports that (integrate/features-20261021 merge rule)."""
    tr, clock = RecordingTransport(), VirtualClock()
    core = AgentCore("r", tr, {"mesh_source": "odometric"}, clock)
    _req(tr, "d", "mesh_lod", {"level": "decimated"})
    _run(clock, core, 5.0)
    assert _acks(tr)[-1]["ok"]
    lod = core._manifest_payload()["lod"]
    assert lod["level"] == "full" and lod["odometric"] is True
