"""End-to-end odometric mesh mode over the emulated link with a REAL Hydra run
(77 loop closures): the agent gets the mesh as staged odometric deltas plus the
deformation graph; after the mesh has converged a "loop closure" delivers the
optimized control-point poses. The base must show the deformed mesh, and the
loop closure must cost the small values message instead of re-sending tiles.

The run's saved mesh stands in for the odometric mesh (only the deformed mesh is
saved); what is checked is that the base reproduces deforming it."""

from __future__ import annotations

import os

import numpy as np
import pytest

from fleetlink import codec, deform
from fleetlink.agent import AgentCore
from fleetlink.mesh_deform_link import tile_vids
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator, VirtualClock

RUN = os.path.expanduser("~/adt4_output/imgstore_lc_on_20261017/hydra/backend")
ROBOT = "lc"
DT = 0.05
STAGES = 10
LINK = dict(bandwidth=2_500_000.0, latency=0.15, loss=0.01, seed=3)  # 20 Mbit/s: fast test


def _load():
    if not os.path.exists(os.path.join(RUN, "mesh.ply")):
        pytest.skip(f"missing {RUN}")
    g = deform.parse_dgrf(os.path.join(RUN, "deformation_graph.dgrf"))["s"]
    with open(os.path.join(RUN, "mesh.ply")) as f:
        nv = nf = 0
        for line in f:
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            if line.startswith("element face"):
                nf = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        a = np.loadtxt(f, max_rows=nv)
        faces = np.loadtxt(f, max_rows=nf, dtype=np.int64)[:, 1:4]
    pos = a[:, :3].astype(np.float32)
    col = a[:, 3:6].astype(np.uint8)
    st = a[:, 7].astype(np.uint64) * np.uint64(1_000_000_000) + a[:, 8].astype(np.uint64)
    return g, pos, col, st, faces


def _deltas(pos, col, st, faces, stages):
    """Reveal vertices in index order; everything archived at once (faces that
    span two stages are dropped: delta-local indices cannot reach back)."""
    n = len(pos)
    cuts = np.linspace(0, n, stages + 1).astype(int)
    stage_of = np.searchsorted(cuts, np.arange(n), side="right") - 1
    fs = stage_of[faces]
    same = (fs[:, 0] == fs[:, 1]) & (fs[:, 1] == fs[:, 2])
    out = []
    for k in range(stages):
        a, b = cuts[k], cuts[k + 1]
        fk = faces[same & (fs[:, 0] == k)] - a
        out.append({
            "prev_active_vertices": 0, "prev_active_faces": 0, "num_archived_vertices": b - a,
            "prev_to_curr": {}, "pos": pos[a:b], "col": col[a:b], "stamp": st[a:b],
            "faces": np.zeros((0, 3), np.int64), "archive_faces": fk,
        })  # fmt: skip
    return out, faces[same]


class Sim:
    def __init__(self, cfg):
        self.clock = VirtualClock()
        self.link = LinkEmulator(self.clock, **LINK)
        self.rx = Receiver(self.link.base, None, clock=self.clock, threaded=False)
        self.rx.start()
        self.agent = AgentCore(ROBOT, self.link.robot, cfg, self.clock)

    def run(self, seconds):
        for _ in range(int(seconds / DT)):
            t = self.clock.advance(DT)
            self.link.advance()
            self.agent.tick(t)
            self.rx.tick(t)

    def c3(self):
        return self.agent.sched.sent_bytes.get("C3", 0)

    def converged(self):
        if ROBOT not in self.rx._robots or not self.agent.tiles:
            return False
        got = {k: t["hash"] for k, t in self.rx.raw_tiles(ROBOT).items()}
        return got == codec.tile_manifest(self.agent.tiles)


@pytest.mark.integration
def test_loop_closure_costs_values_not_tiles():
    g, pos, col, st, faces = _load()
    deltas, kept_faces = _deltas(pos, col, st, faces, STAGES)
    sim = Sim({"mesh_source": "odometric", "mesh_period_s": 2.0, "budgets": {"C3": 1_000_000},
               "budget_max": {"C3": 2_000_000}})  # fmt: skip
    assert sim.agent.odom is not None
    smax = 0
    for d in deltas:
        sim.agent.on_mesh_delta(d)
        smax = max(smax, int(d["stamp"].max()))
        reveal = np.flatnonzero((g["ctrl_stamp"] <= smax))
        sim.agent.on_mesh_graph_incremental(
            [(int(i), int(g["ctrl_stamp"][i]), (*g["ctrl"][i], 0, 0, 0, 1)) for i in reveal]
        )
        sim.run(3.0)
    # Hydra publishes every control point; the last ones' stamps can be newer than
    # the newest vertex of the saved mesh
    sim.agent.on_mesh_graph_incremental(
        [(int(i), int(g["ctrl_stamp"][i]), (*g["ctrl"][i], 0, 0, 0, 1)) for i in range(len(g["ctrl"]))]
    )
    for _ in range(60):
        if sim.converged():
            break
        sim.run(5.0)
    assert sim.converged(), "odometric tiles did not converge"
    before = sim.c3()
    tiles_before = {k: t["hash"] for k, t in sim.rx.tiles(ROBOT).items()}

    # the loop closure: optimized control-point poses
    sim.agent.on_mesh_graph([(int(i), int(g["ctrl_stamp"][i]), tuple(g["pose"][i])) for i in np.flatnonzero(g["valid"])])
    sim.run(5.0)
    lc_bytes = sim.c3() - before
    shown = sim.rx.tiles(ROBOT)
    assert all(shown[k]["hash"] != h for k, h in tiles_before.items() if shown[k]["nv"])
    assert lc_bytes < 40_000, f"a loop closure cost {lc_bytes} B on C3"

    # what the base shows == deforming the odometric mesh (tile-quantized) at once
    raw = sim.rx.raw_tiles(ROBOT)
    vids, got, base = [], [], []
    for k, t in raw.items():
        if not t["nv"]:
            continue
        o = np.asarray(t["origin"])
        vids.append(tile_vids(t))
        base.append(np.frombuffer(t["pos"], np.int16).reshape(-1, 3) * t["scale"] + o)
        got.append(np.frombuffer(shown[k]["pos"], np.int16).reshape(-1, 3) * t["scale"] + o)
    vid = np.concatenate(vids)
    got = np.concatenate(got)
    p0 = np.concatenate(base).astype(np.float32)
    stq = (st[vid] // np.uint64(1_000_000)) * np.uint64(1_000_000)
    back, valid = deform.unpack_poses(deform.pack_poses(g["pose"], g["valid"]))
    want = deform.deform(p0, stq, g["ctrl"], g["ctrl_stamp"], back, valid, 3, 15.0, indices=np.argsort(vid, kind="stable"))
    err = np.linalg.norm(got - want, axis=1)
    assert np.percentile(err, 99.9) <= 0.02 and err.max() < 0.05

    # DSG mode would re-send every tile whose deformed content changed
    dsg_tiles = codec.tile_mesh(want.astype(np.float64), np.zeros((len(want), 3), np.uint8), np.zeros((0, 3), np.int64))
    print(f"\nodometric mode: mesh {sum(len(t['pos']) + len(t['col']) + len(t['idx']) for t in raw.values()) / 1e6:.1f} MB "
          f"raw tiles; loop closure cost {lc_bytes / 1e3:.1f} kB on C3")  # fmt: skip
    assert dsg_tiles is not None
