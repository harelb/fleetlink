"""Deterministic end-to-end run of the REAL Pennovation map over an emulated
2 Mbit/s, 150 ms, 3 % loss radio link (virtual clock, seeded RNG).

The map is revealed to the agent in 20 stages following the agent-trajectory
timestamps (every node / mesh tile appears once the robot has passed its
nearest trajectory pose), emulating a robot mapping.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from helpers import states_equal

from fleetlink import codec
from fleetlink.agent import AgentCore, graph_inputs
from fleetlink.receiver import Receiver
from fleetlink.transport import LinkEmulator, VirtualClock

# a Hydra backend DSG with mesh (dsg_with_mesh.json) of a real run, with the robot's
# agent trajectory; the benchmark is skipped unless FLEETLINK_PENN_MAP points at one
MAP = os.environ.get("FLEETLINK_PENN_MAP", "")
ROBOT = "penn"
STAGES = 20
STAGE_PERIOD = 2.0  # s between graph reveals (Hydra publishes <= 1 Hz)
DT = 0.05
LINK = dict(bandwidth=250_000.0, latency=0.15, loss=0.03, seed=7)  # 2 Mbit/s
GRAPH_BOUND_S = 60.0  # graph must match within this long after the last reveal
MESH_BOUND_S = 600.0  # whole mesh (13.4 MB at 100 kB/s ≈ 134 s) within this


@pytest.fixture(scope="session")
def penn():
    dsg = pytest.importorskip("spark_dsg")
    scipy_spatial = pytest.importorskip("scipy.spatial")
    if not MAP:
        pytest.skip("set FLEETLINK_PENN_MAP to a Hydra dsg_with_mesh.json")
    if not os.path.exists(MAP):
        pytest.skip(f"FLEETLINK_PENN_MAP: missing {MAP}")
    G = dsg.DynamicSceneGraph.load(MAP)  # read-only
    records, edges, ls, mesh = graph_inputs(G)
    raw_full = len(G.to_binary(True))
    raw_nomesh = len(G.to_binary(False))

    traj = []
    for L in G.layer_partitions:
        for n in L.nodes:
            a = n.attributes
            if hasattr(a, "world_R_body"):
                ts = a.timestamp
                ts = (
                    ts.total_seconds()
                    if hasattr(ts, "total_seconds")
                    else float(ts) * 1e-9
                )
                traj.append((ts, str(n.id), np.asarray(a.position, float)))
    traj.sort(key=lambda x: x[0])
    tpos = np.stack([p for _, _, p in traj])
    tree = scipy_spatial.cKDTree(tpos)
    rank = {sym: i for i, (_, sym, _) in enumerate(traj)}
    syms = list(records)
    _, near = tree.query(np.array([records[s].pos for s in syms]))
    node_idx = {s: rank.get(s, int(near[i])) for i, s in enumerate(syms)}

    pos, col, faces = mesh
    ts = 8.0
    ijk = np.floor(pos[faces].mean(axis=1) / ts).astype(np.int64)
    uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
    _, tile_near = tree.query((uniq + 0.5) * ts)
    face_idx = np.asarray(tile_near)[inv.reshape(-1)]
    return dict(
        records=records,
        edges=edges,
        ls=ls,
        mesh=mesh,
        tpos=tpos,
        node_idx=node_idx,
        face_idx=face_idx,
        raw_full=raw_full,
        raw_nomesh=raw_nomesh,
    )


def stage_inputs(penn, k: int):
    """Graph + mesh visible after stage k (1..STAGES)."""
    n = len(penn["tpos"])
    cut = math.ceil(k * n / STAGES) - 1
    recs = {s: r for s, r in penn["records"].items() if penn["node_idx"][s] <= cut}
    edges = {e for e in penn["edges"] if e[0] in recs and e[1] in recs}
    pos, col, faces = penn["mesh"]
    fmask = penn["face_idx"] <= cut
    p = penn["tpos"][cut]
    pose = {"x": p[0], "y": p[1], "z": p[2], "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
    pose.update(frame="map", t=0.0)
    frac = (len(recs) / len(penn["records"]), float(fmask.mean()))
    return recs, edges, (pos, col, faces[fmask]), pose, frac


class Sim:
    def __init__(self, cfg=None, link=None):
        self.clock = VirtualClock()
        self.link = LinkEmulator(self.clock, **(link or LINK))
        # full mesh: this benchmark measures the heavy case (the decimated LOD
        # is ~7x smaller; tests/test_mesh_lod.py covers it)
        self.rx = Receiver(
            self.link.base, None, clock=self.clock, threaded=False, mesh_lod=None
        )
        self.rx.start()
        self.cfg = cfg
        self.agent = AgentCore(ROBOT, self.link.robot, cfg, self.clock)
        self.max_state_age = 0.0
        self.state_age_trace = []
        self.c3_backlogged_steps = 0
        self.steps = 0

    def step(self):
        t = self.clock.advance(DT)
        self.link.advance()
        self.agent.tick(t)
        self.rx.tick(t)
        r = self.rx._robots.get(ROBOT)
        if r is not None and r.last_state_rx is not None:
            age = t - r.last_state_rx
            self.max_state_age = max(self.max_state_age, age)
            self.state_age_trace.append(age)
        self.steps += 1
        if self.agent.sched.queues["C3"]:
            self.c3_backlogged_steps += 1
        return t

    def graph_ok(self):
        if ROBOT not in self.rx._robots:
            return False
        return states_equal(self.rx.graph(ROBOT), self.agent.graph_state())[0]

    def mesh_ok(self):
        if ROBOT not in self.rx._robots or not self.agent.tiles:
            return False
        got = {k: t["hash"] for k, t in self.rx.tiles(ROBOT).items()}
        return got == codec.tile_manifest(self.agent.tiles)


@pytest.fixture(scope="module")
def run(penn):
    sim = Sim()
    # what raw DsgUpdate forwarding would have put on the link at 1 Hz (sizes scaled
    # by the revealed node / face fraction of the full map's to_binary sizes)
    raw_bytes = raw_nomesh_bytes = 0
    stage, t_graph, t_mesh, frac = 0, None, None, (0.0, 0.0)
    c3_busy_first_100s = None
    next_check = 1.0
    while sim.clock() < STAGES * STAGE_PERIOD + MESH_BOUND_S:
        t = sim.clock()
        if stage < STAGES and t >= stage * STAGE_PERIOD:
            stage += 1
            recs, edges, mesh, pose, frac = stage_inputs(penn, stage)
            sim.agent.on_records(recs, edges, penn["ls"], mesh)
            sim.agent.on_pose(pose)
            t_graph = None  # robot graph changed
        t = sim.step()
        if t >= next_check:
            next_check += 1.0
            raw_bytes += (
                penn["raw_nomesh"] * frac[0]
                + (penn["raw_full"] - penn["raw_nomesh"]) * frac[1]
            )
            raw_nomesh_bytes += penn["raw_nomesh"] * frac[0]
            if c3_busy_first_100s is None and t >= 100.0:
                c3_busy_first_100s = sim.c3_backlogged_steps / sim.steps
            if t_graph is None and sim.graph_ok():
                t_graph = t
            if stage == STAGES and t_mesh is None and sim.mesh_ok():
                t_mesh = t
            if stage == STAGES and t_graph is not None and t_mesh is not None:
                break
    end = sim.clock()
    out = dict(
        sim=sim,
        t_graph=t_graph,
        t_mesh=t_mesh,
        t_end=end,
        raw_bytes=raw_bytes,
        raw_nomesh_bytes=raw_nomesh_bytes,
        c3_busy=c3_busy_first_100s,
        sent=dict(sim.agent.sched.sent_bytes),
        wire=dict(sim.link.robot.tx.stats.wire_bytes),
    )
    _print_table(penn, out)
    return out


def _print_table(penn, out):
    sim = out["sim"]
    T = out["t_end"]
    budgets = sim.agent.cfg["budgets"]
    st = sim.link.robot.tx.stats
    lines = [
        "",
        f"fleetlink benchmark — Pennovation courtyard ({len(penn['records'])} nodes, "
        f"{len(penn['edges'])} edges, {len(sim.agent.tiles)} tiles), link 2 Mbit/s / 150 ms / 3 % loss",
        f"graph converged at t={out['t_graph']:.1f}s (last reveal at {STAGES * STAGE_PERIOD - STAGE_PERIOD:.0f}s); "
        f"mesh converged at t={out['t_mesh']:.1f}s; run {T:.0f}s virtual",
        f"{'class':<6}{'app bytes':>14}{'wire bytes':>14}{'avg B/s':>10}{'budget':>9}"
        f"{'retx':>6}{'lost':>6}",
    ]
    for c in ("C0", "C1", "C2", "C3"):
        b = budgets.get(c)
        lines.append(
            f"{c:<6}{out['sent'][c]:>14,}{out['wire'][c]:>14,}{out['sent'][c] / T:>10,.0f}"
            f"{(f'{b:,}' if b else '-'):>9}{st.retransmits[c]:>6}{st.loss_drops[c]:>6}"
        )
    total = sum(out["sent"].values())
    lines += [
        f"{'total':<6}{total:>14,}{sum(out['wire'].values()):>14,}{total / T:>10,.0f}",
        f"raw DsgUpdate @1 Hz for the same {T:.0f}s (with mesh): {out['raw_bytes']:,.0f} B "
        f"→ fleetlink uses {100.0 * total / out['raw_bytes']:.2f} % "
        f"(full map binary: {penn['raw_full']:,} B with mesh, {penn['raw_nomesh']:,} B without)",
        f"raw DsgUpdate @1 Hz without mesh: {out['raw_nomesh_bytes']:,.0f} B vs fleetlink C2 "
        f"{out['sent']['C2']:,} B ({out['raw_nomesh_bytes'] / max(1, out['sent']['C2']):,.0f}x less)",
        f"C1 max state age {sim.max_state_age:.3f}s; C3 backlogged {100 * (out['c3_busy'] or 0):.0f} % of the "
        f"first 100 s; resyncs {sim.rx.link(ROBOT)['resyncs']}, drops {sim.rx.link(ROBOT)['drops']}",
    ]
    print("\n".join(lines))


def test_graph_and_mesh_converge(run):
    last_reveal = (STAGES - 1) * STAGE_PERIOD
    assert run["t_graph"] is not None and run["t_graph"] <= last_reveal + GRAPH_BOUND_S
    assert run["t_mesh"] is not None and run["t_mesh"] <= last_reveal + MESH_BOUND_S
    ok, why = states_equal(run["sim"].rx.graph(ROBOT), run["sim"].agent.graph_state())
    assert ok, why


def test_bytes_within_budget(run):
    sim, T = run["sim"], run["t_end"]
    for c in ("C1", "C2", "C3"):
        budget = sim.agent.cfg["budgets"][c]
        assert run["sent"][c] <= budget * (T * 1.1 + 1.0), c
    assert sum(run["sent"].values()) < 0.05 * run["raw_bytes"]


def test_state_fresh_while_bulk_saturated(run):
    assert run["c3_busy"] > 0.5  # C3 really was saturated
    assert run["sim"].max_state_age < 1.0


def test_agent_restart_triggers_clean_resync(run, penn):
    sim = run["sim"]
    old_epoch = sim.agent.epoch
    resyncs = sim.rx.link(ROBOT)["resyncs"]
    sim.agent.close()  # kill
    recs, edges, mesh, pose, _ = stage_inputs(penn, STAGES)
    sim.agent = AgentCore(ROBOT, sim.link.robot, sim.cfg, sim.clock)
    sim.agent.on_records(recs, edges, penn["ls"], mesh)
    sim.agent.on_pose(pose)
    t0 = sim.clock()
    t_graph = None
    while sim.clock() - t0 < 400.0:
        t = sim.step()
        if (
            t_graph is None
            and sim.rx.link(ROBOT)["resyncs"] > resyncs
            and sim.graph_ok()
        ):
            t_graph = t - t0
        if t_graph is not None and sim.mesh_ok():
            break
    link = sim.rx.link(ROBOT)
    assert link["epoch"] == sim.agent.epoch != old_epoch
    assert link["resyncs"] >= resyncs + 1
    assert t_graph is not None and t_graph < 20.0
    assert sim.mesh_ok()
    assert sim.max_state_age < 1.0


def test_state_fresh_when_link_itself_is_saturated(penn):
    """C3 budget above link capacity: link priorities must keep C1 fresh."""
    sim = Sim(cfg={"budgets": {"C3": 400_000}})
    recs, edges, mesh, pose, _ = stage_inputs(penn, STAGES)
    sim.agent.on_records(recs, edges, penn["ls"], mesh)
    sim.agent.on_pose(pose)
    while sim.clock() < 60.0:
        sim.step()
    wire = sum(sim.link.robot.tx.stats.wire_bytes.values())
    assert wire / 60.0 > 0.9 * LINK["bandwidth"]  # the radio really is full
    assert sim.max_state_age < 1.0
    assert sim.graph_ok()
