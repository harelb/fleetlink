"""Multi-robot fleetlink bench: N robot agents sharing ONE emulated radio channel.

Deterministic (virtual clock, seeded RNG).  Every robot runs the real
:class:`fleetlink.agent.AgentCore` (scheduler, budgets, C1 state, C2 graph
deltas, C3 mesh tiles) against its own :class:`fleetlink.receiver.Receiver`;
all links share one :class:`fleetlink.transport.SharedChannel` (airtime is
shared by every robot, both directions; a robot ``hops`` relays away spends
that many times the air).

Load per robot:

* the real Pennovation courtyard map (graph + mesh), revealed in ``stages``
  over ``reveal_s`` following the robot's trajectory, as in
  ``tests/test_integration_pennovation.py``;
* a forced loop closure at ``lc_s``: every node and mesh vertex moves a little
  (a small rotation + shift), so every mesh tile and most node positions change;
* optional C5 video (frame sizes from the real ZED frames through fleetlink's
  own ffmpeg arguments, ``bench/extract_traces.py``) on the first
  ``video_streams`` robots, at a console preset;
* optional C6 image-store mirror: keyframes at ``kf_hz`` with the real
  ``fast``-mode blob sizes (JPEG q75 + half-res delta-zstd depth).

Measured per robot: bytes per class (put / on the air / delivered / dropped),
C1 state age p50 / p99 / max, C0 command round trip (an acked request every
5 s), C2 catch-up after each reveal and after the loop closure, C3 mesh
catch-up after the last reveal and after the loop closure, video freezes and
delivered fps, image-store mirror delay and backlog.

    python bench/multi_robot_bench.py --traces traces.json --out results.jsonl \\
        --robots 1 2 3 5 --rate-mbps 2 5 10 --video 0 1 2 --mirror 0 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import struct
import sys
import time

import numpy as np

from fleetlink import codec
from fleetlink.agent import AgentCore, graph_inputs
from fleetlink.receiver import Receiver
from fleetlink.scheduler import Item
from fleetlink.transport import LinkEmulator, SharedChannel, VirtualClock

MAP = "~/data/pennovation_2025/courtyard-alleyways/output/hydra/backend/dsg_with_mesh.json"
DT = 1.0 / 30.0  # 15 / 10 / 5 fps video lands on whole ticks
TILE = 8.0
VIDEO_PRESETS = ("low", "med", "high", "tab")


# --------------------------------------------------------------------- map
def load_penn(path: str = MAP) -> dict:
    """The map, each node / mesh face tagged with the trajectory index at which
    the robot has passed it (same reveal as the Pennovation integration test)."""
    import spark_dsg as dsg
    from scipy.spatial import cKDTree

    G = dsg.DynamicSceneGraph.load(path)
    records, edges, ls, mesh = graph_inputs(G)
    traj = []
    for L in G.layer_partitions:
        for n in L.nodes:
            a = n.attributes
            if hasattr(a, "world_R_body"):
                ts = a.timestamp
                ts = ts.total_seconds() if hasattr(ts, "total_seconds") else float(ts) * 1e-9
                traj.append((ts, str(n.id), np.asarray(a.position, float)))
    traj.sort(key=lambda x: x[0])
    tpos = np.stack([p for _, _, p in traj])
    tree = cKDTree(tpos)
    rank = {sym: i for i, (_, sym, _) in enumerate(traj)}
    syms = list(records)
    _, near = tree.query(np.array([records[s].pos for s in syms]))
    node_idx = {s: rank.get(s, int(near[i])) for i, s in enumerate(syms)}
    pos, col, faces = mesh
    ijk = np.floor(pos[faces].mean(axis=1) / TILE).astype(np.int64)
    uniq, inv = np.unique(ijk, axis=0, return_inverse=True)
    _, tile_near = tree.query((uniq + 0.5) * TILE)
    face_idx = np.asarray(tile_near)[inv.reshape(-1)]
    return dict(records=records, edges=edges, ls=ls, mesh=mesh, tpos=tpos,
                node_idx=node_idx, face_idx=face_idx)  # fmt: skip


class Stages:
    """Graph + mesh visible after stage k, built once and shared by every robot
    (the agents only read them), plus the loop-closed copy of the full map."""

    def __init__(self, penn: dict, n_stages: int):
        self.penn, self.n = penn, n_stages
        self._cache: dict = {}
        self._lc = None

    def get(self, k: int):
        if k not in self._cache:
            p = self.penn
            n = len(p["tpos"])
            cut = math.ceil(k * n / self.n) - 1
            recs = {s: r for s, r in p["records"].items() if p["node_idx"][s] <= cut}
            edges = {e for e in p["edges"] if e[0] in recs and e[1] in recs}
            pos, col, faces = p["mesh"]
            mesh = (pos, col, faces[p["face_idx"] <= cut])
            q = p["tpos"][cut]
            pose = {"x": q[0], "y": q[1], "z": q[2], "qw": 1.0, "qx": 0.0, "qy": 0.0,
                    "qz": 0.0, "frame": "map", "t": 0.0}  # fmt: skip
            self._cache[k] = (recs, edges, mesh, pose)
        return self._cache[k]

    def loop_closed(self):
        """The full map after a loop closure: 0.3 deg about z and a 0.15 m shift,
        growing with distance from the start like a corrected drift."""
        if self._lc is None:
            import copy

            recs, edges, (pos, col, faces), pose = self.get(self.n)
            c, s = math.cos(math.radians(0.3)), math.sin(math.radians(0.3))
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            shift = np.array([0.15, -0.1, 0.02])
            new_pos = (pos @ R.T + shift).astype(pos.dtype)
            new_recs = {}
            for sym, r in recs.items():
                r2 = copy.copy(r)
                r2.pos = tuple((np.asarray(r.pos) @ R.T + shift).tolist())
                new_recs[sym] = r2
            self._lc = (new_recs, edges, (new_pos, col, faces), pose)
        return self._lc


_tile_memo: dict = {}
_orig_tile_mesh = codec.tile_mesh


def _memo_tile_mesh(pos, col, faces, tile_size=codec.DEFAULT_TILE_SIZE):
    """Robots share the stage arrays: tile each once (tiling 1.6 M faces is the
    bench's slowest step)."""
    key = (id(pos), id(faces), len(faces), tile_size)
    if key not in _tile_memo:
        _tile_memo[key] = _orig_tile_mesh(pos, col, faces, tile_size)
    return _tile_memo[key]


codec.tile_mesh = _memo_tile_mesh


# ------------------------------------------------------------- synthetic C5/C6
class VideoSource:
    """A live stream's frames at its preset (sizes from real encoded ZED frames).
    Like the real encoder, only the newest frame waits: a frame still queued when
    the next is ready is replaced (counted as skipped)."""

    def __init__(self, agent: AgentCore, trace: dict, cam: str = "rgb"):
        self.agent, self.cam = agent, cam
        self.frames = trace["frames"]
        self.fps = trace["fps"]
        self.i = 0
        self.next_t = 0.0
        self.skipped = 0
        self.key_at = None  # a key frame was requested (the console does on a gap)
        self.key_size = max(f["n"] for f in self.frames if f["key"])
        self.forced_keys = 0

    def tick(self, now: float) -> None:
        if now < self.next_t - 1e-6:
            return
        self.next_t = max(self.next_t + 1.0 / self.fps, now)  # no drift from tick rounding
        f = self.frames[self.i % len(self.frames)]
        if self.key_at is not None and now >= self.key_at and not f["key"]:
            f = {"n": self.key_size, "key": True}  # an extra key frame (bigger)
            self.forced_keys += 1
        if f["key"]:
            self.key_at = None
        seq = self.i
        self.i += 1
        header = struct.pack("<IBd", seq, 1 if f["key"] else 0, now)
        body = header + bytes(max(0, f["n"] - len(header)))
        q = self.agent.sched.queues["C5"]
        if any(it.tag == f"video/{self.cam}" for it in q):
            self.skipped += 1
        self.agent.sched.enqueue(
            Item("C5", f"{self.agent.prefix}/benchvideo/{self.cam}", lambda b=body: b, len(body),
                 tag=f"video/{self.cam}"),  # fmt: skip
        )


class VideoSink:
    """Base side: frames arriving; a lost / skipped delta frame makes the picture
    undecodable until the next key frame (time counted as frozen).  Like the
    console's video relay, a gap asks the robot for a key frame (``on_gap``)."""

    def __init__(self, fps: float, on_gap=None):
        self.fps = fps
        self.on_gap = on_gap
        self.last_seq = None
        self.waiting_key = True
        self.shown: list[float] = []
        self.frozen_s = 0.0
        self.freezes = 0
        self._last_shown_t = None

    def on(self, now: float, data: bytes) -> None:
        seq, key, _t = struct.unpack_from("<IBd", data)
        if self.last_seq is not None and seq != self.last_seq + 1:
            if not self.waiting_key and self.on_gap is not None:
                self.on_gap(now)
            self.waiting_key = True  # a gap: deltas are useless until a key frame
        self.last_seq = seq
        if self.waiting_key and not key:
            return
        self.waiting_key = False
        if self._last_shown_t is not None:
            gap = now - self._last_shown_t
            if gap > 1.0:
                self.freezes += 1
                self.frozen_s += gap
        self._last_shown_t = now
        self.shown.append(now)


class StoreSource:
    """Image-store keyframes at ``kf_hz`` (real ``fast`` blob sizes), queued like
    ``AgentCore._store_tick``: at most ``store_queue_s`` of C6 budget waiting."""

    def __init__(self, agent: AgentCore, store_trace: list, kf_hz: float):
        self.agent = agent
        self.sizes = [x["blob"] for x in store_trace]
        self.kf_hz = kf_hz
        self.next_t = 0.0
        self.made = 0
        self.backlog: list[tuple[int, float, int]] = []  # (id, t made, size)

    def tick(self, now: float) -> None:
        while now >= self.next_t:
            self.backlog.append((self.made, self.next_t, self.sizes[self.made % len(self.sizes)]))
            self.next_t += 1.0 / self.kf_hz
            self.made += 1
        sched = self.agent.sched
        budget = sched.budget("C6") or 0
        limit = budget * self.agent.cfg["store_queue_s"]
        while self.backlog and sched.queued_bytes("C6") + self.backlog[0][2] <= max(limit, self.backlog[0][2]):
            fid, t0, n = self.backlog.pop(0)
            body = struct.pack("<Id", fid, t0) + bytes(max(0, n - 12))
            sched.enqueue(Item("C6", f"{self.agent.prefix}/benchstore/frame", lambda b=body: b, n))


class StoreSink:
    def __init__(self):
        self.delays: list[float] = []
        self.got = 0

    def on(self, now: float, data: bytes) -> None:
        _fid, t0 = struct.unpack_from("<Id", data)
        self.got += 1
        self.delays.append(now - t0)


# --------------------------------------------------------------------- runner
def pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def run(stages: Stages, traces: dict, robots: int, rate_mbps: float, video: int, mirror: bool,
        video_preset: str = "med", latency: float = 0.03, loss: float = 0.01, hops=None,
        reveal_s: float = 300.0, lc_s: float = 330.0, end_s: float = 900.0, kf_hz: float = 1.4,
        seed: int = 1, tcp_hol: bool = False) -> dict:  # fmt: skip
    from helpers import states_equal  # tests/helpers.py

    clock = VirtualClock()
    rate = rate_mbps * 1e6 / 8
    links = [LinkEmulator(clock, bandwidth=rate, latency=latency, loss=loss, seed=seed * 31 + i,
                          tcp_hol=tcp_hol) for i in range(robots)]  # fmt: skip
    channel = SharedChannel(links, clock, airtime=hops or [1.0] * robots)
    names = [f"r{i}" for i in range(robots)]
    agents, rxs, vsrc, vsink, ssrc, ssink = [], [], {}, {}, {}, {}
    for i, (name, lk) in enumerate(zip(names, links)):
        rx = Receiver(lk.base, None, clock=clock, threaded=False)
        rx.start()
        ag = AgentCore(name, lk.robot, None, clock)
        agents.append(ag)
        rxs.append(rx)
        if i < video:
            tr = traces["video"][video_preset]
            vsrc[name] = VideoSource(ag, tr)

            def ask_key(now, n=name):  # C0 request -> robot: ~ one RTT later
                vsrc[n].key_at = now + 2 * latency

            vsink[name] = VideoSink(tr["fps"], ask_key)
            lk.base.subscribe(f"adt4/{name}/benchvideo/**", lambda k, d, n=name: vsink[n].on(clock(), d))
        if mirror:
            ssrc[name] = StoreSource(ag, traces["store"], kf_hz)
            ssink[name] = StoreSink()
            lk.base.subscribe(f"adt4/{name}/benchstore/**", lambda k, d, n=name: ssink[n].on(clock(), d))

    state_age = {n: [] for n in names}
    rtts = {n: [] for n in names}
    pending_cmd: dict = {}
    c2_catch = {n: [] for n in names}
    c2_lc = {n: None for n in names}
    mesh_reveal = {n: None for n in names}
    mesh_lc = {n: None for n in names}
    graph_dirty_since = {n: None for n in names}
    stage, lc_done = 0, False
    stage_period = reveal_s / stages.n
    next_check = 1.0
    next_cmd = 2.0
    wall = time.time()

    def graph_ok(i):
        n = names[i]
        if n not in rxs[i]._robots:
            return False
        return states_equal(rxs[i].graph(n), agents[i].graph_state())[0]

    def mesh_ok(i):
        n = names[i]
        if n not in rxs[i]._robots or not agents[i].tiles:
            return False
        got = {k: t["hash"] for k, t in rxs[i].tiles(n).items()}
        return got == codec.tile_manifest(agents[i].tiles)

    while clock() < end_s:
        t = clock()
        if stage < stages.n and t >= stage * stage_period:
            stage += 1
            recs, edges, mesh, pose = stages.get(stage)
            for i, ag in enumerate(agents):
                ag.on_records(recs, edges, stages.penn["ls"], mesh)
                ag.on_pose(pose)
                if graph_dirty_since[names[i]] is None:
                    graph_dirty_since[names[i]] = t
        if not lc_done and t >= lc_s:
            lc_done = True
            recs, edges, mesh, pose = stages.loop_closed()
            for i, ag in enumerate(agents):
                ag.on_records(recs, edges, stages.penn["ls"], mesh)
                graph_dirty_since[names[i]] = t
        t = clock.advance(DT)
        channel.advance()
        for i, (ag, rx) in enumerate(zip(agents, rxs)):
            n = names[i]
            if n in vsrc:
                vsrc[n].tick(t)
            if n in ssrc:
                ssrc[n].tick(t)
            ag.tick(t)
            rx.tick(t)
            r = rx._robots.get(n)
            if r is not None and r.last_state_rx is not None:
                state_age[n].append(t - r.last_state_rx)
        if t >= next_cmd:  # an acked C0 request per robot every 5 s
            next_cmd += 5.0
            for i, rx in enumerate(rxs):
                n = names[i]
                if n in rx._robots and n not in pending_cmd:
                    pending_cmd[n] = (rx.set_budget(n, "C1", 5000.0), t)
        for n, (rid, t0) in list(pending_cmd.items()):
            rx = rxs[names.index(n)]
            if rid not in rx._pending:
                rtts[n].append(t - t0)
                del pending_cmd[n]
        if t >= next_check:
            next_check += 1.0
            for i, n in enumerate(names):
                if graph_dirty_since[n] is not None and graph_ok(i):
                    dt_c = t - graph_dirty_since[n]
                    if lc_done and c2_lc[n] is None:
                        c2_lc[n] = dt_c
                    else:
                        c2_catch[n].append(dt_c)
                    graph_dirty_since[n] = None
                if stage == stages.n and not lc_done and mesh_reveal[n] is None and mesh_ok(i):
                    mesh_reveal[n] = t - (stages.n - 1) * stage_period
                if lc_done and mesh_lc[n] is None and mesh_ok(i):
                    mesh_lc[n] = t - lc_s
            if lc_done and all(mesh_lc[n] is not None and c2_lc[n] is not None for n in names) \
                    and t > lc_s + 30.0 and not vsrc and not ssrc:
                break

    T = clock()
    per = {}
    for i, n in enumerate(names):
        st = links[i].robot.tx.stats
        down = links[i].base.tx.stats
        vs = vsink.get(n)
        ss = ssink.get(n)
        per[n] = {
            "hops": (hops or [1.0] * robots)[i],
            "put_B": dict(st.put_bytes),
            "air_B": dict(st.wire_bytes),
            "delivered_B": dict(st.delivered_bytes),
            "congestion_drops": dict(st.congestion_drops),
            "loss_drops": dict(st.loss_drops),
            "retransmits": dict(st.retransmits),
            "down_air_B": sum(down.wire_bytes.values()),
            "kbps_up": {c: round(8 * v / T / 1000, 1) for c, v in st.wire_bytes.items() if v},
            "state_age_p50": pct(state_age[n], 50),
            "state_age_p99": pct(state_age[n], 99),
            "state_age_max": max(state_age[n]) if state_age[n] else None,
            "cmd_rtt_p50": pct(rtts[n], 50),
            "cmd_rtt_p99": pct(rtts[n], 99),
            "cmd_rtt_max": max(rtts[n]) if rtts[n] else None,
            "c2_catchup_p50": pct(c2_catch[n], 50),
            "c2_catchup_max": max(c2_catch[n]) if c2_catch[n] else None,
            "c2_after_lc": c2_lc[n],
            "mesh_after_reveal": mesh_reveal[n],
            "mesh_after_lc": mesh_lc[n],
            "video": None if vs is None else {
                "frames_shown": len(vs.shown),
                "fps": round(len(vs.shown) / T, 2),
                "freezes": vs.freezes,
                "frozen_s": round(vs.frozen_s, 1),
                "skipped_at_robot": vsrc[n].skipped,
                "forced_keys": vsrc[n].forced_keys,
            },
            "mirror": None if ss is None else {
                "made": ssrc[n].made,
                "delivered": ss.got,
                "backlog": ssrc[n].made - ss.got,
                "delay_p50": pct(ss.delays, 50),
                "delay_max": max(ss.delays) if ss.delays else None,
            },
            "resyncs": rxs[i].link(n)["resyncs"],
        }
    return {
        "robots": robots, "rate_mbps": rate_mbps, "video": video, "video_preset": video_preset,
        "mirror": mirror, "latency": latency, "loss": loss, "hops": hops, "tcp_hol": tcp_hol,
        "virtual_s": round(T, 1), "wall_s": round(time.time() - wall, 1),
        "channel_busy": round(channel.busy_s / T, 3),
        "per_robot": per,
    }  # fmt: skip


def summary(res: dict) -> dict:
    """One row: worst robot per metric, totals on the air."""
    per = res["per_robot"].values()

    def worst(key, sub=None):
        vals = [(p[sub] or {}).get(key) if sub else p.get(key) for p in per]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def missing(key):
        return sum(1 for p in per if p.get(key) is None)

    air = sum(sum(p["air_B"].values()) + p["down_air_B"] for p in per)
    return {
        "robots": res["robots"], "rate_mbps": res["rate_mbps"], "video": res["video"],
        "mirror": res["mirror"], "hops": res["hops"],
        "air_mbps": round(8 * air / res["virtual_s"] / 1e6, 2),
        "channel_busy": res["channel_busy"],
        "state_age_p99": worst("state_age_p99"), "state_age_max": worst("state_age_max"),
        "cmd_rtt_p99": worst("cmd_rtt_p99"),
        "c2_catchup_max": worst("c2_catchup_max"), "c2_after_lc": worst("c2_after_lc"),
        "mesh_after_reveal": worst("mesh_after_reveal"), "mesh_after_lc": worst("mesh_after_lc"),
        "mesh_reveal_unfinished": missing("mesh_after_reveal"),
        "mesh_lc_unfinished": missing("mesh_after_lc"),
        "video_freezes": worst("freezes", "video"), "video_fps_min": (
            min(p["video"]["fps"] for p in per if p["video"]) if any(p["video"] for p in per) else None),
        "mirror_backlog": worst("backlog", "mirror"), "mirror_delay_max": worst("delay_max", "mirror"),
        "c5_drops": sum(p["congestion_drops"]["C5"] + p["loss_drops"]["C5"] for p in per),
        "c3_drops": sum(p["congestion_drops"]["C3"] + p["loss_drops"]["C3"] for p in per),
        "wall_s": res["wall_s"],
    }  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", required=True, help="JSON lines, one run each (appended)")
    ap.add_argument("--map", default=MAP)
    ap.add_argument("--robots", type=int, nargs="+", default=[1, 2, 3, 5])
    ap.add_argument("--rate-mbps", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    ap.add_argument("--video", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--video-preset", default="med", choices=VIDEO_PRESETS)
    ap.add_argument("--mirror", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--loss", type=float, default=0.01)
    ap.add_argument("--latency", type=float, default=0.03)
    ap.add_argument("--stages", type=int, default=20)
    ap.add_argument("--reveal-s", type=float, default=300.0)
    ap.add_argument("--lc-s", type=float, default=330.0)
    ap.add_argument("--end-s", type=float, default=900.0)
    ap.add_argument("--hops-last", type=float, default=1.0, help="air factor of the last robot (relay hops)")
    ap.add_argument("--tcp-hol", action="store_true", help="one FIFO per link (TCP head-of-line)")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
    traces = json.load(open(a.traces))
    t0 = time.time()
    stages = Stages(load_penn(a.map), a.stages)
    print(f"map loaded in {time.time() - t0:.0f}s", file=sys.stderr)
    for n, rate, v, m in itertools.product(a.robots, a.rate_mbps, a.video, a.mirror):
        if v > n:
            continue
        hops = [1.0] * (n - 1) + [a.hops_last]
        res = run(stages, traces, n, rate, v, bool(m), a.video_preset, a.latency, a.loss, hops,
                  a.reveal_s, a.lc_s, a.end_s, tcp_hol=a.tcp_hol)  # fmt: skip
        res["summary"] = summary(res)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res["summary"]), file=sys.stderr, flush=True)


if __name__ == "__main__":
    random.seed(0)
    main()
