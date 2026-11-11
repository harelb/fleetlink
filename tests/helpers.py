"""Shared test helpers: synthetic graphs/meshes, a recording transport, sim loop."""

from __future__ import annotations

import numpy as np

from fleetlink import codec
from fleetlink.codec import GraphState, NodeRecord
from fleetlink.transport import Transport, key_matches


def synthetic_records(n: int = 50, seed: int = 0, spread: float = 40.0):
    rng = np.random.default_rng(seed)
    recs = {}
    for i in range(n):
        layer, part, prefix = [(2, 0, "O"), (3, 1, "P"), (2, 97, "a")][i % 3]
        p = tuple(float(v) for v in rng.uniform(-spread, spread, 3))
        sym = f"{prefix}({i})"
        bbox = (p[0] - 0.5, p[1] - 0.5, p[2] - 0.5, p[0] + 0.5, p[1] + 0.5, p[2] + 0.5)
        recs[sym] = NodeRecord(
            sym,
            layer,
            part,
            p,
            label=int(i % 7),
            bbox=bbox if prefix == "O" else None,
            color=(i % 255, 10, 20),
            name=f"n{i}" if prefix == "O" else "",
        )
    syms = list(recs)
    edges = {(syms[i], syms[i + 1]) for i in range(0, n - 1, 2)}
    edges |= {
        (syms[i], syms[(i * 7) % n]) for i in range(n) if syms[i] != syms[(i * 7) % n]
    }
    return recs, edges, {"2/0": {"1": "chair", "2": "table"}}


def synthetic_mesh(n_quads: int = 400, extent: float = 30.0, seed: int = 0):
    """A bumpy grid of quads (pos float32[N,3], col uint8[N,3], faces uint32[M,3])."""
    rng = np.random.default_rng(seed)
    k = int(np.sqrt(n_quads)) + 1
    xs = np.linspace(-extent, extent, k)
    X, Y = np.meshgrid(xs, xs)
    Z = rng.normal(0.0, 0.2, X.shape)
    pos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    col = rng.integers(0, 255, (len(pos), 3), dtype=np.uint8)
    faces = []
    for i in range(k - 1):
        for j in range(k - 1):
            a = i * k + j
            faces += [[a, a + 1, a + k], [a + 1, a + k + 1, a + k]]
    return pos, col, np.asarray(faces, np.uint32)


def states_equal(a: GraphState, b: GraphState, tol: float = 0.01) -> tuple[bool, str]:
    if set(a.records) != set(b.records):
        d1 = set(a.records) - set(b.records)
        d2 = set(b.records) - set(a.records)
        return False, f"node sets differ (+{len(d1)} / -{len(d2)})"
    for s, r in a.records.items():
        if np.max(np.abs(np.asarray(r.pos) - np.asarray(b.records[s].pos))) > tol:
            return False, f"position of {s} differs"
    if a.edges != b.edges:
        return False, f"edges differ ({len(a.edges ^ b.edges)})"
    return True, ""


class RecordingTransport(Transport):
    """Captures puts; ``inject`` delivers to local subscribers."""

    def __init__(self):
        self.sent: list[tuple[str, bytes, str]] = []
        self.subs = []

    def put(self, key, data, cls):
        self.sent.append((key, bytes(data), cls))

    def subscribe(self, key_expr, callback):
        self.subs.append((key_expr, callback))
        return (key_expr, callback)

    def declare_queryable(self, key_expr, handler):
        self.queryables = getattr(self, "queryables", []) + [(key_expr, handler)]
        return (key_expr, handler)

    def ask(self, key, params=None):
        """Run a query against the local queryables (no link)."""
        for expr, handler in getattr(self, "queryables", []):
            if key_matches(expr, key):
                return handler(key, dict(params or {}))
        return None

    def inject(self, key, data):
        for expr, cb in self.subs:
            if key_matches(expr, key):
                cb(key, data)

    def messages(self, key_prefix: str = ""):
        return [
            (k, codec.unpack(d), c) for k, d, c in self.sent if k.startswith(key_prefix)
        ]


def env_bytes(robot, stream, seq, epoch, payload, t=0.0) -> bytes:
    env = codec.envelope(robot, stream, seq, epoch, payload)
    env["t"] = t
    return codec.pack(env)
