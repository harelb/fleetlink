"""A5 regression: the agent's DsgUpdate decode path (``decode_dsg`` as used by
``AgentNode._decode_loop``) on real ``to_binary`` payloads reports the right node
count, streams non-trivial C2 deltas, and flags a payload whose nodes the
imported spark_dsg silently dropped."""

import os
import types

import numpy as np
import pytest
from helpers import RecordingTransport, synthetic_mesh

from fleetlink import codec
from fleetlink.agent import AgentCore, decode_dsg
from fleetlink.transport import VirtualClock

dsg = pytest.importorskip("spark_dsg")

REAL_DSG = os.path.expanduser(
    os.environ.get(
        "FLEETLINK_REAL_DSG",
        "~/adt4_output/imgstore_winA_on4_20261017/hydra/backend/dsg_with_mesh.json",
    )
)


def _graph(n_obj=20, n_agents=30, n_places=10):
    G = dsg.DynamicSceneGraph()
    for i in range(n_obj):
        a = dsg.ObjectNodeAttributes()
        a.position = np.array([float(i), 0.5 * i, 0.0])
        a.semantic_label = i % 5
        G.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", i), a)
    for i in range(n_places):
        a = dsg.PlaceNodeAttributes()
        a.position = np.array([float(i), 1.0, 0.0])
        G.add_node(dsg.DsgLayers.PLACES, dsg.NodeSymbol("p", i), a)
    for i in range(n_agents):  # agent keyframes live in a layer partition
        a = dsg.AgentNodeAttributes()
        a.position = np.array([0.1 * i, 0.0, 0.0])
        G.add_node(2, dsg.NodeSymbol("a", i), a, ord("a"))  # layer 2, partition a
    for i in range(n_obj - 1):
        G.insert_edge(dsg.NodeSymbol("O", i), dsg.NodeSymbol("O", i + 1))
    pos, col, faces = synthetic_mesh(100, extent=10)
    m = dsg.Mesh()
    m.set_vertices(np.vstack([pos.T, col.T / 255.0]).astype(np.float64))
    m.set_faces(faces.T.astype(np.int32))
    G.mesh = m
    return G


def _agent_on(data):
    clock = VirtualClock()
    tr = RecordingTransport()
    core = AgentCore("r", tr, {}, clock=clock)
    inputs, info = decode_dsg(dsg, data)
    core.on_records(*inputs)
    core.start_snapshot()
    for _ in range(100):
        clock.advance(0.05)
        core.tick()
    return core, tr, info


def _c2_upserts(tr):
    n = 0
    for _, m, _ in tr.messages("adt4/r/dsg/delta"):
        n += len(codec.GraphDelta.decode(m["p"]).upserts)
    return n


def test_decoding_a_binary_graph_reports_all_nodes_and_streams_c2():
    try:
        G = _graph()
        data = G.to_binary(True)
    except Exception as exc:  # noqa: BLE001  (bindings without partitions/mesh API)
        pytest.skip(f"spark_dsg cannot build the test graph: {exc}")
    core, tr, info = _agent_on(data)
    total = G.num_nodes()
    assert info["nodes"] == total == info["raw_nodes"] == 60
    assert info["has_nodes"] is True and info["warning"] is None
    assert info["mesh_faces"] > 0
    st = core.stats()
    assert st["nodes"] == total and st["tiles"] > 0
    assert _c2_upserts(tr) == total  # the whole graph went out on C2
    assert sum(len(d) for k, d, _c in tr.sent if k.endswith("/dsg/delta")) > 705  # vs ~705 B of empty snapshots seen on hamilton


def test_decoding_the_real_hydra_graph():
    if not os.path.exists(REAL_DSG):
        pytest.skip(f"no real DSG at {REAL_DSG}")
    try:
        G = dsg.DynamicSceneGraph.load(REAL_DSG)
    except Exception as exc:  # noqa: BLE001  (spark_dsg older than the file)
        pytest.skip(f"this spark_dsg cannot load {REAL_DSG}: {exc}")
    data = G.to_binary(True)
    core, tr, info = _agent_on(data)
    assert info["nodes"] == G.num_nodes() > 100 and info["warning"] is None
    assert core.stats()["nodes"] == G.num_nodes()
    assert _c2_upserts(tr) > 100


def test_payload_with_nodes_decoded_to_none_is_flagged():
    try:
        data = _graph().to_binary(False)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"spark_dsg cannot build the test graph: {exc}")
    empty = dsg.DynamicSceneGraph()

    # a spark_dsg that cannot create the writer's attribute types drops every
    # node without an error (what a stale copy on the python path looks like)
    fake = types.SimpleNamespace(
        DynamicSceneGraph=types.SimpleNamespace(from_binary=lambda b: empty)
    )
    inputs, info = decode_dsg(fake, data)
    assert info["nodes"] == 0 and info["has_nodes"] is True
    assert "decoded none" in info["warning"]


def test_binary_header_of_an_empty_graph():
    data = dsg.DynamicSceneGraph().to_binary(False)
    info = codec.dsg_binary_info(data)
    assert info["has_nodes"] is False and "AgentNodeAttributes" in info["node_types"]
    with pytest.raises(ValueError):
        codec.dsg_binary_info(b"\x01\x02")
