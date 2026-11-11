"""Robot-side fleetlink agent.

:class:`AgentCore` is pure logic (no ROS): it takes graphs / poses / guards /
health / action results, and on every :meth:`AgentCore.tick` emits

* C1 ``state`` at ``state_hz`` (latest value only),
* C2 graph deltas against the last *sent* state, split by budget closest to
  the robot first (``codec.split_delta``),
* C3 mesh tiles whose hash differs from the sent manifest, closest first, plus
  a periodic manifest,

* C3 ``grid/occupancy`` (latest occupancy grid) every ``grid_period_s`` when
  its content hash changed,
* C1 ``paths`` (executor smooth / MLP paths + MLP target) only on change,
* C3 ``lidar/points`` (:mod:`fleetlink.pointcloud_codec`) at ``lidar_hz``, only
  while the base asked for it (C0 ``lidar``; off by default),

serves C0 requests (``resync``, ``budget``, ``auto``, ``ping``, ``tiles``,
``resend``, ``video``, ``lidar``) with an ack each (``fb``, the receiver's once-a-second
feedback for auto mode, is fire-and-forget), re-sends lost C2 messages from a
ring buffer of the last ``c2_ring`` ones (``resend {stream: "dsg", from_seq,
to_seq}``), answers C4 camera queries on
``adt4/<r>/media/camera/<name>`` and streams C5 live video on
``adt4/<r>/video/<name>`` while the base keeps a ``video`` request alive
(:mod:`fleetlink.video`).  :class:`AgentNode` wraps it in rclpy.

The C1 state names the robot's ``platform`` ("spot" | "phoenix", see
:func:`fleetlink.relay_ros.platform_of`).  A phoenix robot (Clearpath Husky on
ARL's stack) has no spot_executor_node: its ``guards`` come from the platform
topics (:class:`PhoenixGuards`) and ``phoenix_progress`` reports how far the
robot is from the end of the latest omniplanner Path.
"""

from __future__ import annotations

import importlib.metadata
import os
import secrets
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from fleetlink import codec, media, video
from fleetlink.codec import GraphDelta, GraphState
from fleetlink.scheduler import Item, Scheduler
from fleetlink.transport import Transport, split_endpoints

# rmw_zenoh 0.2.11 links zenoh-c 1.8; eclipse-zenoh must speak the same protocol
ZENOH_EXPECTED = "1.8"

# budget classes reported in the state message (and adjustable from the console)
CLASSES_REPORTED = ("C1", "C2", "C3", "C4", "C5", "C6")

DEFAULT_CONFIG = {
    "budgets": {  # bytes/s
        "C1": 5_000,
        "C2": 50_000,
        "C3": 100_000,
        "C4": 200_000,
        "C5": 400_000,  # live video (all streams together)
        "C6": 150_000,  # live image-store mirror (lowest priority)
    },
    "budget_min": {"C2": 5_000, "C3": 5_000},
    "budget_max": {"C2": 500_000, "C3": 1_000_000},
    "total_bps": None,
    "auto": False,
    "state_hz": 4.0,  # 2 Hz cannot keep C1 staleness < 1 s at 3 % loss (see ADT4 console API.md Deviations)
    "c2_hz": 1.0,
    "mesh_period_s": 5.0,
    # "dsg": tile the deformed mesh from the DSG (a loop closure re-sends the
    # tiles it moves); "odometric": tile Hydra's odometric mesh deltas and send
    # the deformation graph's optimized poses, so the base deforms the mesh
    # itself (fleetlink.mesh_deform_link)
    "mesh_source": "dsg",
    "manifest_period_s": 5.0,
    "tile_size": codec.DEFAULT_TILE_SIZE,
    # vertex-clustering cell of the "decimated" mesh LOD (codec.decimate_mesh);
    # the robot sends full tiles until a base asks for a LOD (``mesh_lod``)
    "mesh_decimate_cell": codec.DEFAULT_DECIMATE_CELL,
    "max_msg_bytes": 64 * 1024,
    "burst_s": 1.0,
    "ack_cache": 256,
    "grid_period_s": 5.0,
    "paths_hz": 2.0,  # how often paths are checked for a change
    "cameras": ["rgb", "overlay"],
    # cameras subscribed all the time; the others only while wanted (a C4 query
    # or a C5 video request within ``camera_idle_s``), so e.g. six Spot cameras
    # behind lazy image_transport republishers cost nothing until viewed
    "eager_cameras": ["rgb", "overlay"],
    "camera_idle_s": 15.0,
    "video_keepalive_s": video.KEEPALIVE_S,  # a stream stops without a request this long
    "ffmpeg": None,  # None -> $FLEETLINK_FFMPEG / ffmpeg on PATH / /usr/bin/ffmpeg
    "c2_ring": 256,  # last C2 messages kept for ``resend`` (gap repair)
    # C6 live image-store mirror (fleetlink.store_mirror); None/"" disables
    "store_path": None,
    # frames are only sent while this is on: off by default (5 robots need ~10
    # Mbit/s with it), the console turns it on per robot (C0 ``store_send``)
    "store_send": False,
    "store_frames": "keyframes",  # keyframes | all
    "store_compression": "fast",  # none | lossless | fast
    "store_poll_s": 1.0,  # how often new store frames are looked for
    "store_queue_s": 2.0,  # C6 frames queued at most (seconds of C6 budget)
    "platform": "spot",  # spot | phoenix (relay_ros.platform_of)
    # C3 lidar stream: None = no lidar (no ``lidar`` state key); frames go out only
    # while ``lidar`` is on (off by default, the console turns it on per robot)
    "lidar_topic": None,
    "lidar": False,
    "lidar_hz": 1.0,
    "lidar_bps": 50_000,  # cap on top of C3: fewer frames when they are big
    "lidar_intensity": False,  # + uint8 intensity (~1 B per point more)
    # safety: stop streaming when no ``lidar`` request (enable / base refresh)
    # came for this long, e.g. the base went away while it was on
    "lidar_ttl_s": 60.0,
}

PHOENIX = "phoenix"
# phoenix_progress: the plan is done within this xy distance of the Path's end,
# once the robot has been farther than that from it since the plan arrived (a
# round trip ends where it starts: not done at dispatch)
PHOENIX_DONE_M = 1.0


def zenoh_version() -> str | None:
    """Installed eclipse-zenoh version (None when it is not installed)."""
    try:
        return importlib.metadata.version("eclipse-zenoh")
    except importlib.metadata.PackageNotFoundError:
        try:
            import zenoh

            return getattr(zenoh, "__version__", None)
        except ImportError:
            return None


def zenoh_version_warning(version: str | None = None) -> str | None:
    """A warning text when eclipse-zenoh is not the 1.8 series rmw_zenoh uses."""
    version = zenoh_version() if version is None else version
    if version is None or str(version).startswith(ZENOH_EXPECTED):
        return None
    return (
        f"eclipse-zenoh {version} installed but rmw_zenoh uses zenoh "
        f"{ZENOH_EXPECTED}.x: pin eclipse-zenoh==1.8.0 in spark_env"
    )


def load_config(path: str | None = None) -> dict:
    """DEFAULT_CONFIG overlaid with a yaml file (``fleetlink.yaml`` layout)."""
    cfg = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()
    }
    if path:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def _merge(cfg: dict | None) -> dict:
    out = load_config(None)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def spark_dsg_location() -> str:
    """Where ``import spark_dsg`` resolves (logged + reported: a stale pip copy
    in spark_env shadowing the workspace build silently changes decoding)."""
    try:
        import spark_dsg

        ver = getattr(spark_dsg, "__version__", None)
        return f"{spark_dsg.__file__}" + (f" ({ver})" if ver else "")
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def decode_dsg(dsg, data: bytes) -> tuple[tuple, dict]:
    """DsgUpdate ``layer_contents`` → (graph_inputs, diagnostics).

    The diagnostics compare what spark_dsg decoded with the payload's own
    header (:func:`codec.dsg_binary_info`): a payload that has nodes but decodes
    to none means the imported spark_dsg cannot read the writer's attribute
    types (it skips such nodes silently), and ``warning`` says so."""
    G = dsg.DynamicSceneGraph.from_binary(data)
    inputs = graph_inputs(G)
    records, _edges, _ls, mesh = inputs
    try:
        raw = int(G.num_nodes())
    except Exception:  # noqa: BLE001
        raw = None
    info = {
        "bytes": len(data),
        "nodes": len(records),
        "raw_nodes": raw,
        "mesh_faces": 0 if mesh is None else int(len(mesh[2])),
        "warning": None,
    }
    try:
        hdr = codec.dsg_binary_info(data)
        info["has_nodes"] = hdr["has_nodes"]
        info["layer_keys"] = [list(k) for k in hdr["layer_keys"]]
        if hdr["has_nodes"] and not records:
            info["warning"] = (
                f"DSG payload has nodes but spark_dsg ({spark_dsg_location()}) "
                f"decoded none: it cannot read the writer's attribute types "
                f"{hdr['node_types']} (stale spark_dsg on the python path?)"
            )
    except ValueError as exc:
        info["has_nodes"] = None
        info["header_error"] = str(exc)
    return inputs, info


def graph_inputs(G) -> tuple:
    """spark_dsg graph → (records, edges, labelspaces, mesh_arrays | None)."""
    records = codec.graph_records(G)
    edges = codec.graph_edges(G)
    labelspaces = codec.graph_labelspaces(G)
    mesh = None
    try:
        if G.has_mesh() and G.mesh is not None and G.mesh.num_faces() > 0:
            mesh = codec.mesh_arrays(G.mesh)
    except Exception:
        mesh = None
    return records, edges, labelspaces, mesh


class AgentCore:
    STREAM_CLASS = {
        "ctl/ack": "C0",
        "state": "C1",
        "dsg/delta": "C2",
        "mesh/tile": "C3",
        "mesh/manifest": "C3",
        "mesh/values": "C3",
        "grid/occupancy": "C3",
        "paths": "C1",
        "store/frame": "C6",
        # bulk, best effort, below graph (C2) and pose (C1); latest frame only
        "lidar/points": "C3",
    }

    def __init__(
        self,
        robot: str,
        transport: Transport,
        config: dict | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.robot = robot
        self.transport = transport
        self.cfg = _merge(config)
        self.clock = clock
        self.epoch = secrets.token_hex(6)
        self.prefix = f"adt4/{robot}"
        self.seq = dict.fromkeys(self.STREAM_CLASS, 0)
        self.sched = Scheduler(
            transport,
            self.cfg["budgets"],
            clock,
            burst_s=self.cfg["burst_s"],
            total_bps=self.cfg["total_bps"],
            budget_min=self.cfg["budget_min"],
            budget_max=self.cfg["budget_max"],
            auto=bool(self.cfg["auto"]),
        )
        self._lock = threading.RLock()
        self._inbox: deque = deque()
        self._inbox_lock = threading.Lock()
        self._acked: dict = {}  # request id -> ack payload (idempotent retries)
        self._fb_seen = False  # a base that sends ``fb`` (ping rx is then ignored)
        # C2 ring buffer: seq -> packed message, for ``resend``
        self._c2_ring: dict[int, bytes] = {}
        self.c2_resent = 0
        self.c2_resend_gone = 0

        # graph
        self._target: tuple | None = None  # (records, edges, labelspaces)
        self.sent = GraphState()
        self._snapshot = False
        self._rec_bytes = 40.0  # learned packed bytes per upsert (edges count 1/2)
        # mesh
        self._mesh = None
        self._mesh_version = 0
        self._tiled_version = 0
        self.tiles: dict[str, dict] = {}
        self.sent_manifest: dict[str, str] = {}
        self._tiles_dirty = False
        # level of detail: None until a base asks (an old base never does, and
        # then gets the full tiles it always had); per-tile overrides on top
        self.mesh_lod: str | None = None
        self.mesh_lod_tiles: dict[str, str] = {}
        self._lod_dirty = False
        self._tiles_by_lod: dict[str, dict] = {}  # LOD -> tiles of _tiled_version
        self.odom = None
        self.mesh_fallback = None  # why odometric mode was given up, if it was
        self._tile = lambda m: codec.tile_mesh(m[0], m[1], m[2], self.cfg["tile_size"])
        if self.cfg["mesh_source"] == "odometric":
            from .mesh_deform_link import OdomMeshAgent, tile_odom

            self.odom = OdomMeshAgent()
            self._tile = lambda m: tile_odom(m, self.cfg["tile_size"])
        # state
        self.pose: dict | None = None
        # global frame -> scene-graph frame (``frame_align``), None until looked up
        self.frame_align: dict | None = None
        # prior-map loop closures + map -> odom jumps (``reloc``); None until
        # the node subscribes to ROMAN (see :meth:`enable_reloc`)
        self.reloc = None
        self.guards: dict = {}
        self.platform = str(self.cfg.get("platform") or "spot")
        # phoenix: latest omniplanner Path (points in ``frame``) for phoenix_progress
        self._plan: dict | None = None
        self.health: dict = {}
        self.results: deque = deque(maxlen=10)
        self._t_state = -1e9
        self._t_c2 = -1e9
        self._t_mesh = -1e9
        self._t_manifest = -1e9
        self._base_rtt: float | None = None
        # occupancy grid (C3) / executor paths (C1)
        self._grid: dict | None = None  # raw: frame, resolution, origin, w, h, data
        self._grid_dirty = False
        self.sent_grid_hash: str | None = None
        self._t_grid = -1e9
        self._paths = {"smooth": None, "mlp": None, "target": None}
        self._paths_frame = ""
        self._paths_t: float | None = None
        self._paths_dirty = False
        self.sent_paths_hash: str | None = None
        self._t_paths = -1e9
        # cameras (C4 request/reply)
        self.camera_names: list[str] = list(self.cfg["cameras"])
        self.eager_cameras: set[str] = set(self.cfg.get("eager_cameras") or [])
        self.camera_wanted_t: dict[str, float] = {}
        self._frames: dict[str, dict] = {}
        self._media_lock = threading.Lock()
        self._c4_tokens = self._c4_cap()
        self._c4_t: float | None = None
        self.media_served = 0
        self.media_refused = 0
        # live video (C5)
        self.video = video.VideoStreams(
            get_frame=self._latest_frame,
            send=video.make_sender(
                transport, robot, lambda: self.epoch, clock, self.prefix
            ),
            budget_fn=lambda: self.sched.budgets.get("C5"),
            clock=clock,
            cameras=lambda: list(self.camera_names),
            burst_s=self.cfg["burst_s"],
            keepalive_s=float(self.cfg["video_keepalive_s"]),
            ffmpeg=self.cfg["ffmpeg"],
        )

        # C6 live image-store mirror
        self.store = None
        self.store_error: str | None = None
        self._t_store = -1e9
        self.store_send = bool(self.cfg.get("store_send"))
        if self.cfg.get("store_path"):
            from fleetlink import store_mirror

            if not store_mirror.tail_mode_available():
                # logged once by AgentNode ("store mirror disabled: ...")
                self.store_error = store_mirror.TAIL_MODE_MISSING
        if self.cfg.get("store_path") and self.store_error is None:
            try:
                from fleetlink.store_mirror import StoreTail

                self.store = StoreTail(
                    self.cfg["store_path"],
                    self.cfg["store_frames"],
                    self.cfg["store_compression"],
                    clock=clock,
                )
            except Exception as exc:  # noqa: BLE001  (bad params)
                self.store_error = f"{type(exc).__name__}: {exc}"
        # C3 lidar (fleetlink.pointcloud_codec frames from AgentNode)
        self.lidar_topic = self.cfg.get("lidar_topic") or None
        self.lidar_enabled = (
            bool(self.cfg.get("lidar")) and self.lidar_topic is not None
        )
        self.lidar_error: str | None = None
        self._lidar: dict | None = None  # newest encoded frame (+ frame, origin, stamp)
        self._lidar_rx_t: float | None = None
        self._lidar_dirty = False
        self._t_lidar = -1e9
        self._lidar_req_t = clock()  # last ``lidar`` request (lidar_ttl_s)
        self.lidar_sent_bytes = 0  # size of the last frame put on the link
        # DsgUpdate decoding diagnostics (filled by AgentNode, reported in stats)
        self.dsg_diag: dict = {}

        self._sub = transport.subscribe(f"{self.prefix}/ctl/req", self._on_req)
        try:
            self._queryable = transport.declare_queryable(
                f"{self.prefix}/media/camera/*", self._on_camera_query
            )
        except NotImplementedError:
            self._queryable = None

    # ------------------------------------------------------------------ inputs

    def on_graph(self, G) -> None:
        records, edges, labelspaces, mesh = graph_inputs(G)
        self.on_records(records, edges, labelspaces, mesh)

    def on_records(self, records, edges, labelspaces=None, mesh_arrays=None) -> None:
        with self._lock:
            self._target = (dict(records), set(edges), labelspaces or {})
            if mesh_arrays is not None and self.odom is None:
                self._mesh = mesh_arrays
                self._mesh_version += 1

    # odometric mesh mode (mesh_source: odometric)
    def on_mesh_delta(self, delta: dict) -> None:
        """Hydra frontend ``full_mesh_update`` (see mesh_deform_link.delta_from_msg)."""
        with self._lock:
            self.odom.on_delta(delta)
            self._refresh_odom_mesh()

    def on_mesh_graph_incremental(self, nodes) -> None:
        """New control points: [(index, stamp_ns, pose7 at the original position)]."""
        with self._lock:
            before = self.odom.version
            self.odom.on_ctrl_incremental(nodes)
            if self.odom.version != before:
                self._refresh_odom_mesh()

    def on_mesh_graph(self, nodes) -> None:
        """Backend ``mesh_graph``: optimized control-point poses."""
        with self._lock:
            self.odom.on_values(nodes)

    def reset_odom_mesh(self) -> None:
        with self._lock:
            if self.odom is not None:
                from .mesh_deform_link import OdomMeshAgent

                self.odom = OdomMeshAgent()

    def fallback_to_dsg_mesh(self, reason: str) -> None:
        """Odometric mode cannot continue (missed deltas, no messages): tile the
        DSG's deformed mesh again, like ``mesh_source: dsg``."""
        with self._lock:
            if self.odom is None:
                return
            self.odom = None
            self.mesh_fallback = reason
            self._tile = lambda m: codec.tile_mesh(
                m[0], m[1], m[2], self.cfg["tile_size"]
            )
            self._mesh = None  # the next DSG brings the deformed mesh

    def _refresh_odom_mesh(self) -> None:
        m = self.odom.mesh_input()
        if m is not None:
            self._mesh = m
            self._mesh_version += 1

    def on_pose(self, pose: dict) -> None:
        """``{x, y, z, qw, qx, qy, qz, frame, t}``."""
        with self._lock:
            self.pose = dict(pose)
            _plan_mark_left(self._plan, self.pose)

    def on_frame_align(self, align: dict | None) -> None:
        """``{parent, child, T: [x, y, z, qw, qx, qy, qz], stamp, ok, reason}``
        (see :func:`frame_align_ok` / :func:`frame_align_failed`)."""
        with self._lock:
            self.frame_align = None if align is None else dict(align)
            if self.reloc is not None and align and align.get("ok"):
                self.reloc.on_align(align.get("T"), self.clock())

    def enable_reloc(self, live_id: int = 0) -> None:
        """Start reporting ``reloc`` (ROMAN's id for this robot: ``live_id``)."""
        from .reloc import RelocTracker

        with self._lock:
            self.reloc = RelocTracker(live_id)

    def on_loop_closure(self, lc: dict) -> None:
        """A ROMAN loop closure (:func:`reloc.loop_closure_to_dict`)."""
        with self._lock:
            if self.reloc is not None:
                self.reloc.on_loop_closure(lc, self.clock())

    def on_guards(self, guards: dict) -> None:
        with self._lock:
            self.guards = dict(guards)

    def on_plan_path(self, points, frame: str, sec: int, nanosec: int) -> None:
        """Phoenix: the latest omniplanner Path (``points`` (N, 3) in ``frame``,
        ideally the pose frame) and its header stamp, which names the plan."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        length = (
            float(np.hypot(*np.diff(pts[:, :2], axis=0).T).sum())
            if len(pts) > 1
            else 0.0
        )
        with self._lock:
            plan_id = f"{int(sec)}.{int(nanosec):09d}"
            if self._plan is not None and self._plan["plan_id"] == plan_id:
                left = self._plan["left"]  # the same plan again: keep its history
            else:
                # a (near) zero-length path is done once the robot is at its end
                left = len(pts) <= 1 or length < PHOENIX_DONE_M
            self._plan = {
                "plan_id": plan_id,
                "stamp": int(sec) + int(nanosec) * 1e-9,
                "frame": frame,
                "end": [float(v) for v in pts[-1]] if len(pts) else None,
                "left": left,
            }
            _plan_mark_left(self._plan, self.pose)

    def on_lidar(self, frame: dict) -> None:
        """Newest lidar frame: :func:`pointcloud_codec.encode_cloud` output plus
        ``frame`` (the pose frame), ``sensor_frame``, ``origin`` [x, y, z, qx,
        qy, qz, qw] (sensor pose in ``frame``) and ``stamp``."""
        with self._lock:
            self._lidar = dict(frame)
            self._lidar_rx_t = self.clock()
            self._lidar_dirty = True
            self.lidar_error = None  # a good frame clears an earlier bad one

    def on_health(self, health: dict) -> None:
        with self._lock:
            self.health = dict(health)

    def on_result(self, result: dict) -> None:
        with self._lock:
            self.results.append(dict(result))

    def on_grid(self, frame, resolution, origin, width, height, data, t=None) -> None:
        """Latest occupancy grid (``data``: int8 bytes, row-major, -1 = unknown)."""
        with self._lock:
            self._grid = {
                "frame": frame,
                "resolution": resolution,
                "origin": list(origin),
                "width": int(width),
                "height": int(height),
                "data": bytes(data),
                "t": t,
            }
            self._grid_dirty = True

    def on_path(self, which: str, value, frame: str = "", t=None) -> None:
        """``which`` in smooth | mlp (Nx3 points) | target ([x, y, z] or None)."""
        if which not in self._paths:
            raise ValueError(f"unknown path {which!r}")
        with self._lock:
            self._paths[which] = value
            if frame:
                self._paths_frame = frame
            if t is not None:
                self._paths_t = t
            self._paths_dirty = True

    def set_cameras(self, names) -> None:
        with self._media_lock:
            self.camera_names = list(names)

    def on_camera_frame(
        self, name, encoding, width, height, step, data, stamp=None
    ) -> None:
        """Keep the newest raw frame per camera; conversion happens on request."""
        frame = {
            "encoding": encoding,
            "width": int(width),
            "height": int(height),
            "step": int(step),
            "data": data,
            "stamp": stamp,
            "rx_t": self.clock(),
        }
        with self._media_lock:
            self._frames[name] = frame

    def want_camera(self, name: str, now: float | None = None) -> None:
        """Note that someone wants ``name`` (keeps a lazy subscription alive)."""
        with self._media_lock:
            self.camera_wanted_t[name] = self.clock() if now is None else now

    def wanted_cameras(self, now: float | None = None) -> set[str]:
        """Cameras that should be subscribed now: eager ones, ones queried or
        requested as video within ``camera_idle_s``, and live video streams."""
        now = self.clock() if now is None else now
        idle = float(self.cfg["camera_idle_s"])
        try:
            streaming = self.video.cameras_in_use()  # a mosaic reads several
        except Exception:  # noqa: BLE001
            streaming = set()
        with self._media_lock:
            names = set(self.camera_names)
            recent = {n for n, t in self.camera_wanted_t.items() if now - t < idle}
        return names & (self.eager_cameras | recent | streaming)

    def drop_frame(self, name: str) -> None:
        """Forget a camera's cached frame (its subscription went idle)."""
        with self._media_lock:
            self._frames.pop(name, None)

    def _latest_frame(self, name: str) -> dict | None:
        with self._media_lock:
            return self._frames.get(name)

    # ----------------------------------------------------------------- helpers

    def origin(self):
        if self.pose is not None:
            return (self.pose["x"], self.pose["y"], self.pose["z"])
        return None

    def _builder(self, stream: str, payload_fn: Callable[[], object]):
        def build():
            payload = payload_fn()
            if payload is None:
                return None
            seq = self.seq[stream]
            self.seq[stream] = seq + 1
            env = codec.envelope(self.robot, stream, seq, self.epoch, payload)
            env["t"] = self.clock()
            data = codec.pack(env)
            if stream == "dsg/delta":
                ring = self._c2_ring
                ring[seq] = data
                while len(ring) > max(0, int(self.cfg["c2_ring"])):
                    ring.pop(next(iter(ring)))
            return data

        return build

    def _queue(
        self,
        stream: str,
        key: str,
        payload_fn,
        size_hint=0,
        on_sent=None,
        tag=None,
        front=False,
    ):
        cls = self.STREAM_CLASS[stream]
        self.sched.enqueue(
            Item(cls, key, self._builder(stream, payload_fn), size_hint, on_sent, tag),
            front,
        )

    # ------------------------------------------------------------------- C0

    def _on_req(self, key: str, data: bytes) -> None:
        try:
            env = codec.unpack(data)
        except Exception:
            return
        with self._inbox_lock:
            self._inbox.append(env.get("p", env))

    def _handle(self, req: dict, now: float) -> None:
        rid = req.get("id")
        op = req.get("op")
        args = req.get("args") or {}
        if op == "fb":  # fire-and-forget feedback: no ack
            try:
                self._execute(op, args, now)
            except Exception:
                pass
            return
        if rid in self._acked and op != "ping":
            ack = self._acked[rid]
        else:
            try:
                result = self._execute(op, args, now)
                ack = {"id": rid, "ok": True, "result": result}
            except Exception as exc:  # bad request: report, never crash the agent
                ack = {"id": rid, "ok": False, "result": {"error": str(exc)}}
            if op != "ping":
                self._acked[rid] = ack
                while len(self._acked) > self.cfg["ack_cache"]:
                    self._acked.pop(next(iter(self._acked)))
        self._queue("ctl/ack", f"{self.prefix}/ctl/ack", lambda a=ack: a, 64)

    def _execute(self, op: str, args: dict, now: float):
        if op == "resync":
            self.start_snapshot()
            if self.odom is not None:
                self.odom.force_values()
            return {"epoch": self.epoch}
        if op == "budget":
            for cls, bps in args.items():
                self.sched.set_budget(cls, bps)
            return {"budgets": dict(self.sched.budgets)}
        if op == "auto":
            self.sched.auto = bool(args.get("enabled", True))
            return {"auto": self.sched.auto}
        if op == "ping":
            rtt = args.get("rtt")
            if rtt is not None:
                self._base_rtt = (
                    rtt if self._base_rtt is None else min(self._base_rtt, rtt)
                )
            if args.get("rx") and not self._fb_seen:  # base without ``fb``
                self.sched.feedback(args["rx"], rtt, now, self._base_rtt)
            return {
                "t": args.get("t"),
                "t_robot": now,
                "budgets": dict(self.sched.budgets),
            }
        if op == "fb":
            self._fb_seen = True
            rtt = args.get("rtt")
            if rtt is not None:
                self._base_rtt = (
                    rtt if self._base_rtt is None else min(self._base_rtt, rtt)
                )
            self.sched.feedback(
                args.get("rx") or {}, rtt, now, self._base_rtt, args.get("loss")
            )
            return None
        if op == "resend":
            out = {}
            if args.get("stream") == "dsg":
                out["dsg"] = self._resend_c2(args)
            streams = list(args.get("streams") or [])
            if "grid" in streams:
                self.sent_grid_hash = None
                self._grid_dirty = True
                self._t_grid = -1e9
            if "paths" in streams:
                self.sent_paths_hash = None
                self._paths_dirty = True
            return {"resend": streams, **out}
        if op == "mesh_lod":
            level = str(args.get("level") or "")
            if level not in codec.MESH_LODS:
                raise ValueError(
                    f"unknown mesh LOD {level!r} (one of {codec.MESH_LODS})"
                )
            keys = args.get("tiles")
            if keys:  # just these tiles; the default stays
                for k in keys:
                    self.mesh_lod_tiles[str(k)] = level
                if self.mesh_lod is None:
                    self.mesh_lod = "full"
            else:
                self.mesh_lod = level
                self.mesh_lod_tiles = {}
            self._lod_dirty = True
            self._tiles_dirty = True
            return {"level": self.mesh_lod, "tiles": dict(self.mesh_lod_tiles)}
        if op == "tiles":
            keys = list(args.get("keys") or [])
            for k in keys:
                self.sent_manifest.pop(k, None)
            self._tiles_dirty = True
            return {"requeued": len(keys)}
        if op == "video":
            if args.get("enable", True) and args.get("camera"):
                cams = args.get("cameras") if args["camera"] == video.MOSAIC else None
                for cam in cams if isinstance(cams, list) else [args["camera"]]:
                    self.want_camera(str(cam), now)
            return self.video.request(args, now)
        if op == "store":
            if self.store is None:
                raise ValueError(self.store_error or "store mirror disabled")
            res = self.store.set_cursor(args.get("id"), args.get("after"))
            self.sched.clear("C6")
            self._t_store = -1e9
            return res
        if op == "store_send":
            if self.store is None:
                raise ValueError(
                    self.store_error or "store mirror disabled (no store_path)"
                )
            enable = bool(args.get("enable", True))
            if self.store_send and not enable:
                # drop what is queued and step the cursor back to the last frame
                # on the wire, so turning it on again continues the chain
                self.sched.clear("C6")
                self.store.pause()
            self.store_send = enable
            self._t_store = -1e9
            return {"send": self.store_send}
        if op == "lidar":
            if self.lidar_topic is None:
                raise ValueError("no lidar on this robot")
            self._lidar_req_t = now
            self._set_lidar(bool(args.get("enable", True)))
            return self._lidar_state()
        if op == "video_keyframe":
            cam = str(args.get("camera") or "")
            return {"camera": cam, "forced": self.video.force_keyframe(cam, now)}
        raise ValueError(f"unknown op {op!r}")

    # ------------------------------------------------------------------- C2

    def _resend_c2(self, args: dict) -> dict:
        """Queue the ring-buffered C2 messages ``from_seq..to_seq`` (inclusive) at
        the front of C2, byte-identical (same seq).  Raises ValueError ("range
        gone": the receiver then falls back to a resync) when any of them has
        already left the ring."""
        lo, hi = int(args["from_seq"]), int(args["to_seq"])
        seqs = range(lo, hi + 1)
        if hi < lo or any(s not in self._c2_ring for s in seqs):
            self.c2_resend_gone += 1
            raise ValueError(f"range gone: dsg {lo}-{hi}")
        key = f"{self.prefix}/dsg/delta"
        for s in reversed(seqs):
            data = self._c2_ring[s]
            self.sched.enqueue(Item("C2", key, lambda d=data: d, len(data)), True)
        self.c2_resent += len(seqs)
        return {"from_seq": lo, "to_seq": hi, "resent": len(seqs)}

    def start_snapshot(self) -> None:
        """Reset the sent mirror and stream the whole graph as a snapshot."""
        with self._lock:
            self.sched.clear("C2")
            self.sent = GraphState()
            self._snapshot = True
            begin = {"kind": "snapshot_begin", **GraphDelta().encode()}
            self._queue("dsg/delta", f"{self.prefix}/dsg/delta", lambda: begin, 64)
            self._t_c2 = -1e9  # stream the first chunk on the next tick

    def _c2_tick(self) -> None:
        budget = self.sched.budget("C2") or float("inf")
        per_tick = budget / max(self.cfg["c2_hz"], 1e-3)
        allow = min(per_tick, self.cfg["max_msg_bytes"]) - self.sched.queued_bytes("C2")
        if allow <= 0:
            return
        if self._target is None:
            delta = GraphDelta()
        else:
            delta = self.sent.diff(*self._target)
        if delta.empty():
            if self._snapshot:
                self._finish_snapshot()
            return
        n = max(1, int(allow / self._rec_bytes))
        now_d, later = codec.split_delta(delta, n, self.origin())
        self.sent.apply(now_d)
        kind = "snapshot_chunk" if self._snapshot else "delta"
        payload = {"kind": kind, **now_d.encode()}
        weight = len(now_d.upserts) + 0.5 * len(now_d.edges_add) + 1

        def learn(nbytes, weight=weight):
            self._rec_bytes = 0.7 * self._rec_bytes + 0.3 * max(4.0, nbytes / weight)

        self._queue(
            "dsg/delta",
            f"{self.prefix}/dsg/delta",
            lambda: payload,
            int(weight * self._rec_bytes),
            on_sent=learn,
        )
        if self._snapshot and later.empty():
            self._finish_snapshot()

    def _finish_snapshot(self) -> None:
        self._snapshot = False
        end = {"kind": "snapshot_end", **GraphDelta().encode()}
        self._queue("dsg/delta", f"{self.prefix}/dsg/delta", lambda: end, 64)
        self._queue_manifest()

    # ------------------------------------------------------------------- C3

    def _manifest_payload(self):
        out = {"tiles": dict(self.sent_manifest), "tile_size": self.cfg["tile_size"]}
        if self.mesh_lod is not None:  # a base that asked for a LOD
            out["lod"] = {
                # odometric mode sends full tiles whatever was asked (see _lod_tiles)
                "level": "full" if self.odom is not None else self.mesh_lod,
                "odometric": self.odom is not None,
                "tiles": dict(self.mesh_lod_tiles),
                "cell": self.cfg["mesh_decimate_cell"],
            }
        return out

    def _lod_of(self, key: str) -> str:
        return self.mesh_lod_tiles.get(key) or self.mesh_lod or "full"

    def _lod_tiles(self, level: str) -> dict:
        """Tiles of the current mesh at ``level`` (built once per mesh version)."""
        if self.odom is not None:
            # odometric mode: the base deforms each vertex by its stamp, so tiles keep
            # every vertex (tile_odom splits big ones); decimation does not apply yet
            level = "full"
        tiles = self._tiles_by_lod.get(level)
        if tiles is None and self.odom is not None:
            tiles = self._tile(self._mesh)
            for t in tiles.values():
                t["lod"] = level
            self._tiles_by_lod[level] = tiles
        if tiles is None:
            pos, col, faces = self._mesh
            if level == "decimated":
                pos, col, faces = codec.decimate_mesh(
                    pos, col, faces, self.cfg["mesh_decimate_cell"]
                )
            tiles = codec.tile_mesh(pos, col, faces, self.cfg["tile_size"])
            for t in tiles.values():
                t["lod"] = level
            self._tiles_by_lod[level] = tiles
        return tiles

    def _queue_manifest(self) -> None:
        # front of C3, built at send time → lists exactly the tiles already put
        self._queue(
            "mesh/manifest",
            f"{self.prefix}/mesh/manifest",
            self._manifest_payload,
            64 + 24 * len(self.sent_manifest),
            tag="manifest",
            front=True,
        )

    def _c3_tick(self) -> None:
        new_mesh = self._mesh is not None and self._tiled_version != self._mesh_version
        if new_mesh or (self._mesh is not None and self._lod_dirty):
            if new_mesh:
                self._tiles_by_lod = {}
                self._tiled_version = self._mesh_version
            self._lod_dirty = False
            levels = {self._lod_of(k) for k in self.mesh_lod_tiles} | {
                self.mesh_lod or "full"
            }
            by_lod = {lv: self._lod_tiles(lv) for lv in levels}
            default = by_lod[self.mesh_lod or "full"]
            keys = set(default) | {
                k for k in self.mesh_lod_tiles if k in by_lod[self._lod_of(k)]
            }
            self.tiles = {
                k: by_lod[self._lod_of(k)][k]
                for k in keys
                if k in by_lod[self._lod_of(k)]
            }
            for k in [k for k in self.sent_manifest if k not in self.tiles]:
                del self.sent_manifest[k]
        elif not self._tiles_dirty:
            return
        self._tiles_dirty = False
        changed = [
            k for k, t in self.tiles.items() if self.sent_manifest.get(k) != t["hash"]
        ]
        o = self.origin()
        if o is not None:
            o = np.asarray(o)
            ts = self.cfg["tile_size"]
            changed.sort(
                key=lambda k: float(
                    np.sum((codec.tile_centre(self.tiles[k], ts) - o) ** 2)
                )
            )
        # re-queue tiles only; the manifest and a pending occupancy grid stay
        self.sched.clear(
            "C3", lambda it: it.tag not in ("manifest", "grid", "values", "lidar")
        )
        for k in changed:
            tile = self.tiles[k]

            def mark(_n, k=k, h=tile["hash"]):
                self.sent_manifest[k] = h

            size = len(tile["pos"]) + len(tile["col"]) + len(tile["idx"])
            self._queue(
                "mesh/tile",
                f"{self.prefix}/mesh/tile/{k}",
                lambda tile=tile: tile,
                size,
                on_sent=mark,
            )

    # ------------------------------------------------- C3 grid / C1 paths

    def _grid_tick(self) -> None:
        if not self._grid_dirty or self._grid is None:
            return
        self._grid_dirty = False
        g = self._grid
        try:
            payload = media.encode_grid(
                g["frame"], g["resolution"], g["origin"], g["width"], g["height"],
                g["data"], g["t"],
            )  # fmt: skip
        except ValueError:
            return
        if payload["hash"] == self.sent_grid_hash:
            return

        def mark(_n, h=payload["hash"]):
            self.sent_grid_hash = h

        self._queue(
            "grid/occupancy",
            f"{self.prefix}/grid/occupancy",
            lambda: payload,
            len(payload["data"]) + 128,
            on_sent=mark,
            tag="grid",
        )

    def _paths_tick(self) -> None:
        if not self._paths_dirty:
            return
        self._paths_dirty = False
        p = self._paths
        payload = media.paths_payload(
            p["smooth"], p["mlp"], p["target"], self._paths_frame, self._paths_t
        )
        if payload["hash"] == self.sent_paths_hash:
            return

        def mark(_n, h=payload["hash"]):
            self.sent_paths_hash = h

        self._queue(
            "paths",
            f"{self.prefix}/paths",
            lambda: payload,
            64 + 20 * (len(payload["smooth"]) + len(payload["mlp"])),
            on_sent=mark,
            tag="paths",
        )

    # --------------------------------------------------------- C4 media

    def _c4_cap(self) -> float:
        b = self.sched.budgets.get("C4")
        return float("inf") if b is None else max(1.0, b * self.cfg["burst_s"])

    def _c4_take(self, now: float) -> bool:
        """Token bucket on the C4 budget (may go into debt by one frame)."""
        b = self.sched.budgets.get("C4")
        if b is None:
            return True
        if self._c4_t is not None:
            self._c4_tokens = min(
                self._c4_cap(), self._c4_tokens + b * max(0.0, now - self._c4_t)
            )
        self._c4_t = now
        return self._c4_tokens > 0

    def _on_camera_query(self, key: str, params: dict):
        name = key.rsplit("/", 1)[-1]
        now = self.clock()
        with self._media_lock:
            names = list(self.camera_names)
            frame = self._frames.get(name)
        if name not in names:
            return media.pack_reply({"error": "unknown camera", "cameras": names})
        self.want_camera(name, now)
        if frame is None:  # lazy camera: subscribed within ~1 s, ask again
            return media.pack_reply(
                {"error": "no frame yet", "subscribing": True, "cameras": names}
            )

        def num(k, default, lo, hi):
            try:
                return int(min(hi, max(lo, int(float(params.get(k, default))))))
            except (TypeError, ValueError):
                return default

        w = num("w", 640, 16, 4096)
        q = num("q", 70, 5, 95)
        with self._media_lock:
            if not self._c4_take(now):
                self.media_refused += 1
                return media.pack_reply({"error": "budget", "cameras": names})
        try:
            rep = media.camera_reply(frame, w, q, now)
        except Exception as exc:  # noqa: BLE001
            return media.pack_reply({"error": f"encode failed: {exc}"})
        rep["camera"] = name
        out = media.pack_reply(rep)
        with self._media_lock:
            if self.sched.budgets.get("C4") is not None:
                self._c4_tokens -= len(out)
            self.media_served += 1
        return out

    # ------------------------------------------------------------------- C1

    def _state_payload(self):
        with self._lock:
            return {
                "pose": self.pose,
                "guards": self.guards,
                "health": self.health,
                "results": list(self.results),
                "stats": self.stats(),
                "cameras": list(self.camera_names),
                "hashes": {"grid": self.sent_grid_hash, "paths": self.sent_paths_hash},
                # last C2 seq handed to the transport: lets the base detect a
                # lost *last* message, which no later seq gap would reveal
                "seqs": {"dsg": self.seq["dsg/delta"] - 1},
                "video": self.video.state(),
                "store": self._store_state(),
                "platform": self.platform,
                **self._frame_align_state(),
                **self._reloc_state(),
                **self._phoenix_state(),
                **({"lidar": self._lidar_state()} if self.lidar_topic else {}),
            }

    def _phoenix_state(self) -> dict:
        """``phoenix_progress`` (phoenix only; None before any plan)."""
        if self.platform != PHOENIX:
            return {}
        return {"phoenix_progress": phoenix_progress(self._plan, self.pose)}

    def _lidar_state(self) -> dict:
        age = None
        if self._lidar_rx_t is not None:
            age = round(max(0.0, self.clock() - self._lidar_rx_t), 2)
        # ``available``: the robot has a lidar (the console keeps its toggle and
        # the receiver keeps reconciling); a bad cloud is only ``error``
        return {
            "available": self.lidar_topic is not None,
            "enabled": self.lidar_enabled,
            "age_s": age,  # since the newest frame reached the agent
            "bytes": self.lidar_sent_bytes,
            "error": self.lidar_error,
        }

    def _set_lidar(self, enable: bool) -> None:
        if not enable:
            self.sched.clear("C3", lambda it: it.tag == "lidar")
            self._lidar = None  # a frame from before must not go out later
        else:
            self.lidar_error = None  # a fresh start: report only new errors
            if not self.lidar_enabled:
                self._t_lidar = -1e9
        self.lidar_enabled = enable

    def _lidar_tick(self, now: float) -> None:
        """Queue the newest frame (one at most: tag ``lidar`` replaces a queued
        one, built at send time) every 1 / ``lidar_hz`` s, slower when frames are
        big (``lidar_bps``)."""
        ttl = self.cfg.get("lidar_ttl_s")
        if self.lidar_enabled and ttl and now - self._lidar_req_t > float(ttl):
            self._set_lidar(False)  # no base kept it alive
        if not self.lidar_enabled or not self._lidar_dirty or self._lidar is None:
            return
        period = 1.0 / max(1e-3, float(self.cfg["lidar_hz"]))
        bps = self.cfg.get("lidar_bps")
        if bps:
            period = max(period, self.lidar_sent_bytes / float(bps))
        if now - self._t_lidar < period:
            return
        self._t_lidar = now
        self._lidar_dirty = False

        def payload():
            return self._lidar  # the newest when it leaves (None: turned off)

        def sent(n):
            self.lidar_sent_bytes = n

        f = self._lidar
        self._queue(
            "lidar/points",
            f"{self.prefix}/lidar/points",
            payload,
            len(f.get("xyz") or b"") + len(f.get("intensity") or b"") + 128,
            on_sent=sent,
            tag="lidar",
        )

    def _reloc_state(self) -> dict:
        """Optional ``reloc`` key (absent unless ROMAN is subscribed, so old
        consoles see the payload they always saw)."""
        if self.reloc is None:
            return {}
        return {"reloc": self.reloc.state(self.clock())}

    def _frame_align_state(self) -> dict:
        """Optional ``frame_align`` key (absent until the first lookup, so old
        consoles see the payload they always saw).  ``age_s`` = robot clock now
        minus the transform stamp, so the base needs no clock-offset estimate."""
        fa = self.frame_align
        if fa is None:
            return {}
        fa = dict(fa)
        st = fa.get("stamp")
        fa["age_s"] = (
            round(max(0.0, self.clock() - st), 3) if st else None
        )  # stamp 0: static TF
        return {"frame_align": fa}

    def stats(self) -> dict:
        s = self.sched
        return {
            "epoch": self.epoch,
            "sent": dict(s.sent_bytes),
            "rate": {c: round(r, 1) for c, r in s.rate.items()},
            "queue": s.queue_stats(),
            # C4 (camera queries) / C5 (video) too, so the console can show
            # and adjust every class's budget
            "budgets": {c: s.budgets[c] for c in CLASSES_REPORTED if c in s.budgets},
            "auto": s.auto,
            "nodes": len(self._target[0]) if self._target else 0,
            "tiles": len(self.tiles),
            "mesh_lod": self.mesh_lod,
            "mesh": {
                "source": "odometric" if self.odom is not None else "dsg",
                "configured": self.cfg["mesh_source"],
                "fallback": self.mesh_fallback,
            },  # fmt: skip
            "c2_resent": self.c2_resent,
            "dsg": {k: v for k, v in dict(self.dsg_diag).items() if k[0] != "_"},
        }

    # ------------------------------------------------------------------- C6

    def _store_state(self):
        if self.store is None:
            return {"id": None, "enabled": False, "error": self.store_error}
        try:
            return {**self.store.state(), "enabled": True, "send": self.store_send}
        except Exception as exc:  # noqa: BLE001
            return {"id": None, "enabled": True, "error": str(exc)}

    def _store_tick(self) -> None:
        """Queue the next store frames within ``store_queue_s`` of C6 budget.
        In auto mode the mirror only fills the link when C2/C3 have nothing
        queued, so it never delays the graph or the mesh."""
        st = self.store
        st.refresh()
        if not st.active or not self.store_send:
            return
        if self.sched.auto and (self.sched.queues["C2"] or self.sched.queues["C3"]):
            return
        budget = self.sched.budget("C6")
        if budget is not None and budget <= 0:
            return
        cap = float("inf") if budget is None else budget * self.cfg["store_queue_s"]
        queued = self.sched.queued_bytes("C6")
        room = max(0, int((cap - queued) // max(1.0, st.avg_bytes)))
        if not self.sched.queues["C6"]:
            room = max(room, 1)  # always one in flight (a frame larger than cap)
        room = min(room, 16)
        for fid, prev in st.take(room):

            def payload(fid=fid, prev=prev):
                return st.encode(fid, prev)

            def sent(n, fid=fid, late=prev == "late"):
                st.mark_sent(fid, n, late=late)

            self._queue(
                "store/frame",
                f"{self.prefix}/store/frame",
                payload,
                int(st.avg_bytes),
                on_sent=sent,
            )

    # ------------------------------------------------------------------ tick

    def tick(self, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        with self._inbox_lock:
            reqs = list(self._inbox)
            self._inbox.clear()
        with self._lock:
            for req in reqs:
                self._handle(req, now)
            if now - self._t_state >= 1.0 / self.cfg["state_hz"]:
                self._t_state = now
                self._queue(
                    "state",
                    f"{self.prefix}/state",
                    self._state_payload,
                    512,
                    tag="state",
                )
            if now - self._t_c2 >= 1.0 / self.cfg["c2_hz"]:
                self._t_c2 = now
                self._c2_tick()
            if self._tiles_dirty or (
                now - self._t_mesh >= self.cfg["mesh_period_s"]
                and self._tiled_version != self._mesh_version
            ):
                self._t_mesh = now
                self._c3_tick()
            if self.odom is not None and self.odom.values_due(now):
                payload = self.odom.values_payload(now)
                self._queue("mesh/values", f"{self.prefix}/mesh/values", lambda p=payload: p,
                            len(payload["data"]) + 32, tag="values", front=True)  # fmt: skip
            if now - self._t_manifest >= self.cfg["manifest_period_s"] and self.tiles:
                self._t_manifest = now
                self._queue_manifest()
            if self._grid_dirty and now - self._t_grid >= self.cfg["grid_period_s"]:
                self._t_grid = now
                self._grid_tick()
            if self._paths_dirty and now - self._t_paths >= 1.0 / self.cfg["paths_hz"]:
                self._t_paths = now
                self._paths_tick()
            self._lidar_tick(now)
            if (
                self.store is not None
                and now - self._t_store >= self.cfg["store_poll_s"]
            ):
                self._t_store = now
                try:
                    self._store_tick()
                except Exception as exc:  # noqa: BLE001  (never stop the tick)
                    self.store_error = f"{type(exc).__name__}: {exc}"
            sent = self.sched.drain(now)
        self.video.tick(now)  # outside the core lock: encoders call back from threads
        return sent

    def graph_state(self) -> GraphState:
        """The robot's current (target) graph as a GraphState (for comparisons)."""
        g = GraphState()
        if self._target:
            g.records = dict(self._target[0])
            g.edges = set(self._target[1])
            g.labelspaces = dict(self._target[2])
        return g

    def close(self) -> None:
        self.video.close()
        if self.store is not None:
            self.store.close()
        unsub = getattr(self.transport, "unsubscribe", None)
        if unsub is not None:
            unsub(self._sub)
        qs = getattr(self.transport, "_queryables", None)
        if qs is not None and self._queryable in qs:
            qs.remove(self._queryable)


# ----------------------------------------------------------------------------
# ROS wrapper
# ----------------------------------------------------------------------------


def guards_to_dict(msg) -> dict:
    return {
        "robot_name": msg.robot_name,
        "battery_percent": float(msg.battery_percent),
        "battery_known": bool(msg.battery_known),
        "estop_pressed": bool(msg.estop_pressed),
        "estop_known": bool(msg.estop_known),
        "powered_on": bool(msg.powered_on),
        "power_known": bool(msg.power_known),
        "lease_owned": bool(msg.lease_owned),
        "lease_known": bool(msg.lease_known),
        # newer field: robots built before it have no lease_owner
        "lease_owner": str(getattr(msg, "lease_owner", "") or ""),
        "faults": list(msg.faults),
        "notes": msg.notes,
        "t": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
    }


def phoenix_progress(plan: dict | None, pose: dict | None) -> dict | None:
    """``{plan_id, stamp, dist_to_goal_m, done}`` from the latest Path (see
    :meth:`AgentCore.on_plan_path`) and pose.  The distance (xy) needs both in
    the same frame: null when they differ (no TF between them) or are unknown.
    ``done``: within PHOENIX_DONE_M of the end *and* the plan's ``left`` mark
    (the robot was farther than that from the end since the plan arrived, or
    the path is shorter than that)."""
    if plan is None:
        return None
    dist = _dist_to_end(plan, pose)
    if dist is not None:
        dist = round(dist, 3)
    return {
        "plan_id": plan["plan_id"],
        "stamp": plan["stamp"],
        "dist_to_goal_m": dist,
        "done": dist is not None and dist < PHOENIX_DONE_M and bool(plan.get("left")),
    }


def _dist_to_end(plan: dict, pose: dict | None) -> float | None:
    """xy distance from ``pose`` to the end of ``plan`` (None: unknown / other frame)."""
    end = plan.get("end")
    if end is None or pose is None or pose.get("frame") != plan.get("frame"):
        return None
    return float(np.hypot(pose["x"] - end[0], pose["y"] - end[1]))


def _plan_mark_left(plan: dict | None, pose: dict | None) -> None:
    """Mark ``plan`` ``left`` once ``pose`` is farther than PHOENIX_DONE_M from its end."""
    if plan is None or plan.get("left"):
        return
    dist = _dist_to_end(plan, pose)
    if dist is not None and dist > PHOENIX_DONE_M:
        plan["left"] = True


class PhoenixGuards:
    """Phoenix guards from the platform topics, in the Spot ``guards`` keys the
    console reads (``battery_percent``, ``estop_pressed``) plus ``safety_stop``
    and ``manual_mode``; None until a topic has been heard."""

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.values = {
            "battery_percent": None,
            "estop_pressed": None,
            "safety_stop": None,
            "manual_mode": None,
        }
        self.t = None

    def battery(self, percentage) -> dict:
        """sensor_msgs/BatteryState ``percentage``: 0-1 (NaN: unknown)."""
        p = float(percentage)
        return self._set(
            "battery_percent", round(100.0 * p, 1) if np.isfinite(p) else None
        )

    def estop(self, pressed) -> dict:
        return self._set("estop_pressed", bool(pressed))

    def safety_stop(self, stopped) -> dict:
        return self._set("safety_stop", bool(stopped))

    def manual_mode(self, manual) -> dict:
        return self._set("manual_mode", bool(manual))

    def _set(self, key, value) -> dict:
        self.values[key] = value
        self.t = self.clock()
        return self.as_dict()

    def as_dict(self) -> dict:
        return {**self.values, "source": PHOENIX, "t": self.t}


def trajectory_points(msg) -> np.ndarray:
    """phx_nav_msgs/TrajectoryArray → (N, 3) points of its first trajectory.

    Duck-typed (the message definition is not available here): the first list
    field holds trajectories, each a list (``poses`` / ``points`` / ...) of
    PoseStamped, Pose or Point-like entries."""

    def lists(m):
        for name in getattr(m, "get_fields_and_field_types", dict)():
            v = getattr(m, name, None)
            if isinstance(v, (list, tuple)) and len(v):
                yield v

    def xyz(e):
        e = getattr(e, "pose", e)
        e = getattr(e, "position", e)
        return [float(e.x), float(e.y), float(e.z)]

    for trajs in lists(msg):
        for entries in lists(trajs[0]):
            try:
                return np.array([xyz(e) for e in entries], dtype=float).reshape(-1, 3)
            except AttributeError:
                continue
    return np.zeros((0, 3))


def result_to_dict(msg) -> dict:
    return {
        "robot_name": msg.robot_name,
        "plan_id": msg.plan_id,
        "action_type": msg.action_type,
        "action_index": int(msg.action_index),
        "status": msg.status,
        "detail": msg.detail,
        "t": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
    }


def status_table_to_dict(msg) -> dict:
    names = {1: "nominal", 2: "warning", 3: "error", 4: "no_hb", 5: "startup"}
    counts = {"nominal": 0, "warning": 0, "error": 0}
    nodes = []
    for n in msg.status:
        st = names.get(int(n.status), str(n.status))
        if st in counts:
            counts[st] += 1
        elif st == "no_hb":
            counts["error"] += 1
        nodes.append(
            {
                "nickname": n.nickname,
                "node_name": n.node_name,
                "status": st,
                "notes": n.notes,
                "required": bool(n.required),
            }
        )
    return {**counts, "nodes": nodes, "monitor": msg.monitor_name}


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def parse_cameras(entries) -> dict[str, str]:
    """``["rgb=euclid_zed/rgb/image_rect_color", ...]`` → ``{name: topic}``."""
    out = {}
    for e in entries or []:
        name, sep, topic = str(e).partition("=")
        if sep and name.strip() and topic.strip():
            out[name.strip()] = topic.strip()
    return out


def path_points(msg) -> np.ndarray:
    """nav_msgs/Path → (N, 3) float array."""
    return np.array(
        [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
        dtype=float,
    ).reshape(-1, 3)


def marker_target(msg):
    """visualization_msgs/MarkerArray → position of the first live marker (or None)."""
    for m in msg.markers:
        if int(m.action) not in (0,):  # ADD/MODIFY only; DELETE/DELETEALL clear
            continue
        if m.points:
            p = m.points[-1]
            base = m.pose.position
            return [base.x + p.x, base.y + p.y, base.z + p.z], m.header.frame_id
        p = m.pose.position
        return [p.x, p.y, p.z], m.header.frame_id
    return None, (msg.markers[0].header.frame_id if msg.markers else "")


def transform_points(pts: np.ndarray, tf) -> np.ndarray:
    """Apply a geometry_msgs/TransformStamped to (N, 3) points."""
    t, q = tf.transform.translation, tf.transform.rotation
    w, x, y, z = q.w, q.x, q.y, q.z
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return pts @ R.T + np.array([t.x, t.y, t.z])


def frame_align_ok(tf, parent: str, child: str) -> dict:
    """``frame_align`` from a geometry_msgs/TransformStamped ``parent <- child``:
    ``T`` maps points in ``child`` (the scene-graph / odom frame every live layer
    is sent in) into ``parent`` (the global frame).  ``stamp`` 0 = static TF."""
    t, q = tf.transform.translation, tf.transform.rotation
    st = tf.header.stamp
    return {
        "parent": parent,
        "child": child,
        "T": [float(v) for v in (t.x, t.y, t.z, q.w, q.x, q.y, q.z)],
        "stamp": st.sec + st.nanosec * 1e-9,
        "ok": True,
        "reason": "",
    }


def frame_align_identity(frame: str) -> dict:
    """The scene graph already lives in the global frame."""
    return {
        "parent": frame,
        "child": frame,
        "T": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "stamp": 0.0,
        "ok": True,
        "reason": "",
    }


def frame_align_failed(parent: str, child: str, reason: str) -> dict:
    """No ``parent <- child`` transform (not relocalized / no global frame)."""
    return {
        "parent": parent,
        "child": child,
        "T": None,
        "stamp": None,
        "ok": False,
        "reason": " ".join(str(reason).split())[:200],
    }


def default_body_frame(robot: str, platform: str) -> str:
    """The frame the pose is looked up for: phoenix (Husky URDF) has no body
    frame, its root link is base_link."""
    return f"{robot}/base_link" if platform == PHOENIX else f"{robot}/body"


def resolve_platform(robot: str, explicit: str = "", topology: str = "") -> str:
    """``explicit`` (param) or the robot's topology ``platform_type``, by the one
    rule (:func:`relay_ros.platform_of`: missing → spot, husky → phoenix)."""
    from fleetlink import relay_ros

    if explicit:
        return relay_ros.platform_of({"platform_type": explicit})
    try:
        topo = relay_ros.load_topology(relay_ros.find_topology(topology))
        machine = ((topo or {}).get("machines") or {}).get(robot)
    except Exception:  # noqa: BLE001  (no topology file on this machine)
        machine = None
    return relay_ros.platform_of(machine)


def lookup_frame_align(tf_buffer, parent: str, child: str) -> dict:
    """Latest ``parent <- child`` from a tf2 buffer as a ``frame_align`` dict."""
    if not child:
        return frame_align_failed(parent, child, "scene-graph frame unknown")
    if child == parent:
        return frame_align_identity(parent)
    if tf_buffer is None:
        return frame_align_failed(parent, child, "tf2_ros unavailable")
    try:
        from rclpy.time import Time

        latest = Time()
    except ImportError:  # unit tests with a fake buffer
        latest = 0
    try:
        tf = tf_buffer.lookup_transform(parent, child, latest)
    except Exception as exc:  # noqa: BLE001  (tf2 Lookup/Connectivity/Extrapolation)
        return frame_align_failed(
            parent, child, f"not relocalized: {type(exc).__name__}: {exc}"
        )
    return frame_align_ok(tf, parent, child)


def lidar_qos():
    """The lidar subscription's QoS: depth 1 (only the newest cloud is wanted,
    none queue up behind a slow callback), best effort (matches a RELIABLE
    publisher, e.g. the Ouster driver's KEEP_LAST 10, as well as a best-effort
    one), volatile."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


class AgentNode:
    """rclpy wrapper: subscriptions + TF pose → :class:`AgentCore`.

    Graph deserialisation runs on a worker thread holding only the newest
    pending DsgUpdate (older pending ones are dropped).  Ticking runs on its
    own thread so a slow executor never delays C1.
    """

    def __init__(
        self, transport: Transport | None = None, node=None, node_name="fleetlink_agent"
    ):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile

        self._rclpy = rclpy
        self.node = node or Node(node_name)
        nd = self.node
        cfg_file = nd.declare_parameter("config", "").value or None
        cfg = load_config(cfg_file)
        p = lambda name, default: nd.declare_parameter(name, default).value  # noqa: E731
        robot = p("robot_name", nd.get_namespace().strip("/") or "robot")
        # platform: ``platform`` (or ``relay_platform``) param, else the topology
        relay_platform = p("relay_platform", "")
        topology = p("topology", "")
        self.platform = resolve_platform(
            robot, p("platform", "") or relay_platform, topology
        )
        phoenix = self.platform == PHOENIX
        cfg["platform"] = self.platform
        # Pose must be expressed in the frame the scene graph lives in, which is
        # the DSG message's header.frame_id (Hydra: <robot>/map).  ``map_frame``
        # overrides that; the global ``map`` frame is a last resort only, since
        # map -> <robot>/map is often a placeholder static transform.
        self.map_frame = p("map_frame", "")
        self._dsg_frame = ""
        self.body_frame = p("body_frame", default_body_frame(robot, self.platform))
        self.fallback_map_frame = p("fallback_map_frame", f"{robot}/map")
        # the global frame prior maps / the planner use; ``global_frame <- DSG
        # frame`` (relocalization: map -> <robot>/odom) goes out as ``frame_align``
        # so the console can draw the live layers over the prior map.  "" = off.
        self.global_frame = p("global_frame", "map")
        frame_align_hz = p("frame_align_hz", 1.0)
        dsg_topic = p("dsg_topic", "hydra/backend/dsg")
        guards_topic = p("guards_topic", "spot_executor_node/runtime_guards")
        result_topic = p("result_topic", "spot_executor_node/action_result")
        health_topic = p("health_topic", "/global/ros_system_monitor/table_in")
        # ROMAN loop closures (prior-map matches) for the ``reloc`` state; "" = off
        roman_lc_topic = p("roman_lc_topic", "roman/roman_lc/loop_closure")
        roman_live_id = p("roman_live_robot_id", 0)
        self.health_monitor = p("health_monitor_name", robot)
        for cls in ("C1", "C2", "C3"):
            cfg["budgets"][cls] = p(f"budget_{cls}", int(cfg["budgets"][cls]))
        cfg["auto"] = p("auto", bool(cfg["auto"]))
        tick_hz = p("tick_hz", 20.0)
        cfg["mesh_period_s"] = p("mesh_period_s", float(cfg["mesh_period_s"]))
        cfg["tile_size"] = p("tile_size", float(cfg["tile_size"]))
        cfg["mesh_decimate_cell"] = p(
            "mesh_decimate_cell", float(cfg["mesh_decimate_cell"])
        )
        # "dsg" | "odometric"; ADT4_FLEETLINK_MESH_SOURCE sets the default per robot
        cfg["mesh_source"] = p(
            "mesh_source",
            os.environ.get("ADT4_FLEETLINK_MESH_SOURCE") or str(cfg["mesh_source"]),
        )
        mesh_delta_topic = p("mesh_delta_topic", "hydra/frontend/full_mesh_update")
        mesh_graph_inc_topic = p(
            "mesh_graph_incremental_topic", "hydra/frontend/mesh_graph_incremental"
        )
        mesh_graph_topic = p("mesh_graph_topic", "hydra/backend/mesh_graph")
        # one endpoint or a comma-separated list (zenoh keeps retrying them all)
        connect = split_endpoints(p("zenoh_connect", "tcp/127.0.0.1:7447"))
        cfg["c2_ring"] = p("c2_ring", int(cfg["c2_ring"]))
        cfg["budgets"]["C4"] = p("budget_C4", int(cfg["budgets"]["C4"]))
        cfg["budgets"]["C5"] = p("budget_C5", int(cfg["budgets"]["C5"]))
        cfg["ffmpeg"] = p("ffmpeg", cfg["ffmpeg"] or "") or None
        grid_topic = p("grid_topic", "hydra/tsdf/occupancy")
        cfg["grid_period_s"] = p("grid_period_s", float(cfg["grid_period_s"]))
        smooth_topic = p(
            "smooth_path_topic", "spot_executor_node/smooth_path_publisher"
        )
        mlp_topic = p("mlp_path_topic", "spot_executor_node/mlp_path_publisher")
        target_topic = p("mlp_target_topic", "spot_executor_node/mlp_target_publisher")
        cameras = parse_cameras(
            p(
                "cameras",
                [
                    # robot1: euclid_zed/rgb/image_rect_color
                    f"rgb={robot}_zed/rgb/image_rect_color",
                    "overlay=semantic_overlay/image_raw",
                ],
            )
        )
        cfg["cameras"] = list(cameras)
        cfg["eager_cameras"] = list(
            p("eager_cameras", list(cfg["eager_cameras"])) or []
        )
        cfg["camera_idle_s"] = p("camera_idle_s", float(cfg["camera_idle_s"]))
        # C6 live image-store mirror: "" disables
        default_store = ""
        if os.environ.get("ADT4_OUTPUT_DIR"):
            default_store = os.path.join(os.environ["ADT4_OUTPUT_DIR"], "image_store")
        cfg["store_path"] = p("store_path", default_store) or None
        cfg["store_send"] = p("store_send", bool(cfg["store_send"]))
        cfg["store_frames"] = p("store_frames", str(cfg["store_frames"]))
        cfg["store_compression"] = p("store_compression", str(cfg["store_compression"]))
        cfg["budgets"]["C6"] = p("budget_C6", int(cfg["budgets"]["C6"]))
        # C3 lidar stream (off until the console asks): "" = no lidar
        cfg["lidar_topic"] = (
            p("lidar_topic", f"/{robot}/ouster/points" if phoenix else "") or None
        )
        cfg["lidar_hz"] = p("lidar_hz", float(cfg["lidar_hz"]))
        cfg["lidar_bps"] = p("lidar_bps", int(cfg["lidar_bps"]))
        cfg["lidar_intensity"] = p("lidar_intensity", bool(cfg["lidar_intensity"]))
        cfg["lidar_ttl_s"] = p("lidar_ttl_s", float(cfg["lidar_ttl_s"]))
        # phoenix platform topics (guards, plan progress, executor path)
        phx_topics = {
            "battery": p("battery_topic", "platform/bms/state"),
            "estop": p("estop_topic", "platform/emergency_stop"),
            "safety_stop": p("safety_stop_topic", "platform/safety_stop"),
            "manual_mode": p("manual_mode_topic", "manual_mode"),
            "mcu": p("mcu_status_topic", "platform/mcu/status"),
            "plan": p("plan_topic", "omniplanner_node/compiled_plan_out"),
            "trajectories": p("trajectories_topic", "mrp/trajectories"),
        }

        self.log = nd.get_logger()
        warning = zenoh_version_warning()
        if warning:
            self.log.warn(warning)
        if transport is None:
            transport = self._open_zenoh(connect)
        self.transport = transport
        self.core = AgentCore(robot, transport, cfg)
        if self.core.store_error:
            self.log.warn(f"store mirror disabled: {self.core.store_error}")
        elif self.core.store is not None:
            self.log.info(
                f"store mirror: {cfg['store_path']} frames={cfg['store_frames']} "
                f"compression={cfg['store_compression']} C6={cfg['budgets']['C6']} B/s "
                f"send={'on' if cfg['store_send'] else 'off (console toggles it)'}"
            )
        self.spark_dsg_path = spark_dsg_location()
        self.log.info(f"spark_dsg: {self.spark_dsg_path}")
        self.core.dsg_diag.update(
            {"spark_dsg": self.spark_dsg_path, "received": 0, "processed": 0,
             "dropped": 0, "failed": 0}
        )  # fmt: skip

        self._pending = None
        self._pending_cv = threading.Condition()
        self._running = True
        self.graphs_received = 0
        self.graphs_dropped = 0
        self.graphs_processed = 0

        from hydra_msgs.msg import DsgUpdate

        nd.create_subscription(DsgUpdate, dsg_topic, self._on_dsg, QoSProfile(depth=1))
        if self.core.odom is not None:
            self._subscribe_odom_mesh(
                mesh_delta_topic, mesh_graph_inc_topic, mesh_graph_topic
            )
        # best-effort subscribers match both reliable and best-effort publishers
        from rclpy.qos import qos_profile_sensor_data

        sensor_qos = qos_profile_sensor_data
        try:
            from nav_msgs.msg import OccupancyGrid, Path

            nd.create_subscription(OccupancyGrid, grid_topic, self._on_grid, sensor_qos)
            if not phoenix:  # spot_executor_node paths
                nd.create_subscription(
                    Path, smooth_topic, lambda m: self._on_path("smooth", m), sensor_qos
                )
                nd.create_subscription(
                    Path, mlp_topic, lambda m: self._on_path("mlp", m), sensor_qos
                )
        except ImportError:
            self.log.warn("nav_msgs unavailable: no occupancy grid / paths")
        if not phoenix:
            try:
                from visualization_msgs.msg import MarkerArray

                nd.create_subscription(
                    MarkerArray, target_topic, self._on_target, sensor_qos
                )
            except ImportError:
                self.log.warn("visualization_msgs unavailable: no MLP target")
        # cameras: subscribed lazily (eager_cameras always), see _update_cameras
        self._camera_topics = dict(cameras)
        self._camera_subs: dict = {}
        self._sensor_qos = sensor_qos
        try:
            from sensor_msgs.msg import CompressedImage, Image

            self._image_type = Image
            self._compressed_type = CompressedImage
        except ImportError:
            self._image_type = None
            self.log.warn("sensor_msgs unavailable: no cameras")
        self._update_cameras()
        nd.create_timer(1.0, self._update_cameras)
        # lidar: subscribed only while the base wants the stream (_update_lidar)
        self._lidar_sub = None
        self._lidar_t = -1e9
        self._lidar_warned = ""
        self._cloud_type = None
        self._lidar_group = None
        if self.core.lidar_topic:
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

            # its own group: a big cloud never holds up pose / graph / C0
            # callbacks (MultiThreadedExecutor), and one cloud at a time
            self._lidar_group = MutuallyExclusiveCallbackGroup()
            try:
                from sensor_msgs.msg import PointCloud2

                self._cloud_type = PointCloud2
            except ImportError as exc:
                self.core.lidar_error = f"sensor_msgs unavailable: {exc}"
            nd.create_timer(1.0, self._update_lidar)
        if phoenix:
            self._subscribe_phoenix(phx_topics, sensor_qos)
        else:
            self._subscribe_spot_executor(guards_topic, result_topic)
        try:
            from ros_system_monitor_msgs.msg import StatusTable

            nd.create_subscription(StatusTable, health_topic, self._on_health, 10)
        except ImportError:
            self.log.warn("ros_system_monitor_msgs unavailable: no health")
        if roman_lc_topic:
            try:
                from roman_msgs.msg import LoopClosure

                from .reloc import loop_closure_to_dict

                self.core.enable_reloc(int(roman_live_id))
                nd.create_subscription(
                    LoopClosure,
                    roman_lc_topic,
                    lambda m: self.core.on_loop_closure(loop_closure_to_dict(m)),
                    10,
                )
            except ImportError:
                self.log.warn("roman_msgs unavailable: no relocalization evidence")

        self._tf_buffer = None
        try:
            import tf2_ros

            self._tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, nd)
        except ImportError:
            self.log.warn("tf2_ros unavailable: no pose")
        nd.create_timer(1.0 / max(cfg["state_hz"], 1.0), self._update_pose)
        if self.global_frame and frame_align_hz > 0:
            nd.create_timer(1.0 / frame_align_hz, self._update_frame_align)

        # generic ROS relay (relay.yaml); off unless ``relay_config`` is set, so
        # router-mode deployments behave exactly as before
        self.relay = None
        relay_config = p("relay_config", "")
        if relay_config:
            self.relay = self._start_relay(robot, relay_config, self.platform, topology)

        self._decoder = threading.Thread(target=self._decode_loop, daemon=True)
        self._decoder.start()
        self._tick_period = 1.0 / tick_hz
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True)
        self._ticker.start()
        # video frames are sampled on their own thread so 15 fps is not quantised
        # to the 20 Hz tick
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._video_thread.start()
        self.log.info(
            f"fleetlink agent robot={robot} platform={self.platform} "
            f"body={self.body_frame} epoch={self.core.epoch} dsg={dsg_topic} "
            f"budgets={cfg['budgets']}"
        )

    def _subscribe_spot_executor(self, guards_topic: str, result_topic: str) -> None:
        """Spot: guards and action results from spot_executor_node."""
        try:
            from robot_executor_msgs.msg import ActionResultMsg, RuntimeGuardsMsg

            self.node.create_subscription(
                RuntimeGuardsMsg,
                guards_topic,
                lambda m: self.core.on_guards(guards_to_dict(m)),
                10,
            )
            self.node.create_subscription(
                ActionResultMsg,
                result_topic,
                lambda m: self.core.on_result(result_to_dict(m)),
                10,
            )
        except ImportError:
            self.log.warn("robot_executor_msgs unavailable: no guards/results")

    def _subscribe_phoenix(self, topics: dict, qos) -> None:
        """Phoenix: guards from the platform topics, plan progress from the
        omniplanner Path, and (if phx_nav_msgs imports) the MRP trajectory as the
        executor path overlay."""
        nd = self.node
        self._phx_guards = PhoenixGuards()
        g = self._phx_guards
        guards = self.core.on_guards
        guards(g.as_dict())  # every key present (null) before the first message
        try:
            from sensor_msgs.msg import BatteryState
            from std_msgs.msg import Bool

            nd.create_subscription(
                BatteryState,
                topics["battery"],
                lambda m: guards(g.battery(m.percentage)),
                qos,
            )
            for key, fn in (("estop", g.estop), ("safety_stop", g.safety_stop),
                            ("manual_mode", g.manual_mode)):  # fmt: skip
                nd.create_subscription(
                    Bool, topics[key], lambda m, fn=fn: guards(fn(m.data)), qos
                )
        except ImportError as exc:
            self.log.warn(f"phoenix guards unavailable: {exc}")
        try:  # optional: proves the MCU link is alive (its fields are not used)
            from clearpath_platform_msgs.msg import Status

            def on_mcu(_m):
                g.values["mcu_t"] = time.time()

            nd.create_subscription(Status, topics["mcu"], on_mcu, qos)
        except ImportError:
            pass
        try:
            from nav_msgs.msg import Path

            nd.create_subscription(Path, topics["plan"], self._on_plan, qos)
        except ImportError:
            self.log.warn("nav_msgs unavailable: no phoenix plan progress")
        try:
            from phx_nav_msgs.msg import TrajectoryArray

            nd.create_subscription(
                TrajectoryArray, topics["trajectories"], self._on_trajectories, qos
            )
        except ImportError:
            self.log.warn("phx_nav_msgs unavailable: no executor path overlay")

    def _on_plan(self, msg) -> None:
        """Phoenix omniplanner Path → phoenix_progress, in the pose frame when TF
        allows (else its own frame: the distance is then null)."""
        target = (self.core.pose or {}).get("frame") or self._map_frame()
        pts, frame = self._to_frame(path_points(msg), msg.header.frame_id, target)
        st = msg.header.stamp
        sec, nsec = int(st.sec), int(st.nanosec)
        if sec == 0 and nsec == 0:  # unstamped: name the plan by its arrival
            now = time.time()
            sec, nsec = int(now), int((now % 1.0) * 1e9)
        self.core.on_plan_path(pts, frame, sec, nsec)

    def _on_trajectories(self, msg) -> None:
        pts = trajectory_points(msg)
        header = getattr(msg, "header", None)
        frame = getattr(header, "frame_id", "") if header is not None else ""
        pts, frame = self._to_map(pts, frame)
        self.core.on_path("smooth", pts, frame, stamp_s(header) if header else None)

    def _update_lidar(self) -> None:
        """Subscribe the lidar while the base wants it, drop it otherwise (1 Hz)."""
        if self.core.lidar_enabled and self._cloud_type is None:
            self.core.lidar_error = "sensor_msgs unavailable: no PointCloud2"
        want = self.core.lidar_enabled and self._cloud_type is not None
        if want and self._lidar_sub is None:
            self._lidar_sub = self.node.create_subscription(
                self._cloud_type, self.core.lidar_topic, self._on_cloud, lidar_qos(),
                callback_group=self._lidar_group,
            )  # fmt: skip
        elif not want and self._lidar_sub is not None:
            self.node.destroy_subscription(self._lidar_sub)
            self._lidar_sub = None

    def _on_cloud(self, msg) -> None:
        """Sample the lidar at ``lidar_hz``: encode it in the sensor frame, with the
        sensor's pose in the robot pose's frame as ``origin``.  The time gate
        comes first: the ~10 Hz clouds in between cost next to nothing."""
        now = time.time()
        if not self.core.lidar_enabled or now - self._lidar_t < 0.9 / max(
            1e-3, float(self.core.cfg["lidar_hz"])
        ):
            return
        from fleetlink import pointcloud_codec

        parent = (self.core.pose or {}).get("frame") or self._map_frame()
        sensor = msg.header.frame_id
        try:
            from rclpy.time import Time

            tf = self._tf_buffer.lookup_transform(parent, sensor, Time())
        except Exception as exc:  # noqa: BLE001  (no TF yet / tf2 missing)
            reason = f"no transform {parent} <- {sensor}: {type(exc).__name__}"
            if reason != self._lidar_warned:
                self._lidar_warned = reason
                self.log.warn(f"lidar: {reason}")
            return
        self._lidar_t = now
        try:
            xyz, intensity = pointcloud_codec.cloud_from_msg(msg)
            if not self.core.cfg.get("lidar_intensity"):
                intensity = None  # the console colours by height
            frame = pointcloud_codec.encode_cloud(xyz, intensity)
        except Exception as exc:  # noqa: BLE001
            self.core.lidar_error = f"{type(exc).__name__}: {exc}"
            return
        t, q = tf.transform.translation, tf.transform.rotation
        frame.update(
            frame=parent,
            sensor_frame=sensor,
            origin=[float(v) for v in (t.x, t.y, t.z, q.x, q.y, q.z, q.w)],
            stamp=stamp_s(msg.header),
        )
        self.core.on_lidar(frame)  # clears lidar_error

    def _start_relay(self, robot, config_path, platform, topology):
        from fleetlink import relay_ros
        from fleetlink.relay import RelayConfig

        relay_ros.require_file(config_path, "relay_config", relay_ros.RELAY_CONFIG_HINT)
        if not platform:
            topo = relay_ros.load_topology(relay_ros.find_topology(topology))
            platform = relay_ros.topology_robots(topo).get(
                robot, relay_ros.DEFAULT_PLATFORM
            )
        return relay_ros.RobotRelay(
            self.node,
            self.transport,
            robot,
            RelayConfig.load(config_path),
            platform=platform,
            tf_buffer=self._tf_buffer,
        )

    def _open_zenoh(self, connect: list[str]):
        """Never fails for want of a router: the transport keeps retrying the
        open in the background (and reconnects after router restarts)."""
        from fleetlink.transport import ZenohTransport

        transport = ZenohTransport(connect)
        if not transport.opened.is_set():
            self.log.warn(
                f"no zenoh router at {connect} yet ({transport.open_error}); "
                "retrying in the background"
            )
        return transport

    def _update_cameras(self) -> None:
        """Subscribe wanted cameras, unsubscribe idle ones (1 Hz timer)."""
        if self._image_type is None:
            return
        want = self.core.wanted_cameras(time.time())
        for name, topic in self._camera_topics.items():
            sub = self._camera_subs.get(name)
            if name in want and sub is None:
                compressed = topic.rstrip("/").endswith("/compressed")
                self._camera_subs[name] = self.node.create_subscription(
                    self._compressed_type if compressed else self._image_type,
                    topic,
                    (lambda m, name=name: self._on_compressed(name, m))
                    if compressed
                    else (lambda m, name=name: self._on_image(name, m)),
                    self._sensor_qos,
                )
            elif name not in want and sub is not None:
                self.node.destroy_subscription(sub)
                del self._camera_subs[name]
                self.core.drop_frame(name)

    def _subscribe_odom_mesh(self, delta_topic, inc_topic, graph_topic) -> None:
        """Odometric mesh mode: Hydra's mesh deltas + deformation graph."""
        from rclpy.qos import DurabilityPolicy, QoSProfile

        from .mesh_deform_link import delta_from_msg, nodes_from_pose_graph

        try:
            from kimera_pgmo_msgs.msg import MeshDelta
            from pose_graph_tools_msgs.msg import PoseGraph
        except ImportError as exc:
            self.log.warn(f"odometric mesh unavailable ({exc}); tiling the DSG mesh")
            self.core.fallback_to_dsg_mesh(str(exc))
            return
        # Hydra publishes both with QoS(100).transient_local(): match it to get the
        # last 100 on (re)connect
        latched = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._delta_seq = None

        def on_delta(msg):
            seq = int(msg.seq_number)
            if self._delta_seq is not None and seq != (self._delta_seq + 1) % 65536:
                if seq <= 1 and int(msg.prev_active_vertices) == 0:
                    self.core.reset_odom_mesh()  # Hydra restarted
                else:
                    self.core.fallback_to_dsg_mesh(
                        f"mesh delta gap {self._delta_seq} -> {seq}"
                    )
                    return
            self._delta_seq = seq
            try:
                self.core.on_mesh_delta(delta_from_msg(msg))
            except ValueError as exc:  # a delta that does not fit: started too late
                self.core.fallback_to_dsg_mesh(str(exc))

        self.node.create_subscription(MeshDelta, delta_topic, on_delta, latched)
        self.node.create_subscription(
            PoseGraph,
            inc_topic,
            lambda m: self.core.on_mesh_graph_incremental(nodes_from_pose_graph(m)),
            latched,
        )
        self.node.create_subscription(
            PoseGraph,
            graph_topic,
            lambda m: self.core.on_mesh_graph(nodes_from_pose_graph(m)),
            QoSProfile(depth=1),
        )
        self.log.info(f"mesh: odometric ({delta_topic}, {inc_topic}, {graph_topic})")

    def _on_dsg(self, msg) -> None:
        self._dsg_frame = msg.header.frame_id or self._dsg_frame
        with self._pending_cv:
            if self._pending is not None:
                self.graphs_dropped += 1
            self._pending = bytes(msg.layer_contents)
            self.graphs_received += 1
            self._pending_cv.notify()

    def _decode_loop(self) -> None:
        import spark_dsg as dsg

        while self._running:
            with self._pending_cv:
                while self._pending is None and self._running:
                    self._pending_cv.wait(0.5)
                data, self._pending = self._pending, None
            if data is None:
                continue
            diag = self.core.dsg_diag
            diag.update(received=self.graphs_received, dropped=self.graphs_dropped)
            try:
                inputs, info = decode_dsg(dsg, data)
            except Exception as exc:
                diag["failed"] = diag.get("failed", 0) + 1
                diag["last_error"] = f"{type(exc).__name__}: {exc}"
                self.log.error(f"failed to decode DSG ({len(data)} B): {exc}")
                continue
            prev = diag.get("nodes")
            self.core.on_records(*inputs)
            self.graphs_processed += 1
            diag.update(info, processed=self.graphs_processed)
            if info.get("warning") and info["warning"] != diag.get("_warned"):
                diag["_warned"] = info["warning"]
                self.log.error(info["warning"])
            if prev != info["nodes"] and (not prev or not info["nodes"]):
                self.log.info(
                    f"DSG: {info['nodes']} nodes, {info['raw_nodes']} in the graph, "
                    f"{info['bytes']} B, mesh faces {info['mesh_faces']}"
                )

    def _on_grid(self, msg) -> None:
        from fleetlink.media import yaw_from_quat

        o = msg.info.origin
        q = o.orientation
        self.core.on_grid(
            msg.header.frame_id,
            msg.info.resolution,
            [
                o.position.x,
                o.position.y,
                o.position.z,
                yaw_from_quat(q.w, q.x, q.y, q.z),
            ],
            msg.info.width,
            msg.info.height,
            np.asarray(msg.data, dtype=np.int8).tobytes(),
            stamp_s(msg.header),
        )

    def _map_frame(self) -> str:
        return self.map_frame or self._dsg_frame or self.fallback_map_frame

    def _ros_frame(self, frame: str) -> str:
        """Spot SDK frame names (the executor publishes paths in ``vision``) →
        their ROS TF names: Spot's vision/odom frames are ``<robot>/odom`` here."""
        robot = self.core.robot
        return {
            "vision": f"{robot}/odom",
            "odom": f"{robot}/odom",
            "body": f"{robot}/body",
        }.get(frame, frame)

    def _to_map(self, pts: np.ndarray, frame: str):
        """Express points in the scene-graph frame when TF allows (else unchanged)."""
        return self._to_frame(pts, frame, self._map_frame())

    def _to_frame(self, pts: np.ndarray, frame: str, target: str):
        """Express points in ``target`` when TF allows (else unchanged)."""
        frame = self._ros_frame(frame)
        if not frame or frame == target or self._tf_buffer is None or not len(pts):
            return pts, frame or target
        from rclpy.time import Time

        try:
            tf = self._tf_buffer.lookup_transform(target, frame, Time())
        except Exception:
            return pts, frame
        return transform_points(pts, tf), target

    def _on_path(self, which: str, msg) -> None:
        pts, frame = self._to_map(path_points(msg), msg.header.frame_id)
        self.core.on_path(which, pts, frame, stamp_s(msg.header))

    def _on_target(self, msg) -> None:
        target, frame = marker_target(msg)
        if target is not None:
            pts, frame = self._to_map(np.array([target], float), frame)
            target = [float(v) for v in pts[0]]
        self.core.on_path("target", target, frame)

    def _on_image(self, name: str, msg) -> None:
        self.core.on_camera_frame(
            name,
            msg.encoding,
            msg.width,
            msg.height,
            msg.step,
            msg.data,  # converted lazily, only when a frame is requested
            stamp_s(msg.header),
        )

    def _on_compressed(self, name: str, msg) -> None:
        from fleetlink.media import compressed_encoding

        # stored as JPEG/PNG bytes, decoded only when a frame is requested
        self.core.on_camera_frame(
            name,
            compressed_encoding(msg.format),
            0,
            0,
            0,
            msg.data,
            stamp_s(msg.header),
        )

    def _on_health(self, msg) -> None:
        if self.health_monitor and msg.monitor_name != self.health_monitor:
            return
        self.core.on_health(status_table_to_dict(msg))

    def _update_pose(self) -> None:
        if self._tf_buffer is None:
            return
        from rclpy.time import Time

        parents = [self.map_frame or self._dsg_frame, self.fallback_map_frame, "map"]
        for parent in dict.fromkeys(f for f in parents if f):
            try:
                tf = self._tf_buffer.lookup_transform(parent, self.body_frame, Time())
            except Exception:
                continue
            tr, q = tf.transform.translation, tf.transform.rotation
            st = tf.header.stamp
            self.core.on_pose(
                {
                    "x": tr.x,
                    "y": tr.y,
                    "z": tr.z,
                    "qw": q.w,
                    "qx": q.x,
                    "qy": q.y,
                    "qz": q.z,
                    "frame": parent,
                    "t": st.sec + st.nanosec * 1e-9,
                }
            )
            return

    def _update_frame_align(self) -> None:
        self.core.on_frame_align(
            lookup_frame_align(self._tf_buffer, self.global_frame, self._map_frame())
        )

    def _tick_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                self.core.tick(time.time())
            except Exception as exc:
                self.log.error(f"tick failed: {exc}")
            time.sleep(max(0.0, self._tick_period - (time.monotonic() - t0)))

    def _video_loop(self) -> None:
        while self._running:
            now = time.time()
            try:
                self.core.video.tick(now)
            except Exception as exc:
                self.log.error(f"video tick failed: {exc}")
            due = self.core.video.next_due()
            wait = 0.05 if due is None else min(0.05, max(0.002, due - time.time()))
            time.sleep(wait)

    def shutdown(self) -> None:
        self._running = False
        with self._pending_cv:
            self._pending_cv.notify_all()
        self._ticker.join(timeout=2.0)
        self._video_thread.join(timeout=2.0)
        self._decoder.join(timeout=2.0)
        if self.relay is not None:
            self.relay.shutdown()
        self.core.close()


def main(args=None) -> None:
    import rclpy

    from fleetlink.relay_ros import make_executor

    rclpy.init(args=args)
    agent = AgentNode()
    # multi-threaded: the relay's proxy service servers block (reentrant group)
    # while the base answers, at most relay_ros.MAX_PROXY_CALLS of
    # EXECUTOR_THREADS at once; the agent's own callbacks stay mutually exclusive
    executor = make_executor()
    executor.add_node(agent.node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        agent.shutdown()
        agent.transport.close()
        agent.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
