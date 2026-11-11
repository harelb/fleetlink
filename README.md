# fleetlink

fleetlink carries a robot fleet's live state to a base station over lossy, low-bandwidth radio
links, on zenoh, with ROS 2 at both ends. On each robot an **agent** turns ROS topics (Hydra 3D
scene graphs and meshes, pose and TF, executor guards and results, health, occupancy grid,
paths, cameras and the image store) into compact messages in seven QoS classes, each with a
byte budget and its own repair rule, and serves camera frames and live video on demand. On the
base a **receiver** mirrors every robot and repairs what got lost, without the transport ever
blocking on a congested radio. A generic **ROS relay** (`fleetlink-agent` on the robot,
`fleetlink-bridge` on the base) carries a configured list of ROS topics, transforms and
services across, so each machine's ROS graph can stay on that machine.

fleetlink was written for [Awesome-DCIST-T4](https://github.com/MIT-SPARK/Awesome-DCIST-T4)
(ADT4), whose web console embeds the receiver; nothing in it depends on ADT4 at runtime. The
wire contract is documented in ADT4's console API (`adt4_console/docs/API.md` §8, §11-§13).

* [Install](#install), [Run](#run), [Environment variables](#environment-variables), [Tests](#tests)
* [Design](#design) and [stream classes](#stream-classes) (C0-C6)
* [Generic ROS relay](#generic-ros-relay-fleetlink-mode), incl. large messages relayed in parts
* [Router modes](#router-modes): `router` and `fleetlink` (dedicated base router)
* [Mesh level of detail](#mesh-level-of-detail), [mesh deformation](docs/mesh_deformation.md)

## Install

fleetlink is a ROS 2 (Jazzy) `ament_python` package. In a colcon workspace:

```bash
git clone https://github.com/harelb/fleetlink src/fleetlink
rosdep install --from-paths src/fleetlink --ignore-src -y   # ROS and system deps (package.xml)
colcon build --packages-select fleetlink
```

The zenoh transport needs **`eclipse-zenoh==1.8.0`** (the zenoh of `rmw_zenoh` 0.2.x) in the
Python that runs the agent and bridge; it has no rosdep key, so install it with pip, typically
into a virtualenv:

```bash
pip install 'eclipse-zenoh==1.8.0' numpy msgpack zstandard PyYAML
pip install opencv-python-headless Pillow scipy   # optional, see below
export FLEETLINK_PYTHON=/path/to/venv/bin/python3  # the entry points re-execute there
```

Runtime dependencies:

| | needed for |
|---|---|
| `rclpy`, `std_msgs`, `tf2_ros`, `tf2_msgs`, `ament_index_python` | agent and bridge |
| `hydra_msgs`, `spark_dsg` | agent: the scene-graph input and its decoding |
| `nav_msgs`, `sensor_msgs`, `visualization_msgs` | agent: grid, paths, markers, cameras (skipped with a warning when missing) |
| `rmw_zenoh_cpp` | `fleetlink-router` (`rmw_zenohd`) |
| numpy, msgpack, zstandard, PyYAML | everything (the wire codec) |
| eclipse-zenoh 1.8.0 | the zenoh transport (agent, bridge, receiver) |
| *optional:* OpenCV or Pillow | camera frames (C4) and the image-store codec |
| *optional:* scipy | `mesh_source: odometric` (base-side mesh deformation) |
| *optional:* `ffmpeg` with libx264 / libvpx | live video (C5; JPEG frames otherwise) |
| *optional:* `hydra_python.image_store` (Hydra) | the image-store mirror (C6) and `fleetlink-image-pack` |
| *optional:* `robot_executor_msgs`, `ros_system_monitor_msgs`, `roman_msgs`, `kimera_pgmo_msgs`, `pose_graph_tools_msgs`, `phx_nav_msgs`, `clearpath_platform_msgs` | executor guards/results, health, relocalization evidence, odometric mesh, phoenix platform |

Optional imports are lazy: a missing one turns its feature off with a log line. The relay
imports each relayed message type from its package and skips types that are not installed.

## Run

Installed executables (`lib/fleetlink/`, so `ros2 run fleetlink <name>` works):

| executable | where | what |
|---|---|---|
| `fleetlink-agent` | robot | the agent (+ the ROS relay with `relay_config`) |
| `fleetlink-bridge` | base | the ROS relay's base side, one process for all robots |
| `fleetlink-router [PORT]` | base | the dedicated fleetlink zenoh router (fleetlink mode) |
| `fleetlink-image-pack` | any | pack / unpack a run's image store for a map transfer |

`config/` (`fleetlink.yaml`, `fleetlink_router.json5`, `relay.example.yaml`) is installed to
`share/fleetlink/config/`. `fleetlink-agent --help` and `fleetlink-bridge --help` list the main
parameters.

```bash
# robot (router mode: the local ROS router)
ros2 run fleetlink fleetlink-agent --ros-args -r __ns:=/robot1 -p robot_name:=robot1 \
    -p zenoh_connect:=tcp/127.0.0.1:7447

# fleetlink mode: base
ros2 run fleetlink fleetlink-router 7450
ros2 run fleetlink fleetlink-bridge --ros-args -p zenoh_connect:=tcp/127.0.0.1:7450 \
    -p topology:=topology.yaml -p relay_config:=relay.yaml -p base_name:=base1
# fleetlink mode: robot, over the radio (one endpoint per network, in order)
ros2 run fleetlink fleetlink-agent --ros-args -r __ns:=/robot1 -p robot_name:=robot1 \
    -p zenoh_connect:=tcp/192.0.2.10:7450,tcp/198.51.100.10:7450 \
    -p topology:=topology.yaml -p relay_config:=relay.yaml
```

On the base, a `fleetlink.receiver.Receiver` over a `fleetlink.transport.ZenohTransport`
(connected to the same router) mirrors the robots; the ADT4 console is one such consumer.

## Environment variables

| variable | read by | meaning |
|---|---|---|
| `FLEETLINK_PYTHON` | agent, bridge entry points | interpreter to re-execute in (the one with eclipse-zenoh) |
| `ADT4_ENV` | agent, bridge entry points | ADT4 compatibility: without `FLEETLINK_PYTHON`, re-execute in `$ADT4_ENV/spark_env/bin/python3` when it exists |
| `_FLEETLINK_REEXEC` | entry points | internal guard: set after the re-exec, so it happens once |
| `FLEETLINK_FFMPEG` | agent (video) | ffmpeg binary when the `ffmpeg` parameter is unset (else `ffmpeg` on `PATH`, `/usr/bin/ffmpeg`) |
| `FLEETLINK_ROUTER_CONFIG` | `fleetlink-router` | router json5 (default: the installed `fleetlink_router.json5`) |
| `ZENOH_CONFIG_OVERRIDE` | `fleetlink-router` | kept, with the port override appended |
| `ROS_DISTRO` | `fleetlink-router` | the ROS setup sourced when `ros2` is not on `PATH` (default `jazzy`) |
| `ADT4_TOPOLOGY` | agent, bridge | topology file when the `topology` parameter is unset |
| `ADT4_ROBOT_NAME` | bridge | base name when the `base_name` parameter is unset |
| `ADT4_OUTPUT_DIR` | agent | `store_path` default: `$ADT4_OUTPUT_DIR/image_store` |
| `ADT4_FLEETLINK_MESH_SOURCE` | agent | `mesh_source` default (`dsg` / `odometric`) |
| `FLEETLINK_TEST_PORT`, `FLEETLINK_ROUTER_TEST_PORT` | tests | throwaway router ports (17448, 17460) |
| `FLEETLINK_HARNESS_PORT`, `FLEETLINK_HARNESS_DOMAIN`, `FLEETLINK_HARNESS_KEEP_LOGS`, `FLEETLINK_HARNESS_RUST_LOG`, `FLEETLINK_STALL_S`, `FLEETLINK_RESUME_S`, `FLEETLINK_THROTTLE_BPS` | tests | fault-harness ports, domains, logs and fault sizes |
| `FLEETLINK_PENN_MAP`, `FLEETLINK_REAL_DSG`, `FLEETLINK_LC_RUN`, `PGMO_DEFORM_HARNESS`, `AGENTIC_NAV_SRC` | tests | real data for the data-driven tests (skipped when unset) |

fleetlink does not read `ADT4_FLEET_LINK`: the mode is whatever the launcher starts and
whatever `zenoh_connect` points at.

## Design

```
 robot                                            base station
 ┌───────────────────────────────┐   zenoh       ┌─────────────────────────────┐
 │ AgentNode (rclpy)             │   adt4/<r>/…  │ Receiver                    │
 │  DsgUpdate, TF, guards,       │──────────────▶│  GraphState mirror, tiles,  │
 │  health, results, grid,       │◀──────────────│  state, grid, paths, link   │
 │  paths, images                │  ctl/req (C0) │  stats; resync / tile /     │
 │      │                        │               │  resend requests with retry │
 │ AgentCore (pure, no ROS)      │  C4 query ⇄   │  camera queries             │
 │  diff → Scheduler → Transport │   reply       └─────────────────────────────┘
 └───────────────────────────────┘
```

* **`codec.py`.** The shared wire format: msgpack + zstd envelopes
  `{v, robot, stream, seq, epoch, t, p}`, compact columnar node records, graph deltas, and mesh
  tiles (faces bucketed into 8 m cubes, vertices quantised to 1 cm int16). Consumers import it
  (`from fleetlink import codec`); the ADT4 web console's browser decoder mirrors it.
* **`media.py`.** Occupancy-grid, path and camera-frame payloads: zlib grids, ≤150-point paths,
  and frame conversion to JPEG (OpenCV) or PNG (numpy + zlib).
* **`video.py`.** Live camera video (C5). `VideoStreams` runs one encoder per requested camera:
  the system `ffmpeg` as a subprocess (libx264 Annex-B with an AUD per access unit, or libvpx
  in IVF), raw RGB frames on stdin at the requested fps. `AnnexBSplitter` / `IvfParser` cut the
  output into frames and flag key frames. Without a working ffmpeg the stream falls back to one
  JPEG per frame. `C5Bucket` is the class's token bucket: delta frames need budget, key frames
  may borrow up to one burst, and after a drop a stream skips deltas until its next key frame.
* **`agent.py`.** `AgentCore` keeps "what the base has" (the sent `GraphState`, the sent tile
  manifest, and the sent grid/paths hashes) and on every `tick()` queues only the difference.
  Graph deltas are split to fit the C2 budget, closest to the robot first. Mesh tiles go only
  when their hash changes, and a periodic manifest lets the base detect lost tiles. Payloads are
  built at send time, so dropping or replacing queued items never creates sequence gaps.
  `AgentNode` is the rclpy wrapper: it decodes DSGs on a worker thread (newest only), ticks on
  its own thread, looks up the pose in the DSG frame, and transforms paths into that frame.
  At `frame_align_hz` it also looks up `global_frame` (default `map`) <- the DSG frame (e.g.
  `map -> robot3/odom`, published by hydra_multi / ROMAN relocalization) and sends it in the
  C1 state as `frame_align {parent, child, T: [x, y, z, qw, qx, qy, qz], stamp, ok, reason, age_s}`
  (`ok: false`, `T: null` and a `reason` when the lookup fails, i.e. not relocalized; `stamp` 0 =
  static TF). Everything else stays in its native frame; the console applies the transform.
* **`scheduler.py`.** A token bucket per class, drained in strict priority C0 > C1 > C2 > C3.
  Each bucket allows a `burst_s` burst and goes into debt by at most one message, so long-run
  throughput equals the budget. An optional `total_bps` caps all classes together. *Auto* mode
  runs AIMD on the C2/C3 budgets from the receiver's `fb` feedback: multiplicative decrease
  (x0.7, at most every 2 s) when a class's loss (seq gaps) exceeds 5 %, its goodput falls below
  80 % of what was put, or the RTT inflates; additive increase (+10 % of `budget_max`) while the
  class is backlogged and the link keeps up.
* **`receiver.py`.** One mirror per robot. A new `epoch` (an agent restart) clears the mirror and
  requests a resync. C2 is applied strictly in seq order: after a gap the later messages are
  held back and the missing range is asked for with `resend {stream: "dsg", from_seq, to_seq}`
  (served byte-identical from the agent's ring of the last `c2_ring` messages). A lost *last*
  message is found through `seqs.dsg` in the C1 state. After 3 unanswered attempts (2 s or 3 RTT
  each), or when the agent acks "range gone", the receiver applies what it holds and falls back
  to a resync, rate-limited to one per 5 s (deferred, not dropped, when rate-limited). A manifest
  mismatch triggers a `tiles` re-request. A grid/paths hash mismatch that lasts more than 10 s
  triggers a `resend {streams}`. C0 requests are retried until acked; they are queued in an
  outbox and put only after the receiver's lock is released. Pings every second measure RTT and
  clock offset; the `fb` op every second carries goodput, loss (seq gaps per class, also in
  `link()` as `loss` / `lost`) and RTT to the agent's auto mode.
* **`transport.py`.** `ZenohTransport` maps the classes to zenoh priority and reliability; every
  class is congestion-control **DROP** (a must-deliver message on a congested radio stalls the
  whole zenoh transport and then closes it, eclipse-zenoh/zenoh#1876), and reliability is the application's
  job (C0 retry, C2 `resend`/resync, C3 manifest). It is a zenoh *client* (a router does not
  forward between peers) that never fails for want of a router: zenoh 1.8 cannot open a client
  session without one, so a failed open is retried on a background thread (puts dropped and
  counted meanwhile, subscriptions declared on open), and an open session reconnects by itself
  after a router restart. `connect` is one endpoint, a list or a comma-separated list;
  `zenoh_overrides` sets any other config path. The config keys it sets (`ZENOH_DEFAULTS`,
  checked against eclipse-zenoh 1.8.0): `connect/timeout_ms: 0`, `connect/exit_on_failure: false`,
  `connect/retry {period_init_ms: 500, period_max_ms: 1000, period_increase_factor: 2}`,
  `transport/unicast/open_timeout: 3000`,
  `transport/link/tx/queue/congestion_control/drop/wait_before_drop: 1000` (µs),
  `.../drop/max_wait_before_drop_fragments: 50000`, `scouting/multicast/enabled: false`,
  `transport/link/tcp/so_sndbuf: 16384`, `transport/link/tx/batch_size: 8192` and
  `transport/link/tx/queue/size/{data_high,data,data_low}: 16`.
  `so_sndbuf` and `batch_size` keep a congested radio's backlog in zenoh's priority queues
  instead of the kernel's TCP send buffer (autotuned to MBs, where C0 waits behind C3:
  head-of-line), and `period_max_ms: 1000` brings C1 back within ~2 s of a long dropout
  ending; both were measured with the fleet harness (see *Tests*).
  The queue sizes: zenoh sends a message larger than one batch as 8 kB fragments, and a DROP
  message whose next fragment finds no free batch in time (zenoh grows that wait 1, 2, 4 …
  ms per fragment, capped at `max_wait_before_drop_fragments` in all) is dropped whole.
  With zenoh's default of 2 batches per priority, every message over ~16 kB (C2 deltas and
  snapshot chunks, C3 tiles and grids, C4 JPEG replies, C5 key frames) was dropped on a busy
  link, and so was its resend. 16 batches (zenoh's maximum) hold 128 kB per data priority,
  so such a message is queued whole; C0/C1 have their own, strictly higher, queues.
  Measured on a 30 kB/s radio saturated by 64 kB grids (fleet harness): every grid and a
  28 kB camera reply (in 1.0 s) arrive, plan→result 1.0-1.6 s, the relayed service 0.8 s.
  Raising `max_wait_before_drop_fragments` does not help: a put blocks the agent's tick
  thread for that long (C1 gaps over 3 s at 2 s) and a 200 kB grid still does not fit.
  The limit that remains: one message over ~128 kB cannot cross a slow radio (30 kB/s) at
  all, and bulk budgets above the link rate keep the data queues full (use `auto`).
  The fleetlink router's json5 sets the same queue sizes, since it re-fragments what it
  forwards (to the base consumers, and base→robot over the radio).
  Publishers are cached per key and class, LRU-capped at 1024 (`undeclare(key)` frees one
  early). A publisher evicted while a put on another thread uses it is undeclared when that
  put returns. `can_send` is only a local in-flight counter (a put of that class still running on
  another thread); the real congestion signal is the receiver's `fb`.
  `LinkEmulator` is two `MemoryTransport` endpoints joined by a simulated radio: bandwidth,
  latency, per-fragment loss, bounded queues and fragment-level strict priority, driven by a
  `VirtualClock` and a seeded RNG, so tests and benchmarks are fully deterministic. Options:
  `block_classes` (model zenoh BLOCK; default none), `backpressure=False` (`can_send` cannot see
  the queue, like zenoh), `tcp_hol=True` (one byte-ordered FIFO for all classes, like a TCP send
  buffer: a big message delays everything behind it) and `close_transport(duration_s)` (drop
  everything queued and in flight, refuse puts for that long, then reopen). Both transports
  implement pub/sub and request/reply (`declare_queryable` / `query`).

## Stream classes

| class | key(s) under `adt4/<robot>/` | zenoh priority, reliability | congestion control | default budget | contents | delivery guarantee |
|---|---|---|---|---|---|---|
| C0 control | `ctl/req` (base→robot), `ctl/ack` | INTERACTIVE_HIGH, express, reliable | DROP | unlimited | `resync`, `budget {cls: bps}`, `auto`, `ping`, `fb`, `tiles {keys}`, `mesh_lod {level, tiles?}`, `resend {streams}` / `resend {stream: "dsg", from_seq, to_seq}`, `video`, `video_keyframe` | retried every 1 s until acked (`fb`, `ping` fire-and-forget) |
| C1 state | `state`, `paths` | INTERACTIVE_LOW | DROP | 5 kB/s | pose, guards, health, results, stats, `cameras`, `hashes`, `seqs`, `frame_align` at 4 Hz; executor paths on change | latest value; paths re-sent on hash mismatch |
| C2 graph | `dsg/delta` | DATA_HIGH, reliable | DROP | 50 kB/s | snapshot begin/chunk/end, deltas (1 Hz tick) | in order; gap → `resend` from ring → resync |
| C3 mesh | `mesh/tile/<key>`, `mesh/manifest`, `grid/occupancy` | DATA_LOW, best effort | DROP | 100 kB/s | mesh tiles, manifest every 5 s, occupancy grid (checked every 5 s, sent on change) | manifest → `tiles`; grid hash → `resend` |
| C4 media | queryable `media/camera/<name>` | DATA (request/reply) | DROP | 200 kB/s | camera frames on request (`?w=640;q=70`) | none (the caller retries) |
| C5 video | `video/<name>` | DATA_LOW, best effort | DROP | 400 kB/s | live h264 / vp8 / jpeg frames while a `video` request is kept alive | none; a gap requests a key frame |
| C6 store | `store/frame` | BACKGROUND, reliable | DROP | 150 kB/s | image-store frames (`{id, frame_id, prev, blob}`) of the live store mirror | `prev` chain; gap or lost tail → C0 `store {id, after: watermark}` rewinds the agent |

Budgets are **bytes per second**. On the emulated link the transmit order is
C0 > C1 > C2 > C4 > C3 > C5 > C6: an operator waiting for a frame outranks bulk mesh, but not the
graph; the image-store mirror only uses what is left.

### Live image-store mirror (C6)

`store_mirror.py`. The agent tails the robot's image store read-only (`store_path`, default
`$ADT4_OUTPUT_DIR/image_store`, built from Hydra's keyframe zips / track logs / event logs by
Hydra's image-store tailer) and, **only while sending is on**, sends each new settled frame
(`store_frames`: `keyframes` = the agent nodes' own keyframes, i.e. the KEYFRAMEs anchored with an
identity `anchor_T_body` (+ kept object observation frames, with their observations and node
binding) | `all` = every keyframe)
encoded with `fleetlink.image_transfer.encode_frame` (`store_compression`: `none` | `lossless` |
`fast`) within the C6 budget. The base's `StoreMirror` writes them into
`<live_store_root>/<robot>/<store_id>/image_store`, a valid store that
`hydra_python.image_store` (and SAM3 search on top of it) can read. The base tells the agent its watermark (C0 `store {id, after}`) on first contact,
after an agent restart and after a gap, so nothing is lost or duplicated across restarts; a new
run or Hydra session (`indexer.tail.session`) starts a new mirror directory. Protocol details: ADT4 console API.md §13.
On a Penn-bag run (`fast`): 64 kB per keyframe, 115 keyframes mirrored at ~61 kB/s.

**Needs the store's tail (live) mode.** The tailer is Hydra PR #221, which is not on Hydra
`develop`. Without `hydra_python.image_store.indexer.tail_state` the agent logs
`store mirror disabled: ...` once and runs without the mirror; everything else (prior-map
pack/unpack included) works on `develop`.

**Off by default.** The mirror is what fills the radio (5 robots need ~10 Mbit/s with it on), so
the agent starts with sending off (`store_send: false`). The store is still tailed and armed, so
the state shows the backlog. The base turns it on per robot with C0 `store_send {enable}` (the ADT4 console's Live link
panel has a switch). The receiver remembers it and asks again when the robot's state
disagrees, e.g. after an agent restart. Turning it off drops queued C6 frames and steps the cursor back
to the last frame sent, so turning it on again continues the chain with nothing lost.

### Image-store codec and prior-map pack / unpack

`fleetlink/image_transfer.py` encodes one store frame
(row, calibration, RGB, depth) as a self-describing blob: `b"HISF"`, version byte, JSON header,
payloads. It needs `zstandard` (`lossless` / `fast`) and cv2 or Pillow; a missing one raises
`image_transfer.MissingDependency`. Hydra's `hydra_python.image_store` stays the store reader and writer.

| compression | RGB | depth |
|---|---|---|
| `none` | stored bytes | stored PNG bytes |
| `lossless` | stored bytes | uint16 row-delta + zstd-3 (bit-exact) |
| `fast` (default) | JPEG q75 (~40% smaller) | half resolution (nearest, exact `fx/2 ...` intrinsics) + row-delta + zstd-3 |

Writing a decoded frame (`write_frame` for the mirror, `write_decoded_images`) restores depth to
full resolution, keeps safe source-relative paths, and otherwise names files like Hydra's C++ writer
(`frames/r<robot>/<ts>[_s<seq>]_{rgb,depth}.*`). Headers come from another machine, so ids and
timestamps must be integers and every path must resolve inside the destination store
(`image_transfer.UnsafeFrame` otherwise).

`fleetlink/image_pack.py` packs a run's store into one tar
(optionally zstd-wrapped): `manifest.json`, `frames/<id>.hisf`, kept mask PNGs, then a
`store.sqlite` snapshot with every row (paths of frames not shipped are NULL, as after `gc`).
`unpack` builds `DEST/image_store.partial`, runs `hydra_python.image_store.check_store` and only then
renames it to `DEST/image_store`. A map transfer streams one into the other:

```
python -m fleetlink.image_pack pack RUN_DIR OUT.tar[.zst]|- [--frames keyframes|all]
       [--compression none|lossless|fast] [--zstd-level N] [--no-masks] [--quiet]
python -m fleetlink.image_pack unpack ARCHIVE|- DEST_RUN_DIR [--overwrite] [--no-check] [--dsg dsg.json]
python -m hydra_python.image_store check DEST_RUN_DIR   # the same checks, standalone
```

(`fleetlink-image-pack` is the installed command.) Python: `pack_store(run, out_path_or_file, ...)`,
`unpack_store(path_or_stream, dest, ...)`. A corrupt or truncated archive is reported as
`unpack failed: ...` (exit 1).

### Lazy cameras

Only `eager_cameras` (default `[rgb, overlay]`) are subscribed all the time. Other `cameras` entries
(e.g. the six Spot cameras, behind lazy image_transport republishers) are subscribed while a C4
query or C5 video request wanted them within `camera_idle_s` (15 s), then released.

### Live video (C5)

The base sends the C0 op `video {camera, enable, fps, kbps, width, codec}` and repeats it at
least every 6 s (`video_keepalive_s`); the ADT4 console does that every 2 s while a browser watches.
fps 0.5–15 (default 5), kbps 50–2000 kilobits/s (default 300), width default 640 (never upscaled,
aspect kept, even), codec `h264` (default), `vp8` or `jpeg`. Changed parameters restart the
encoder, which starts with a key frame. Key frames come every 2 s (`-g 2*fps`). Frames are sampled
from the newest image the agent keeps per camera (the same one C4 uses), on a dedicated thread so
15 fps is not quantised to the 20 Hz tick. Each frame is one zenoh message
`{camera, codec, seq, t, stamp, key, width, height, fps, data}`; frames are sent as they leave the
encoder, never queued. The C1 state lists active streams under `video`. The encoders, on synthetic
640x360 content for 10 s (`tests/test_video.py`):

| target | h264 (x264 ultrafast, zerolatency) | vp8 (libvpx realtime, cpu-used 8) |
|---|---|---|
| 150 kbit/s, 5 fps | 152 kbit/s | 151 kbit/s |
| 300 kbit/s, 5 fps | 305 kbit/s | 303 kbit/s |
| 800 kbit/s, 10 fps | 834 kbit/s | 816 kbit/s |

Encoder latency is about 25 ms for h264 (the splitter emits an access unit when x264 has been quiet
for 20 ms instead of waiting for the next AUD) and about 10 ms for vp8.

## Parameters

The built-in defaults (`agent.DEFAULT_CONFIG`) are listed in `config/fleetlink.yaml` (installed
to `share/fleetlink/config/`). Pass a file overriding any of them with `-p config:=FILE`.

| key | default | |
|---|---|---|
| `budgets.{C1,C2,C3,C4}` | 5000 / 50000 / 100000 / 200000 | bytes/s |
| `budget_min`, `budget_max` (C2, C3) | 5000 / 500000, 1000000 | auto-mode clamps |
| `total_bps` | null | optional cap shared by all classes |
| `auto` | false | AIMD on C2/C3 |
| `state_hz` | 4.0 | C1 rate (2 Hz is not enough for < 1 s staleness at 3 % loss) |
| `c2_hz` | 1.0 | graph tick |
| `mesh_period_s`, `manifest_period_s` | 5.0, 5.0 | |
| `mesh_decimate_cell` | 0.4 | vertex-clustering cell (m) of the `decimated` mesh LOD; divides `tile_size` |
| `tile_size` | 8.0 | m |
| `mesh_source` | `dsg` | `odometric`: tile Hydra's odometric mesh deltas and send the deformation graph's optimized poses; the base deforms the mesh, so a loop closure costs ~10 kB instead of re-sending most tiles ([`docs/mesh_deformation.md`](docs/mesh_deformation.md)). Default from `$ADT4_FLEETLINK_MESH_SOURCE`. Topics: `mesh_delta_topic`, `mesh_graph_incremental_topic`, `mesh_graph_topic` (Hydra's `hydra/frontend/full_mesh_update`, `hydra/frontend/mesh_graph_incremental`, `hydra/backend/mesh_graph`) |
| `max_msg_bytes` | 65536 | largest C2 message |
| `burst_s` | 1.0 | token-bucket depth, in seconds of budget |
| `grid_period_s` | 5.0 | occupancy grid check period |
| `paths_hz` | 2.0 | executor path check rate |
| `global_frame` (ROS param) | `map` | parent of `frame_align`; "" disables the lookup |
| `frame_align_hz` (ROS param) | 1.0 | `frame_align` lookup rate |
| `budgets.C5` | 400000 | live video, all streams together |
| `video_keepalive_s` | 6.0 | a stream stops without a `video` request this long |
| `ffmpeg` | null | ffmpeg binary (null: `$FLEETLINK_FFMPEG`, `ffmpeg` on `PATH`, `/usr/bin/ffmpeg`) |
| `c2_ring` | 256 | last C2 messages kept to serve `resend` (0: every C2 gap ends in a resync) |
| `budgets.C6` | 150000 | live image-store mirror |
| `store_frames`, `store_compression` | keyframes, fast | C6 frame selection and codec |
| `store_poll_s`, `store_queue_s` | 1.0, 2.0 | store check period; C6 queue depth in seconds of budget |
| `eager_cameras`, `camera_idle_s` | [rgb, overlay], 15.0 | always-subscribed cameras; idle time before a lazy one is released |

Receiver kwargs (base side, `Receiver(...)`): `resend_attempts` (3), `resend_timeout_s` (2.0,
at least 3 RTT), `hold_max` (1024 held C2 messages before giving up on a gap), `fb_period` (1.0 s),
plus `retry_s`, `max_retries`, `resync_min_interval` (5 s), `ping_period`, `resend_grace_s`.

ROS params of `fleetlink-agent` (topics are relative to the robot namespace unless absolute):
`robot_name` (default: the namespace), `config`, `zenoh_connect` (`tcp/127.0.0.1:7447`; one
endpoint or a comma-separated list, retried forever, so the agent starts fine before the router),
`c2_ring`,
`tick_hz` (20), `budget_C1`…`budget_C6`, `store_path` (`$ADT4_OUTPUT_DIR/image_store`, `""` off),
`store_frames`, `store_compression`, `eager_cameras`, `camera_idle_s`, `ffmpeg`, `auto`, `mesh_period_s`, `tile_size`, `grid_period_s`,
`dsg_topic` (`hydra/backend/dsg`), `guards_topic`, `result_topic`, `health_topic`,
`health_monitor_name`, `status_nickname` / `status_period_s` / `status_rtt_warn_s` (see
[Status monitor](#status-monitor-node_status)), `map_frame` (default: the DSG header frame), `body_frame` (`<robot>/body`),
`fallback_map_frame` (`<robot>/map`), `grid_topic` (`hydra/tsdf/occupancy`),
`smooth_path_topic` / `mlp_path_topic` / `mlp_target_topic` (`spot_executor_node/…_publisher`),
and `cameras` (list of `name=topic`, default
`[rgb=<robot>_zed/rgb/image_rect_color, overlay=semantic_overlay/image_raw]`).

The agent logs a warning at startup when the installed eclipse-zenoh is not 1.8.x (the zenoh
of rmw_zenoh 0.2.x). It also logs which `spark_dsg` it imported and reports DSG decoding in
`state.stats.dsg` (received / processed / failed, node counts, and a `warning` when a payload
that has nodes decodes to none, e.g. a stale spark_dsg shadowing the workspace build). Budgets
can also be changed live by the base (C0 `budget`).

## Status monitor (`~/node_status`)

The agent and the bridge report their health once per `status_period_s` (1.0; 0 = off) as a
`ros_system_monitor_msgs/NodeInfoMsg` on `~/node_status`, with the nicknames
`fleetlink_agent` and `fleetlink_bridge` (`status_nickname` overrides). Remap `~/node_status`
to your monitor's collector (in ADT4: `/<robot>/ros_system_monitor/node_diagnostic_collector`)
and track the nickname there. `ros_system_monitor_msgs` is optional: without it the node logs
one line and publishes nothing. The status is computed from what the link already tracks (the
relay heartbeat's liveness and RTT, the console's `ping` / `fb` requests, the zenoh session,
the mesh mode); nothing is added to the wire. The notes are one line, e.g.
`fleetlink | base up rtt 84ms | tcp/192.0.2.10:7450 (+1 failover) | mesh odometric`.

| node | NOMINAL | WARNING | ERROR |
|---|---|---|---|
| agent, fleetlink mode (`relay_config` set) | relay heartbeat from the base within 3 s, RTT under `status_rtt_warn_s` (1.5 s) | no base heartbeat yet, base link down (heartbeat older than 3 s), RTT high, `relay.yaml` differs from the base's, odometric mesh fell back to `dsg` | zenoh session not open |
| agent, router mode (no relay) | session open; the console's last request and RTT are a note only (nothing says a console should be there) | odometric mesh fell back to `dsg` | zenoh session not open |
| bridge | every robot it has heard from is still sending heartbeats (robots never heard from are listed as `waiting`) | a robot that was connected went silent, or its `relay.yaml` differs | zenoh session not open |
| bridge, `idle:=true` | always (`Idle: router mode, ...`): for launch files that start the bridge in every mode, so the monitor never sees it go missing | | |

The notes also carry the configured endpoint(s), the mesh source, the scheduler's queued bytes
when non-zero, and the reason when the image-store mirror is disabled (a note, not a warning).

## Generic ROS relay (fleetlink mode)

In fleetlink mode each robot's ROS graph stays on the robot, and the ROS topics, services and
transforms that must cross the radio are relayed by fleetlink. They are listed in a relay
table that the deployment provides (`relay_config`); [`config/relay.example.yaml`](config/relay.example.yaml)
documents the schema. Both ends load the same file (the heartbeat compares hashes). Relayed
message types that are not installed on a machine are skipped with a warning.

* **`relay.py`** (pure, no ROS). `RelayConfig` validates the table: it rejects unknown keys,
  modes, directions or classes, a topic or service configured in **both** directions (the loop
  prevention), and two same-direction entries whose platforms overlap. It templates `{robot}`
  (and `{base}` in a service's `base_name`) and filters entries by the robot's topology
  `platform_type`; a robot without one counts as `spot`. `RelayCore` is one end of one
  robot↔base link and runs the delivery modes:

  | mode | behavior |
  |---|---|
  | `reliable` | in order; queue cap 32 (oldest dropped); retried every 1 s, 30 tries; the receiver acks, dedupes and applies in `tseq` order |
  | `latest_reliable` | one pending slot: a newer message replaces the pending retry; the receiver applies only `tseq` above the last applied |
  | `latest` | rate-limited to `rate_hz` (the newest waiting value goes at the next slot), fire and forget |

  `ttl_s` is checked by the sender (it stops retrying) and by the receiver (it rejects with a
  negative ack), using the envelope `t` corrected by the clock offset from the relay's own
  heartbeats. Until that offset is measured (the first heartbeat round trip after link-up or
  a peer restart) the receiver uses the last heartbeat's one-way estimate, and before any
  heartbeat it does not check the TTL at all: the sender's TTL still bounds the retries, and
  a raw `t` would reject everything on machines whose clocks differ by more than `ttl_s`.
  Each end has a random epoch. When the peer **restarts**, only `replay_on_resync: true`
  entries re-send: their pending messages, re-addressed to the new epoch, or with nothing
  pending the last value of a `latest` / `latest_reliable` entry (state, e.g. `/tf_static`).
  A `reliable` entry (events, e.g. action results) does not re-send a message the old peer
  acked, since it was already published there. Everything else pending is dropped, and the
  new peer rejects retries still addressed to the old epoch, so a rebooted robot never re-runs
  a stale plan or pause. That `stale epoch` nack never cancels a pending message: it is how a
  sender may first learn of the restart, and by then the pending was dropped or re-addressed.
  A result applied by the old peer whose ack was lost is re-sent to the new one (at least
  once across a restart of the receiving process). Services use the same request/retry/dedupe
  pattern (`srv/req`, `srv/rep`, never zenoh `get`) and give up after `timeout_s`. A type-hash
  mismatch (ROS 2 `RIHS01`, from the package's generated `.json`) rejects and counts the message.
  The heartbeat carries the config hash, and a mismatch is logged and shown as
  `cfg_match: false`. Everything is congestion-control DROP. Relay messages do not go through
  the agent's scheduler; `rate_hz` is their only budget.

  **Large messages are relayed in parts.** In the acked modes a message over `chunk_bytes`
  (16 kB) goes as parts `{..., id, part, parts, data}`, at most `chunk_window` (6)
  unacknowledged at a time, so a few hundred kB never overflows zenoh's 128 kB per-priority
  queue (which drops a message that does not fit whole). The receiver acks each part
  (`pack`), reassembles the message and handles it as one (type, ttl, order, the usual
  `ack`). The sender resends only unacked parts: on a timer paced by the part acks (at least
  `retry_s`, doubled on a timeout without progress, at most 8 s), or at once when a later part
  is acked first. `latest` (fire and forget) is never split.
* **`relay_ros.py`.** `RobotRelay` runs inside `fleetlink-agent` on the agent's transport. It
  is enabled with `-p relay_config:=<relay.yaml>`, plus optional `relay_platform` and
  `topology`. Without `relay_config` the agent behaves as before. It
  subscribes the `up` topics with `raw=True` and the configured QoS, republishes `down` topics,
  and looks up each configured TF pair (`map → {robot}/base_link`, 5 Hz) in a tf2 buffer.
  `/tf_static` transforms under `frames_prefix` are sent as one message when they change and
  again to a restarted base; a frame that is the child of a composed pair is left out, so it
  never gets two parents. For `dir: up` services (the robot calls, the base answers) it serves
  proxy servers, and for `dir: down` services it calls the robot's real servers.
  `BaseBridge` is the `fleetlink-bridge` node, one process for every robot:

  ```bash
  ros2 run fleetlink fleetlink-bridge --ros-args -p zenoh_connect:=tcp/127.0.0.1:7450 \
      -p topology:=<topology.yaml> -p relay_config:=<relay.yaml> [-p base_name:=base1] [-p robots:=robot1,robot3]
  ```

  Robots come from the topology (`role: robot`) and from any relay heartbeat that arrives. On
  the base graph the bridge subscribes the `down` topics and republishes `up` topics under the
  same names. Transforms go to `/tf`; static transforms go to `/tf_static`, all robots in one
  latched message. The bridge serves proxies for `dir: down` services and calls the real base
  servers (`base_name`, e.g. `/{base}/update_holding_state`) for `dir: up` ones. It
  publishes `/fleetlink/link_status`: one `std_msgs/String` JSON per robot, at 1 Hz and right
  after a down message is acked. The JSON is `RelayCore.status()`: `connected`, `rtt`, `loss`,
  `last_seen`, `clock_offset`, `epoch`, `peer_epoch`, `cfg_match`, and per topic `sent`,
  `acked`, `pending` and `subs`, the subscribers on the far side when the message was applied.
  A base UI can take delivery acks and robot liveness from this topic, and `~/node_status`
  summarizes it for a status monitor ([Status monitor](#status-monitor-node_status); `-p
  idle:=true` runs status only, needing no topology or relay table). ROS subscriber
  counts mean nothing once the bridge subscribes everything. Both ends spin multi-threaded
  (`relay_ros.make_executor()`, 8 threads): proxy servers block in a reentrant group while
  the far side answers. At most 2 relayed calls per end wait at once (`MAX_PROXY_CALLS`);
  more, or a call while the link is down (`connected: false`), fail at once with the default
  response and a warning, so proxies never take the threads that DSG, TF and pose callbacks
  need. Service clients and publishers are created on the executor thread or before it spins,
  never on zenoh's callback thread.

## Router modes

A deployment chooses how the robots and the base talk; fleetlink works in both:

* **router**: the base's ROS `rmw_zenohd` links router-to-router to every robot, so all ROS
  graphs merge across the radio. Each agent connects to its local ROS router
  (`tcp/127.0.0.1:7447`) and the relay is off.
* **fleetlink**: only fleetlink crosses the radio, and every ROS graph stays on its machine.
  * The base runs a dedicated zenoh router, `fleetlink-router [PORT]`: `rmw_zenohd` with
    [`config/fleetlink_router.json5`](config/fleetlink_router.json5) (listen `tcp/[::]:7450`,
    connect to nothing, no scouting, gossip, adminspace or plugins, and 16-batch data queues,
    see `transport.py` above). It refuses 7447, the ROS router's port.
  * The base's `fleetlink-bridge` and receiver connect to it over loopback.
  * Each robot's agent connects to it as a zenoh client over the radio (`zenoh_connect`, one
    endpoint per network) and runs the ROS relay (`relay_config`).
  * The machines' own ROS routers must not dial each other (e.g.
    `ZENOH_CONFIG_OVERRIDE='connect/endpoints=[]'`).

**Topology.** The bridge (and, for the platform, the agent) read a topology YAML: every
`machines.<name>` with `role: robot` is a robot, its `platform_type` (`spot` when missing,
`husky` = `phoenix`) selects relay rows, and `fleet_defaults.fleetlink_base` names the base:

```yaml
machines:
  robot1: {role: robot, platform_type: spot}
  robot2: {role: robot, platform_type: husky}
  base1: {role: base_station}
fleet_defaults:
  fleetlink_base: base1
```

It is the `topology` parameter, else `$ADT4_TOPOLOGY`, else `config/topology.yaml` in the share
directory of an installed ADT4 `dcist_launch_system` (when it ships one). fleetlink reads no
addresses from it; endpoints are always passed in (`zenoh_connect`).

**Failover across networks.** Give the agent all of the base's fleetlink endpoints, in order of
preference. Verified on zenoh 1.8 (`tests/test_fleetlink_router.py`):

* A zenoh *client* opens **one** session, to the first endpoint that answers. When two
  endpoints reach the same router, there is one TCP link and each message arrives once.
* When that link drops, the client reconnects to the first endpoint that answers again.
* It does **not** fail back: once on the second network, it stays there until that link drops.
* A refused endpoint (router down) is skipped at once. A black-holed one (radio out of range)
  costs zenoh's `transport/unicast/open_timeout` before the next one is tried.

**In ADT4** (Awesome-DCIST-T4), `topology.yaml`'s `fleet_defaults.fleet_link: router | fleetlink`
picks the mode, `run-adt4` derives `ADT4_FLEET_LINK`, `ADT4_FLEETLINK_CONNECT` and
`ADT4_FLEETLINK_PORT` from it, and the generated `fleetlink` / `fleetlink_base` tmux windows start
the agent, the router and the bridge; ADT4 keeps its relay table in
`dcist_launch_system/fleetlink/relay.yaml`.

Not done: zenoh access control, so that a robot can only put `adt4/<self>/**`.

## Tests

```bash
pip install -e '.[test]'          # or: numpy msgpack zstandard PyYAML pytest
python -m pytest -q               # pure Python: what CI runs; ROS / zenoh tests skip
```

With ROS 2 Jazzy, `rmw_zenoh_cpp`, eclipse-zenoh 1.8.0 and the workspace's message packages
(`hydra_msgs`, `robot_executor_msgs`, `heracles_ros_interfaces`, `dcist_launch_system_msgs`,
...) installed, the integration tests run too:

```bash
source /opt/ros/jazzy/setup.bash && source <workspace>/install/setup.bash
python -m pytest -q               # with the interpreter that has eclipse-zenoh
python -m pytest -q -m "not integration"   # unit tests only
```

`pyproject.toml` puts the checkout and `tests/` on the path and disables ROS's
`launch_testing` pytest plugins (they end the session at the first module-level skip).
Every integration test uses throwaway routers on free local ports and private ROS domains and
never touches a router on 7447; each skips when ROS, zenoh or its port is unavailable.

* `test_codec`, `test_scheduler`, `test_transport`, `test_agent`, `test_receiver`, `test_media`,
  `test_video` are unit tests. `test_reliability` runs agent + receiver over the emulator with
  every class DROP: C2 gap repaired by `resend` without a resync, "range gone" escalating to a
  resync, a 5 s transport close, `tcp_hol` C1 staleness, and `fb`-driven AIMD converging to the
  link rate with a blind sender. `test_video` runs the real ffmpeg encoders (frame splitting,
  key-frame flags, bitrate within 30 % of target, decodable output) and skips them when ffmpeg
  lacks libx264 / libvpx. They use the `RecordingTransport` and the deterministic `LinkEmulator`
  (latency, bandwidth, loss, congestion, C4 query timeouts and budget).
* `test_zenoh_smoke` (graph, mesh, C0, camera query, grid, paths, live h264 video) and
  `test_ros_smoke` (a real `hydra_ros.DsgPublisher` plus OccupancyGrid/Path/MarkerArray/Image
  publishers into `AgentNode`) start a throwaway `rmw_zenohd` on `127.0.0.1:17448`
  (`FLEETLINK_TEST_PORT` moves it; the late-router tests use the next port) and private ROS
  domains (88, 89).
* `test_relay` exercises `RelayConfig` / `RelayCore` over the emulator: config rejection, the
  platform filter, reliable order under 30 % loss and the queue cap, latest_reliable
  supersede, the latest rate limit, TTL with clock skew before and after the offset is known,
  no replay after a robot restart, replay to a restarted base, type and config hash mismatch,
  service round trips and timeouts. `test_relay_chunks` covers messages relayed in parts.
  `test_relay_ros` runs a robot graph (domain 90) with `AgentNode` + `relay_config` and a base
  graph (domain 91) with `BaseBridge`, joined only through a throwaway router, with
  `tests/data/relay_test.yaml`.
* `test_fleetlink_router` starts `scripts/fleetlink-router` on ports 17460/17461
  (`FLEETLINK_ROUTER_TEST_PORT`) and pins the zenoh client failover behaviour.
* `test_fleet_harness` is the **multi-router fault harness**: fleetlink mode with every part a
  separate process (`tests/harness.py`): a base `rmw_zenohd`, the fleetlink router and two robot
  `rmw_zenohd` on ports 17470-17477 (`FLEETLINK_HARNESS_PORT`) and ROS domains 94-96
  (`FLEETLINK_HARNESS_DOMAIN`), a real agent per robot (relay on) and a real bridge, fake robot
  and base nodes (`tests/harness_nodes/node_host.py`), and a `FaultProxy` on each robot's link
  that can stall, throttle or reset it. It checks graph isolation and the per-robot relay round
  trips, a 30 kB/s link saturated by ~64 kB grids (every grid and a ~28 kB camera reply arrive,
  commands still take under 2 s), a stall (`FLEETLINK_STALL_S`, 20 s by default) with no
  transport close, C1 back within `FLEETLINK_RESUME_S` and the other robot unaffected, and a
  fleetlink router kill -9 mid-plan with no re-execution. About 50 s in all; teardown kills every
  process group and checks that none is left.
* Data-driven tests run only when pointed at real Hydra outputs: `test_integration_pennovation`
  (`FLEETLINK_PENN_MAP`, a `dsg_with_mesh.json`; streams it over an emulated 2 Mbit/s, 150 ms,
  3 % loss link in 20 reveal stages and asserts graph convergence ≤ 60 s, full mesh ≤ 600 s,
  C1 staleness < 1 s and budget compliance; deterministic), `test_dsg_decode`
  (`FLEETLINK_REAL_DSG`), `test_mesh_deform_e2e` and the Kimera-PGMO ground truth in
  `test_deform` (`FLEETLINK_LC_RUN`, `PGMO_DEFORM_HARNESS`), and the SAM3 keyframe-loader check in
  `test_store_mirror` (`AGENTIC_NAV_SRC`). The image-store tests need `hydra_python.image_store`.

## Mesh level of detail

A robot sends its mesh at one of two levels of detail (`codec.MESH_LODS`):

* `decimated`: vertex clustering on a global `mesh_decimate_cell` grid (0.4 m), every vertex of a cell
  replaced by the cell's mean vertex and colour, collapsed and repeated faces dropped
  (`codec.decimate_mesh`, numpy only). The grid divides the 8 m tile size, so neighbouring tiles share
  their border vertices exactly. Pennovation courtyard: 1.62 M -> 0.19 M faces, 13.4 -> 1.8 MB
  (zstd) for the whole mesh, 0.19 s + 0.08 s tiling on a laptop.
* `full`: Hydra's mesh as is.

The base asks with C0 `mesh_lod {level}` (whole mesh) or `mesh_lod {level, tiles: [keys]}` (just
those tiles, e.g. full detail where the operator looks). Each tile carries its `lod` and its hash
covers its content, so a change re-sends exactly the tiles that differ, closest to the robot first,
within the C3 budget; the manifest carries `lod {level, tiles, cell}`. The Receiver asks every robot
for `decimated` when its epoch starts (`Receiver(mesh_lod=...)`, None = never ask) and re-sends the
console's choice (`set_mesh_lod`) to a restarted robot. Compatibility: a robot sends full tiles until
a base asks (an old base never does); an old robot answers the unknown op with an error ack and keeps
sending full tiles (the link status `mesh.lod` is then null).

## License

Apache License 2.0, see [LICENSE](LICENSE). Copyright 2026 Harel Biggie.
