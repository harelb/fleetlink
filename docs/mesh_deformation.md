# Base-side mesh deformation (fleetlink `mesh_source: odometric`)

Validated offline and over the emulated link. No Hydra / Kimera-PGMO change is needed.

## Why

Default fleetlink (`mesh_source: dsg`) tiles the *deformed* mesh from Hydra's DSG and
re-sends every tile whose content changed. A loop closure moves most vertices, so it
re-sends most of the mesh. On a recorded Hydra run with 77 loop closures (641,205
vertices):

| | bytes on the wire |
|---|---|
| DSG mode, one loop closure | 77 of 100 tiles change: **8.04 MB** |
| odometric mode, the same loop closure | **15.0 kB** on C3 over 5 s (the poses message plus manifests) |

## How Hydra deforms the mesh

- The frontend publishes the **odometric** mesh as incremental deltas on
  `~/full_mesh_update` (`kimera_pgmo_msgs/MeshDelta`, frame odom, `QoS(100).transient_local()`,
  always). This is what the backend's `original_vertices_` is built from.
- The deformation graph's mesh nodes ("control points", one per ~`d_graph_resolution` = 2.5 m)
  appear on `~/mesh_graph_incremental` (frontend, transient local) and their optimized poses on
  the backend's `~/mesh_graph` (when subscribed). Both are `pose_graph_tools_msgs/PoseGraph`.
  The node stamp travels in `header.stamp`.
- The backend's `updateDsgMesh` calls `deformPoints(mesh, original_vertices_, 's', values,
  num_interp_pts=3, interp_horizon=15 s, start_idx=last_deformed_vertices_)`:
  - vertices are visited in index order with a sliding control-point window (stamps within
    ±15 s, at least k+1 points);
  - each vertex blends the corrections of its **k+1 = 4** nearest control points
    (`search(k+1)` → `nearestKSearch(k+2)`, the farthest sets `d_max`, weights `1 - d/d_max`);
  - rotations are averaged as raw quaternion coefficients, translations as `t_j - R_j c_j`.
- **Archived vertices are deformed only once.** `last_deformed_vertices_` becomes the archived
  count after each deformation, so later loop closures only move recently archived and active
  vertices. Hydra's own DSG mesh therefore keeps stale corrections in old areas. The base
  deforms every vertex with the latest poses (consistent with the optimized graph), so the two
  can differ there, by design.

## What fleetlink does in odometric mode

robot (`AgentNode`, `mesh_source: odometric`)
- subscribes the three topics above (QoS matched to Hydra's), rebuilds the odometric mesh
  from the deltas (`OdomMesh`: PGMO's archived / pending / active bands);
- tiles it (`tile_odom_mesh`): the normal tile format plus per-vertex stamps (u32 ms) and
  global vertex indices (u32 deltas). A tile that packs to more than 96 kB is split into half-size
  sub-tiles (`i_j_k@4`, ...), so no tile exceeds what zenoh carries whole on a busy link.
  Control points ride in small carrier tiles (`i_j_k~ctrl`). Tiles change only where the
  robot adds geometry;
- sends the optimized poses as `mesh/values` (C3, front of the queue) when they change by more
  than the wire quantization (1 mm, 1/32767), and every 30 s: int16 mm relative to the original
  control points + int16 quaternions, zstd. That is 10 kB for 3,266 control points, with
  ≤ 0.5 mm error;
- falls back to DSG mode (and reports `mesh.fallback` in its stats) on a delta sequence gap
  or a delta that does not fit (agent started > 100 deltas after Hydra). A Hydra restart
  (sequence back to 0) resets the odometric mesh.

base (`Receiver`)
- keeps the odometric tiles as received (`raw_tiles`, hash convergence unchanged);
- `tiles(robot)` hands the console ordinary tiles: all vertices gathered in global index
  order and deformed with the latest poses (`fleetlink.deform`, a numpy port of PGMO), then
  requantized at 1 cm. The result is cached per (tile hashes, values).

## Validation

- `tests/test_deform.py` covers:
  - the port against a direct reimplementation (windows, invalid points, out-of-order stamps);
  - a uniform correction giving an exact rigid motion;
  - pose packing;
  - MeshDelta bookkeeping;
  - odometric tiles plus the base view.
- **C++ ground truth:** a small harness links the installed `libkimera_pgmo.so` and calls
  `kimera_pgmo::deformation::deformPoints` (the call Hydra makes). On the 77-LC run (641,205
  vertices, 3,266 control points) the port agrees within 1 mm for 97.4 % of vertices, within
  1 cm for 99.956 %, within 5 cm for 100 % (max 25.9 mm). The residual comes from PGMO's
  octree, which counts and removes 1 m voxels rather than points. numpy takes 1.4 s against
  C++'s 2.9 s.
  ```
  g++ -O2 -std=c++17 pgmo_deform.cpp -o pgmo_deform -I$I/kimera_pgmo/include -I/usr/include/eigen3 \
    -I/usr/include/pcl-1.14 -I/opt/ros/jazzy/include -I/opt/ros/jazzy/include/gtsam/3rdparty/metis \
    -I$I/spark_dsg/include -L$I/kimera_pgmo/lib -L$I/kimera_rpgo/lib -L$I/config_utilities/lib \
    -L$I/spark_dsg/lib -lkimera_pgmo -lkimera_rpgo -lconfig_utilities \
    -L/opt/ros/jazzy/lib/x86_64-linux-gnu -lgtsam -lgtsam_unstable -ltbb
  PGMO_DEFORM_HARNESS=./pgmo_deform FLEETLINK_LC_RUN=<run>/hydra/backend \
    pytest tests/test_deform.py   # with LD_LIBRARY_PATH set
  ```
  (`I` is the colcon workspace's `install` directory; the harness source is
  `tools/pgmo_deform.cpp`.)
- `tests/test_mesh_deform_e2e.py` (`FLEETLINK_LC_RUN`) runs the real 77-LC mesh as staged odometric deltas over the
  emulated link (20 Mbit/s, 1 % loss). The tiles converge, the loop closure costs 15 kB, and the
  base's deformed vertices match a global deformation (99.9th percentile ≤ 2 cm).
- `tests/test_ros_odom_mesh.py` uses real `MeshDelta` / `PoseGraph` ROS messages, including
  transient-local delivery to a late agent, deformation after a `mesh_graph` update, and
  fallback on a sequence gap.

## Side finding: big tiles in DSG mode

On the 77-LC mesh, 35 of the 99 default 8 m tiles pack to more than 128 kB (max 384 kB). On
a busy link zenoh drops a message that large whole (16 × 8 kB send queue), and the emulator
drops anything over its 256 kB queue. In dense areas such a tile can fail to arrive until the
link is quiet. Odometric mode splits tiles at 96 kB; DSG mode does not yet (the LOD /
decimation work is the natural place to bound it).

## Not done / limits

- Not run with a live Hydra. Relocalize mode (`hydra_multi`) uses a different backend; only
  mapping mode (`hydra`) is covered.
- An agent started more than 100 mesh deltas after Hydra cannot rebuild the odometric mesh
  (Hydra keeps the last 100: transient-local depth and `mesh_delta_queue_size`). It falls back
  to DSG mode; querying `~/mesh_delta_query` for older deltas is not implemented.
- Base deformation is a full pass (1.4 s for 641 k vertices) on every new pose set, outside
  the receiver lock. It is fine at loop-closure rates but not meant for per-second updates.
- Deforming decimated (level-of-detail) tiles works the same, as long as each kept vertex
  carries its stamp and global index.

## How to try it

Odometric mode is chosen per robot; the base needs nothing.

1. Enable it on the robot's agent: `-p mesh_source:=odometric`, or set
   `ADT4_FLEETLINK_MESH_SOURCE=odometric` in the agent's environment (the parameter wins).
2. Check it took: the agent logs `mesh: odometric (hydra/frontend/full_mesh_update, ...)`;
   its stats (`/fleetlink/link_status`, or the receiver's robot state) show `mesh.source:
   odometric` and `mesh.fallback: null`. `ros2 topic info /<robot>/hydra/frontend/full_mesh_update`
   lists the agent as a subscriber.
3. Map normally. The base's live mesh appears as before (tile keys may carry `@4` /
   `@2` where the mesh is dense).
4. At a loop closure, watch the C3 bytes: the robot sends only `mesh/values` (~10 kB for
   ~3,000 control points); in `dsg` mode the same loop closure re-sends most tiles (MBs).
5. If `mesh.fallback` is set (e.g. "mesh delta gap"), the agent went back to DSG tiles;
   restart the agent together with Hydra to try again.

Offline checks:
```
python -m pytest -q tests/test_deform.py tests/test_mesh_deform_e2e.py
python -m pytest -q tests/test_ros_odom_mesh.py   # with ROS and the workspace sourced
```
