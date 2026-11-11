"""fleetlink.pointcloud_codec: lidar frames for the C3 ``lidar/points`` stream."""

import numpy as np
import pytest

from fleetlink import codec
from fleetlink import pointcloud_codec as pc


def room_scan(rows=128, cols=1024, seed=0):
    """A 128-beam spinning lidar in a 37 x 14 x 3.4 m room (every ray returns:
    a dense, worst-ish case), 1 cm range noise; plus a uniform intensity."""
    rng = np.random.default_rng(seed)
    az = np.linspace(-np.pi, np.pi, cols, endpoint=False)
    el = np.radians(np.linspace(-22.5, 22.5, rows))
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    t = np.full(len(d), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for ax, lo, hi in ((0, -12, 25), (1, -6, 8), (2, -0.8, 2.6)):
            for b in (lo, hi):
                tt = b / d[:, ax]
                tt[~(tt > 0)] = np.inf
                t = np.minimum(t, tt)
    p = d * t[:, None] + rng.normal(0, 0.01, (len(d), 3))
    return p.astype(np.float32), rng.uniform(0, 500, len(d)).astype(np.float32)


def test_round_trip_within_a_centimetre_without_voxels():
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-20, 20, (5000, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(-0.9, 3.9, 5000)
    w = pc.encode_cloud(xyz, voxel=0)
    # what the envelope carries survives msgpack + zstd
    w = codec.unpack(codec.pack(w))
    out, inten = pc.decode_cloud(w)
    assert inten is None and w["n"] == len(out)
    kept, _ = pc.crop(xyz)
    assert len(out) == len(kept)
    assert np.abs(out - kept).max() <= 0.005 + 1e-6  # half a cm per axis
    assert np.linalg.norm(out - kept, axis=1).max() <= 0.01


def test_xyz_int16_is_the_interleaved_contract_format():
    xyz = np.array([[1.0, -2.0, 0.5], [29.0, 0.01, -0.99], [-0.31, 0.0, 3.99]], np.float32)
    w = pc.encode_cloud(xyz, voxel=0)
    raw = pc.xyz_int16(w)
    assert len(raw) == 6 * w["n"]
    q = np.frombuffer(raw, "<i2").reshape(-1, 3)
    assert q.tolist() == [[100, -200, 50], [2900, 1, -99], [-31, 0, 399]]


def test_crop_range_height_and_invalid():
    xyz = np.array(
        [
            [1, 0, 0],  # kept
            [0, 0, 0],  # no return
            [31, 0, 0],  # beyond 30 m
            [5, 0, -1.5],  # below z_min
            [5, 0, 4.5],  # above z_max
            [np.nan, 0, 0],
            [0, 29.5, 3.9],  # kept (range is 3D: 29.76 m)
        ],
        np.float32,
    )
    kept, inten = pc.crop(xyz, np.arange(7.0))
    assert kept.tolist() == [[1, 0, 0], [0, pytest.approx(29.5), pytest.approx(3.9)]]
    assert inten.tolist() == [0.0, 6.0]


def test_voxel_downsample_one_centroid_per_voxel():
    rng = np.random.default_rng(2)
    xyz = rng.uniform(0, 3, (20000, 3)).astype(np.float32)
    v, _ = pc.voxel_downsample(xyz, voxel=0.15)
    keys = {tuple(k) for k in np.floor(xyz / 0.15).astype(int)}
    assert len(v) == len(keys) <= 20 ** 3
    # each centroid lies in its own voxel
    assert len({tuple(k) for k in np.floor(v / 0.15).astype(int)}) == len(v)
    w = pc.encode_cloud(xyz, voxel=0.15, min_range=0)
    out, _ = pc.decode_cloud(w)
    assert len(out) == len(v) and np.abs(out - v).max() <= 0.005 + 1e-6


def test_size_of_a_dense_scan():
    xyz, inten = room_scan()
    w = pc.encode_cloud(xyz)
    n = w["n"]
    assert 10_000 < n < len(xyz) // 2  # 131k beams → ~35k voxels
    packed = len(codec.pack(w))
    # ~1.9 B per point after the envelope's zstd (plain int16 would be ~3.7)
    assert packed < 2.2 * n
    assert packed < 80_000
    wi = pc.encode_cloud(xyz, inten)
    _, i8 = pc.decode_cloud(wi)
    assert i8.dtype == np.uint8 and len(i8) == n and i8.max() == 255


def test_cloud_arrays_parses_pointcloud2_layouts():
    # Ouster-like layout: x y z pad intensity(float32) t(uint32) ring(uint16) ...
    n = 6
    dt = np.dtype(
        {"names": ["x", "y", "z", "intensity", "t", "ring"],
         "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2"],
         "offsets": [0, 4, 8, 16, 20, 24], "itemsize": 48}
    )  # fmt: skip
    a = np.zeros(n, dt)
    a["x"], a["y"], a["z"] = np.arange(n), -np.arange(n), 0.5
    a["intensity"] = 10 * np.arange(n)
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 16, 7),
              ("t", 20, 6), ("ring", 24, 4)]  # fmt: skip
    xyz, inten = pc.cloud_arrays(fields, 48, a.tobytes(), n // 2, 2, row_step=48 * n // 2)
    assert xyz[:, 0].tolist() == list(range(n)) and xyz[:, 1].tolist() == [-v for v in range(n)]
    assert inten.tolist() == [10.0 * v for v in range(n)]
    # padded rows (row_step > width * point_step) and big-endian data
    rows = [np.concatenate([a[r * 3 : r * 3 + 3].astype(dt.newbyteorder(">")).view(np.uint8),
                            np.zeros(8, np.uint8)]) for r in range(2)]  # fmt: skip
    xyz2, _ = pc.cloud_arrays(fields, 48, b"".join(r.tobytes() for r in rows), 3, 2,
                              row_step=48 * 3 + 8, bigendian=True)  # fmt: skip
    assert np.array_equal(xyz2, xyz)
    xyz3, i3 = pc.cloud_arrays(fields[:3], 48, a.tobytes(), n)
    assert i3 is None and len(xyz3) == n
    with pytest.raises(ValueError):
        pc.cloud_arrays(fields[:2], 48, a.tobytes(), n)
