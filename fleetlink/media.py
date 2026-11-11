"""Payload helpers for the RViz-replacement streams: occupancy grid, executor
paths, and on-demand camera frames (API.md §11).

Pure numpy / zlib / msgpack.  cv2 is used for resizing and JPEG when it can be
imported; otherwise frames are encoded as PNG with numpy + zlib.
"""

from __future__ import annotations

import hashlib
import struct
import zlib

import msgpack
import numpy as np

MAX_PATH_POINTS = 150

# ----------------------------------------------------------------------------
# Occupancy grid
# ----------------------------------------------------------------------------


def grid_hash(meta: tuple, data: bytes) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(repr(meta).encode())
    h.update(data)
    return h.hexdigest()


def encode_grid(
    frame: str,
    resolution: float,
    origin,
    width: int,
    height: int,
    data: bytes,
    t: float | None = None,
) -> dict:
    """OccupancyGrid → wire dict ``{frame, resolution, origin: [x,y,z,yaw], width,
    height, data: zlib(int8[height*width]), hash, t}``.  ``data`` is row-major
    (x fastest) like nav_msgs/OccupancyGrid: -1 unknown, 0..100 occupancy."""
    raw = bytes(data)
    if len(raw) != int(width) * int(height):
        raise ValueError(f"grid data has {len(raw)} cells, expected {width}x{height}")
    origin = [float(v) for v in origin]
    meta = (frame, round(float(resolution), 6), tuple(round(v, 4) for v in origin),
            int(width), int(height))  # fmt: skip
    return {
        "frame": frame,
        "resolution": float(resolution),
        "origin": origin,
        "width": int(width),
        "height": int(height),
        "data": zlib.compress(raw, 6),
        "hash": grid_hash(meta, raw),
        "t": t,
    }


def decode_grid(enc: dict) -> np.ndarray:
    """Wire dict → int8 array of shape (height, width)."""
    raw = zlib.decompress(enc["data"])
    return np.frombuffer(raw, np.int8).reshape(int(enc["height"]), int(enc["width"]))


def yaw_from_quat(qw, qx, qy, qz) -> float:
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------


def downsample(points, n: int = MAX_PATH_POINTS) -> list:
    """At most ``n`` points, evenly spaced by index, first and last kept."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) > n:
        idx = np.unique(np.round(np.linspace(0, len(pts) - 1, n)).astype(int))
        pts = pts[idx]
    return [[round(float(v), 3) for v in p] for p in pts]


def paths_hash(smooth, mlp, target, frame) -> str:
    """Change detection at 1 cm (the timestamp is not part of the identity)."""

    def q(pts):
        return [tuple(round(v * 100) for v in p) for p in pts or []]

    tgt = tuple(round(v * 100) for v in target) if target is not None else None
    h = hashlib.blake2b(digest_size=8)
    h.update(repr((q(smooth), q(mlp), tgt, frame)).encode())
    return h.hexdigest()


def paths_payload(smooth, mlp, target, frame: str, t: float | None) -> dict:
    smooth = downsample(smooth) if smooth is not None and len(smooth) else []
    mlp = downsample(mlp) if mlp is not None and len(mlp) else []
    target = [round(float(v), 3) for v in target] if target is not None else None
    return {
        "smooth": smooth,
        "mlp": mlp,
        "target": target,
        "frame": frame,
        "t": t,
        "hash": paths_hash(smooth, mlp, target, frame),
    }


# ----------------------------------------------------------------------------
# Camera frames
# ----------------------------------------------------------------------------

try:  # spark_env has opencv; robots without it fall back to numpy PNG
    import cv2 as _cv2
except Exception:  # noqa: BLE001
    _cv2 = None


def image_to_array(
    encoding: str, width: int, height: int, step: int, data
) -> np.ndarray:
    """sensor_msgs/Image fields → uint8 RGB (H, W, 3) or grey (H, W).

    rgb8 / bgr8 / rgba8 / bgra8 / mono8 pass through (channel-swapped to RGB);
    16UC1 / mono16 / 32FC1 (depth) are normalised to 0..255 between the 2nd and
    98th percentile of the valid (nonzero, finite) pixels."""
    enc = (encoding or "").lower()
    buf = np.frombuffer(bytes(data), np.uint8)
    if enc in ("jpeg", "png"):  # sensor_msgs/CompressedImage payload
        return decode_compressed(buf)
    h, w = int(height), int(width)
    step = int(step) or 0
    chans = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1,
             "8uc3": 3, "8uc4": 4}  # fmt: skip
    if enc in chans:
        c = chans[enc]
        step = step or w * c
        img = buf[: h * step].reshape(h, step)[:, : w * c].reshape(h, w, c)
        if c == 1:
            return np.ascontiguousarray(img[:, :, 0])
        if enc.startswith("bgr") or enc in ("8uc3", "8uc4"):
            img = img[:, :, [2, 1, 0]]
        else:
            img = img[:, :, :3]
        return np.ascontiguousarray(img)
    if enc in ("16uc1", "mono16", "32fc1"):
        dt = np.float32 if enc == "32fc1" else np.uint16
        size = np.dtype(dt).itemsize
        step = step or w * size
        rows = buf[: h * step].reshape(h, step)[:, : w * size]
        img = np.ascontiguousarray(rows).view(dt).reshape(h, w).astype(np.float32)
        valid = np.isfinite(img) & (img > 0)
        out = np.zeros((h, w), np.uint8)
        if valid.any():
            lo, hi = np.percentile(img[valid], [2, 98])
            hi = hi if hi > lo else lo + 1.0
            out[valid] = np.clip((img[valid] - lo) / (hi - lo) * 254 + 1, 1, 255)
        return out
    raise ValueError(f"unsupported image encoding {encoding!r}")


def compressed_encoding(fmt: str) -> str:
    """sensor_msgs/CompressedImage ``format`` (e.g. "rgb8; jpeg compressed bgr8") → jpeg | png."""
    return "png" if "png" in (fmt or "").lower() else "jpeg"


def decode_compressed(buf: np.ndarray) -> np.ndarray:
    """JPEG/PNG bytes → uint8 RGB (H, W, 3) or grey (H, W)."""
    if _cv2 is not None:
        img = _cv2.imdecode(buf, _cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("undecodable compressed image")
        if img.ndim == 3:
            img = _cv2.cvtColor(img, _cv2.COLOR_BGRA2RGB if img.shape[2] == 4 else _cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)
    import io

    from PIL import Image as _PIL

    pil = _PIL.open(io.BytesIO(buf.tobytes()))
    return np.asarray(pil.convert("L" if pil.mode in ("L", "I;16") else "RGB"))


def resize_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max_w <= 0 or w <= max_w:
        return img
    new_w = int(max_w)
    new_h = max(1, round(h * new_w / w))
    if _cv2 is not None:
        return _cv2.resize(img, (new_w, new_h), interpolation=_cv2.INTER_AREA)
    ys = np.minimum((np.arange(new_h) * h / new_h).astype(int), h - 1)
    xs = np.minimum((np.arange(new_w) * w / new_w).astype(int), w - 1)
    return np.ascontiguousarray(img[ys][:, xs])


def _png(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    color = 2 if img.ndim == 3 else 0
    rows = img.reshape(h, -1)
    raw = b"".join(b"\x00" + rows[i].tobytes() for i in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        c = struct.pack(">I", len(payload)) + tag + payload
        return c + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def encode_image(img: np.ndarray, quality: int = 70, force_png: bool = False):
    """→ (bytes, "jpeg" | "png")."""
    if _cv2 is not None and not force_png:
        bgr = img if img.ndim == 2 else img[:, :, ::-1]
        ok, buf = _cv2.imencode(".jpg", bgr, [_cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if ok:
            return buf.tobytes(), "jpeg"
    return _png(img), "png"


def camera_reply(
    frame: dict, max_w: int, quality: int, now: float, force_png=False
) -> dict:
    """A stored frame (``{encoding, width, height, step, data, stamp, rx_t}``) →
    reply dict ``{format, data, width, height, src_width, src_height, stamp, age_s}``."""
    img = image_to_array(
        frame["encoding"], frame["width"], frame["height"], frame["step"], frame["data"]
    )
    src_h, src_w = img.shape[:2]  # compressed frames don't know their size until decoded
    img = resize_max_width(img, max_w)
    data, fmt = encode_image(img, quality, force_png)
    return {
        "format": fmt,
        "data": data,
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "src_width": int(src_w),
        "src_height": int(src_h),
        "encoding": frame["encoding"],
        "stamp": frame.get("stamp"),
        "age_s": None if frame.get("rx_t") is None else max(0.0, now - frame["rx_t"]),
    }


def pack_reply(obj: dict) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack_reply(data: bytes) -> dict:
    return msgpack.unpackb(data, raw=False)
