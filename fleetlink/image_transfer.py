"""Compact, self-describing encoding of hydra image-store frames for the radio.

The store reader/writer is Hydra's ``hydra_python.image_store``. Used by the prior-map
transfer (:mod:`fleetlink.image_pack`, ``python -m fleetlink.image_pack pack`` /
``unpack``) and the live store mirror
(:mod:`fleetlink.store_mirror`). One frame (its row,
calibration, RGB and depth) becomes one ``bytes`` blob::

    b"HISF" | u8 version (1) | u32 LE header length | header JSON (utf-8) | parts...

The header holds ``meta`` (frame row, calibration, codec choices) and ``parts``
(``[{"name": "rgb"|"depth", "codec": ..., "size": n, ...}]``); the part payloads follow
in that order.

Compression modes:

=============  ======================  ===============================================
mode           RGB                     depth
=============  ======================  ===============================================
``none``       stored file bytes       stored file bytes (``png16``)
``lossless``   stored file bytes       uint16 row-delta + zstd-3 (``delta-zstd``)
``fast``       JPEG q75 (re-encoded)   half resolution, then ``delta-zstd``
=============  ======================  ===============================================

``fast`` halves the depth image by nearest sampling: the depth at half-resolution pixel
``(i, j)`` is the full-resolution sample at ``(2i, 2j)`` (the top-left pixel of each 2x2
block). Only measured values survive (no blending of foreground and background across
depth edges, zeros stay "no measurement"), and the scaled intrinsics are exact for this
sampling: ``fx/2, fy/2, cx/2, cy/2``, width ``ceil(w/2)``, height ``ceil(h/2)``. The
blob records ``depth_downscale`` and the scaled ``depth_calib``. :func:`restore_depth`
(used when writing into a store) repeats each sample over its 2x2 block so depth is
pixel-aligned with RGB and the sensor calibration again; back-projection then uses the
full-resolution calibration unchanged.

Optional dependencies: ``zstandard`` (``lossless`` / ``fast``) and ``opencv-python``
(``cv2``) or Pillow for JPEG re-encoding and PNG decoding. Missing ones raise
:class:`MissingDependency` naming the package.

Frames arrive from another machine, so :func:`decoded_paths` treats the header as
untrusted: ids and timestamps must be integers and every path must stay inside the
destination store (:class:`UnsafeFrame` otherwise).
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import struct
from collections.abc import Iterable

import numpy as np

# track ids from EXTERNAL_TRACK_BASE up belong to writers other than Khronos (SAM3 write-back)
from hydra_python.image_store.codec import EXTERNAL_TRACK_BASE as SAM3_TRACK_BASE
from hydra_python.image_store.codec import rle_encode, transform_to_blob
from hydra_python.image_store.store import (
    _FRAME_COLS,
    Frame,
    ImageStore,
    ImageStoreWriter,
    frame_stem,
)

MAGIC = b"HISF"
VERSION = 1
COMPRESSION_MODES = ("none", "lossless", "fast")
FRAME_SELECTIONS = ("keyframes", "all")
# "keyframes" = the agent nodes' own keyframes: of the image store's KEYFRAMEs (every
# keyframe archive, anchored to the agent node nearest in time) the ones taken at their
# anchor's stamp, whose anchor_T_body is therefore the identity (what used to be kind 0
# before the sub-keyframes, now the other keyframes, became KEYFRAMEs too). "all" adds
# the other keyframes. SQL: needs register_sql_functions(store) on the connection.
OWN_KEYFRAME_SQL = "(kind = 0 AND own_keyframe(anchor_node_id, anchor_T_body))"
OWN_KEYFRAME_ATOL = 1e-3
_EYE = np.eye(4).reshape(-1)


def _own_keyframe(anchor_node_id, anchor_T_body) -> int:
    if anchor_node_id is None:
        return 0
    if anchor_T_body is None:
        return 1
    a = np.frombuffer(anchor_T_body, dtype="<f8")
    return int(a.size == 16 and float(np.max(np.abs(a - _EYE))) < OWN_KEYFRAME_ATOL)


def register_sql_functions(store) -> None:
    """Register ``own_keyframe(anchor_node_id, anchor_T_body)`` (see OWN_KEYFRAME_SQL)
    on the store's sqlite connections (idempotent)."""
    for db in getattr(store, "_dbs", ()):
        db.conn.create_function("own_keyframe", 2, _own_keyframe, deterministic=True)


FAST_JPEG_QUALITY = 75
ZSTD_LEVEL = 3
_HEADER = struct.Struct("<4sBI")


class MissingDependency(ImportError):
    pass


class UnsafeFrame(ValueError):
    """An encoded frame whose header would write outside the destination store."""


# --- optional dependencies -------------------------------------------------------------


def _zstd():
    try:
        import zstandard
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependency(
            "compression 'lossless'/'fast' needs the 'zstandard' package "
            "(pip install zstandard)"
        ) from exc
    return zstandard


def _cv2():
    try:
        import cv2

        return cv2
    except ImportError:
        return None


def _need_image_lib(what: str):
    cv2 = _cv2()
    if cv2 is not None:
        return cv2, None
    try:
        from PIL import Image

        return None, Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise MissingDependency(
            f"{what} needs opencv-python (cv2) or Pillow "
            "(pip install opencv-python-headless)"
        ) from exc


def _sniff(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return "bin"


def _decode_rgb(data: bytes) -> np.ndarray:
    """RGB (HxWx3 uint8, R first) from an encoded image."""
    cv2, pil = _need_image_lib("re-encoding RGB")
    if cv2 is not None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("could not decode the RGB image")
        return img[..., ::-1]
    with pil.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"))


def encode_jpeg(rgb: np.ndarray, quality: int = FAST_JPEG_QUALITY) -> bytes:
    """JPEG bytes of an RGB (or grayscale) uint8 image (cv2/libjpeg-turbo, else Pillow)."""
    cv2, pil = _need_image_lib("JPEG encoding")
    rgb = np.ascontiguousarray(rgb)
    if cv2 is not None:
        img = rgb[..., ::-1] if rgb.ndim == 3 else rgb
        ok, buf = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(img),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )
        if not ok:
            raise ValueError("JPEG encoding failed")
        return buf.tobytes()
    out = io.BytesIO()
    pil.fromarray(rgb).save(out, format="JPEG", quality=int(quality))
    return out.getvalue()


def decode_png16(data: bytes) -> np.ndarray:
    """uint16 HxW array from a 16-bit (or 8-bit) PNG."""
    cv2, pil = _need_image_lib("decoding depth PNGs")
    if cv2 is not None:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("could not decode the depth PNG")
    else:
        with pil.open(io.BytesIO(data)) as im:
            arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr.astype(np.uint16, copy=False))


def decode_depth_mm(data: bytes) -> np.ndarray:
    """uint16 millimetre depth from a stored depth file: a 16-bit PNG (the store's own
    format), else any float image in metres (the TIFF a raw Hydra InputData archive
    holds, or the EXR of an older archive), converted to millimetres."""
    if _sniff(data) == "png":
        return decode_png16(data)
    arr = None
    if data[:4] == b"\x76\x2f\x31\x01":  # OpenEXR (archives before Hydra's TIFF depth)
        try:
            from hydra_python.image_store.inputdata import decode_exr

            arr = decode_exr(data)
        except Exception:  # noqa: BLE001  (no EXR support: try OpenCV below)
            arr = None
    if arr is not None:
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        mm = np.nan_to_num(
            arr.astype(np.float64) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.ascontiguousarray(np.clip(np.rint(mm), 0, 65535).astype(np.uint16))
    cv2 = _cv2()
    if cv2 is None:
        raise MissingDependency("decoding non-PNG depth needs opencv-python (cv2)")
    arr = cv2.imdecode(
        np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH
    )
    if arr is None:
        raise ValueError(
            "could not decode the depth image (EXR needs OPENCV_IO_ENABLE_OPENEXR=1)"
        )
    if arr.ndim == 3:
        arr = arr[..., 0]
    if np.issubdtype(arr.dtype, np.floating):
        mm = np.nan_to_num(
            arr.astype(np.float64) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.ascontiguousarray(np.clip(np.rint(mm), 0, 65535).astype(np.uint16))
    return np.ascontiguousarray(arr.astype(np.uint16, copy=False))


def encode_png16(depth: np.ndarray) -> bytes:
    """16-bit PNG bytes of a uint16 HxW array."""
    cv2, pil = _need_image_lib("encoding depth PNGs")
    depth = np.ascontiguousarray(depth, dtype=np.uint16)
    if cv2 is not None:
        ok, buf = cv2.imencode(".png", depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if not ok:
            raise ValueError("PNG encoding failed")
        return buf.tobytes()
    out = io.BytesIO()
    pil.fromarray(depth).save(out, format="PNG")
    return out.getvalue()


# --- depth codec -------------------------------------------------------------------------


def delta_zstd_encode(depth: np.ndarray, level: int = ZSTD_LEVEL) -> bytes:
    """Row-wise delta (mod 2**16) of a uint16 image, then zstd."""
    d = np.ascontiguousarray(depth, dtype=np.uint16)
    delta = np.empty_like(d)
    delta[:, :1] = d[:, :1]
    np.subtract(
        d[:, 1:], d[:, :-1], out=delta[:, 1:]
    )  # uint16 wraps: exact inverse below
    return (
        _zstd()
        .ZstdCompressor(level=level)
        .compress(delta.astype("<u2", copy=False).tobytes())
    )


def delta_zstd_decode(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    raw = _zstd().ZstdDecompressor().decompress(data, max_output_size=2 * h * w)
    delta = np.frombuffer(raw, dtype="<u2").reshape(h, w)
    return np.cumsum(delta, axis=1, dtype=np.uint16)


def downscale_depth(depth: np.ndarray, factor: int = 2) -> np.ndarray:
    """Nearest sampling: half-res pixel (i, j) = full-res pixel (factor*i, factor*j)."""
    return np.ascontiguousarray(depth[::factor, ::factor])


def restore_depth(depth: np.ndarray, factor: int, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`downscale_depth`: repeat every sample over its block, crop to ``shape``."""
    if factor == 1:
        return depth
    up = np.repeat(np.repeat(depth, factor, axis=0), factor, axis=1)
    return np.ascontiguousarray(up[: int(shape[0]), : int(shape[1])])


def scale_calib(calib: dict, factor: int) -> dict:
    """Calibration dict (``Calib.to_json`` layout) for an image downscaled by ``factor``
    with :func:`downscale_depth` sampling."""
    out = dict(calib)
    for k in ("fx", "fy", "cx", "cy"):
        out[k] = float(calib[k]) / factor
    out["width"] = -(-int(calib["width"]) // factor)
    out["height"] = -(-int(calib["height"]) // factor)
    return out


# --- frames -------------------------------------------------------------------------------


def select_frames(
    store: ImageStore,
    frames: str = "keyframes",
    after: int | None = None,
    *,
    include_kept_observations: bool = False,
    require_images: bool = True,
    limit: int | None = None,
) -> list[Frame]:
    """Frames to send, ordered by ``frame_id`` (the sqlite rowid, so a watermark).

    ``frames``: ``"keyframes"`` (the agent nodes' own keyframes, see
    ``OWN_KEYFRAME_SQL``; what SAM3 search uses) or ``"all"``.
    ``after``: only frames with ``frame_id > after`` (resume / tail). Note that frame ids
    are per robot (``robot_id << 48 | seq``), so a watermark is per robot.
    ``include_kept_observations``: with ``"keyframes"``, also the frames of kept object
    observations (the object images; ``pack`` uses this).
    ``require_images``: skip rows that have neither an RGB nor a depth file yet.
    """
    if frames not in FRAME_SELECTIONS:
        raise ValueError(f"frames must be one of {FRAME_SELECTIONS}, got {frames!r}")
    where, params = [], []
    if frames == "keyframes":
        register_sql_functions(store)
        cond = OWN_KEYFRAME_SQL
        if include_kept_observations:
            cond = f"({OWN_KEYFRAME_SQL} OR frame_id IN (SELECT frame_id FROM observations WHERE kept != 0))"
        where.append(cond)
    if after is not None:
        where.append("frame_id > ?")
        params.append(int(after))
    if require_images:
        where.append("(rgb_path IS NOT NULL OR depth_path IS NOT NULL)")
    sql = f"SELECT {_FRAME_COLS} FROM frames"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY frame_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    out = [db.frame_from_row(r) for db, r in store._query(sql, params)]
    out.sort(key=lambda f: f.frame_id)
    return out if limit is None else out[: int(limit)]


def _read(path: pathlib.Path | None) -> bytes | None:
    if path is None or not os.path.exists(path):
        return None
    return pathlib.Path(path).read_bytes()


def _rel(frame: Frame, path: pathlib.Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(pathlib.Path(path).relative_to(frame.store_dir))
    except ValueError:
        return None


def frame_meta(store: ImageStore, frame: Frame) -> dict:
    calib = store.calib(frame.sensor_id)
    return {
        "frame_id": int(frame.frame_id),
        "robot_id": int(frame.robot_id),
        "sensor_id": int(frame.sensor_id),
        "timestamp_ns": int(frame.timestamp_ns),
        "kind": int(frame.kind),
        "world_T_body_odom": np.asarray(frame.world_T_body_odom, float)
        .reshape(16)
        .tolist(),
        "anchor_node_id": None
        if frame.anchor_node_id is None
        else int(frame.anchor_node_id),
        "anchor_T_body": None
        if frame.anchor_T_body is None
        else np.asarray(frame.anchor_T_body, float).reshape(16).tolist(),
        "rgb_path": _rel(frame, frame.rgb_path),
        "depth_path": _rel(frame, frame.depth_path),
        "calib": None
        if calib is None
        else {**calib.to_json(), "robot_id": int(calib.robot_id)},
    }


_OBS_SQL = (
    "SELECT o.obs_id, o.robot_id, o.track_id, o.timestamp_ns, "
    "o.bbox2d_min_x, o.bbox2d_min_y, o.bbox2d_max_x, o.bbox2d_max_y, "
    "o.bbox3d_min_x, o.bbox3d_min_y, o.bbox3d_min_z, o.bbox3d_max_x, o.bbox3d_max_y, o.bbox3d_max_z, "
    "o.mask_rle, o.mask_path, o.score, o.kept, b.node_id FROM observations o "
    "LEFT JOIN track_bindings b ON b.robot_id = o.robot_id AND b.track_id = o.track_id "
    "WHERE o.frame_id = ? AND o.kept != 0 ORDER BY o.obs_id"
)


def _mask_rle_from_png(path) -> str | None:
    data = _read(path)
    if data is None:
        return None
    try:
        arr = decode_png16(data)
    except Exception:  # noqa: BLE001
        return None
    return json.dumps(rle_encode(arr > 0), separators=(",", ":"))


def frame_observations(store: ImageStore, frame_id: int) -> list[dict]:
    """The kept object observations of a frame with the node their track is bound to
    right now (``node_id`` None = not bound yet). The mask goes as RLE (sensor-sized,
    from the mask PNG when the row has none), so the observation is self-contained.
    Used by the live store mirror: the base then shows object images during the run."""
    out = []
    for db, r in store._query(_OBS_SQL, (int(frame_id),)):
        rle = r[14]
        if rle is None and r[15] is not None:
            rle = _mask_rle_from_png(db.abs(r[15]))
        out.append(
            {
                "obs_id": int(r[0]),
                "robot_id": int(r[1]),
                "track_id": int(r[2]),
                "timestamp_ns": int(r[3]),
                "bbox2d": None if r[4] is None else [int(v) for v in r[4:8]],
                "bbox3d": None if r[8] is None else [float(v) for v in r[8:14]],
                "mask_rle": rle,
                "score": float(r[16]),
                "kept": bool(r[17]),
                "node_id": None if r[18] is None else int(r[18]),
            }
        )
    return out


def encode_frame(
    store: ImageStore,
    frame: Frame | int,
    compression: str = "fast",
    *,
    observations: bool = False,
) -> bytes:
    """One frame (row, calibration, RGB, depth) as a self-describing blob.

    ``frame`` is a :class:`Frame` or a frame id. Missing image files are simply left out
    (``parts`` then lacks that entry). ``observations``: also carry the frame's kept
    object observations and their track bindings (:func:`frame_observations`), which
    :func:`write_frame` inserts.
    """
    if compression not in COMPRESSION_MODES:
        raise ValueError(
            f"compression must be one of {COMPRESSION_MODES}, got {compression!r}"
        )
    if not isinstance(frame, Frame):
        fid = int(frame)
        frame = store.frame(fid)
        if frame is None:
            raise KeyError(f"unknown frame {fid}")
    meta = frame_meta(store, frame)
    meta["compression"] = compression
    if observations:
        obs = frame_observations(store, frame.frame_id)
        if obs:
            meta["observations"] = obs
    meta["depth_downscale"] = 1
    parts: list[tuple[dict, bytes]] = []

    for path in (frame.rgb_path, frame.depth_path):
        if path is not None and pathlib.Path(path).suffix.lower() == ".zip":
            # raw InputData archives are indexed (and re-encoded) by the store's indexer;
            # a store row pointing at one is a bug, never ship it
            raise ValueError(f"frame {frame.frame_id}: refusing to send archive {path}")
    rgb = _read(frame.rgb_path)
    if rgb is not None:
        fmt = _sniff(rgb)
        if fmt == "bin" and compression != "fast":  # not JPEG/PNG: send a JPEG
            rgb = encode_jpeg(_decode_rgb(rgb), 95)
            fmt = "jpeg"
        if compression == "fast":  # stored JPEGs are q90+; q75 is ~40% smaller
            rgb = encode_jpeg(_decode_rgb(rgb), FAST_JPEG_QUALITY)
            fmt = "jpeg"
            meta["rgb_quality"] = FAST_JPEG_QUALITY
        parts.append(({"name": "rgb", "codec": fmt}, rgb))

    depth = _read(frame.depth_path)
    if depth is not None and _sniff(depth) != "png":
        depth = encode_png16(decode_depth_mm(depth))  # e.g. EXR metres -> PNG mm
        if meta["calib"] is not None:
            meta["calib"]["depth_scale"] = 0.001
    if depth is not None:
        if compression == "none":
            parts.append(
                (
                    {
                        "name": "depth",
                        "codec": "png16" if _sniff(depth) == "png" else "file",
                        "ext": pathlib.Path(frame.depth_path).suffix,
                    },
                    depth,
                )
            )
        else:
            arr = decode_png16(depth)
            full_shape = list(arr.shape)
            if compression == "fast":
                arr = downscale_depth(arr, 2)
                meta["depth_downscale"] = 2
                if meta["calib"] is not None:
                    meta["depth_calib"] = scale_calib(meta["calib"], 2)
            parts.append(
                (
                    {
                        "name": "depth",
                        "codec": "delta-zstd",
                        "shape": list(arr.shape),
                        "full_shape": full_shape,
                        "dtype": "<u2",
                    },
                    delta_zstd_encode(arr),
                )
            )
    if meta.get("depth_calib") is None and meta["calib"] is not None:
        meta["depth_calib"] = dict(meta["calib"])

    header = {"meta": meta, "parts": [dict(p, size=len(b)) for p, b in parts]}
    hdr = json.dumps(header, separators=(",", ":")).encode()
    return b"".join(
        [_HEADER.pack(MAGIC, VERSION, len(hdr)), hdr, *(b for _, b in parts)]
    )


def decode_header(blob: bytes) -> dict:
    """The header (``meta`` + ``parts``) of an encoded frame without decoding pixels."""
    magic, version, n = _HEADER.unpack_from(blob, 0)
    if magic != MAGIC:
        raise ValueError("not an encoded store frame (bad magic)")
    if version != VERSION:
        raise ValueError(
            f"encoded frame version {version}, this reader supports {VERSION}"
        )
    return json.loads(blob[_HEADER.size : _HEADER.size + n])


def decode_frame(blob: bytes, *, decode_depth_png: bool = True) -> dict:
    """Inverse of :func:`encode_frame`.

    Returns ``{"meta", "rgb", "rgb_format", "depth", "depth_png", "depth_downscale"}``:
    ``rgb`` is the encoded image (JPEG bytes unless the store held something else,
    ``rgb_format`` says which); ``depth`` is a uint16 ndarray at the transmitted
    resolution (``depth_downscale`` 2 in ``fast`` mode, calibration in
    ``meta["depth_calib"]``; full-resolution calibration in ``meta["calib"]``);
    ``depth_png`` holds the stored PNG bytes in ``none`` mode (``depth`` is then decoded
    from it only if ``decode_depth_png``).
    """
    _, _, n = _HEADER.unpack_from(blob, 0)
    header = decode_header(blob)
    meta = header["meta"]
    out = {
        "meta": meta,
        "rgb": None,
        "rgb_format": None,
        "depth": None,
        "depth_png": None,
        "depth_downscale": int(meta.get("depth_downscale", 1)),
    }
    pos = _HEADER.size + n
    for part in header["parts"]:
        data = bytes(blob[pos : pos + int(part["size"])])
        pos += int(part["size"])
        if part["name"] == "rgb":
            out["rgb"], out["rgb_format"] = data, part["codec"]
        elif part["name"] == "depth":
            if part["codec"] == "delta-zstd":
                out["depth"] = delta_zstd_decode(data, part["shape"])
                out["depth_full_shape"] = tuple(part.get("full_shape", part["shape"]))
            else:
                out["depth_png"] = data
                out["depth_ext"] = part.get("ext") or ".png"
                if decode_depth_png:
                    out["depth"] = decode_png16(data)
                    out["depth_full_shape"] = out["depth"].shape
    if pos != len(blob):
        raise ValueError(f"encoded frame has {len(blob) - pos} trailing bytes")
    return out


def full_res_depth(decoded: dict) -> np.ndarray | None:
    """Depth pixel-aligned with RGB / the sensor calibration (upsampled if downscaled)."""
    depth = decoded.get("depth")
    if depth is None and decoded.get("depth_png") is not None:
        depth = decode_png16(decoded["depth_png"])
    if depth is None:
        return None
    factor = int(decoded.get("depth_downscale", 1))
    if factor == 1:
        return depth
    calib = decoded["meta"].get("calib")
    shape = decoded.get("depth_full_shape") or (
        (calib["height"], calib["width"])
        if calib
        else (depth.shape[0] * factor, depth.shape[1] * factor)
    )
    return restore_depth(depth, factor, shape)


_RGB_EXT = {"jpeg": ".jpg", "png": ".png"}


def _safe_rel(rel: str | None) -> str | None:
    if not rel or not isinstance(rel, str) or "\\" in rel or "\0" in rel:
        return None
    p = pathlib.PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or p.name in ("", ".", ".."):
        return None
    return str(p)


def _int_field(meta: dict, key: str) -> int:
    """``meta[key]`` as an int; rejects strings, floats and bools (untrusted header)."""
    value = meta.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UnsafeFrame(f"frame header {key} must be an integer, got {value!r}")
    return value


def _inside(root: pathlib.Path, rel: str) -> pathlib.Path:
    """``root / rel``, provided it resolves inside ``root`` (no ``..``, absolute paths
    or symlinked directories leading out)."""
    base = os.path.realpath(root)
    dst = os.path.realpath(os.path.join(base, rel))
    if os.path.commonpath([base, dst]) != base or dst == base:
        raise UnsafeFrame(f"frame path {rel!r} leaves the store {root}")
    return pathlib.Path(dst)


def decoded_paths(decoded: dict) -> tuple[str | None, str | None]:
    """Store-relative paths the decoded images are written to.

    The source paths are kept when they are safe relative paths (same layout as the
    source store), else ``frames/r<robot>/<stem>_{rgb,depth}<ext>`` with the C++
    writer's stem (``<ts>``, or ``<ts>_s<seq>`` for the robot's other sensors). The RGB
    extension follows the transmitted format; depth is always written as a 16-bit PNG.
    ``robot_id``, ``sensor_id`` and ``timestamp_ns`` must be integers
    (:class:`UnsafeFrame`).
    """
    meta = decoded["meta"]
    robot = _int_field(meta, "robot_id")
    ts = _int_field(meta, "timestamp_ns")
    sensor = _int_field(meta, "sensor_id")
    base = f"frames/r{robot}/{frame_stem(ts, sensor)}"
    rgb_rel = depth_rel = None
    if decoded.get("rgb") is not None:
        ext = _RGB_EXT.get(decoded["rgb_format"], ".bin")
        src = _safe_rel(meta.get("rgb_path"))
        rgb_rel = (
            str(pathlib.PurePosixPath(src).with_suffix(ext))
            if src
            else f"{base}_rgb{ext}"
        )
    if decoded.get("depth") is not None or decoded.get("depth_png") is not None:
        src = _safe_rel(meta.get("depth_path"))
        ext = (
            decoded.get("depth_ext", ".png")
            if decoded.get("depth_png") is not None
            else ".png"
        )
        depth_rel = (
            str(pathlib.PurePosixPath(src).with_suffix(ext))
            if src
            else f"{base}_depth{ext}"
        )
    return rgb_rel, depth_rel


def write_decoded_images(
    root: str | os.PathLike, decoded: dict
) -> tuple[str | None, str | None]:
    """Write a decoded frame's images under the store directory ``root``; returns the
    relative ``(rgb_path, depth_path)``. Downscaled depth is restored to full resolution
    (see :func:`restore_depth`), so the store's single sensor calibration stays valid.
    Raises :class:`UnsafeFrame` (writing nothing) when a path would leave ``root``."""
    root = pathlib.Path(root)
    rgb_rel, depth_rel = decoded_paths(decoded)
    rgb_dst = None if rgb_rel is None else _inside(root, rgb_rel)
    depth_dst = None if depth_rel is None else _inside(root, depth_rel)
    if rgb_dst is not None:
        rgb_dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(rgb_dst, decoded["rgb"])
    if depth_dst is not None:
        dst = depth_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        if (
            decoded.get("depth_png") is not None
            and int(decoded.get("depth_downscale", 1)) == 1
        ):
            data = decoded["depth_png"]
        else:
            data = encode_png16(full_res_depth(decoded))
        _atomic_write(dst, data)
    return rgb_rel, depth_rel


def _atomic_write(dst: pathlib.Path, data: bytes) -> None:
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dst)


def write_frame(writer: ImageStoreWriter, decoded: dict) -> int:
    """Insert a decoded frame into a store, keeping the source ids (mirror use).

    The sensor row (full-resolution calibration) and the frame row are inserted with
    their source ``sensor_id`` / ``frame_id`` (so the mirror's ids equal the robot's, and
    a resume watermark is the max ``frame_id``); an existing frame row only gets its
    missing image paths and anchor filled in. Returns the frame id.
    """
    meta = decoded["meta"]
    rgb_rel, depth_rel = write_decoded_images(writer.root, decoded)
    calib = meta.get("calib")
    with writer.batch():
        c = writer.conn
        if calib is not None:
            c.execute(
                "INSERT OR IGNORE INTO sensors (sensor_id, robot_id, name, width, height, fx, fy, cx, cy, "
                "depth_scale, body_T_sensor) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(calib["sensor_id"]),
                    int(calib.get("robot_id", meta["robot_id"])),
                    calib["name"],
                    int(calib["width"]),
                    int(calib["height"]),
                    float(calib["fx"]),
                    float(calib["fy"]),
                    float(calib["cx"]),
                    float(calib["cy"]),
                    float(calib["depth_scale"]),
                    transform_to_blob(
                        np.asarray(calib["body_T_sensor"], float).reshape(4, 4)
                    ),
                ),
            )
        anchor_blob = None
        if meta.get("anchor_T_body") is not None:
            anchor_blob = transform_to_blob(
                np.asarray(meta["anchor_T_body"], float).reshape(4, 4)
            )
        row = c.execute(
            "SELECT frame_id FROM frames WHERE frame_id = ?", (int(meta["frame_id"]),)
        ).fetchone()
        if row is None:
            c.execute(
                f"INSERT INTO frames ({_FRAME_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    int(meta["frame_id"]),
                    int(meta["robot_id"]),
                    int(meta["sensor_id"]),
                    int(meta["timestamp_ns"]),
                    int(meta["kind"]),
                    transform_to_blob(
                        np.asarray(meta["world_T_body_odom"], float).reshape(4, 4)
                    ),
                    meta.get("anchor_node_id"),
                    anchor_blob,
                    rgb_rel,
                    depth_rel,
                ),
            )
        else:
            c.execute(
                "UPDATE frames SET rgb_path = COALESCE(rgb_path, ?), depth_path = COALESCE(depth_path, ?), "
                "anchor_node_id = COALESCE(anchor_node_id, ?), anchor_T_body = COALESCE(anchor_T_body, ?) "
                "WHERE frame_id = ?",
                (
                    rgb_rel,
                    depth_rel,
                    meta.get("anchor_node_id"),
                    anchor_blob,
                    int(meta["frame_id"]),
                ),
            )
        _write_observations(c, meta)
    return int(meta["frame_id"])


def _write_observations(c, meta: dict) -> None:
    """Rows of :func:`frame_observations` carried by a frame: observations keep their
    source ids (a re-sent one is ignored), bindings follow the robot's latest."""
    for o in meta.get("observations") or ():
        b2 = o.get("bbox2d") or [None] * 4
        b3 = o.get("bbox3d") or [None] * 6
        rle = o.get("mask_rle")
        if rle is not None and not isinstance(rle, str):
            rle = json.dumps([int(v) for v in rle], separators=(",", ":"))
        c.execute(
            "INSERT OR IGNORE INTO observations (obs_id, robot_id, track_id, frame_id, timestamp_ns, "
            "bbox2d_min_x, bbox2d_min_y, bbox2d_max_x, bbox2d_max_y, bbox3d_min_x, bbox3d_min_y, "
            "bbox3d_min_z, bbox3d_max_x, bbox3d_max_y, bbox3d_max_z, mask_rle, mask_path, score, kept) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                int(o["obs_id"]),
                int(o["robot_id"]),
                int(o["track_id"]),
                int(meta["frame_id"]),
                int(o.get("timestamp_ns", meta["timestamp_ns"])),
                *[None if v is None else int(v) for v in b2],
                *[None if v is None else float(v) for v in b3],
                rle,
                float(o.get("score", 0.0)),
                int(bool(o.get("kept", True))),
            ),
        )
        node = o.get("node_id")
        if node is not None and int(o["track_id"]) < SAM3_TRACK_BASE:
            c.execute(
                "INSERT OR REPLACE INTO track_bindings (robot_id, track_id, node_id) VALUES (?, ?, ?)",
                (int(o["robot_id"]), int(o["track_id"]), int(node)),
            )


def encode_frames(
    store: ImageStore, frames: Iterable[Frame], compression: str = "fast"
):
    """Generator of ``(frame, blob)`` for :func:`select_frames` output."""
    for f in frames:
        yield f, encode_frame(store, f, compression)


__all__ = [
    "COMPRESSION_MODES",
    "FRAME_SELECTIONS",
    "MissingDependency",
    "OWN_KEYFRAME_SQL",
    "UnsafeFrame",
    "decode_frame",
    "decode_header",
    "decoded_paths",
    "encode_frame",
    "encode_frames",
    "frame_observations",
    "decode_depth_mm",
    "full_res_depth",
    "register_sql_functions",
    "restore_depth",
    "scale_calib",
    "select_frames",
    "write_decoded_images",
    "write_frame",
]
