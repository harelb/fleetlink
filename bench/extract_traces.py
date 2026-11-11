"""Realistic per-message size traces for the multi-robot bench, from the
Pennovation bag's real ZED frames:

* image-store mirror (C6): each keyframe as ``fleetlink.image_transfer``
  encodes it in ``fast`` mode (JPEG q75 RGB + half-res delta-zstd depth);
* live video (C5): the frames through fleetlink's own ffmpeg arguments
  (``video.ffmpeg_args``) at the console presets, split with the same parser.

    python bench/extract_traces.py <bag dir> <out.json> [--seconds 90]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np


def read_frames(bag: str, seconds: float, rgb_topic: str, depth_topic: str):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))  # fmt: skip
    r.set_filter(rosbag2_py.StorageFilter(topics=[rgb_topic, depth_topic]))
    rgb, depth, t0 = [], [], None
    while r.has_next():
        topic, data, t = r.read_next()
        t0 = t if t0 is None else t0
        if (t - t0) * 1e-9 > seconds:
            break
        m = deserialize_message(data, Image)
        a = np.frombuffer(m.data, np.uint8)
        if topic == rgb_topic:
            ch = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3}[m.encoding]
            img = a.reshape(m.height, m.width, ch)[:, :, :3]
            if m.encoding.startswith("bgr"):
                img = img[:, :, ::-1]
            rgb.append(((t - t0) * 1e-9, np.ascontiguousarray(img)))
        else:
            if m.encoding == "32FC1":
                d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
                d = np.nan_to_num(d * 1000.0, nan=0.0, posinf=0.0).clip(0, 65535).astype(np.uint16)
            else:
                d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
            depth.append(((t - t0) * 1e-9, d))
    return rgb, depth


def store_sizes(rgb, depth, kf_hz: float):
    from fleetlink.image_transfer import FAST_JPEG_QUALITY, delta_zstd_encode, downscale_depth, encode_jpeg

    out, next_t, di = [], 0.0, 0
    for t, img in rgb:
        if t < next_t:
            continue
        next_t = t + 1.0 / kf_hz
        while di + 1 < len(depth) and depth[di + 1][0] <= t:
            di += 1
        j = len(encode_jpeg(img, FAST_JPEG_QUALITY))
        d = len(delta_zstd_encode(downscale_depth(depth[di][1], 2))) if depth else 0
        out.append({"t": round(t, 3), "rgb": j, "depth": d, "blob": j + d + 600})  # +header
    return out


def video_sizes(rgb, fps: float, kbps: int, width: int):
    from fleetlink.video import FFMPEG, AnnexBSplitter, out_size, ffmpeg_args, scale

    h0, w0 = rgb[0][1].shape[:2]
    w, h = out_size(w0, h0, width)
    frames, next_t = [], 0.0
    for t, img in rgb:
        if t >= next_t:
            next_t = t + 1.0 / fps
            frames.append(scale(img, w, h) if (w, h) != (w0, h0) else img)
    p = subprocess.Popen(ffmpeg_args("h264", w, h, fps, kbps, FFMPEG), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)  # fmt: skip
    out, _ = p.communicate(b"".join(np.ascontiguousarray(f).tobytes() for f in frames))
    split = AnnexBSplitter()
    aus = split.feed(out) + split.flush()
    sizes = [{"n": len(data), "key": bool(key)} for data, key in aus]
    return {"fps": fps, "kbps": kbps, "width": w, "height": h, "frames": sizes}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--rgb", default="/euclid/euclid_zed/rgb/image_rect_color")
    ap.add_argument("--depth", default="/euclid/euclid_zed/depth/depth_registered")
    a = ap.parse_args(argv)
    t = time.time()
    rgb, depth = read_frames(a.bag, a.seconds, a.rgb, a.depth)
    print(f"read {len(rgb)} rgb, {len(depth)} depth ({rgb[0][1].shape}) in {time.time() - t:.0f}s", file=sys.stderr)
    traces = {
        "source": a.bag, "seconds": a.seconds, "rgb_shape": list(rgb[0][1].shape),
        "store": store_sizes(rgb, depth, kf_hz=1.4),
        "video": {
            name: video_sizes(rgb, fps, kbps, w)
            for name, (fps, kbps, w) in {
                "low": (10, 150, 480), "med": (15, 400, 640), "high": (15, 1000, 640), "tab": (5, 300, 640),
            }.items()
        },
    }
    json.dump(traces, open(a.out, "w"))
    s = traces["store"]
    print(f"store keyframes: {len(s)}, mean blob {np.mean([x['blob'] for x in s]) / 1e3:.1f} kB "
          f"(rgb {np.mean([x['rgb'] for x in s]) / 1e3:.1f} + depth {np.mean([x['depth'] for x in s]) / 1e3:.1f})", file=sys.stderr)  # fmt: skip
    for k, v in traces["video"].items():
        n = sum(f["n"] for f in v["frames"])
        print(f"video {k}: {len(v['frames'])} frames, {8 * n / a.seconds / 1e3:.0f} kbit/s actual "
              f"(target {v['kbps']}), max frame {max(f['n'] for f in v['frames']) / 1e3:.1f} kB", file=sys.stderr)  # fmt: skip


if __name__ == "__main__":
    main()
