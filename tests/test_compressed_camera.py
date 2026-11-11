"""Compressed (JPEG/PNG) camera frames: stored undecoded, decoded on request."""

import numpy as np
import pytest

from fleetlink import media

cv2 = pytest.importorskip("cv2")


def _jpeg(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def test_compressed_encoding_from_format():
    assert media.compressed_encoding("rgb8; jpeg compressed bgr8") == "jpeg"
    assert media.compressed_encoding("bgr8; png compressed bgr8") == "png"
    assert media.compressed_encoding("") == "jpeg"


def test_jpeg_frame_decodes_to_rgb():
    rgb = np.zeros((48, 64, 3), np.uint8)
    rgb[..., 0] = 200  # red
    img = media.image_to_array("jpeg", 0, 0, 0, _jpeg(rgb))
    assert img.shape == (48, 64, 3)
    assert abs(int(img[..., 0].mean()) - 200) < 6 and img[..., 2].mean() < 10


def test_camera_reply_reports_decoded_size():
    rgb = np.full((48, 64, 3), 90, np.uint8)
    frame = {"encoding": "jpeg", "width": 0, "height": 0, "step": 0, "data": _jpeg(rgb), "stamp": 1.0, "rx_t": 1.0}
    reply = media.camera_reply(frame, max_w=32, quality=80, now=2.0)
    assert (reply["src_width"], reply["src_height"]) == (64, 48)
    assert (reply["width"], reply["height"]) == (32, 24)
    assert reply["age_s"] == pytest.approx(1.0)
