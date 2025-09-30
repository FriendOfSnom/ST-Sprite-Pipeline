# face_detect_ssd.py
from __future__ import annotations
import os
from typing import Optional, Tuple

import cv2
import numpy as np

# Our refactored FaceBoxes wrapper (PyTorch .pth)
from third_party.anime_face_boxes import detect_boxes_bgr

# Optional: only used to decide whether to try CUDA first
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Where your weights live
SSD_WEIGHTS_PATH = os.path.join("models", "ssd_anime_face_detect.pth")

# Thresholds (tweak here if needed)
CONF_THRESH = 0.10
NMS_THRESH  = 0.30

# Haar fallback
_HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def _haar_detect(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Largest Haar face; returns (x, y, w, h) or None."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
    x, y, w, h = map(int, faces[0])
    return (x, y, w, h)


def _try_faceboxes(image_bgr: np.ndarray, device: str) -> Optional[Tuple[int, int, int, int]]:
    """
    Try FaceBoxes on the requested device ("cuda" or "cpu").
    This is resilient to older wrappers that don't accept 'device'/'fp16'.
    """
    if not os.path.isfile(SSD_WEIGHTS_PATH):
        return None

    kwargs = dict(
        image_bgr=image_bgr,
        weights_path=SSD_WEIGHTS_PATH,
        conf_thresh=CONF_THRESH,
        nms_thresh=NMS_THRESH,
    )

    # First attempt: pass device (and fp16 on CUDA) if supported
    try:
        dets = detect_boxes_bgr(
            **kwargs,
            device=device,           # many wrappers support this
            fp16=(device == "cuda")  # safe hint; ignored if wrapper doesn't support
        )
    except TypeError:
        # Wrapper doesn't accept device/fp16 → call without them
        dets = detect_boxes_bgr(**kwargs)
    except Exception:
        # Any other runtime error on this device → give up on this device
        return None

    if not dets:
        return None

    x1, y1, x2, y2, sc = dets[0]
    x = int(round(x1)); y = int(round(y1))
    w = max(0, int(round(x2 - x1))); h = max(0, int(round(y2 - y1)))
    if w > 0 and h > 0:
        return (x, y, w, h)
    return None


def detect_face_box_bgr(image_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect a single best anime face in a BGR image using FaceBoxes weights.
    Tries CUDA first (if available), then falls back to CPU, then to Haar.
    Returns (x, y, w, h) or None.
    """
    # Decide whether to try CUDA first
    want_cuda = _HAS_TORCH and torch.cuda.is_available()

    # 1) FaceBoxes on CUDA
    if want_cuda:
        box = _try_faceboxes(image_bgr, device="cuda")
        if box is not None:
            return box

    # 2) FaceBoxes on CPU (fallback)
    box = _try_faceboxes(image_bgr, device="cpu")
    if box is not None:
        return box

    # 3) Haar (final fallback)
    return _haar_detect(image_bgr)
