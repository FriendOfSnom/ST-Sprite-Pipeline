# pose_backend_yolov8.py
"""
Thin wrapper around Ultralytics YOLOv8-Pose to get body keypoints.
We expose: estimate_keypoints(image_bgr) -> np.ndarray of shape (K,2) in pixel coords or None
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from ultralytics import YOLO

# Load once (smallest pose model; good enough on CPU)
# You can swap "yolov8n-pose.pt" -> "yolov8s-pose.pt" if you want a bit more accuracy.
_MODEL = YOLO("yolov8n-pose.pt")

def estimate_keypoints(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Run YOLOv8-Pose and return the best person's keypoints as (K,2) array.
    Returns None if nothing is detected.
    """
    # Ultralytics expects RGB
    rgb = image_bgr[:, :, ::-1]
    results = _MODEL.predict(source=rgb, verbose=False, imgsz=max(256, max(rgb.shape[:2])))
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None
    # Take the highest-conf instance
    kps_xy = results[0].keypoints.xy[0].cpu().numpy()  # (K,2), float32
    return kps_xy
