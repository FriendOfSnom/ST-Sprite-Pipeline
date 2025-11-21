#!/usr/bin/env python3
"""
auto_sort_helper.py

Step 2 replacement: fully automatic grouping of downloaded sprite images
by Character -> Pose -> Outfit, IN PLACE, inside the given folder.

USAGE:
    python auto_sort_helper.py /path/to/unsorted_folder

Behavior:
- Takes exactly one argument: the folder containing all unsorted images.
- Reorganizes those image files in place into:
      <root>/
        char_000/
          pose_000/
            outfit_000/
              <moved images>
            outfit_001/
              ...
          pose_001/
            ...
        char_001/
          ...
- Prints the folder path to stdout at the end (for Step 3 to consume).
- All thresholds and options are defined in the CONFIG block below.

Notes:
- Face detection: uses face_detect_ssd.detect_face_box_bgr() (SSD first, Haar fallback).
- Pose detection: stub that you can later swap with a real keypoint backend.
- Outfit detection: uses ONLY the silhouette “ray signature” (matches user spec).

Author: Auto Sprite Tools
"""

from __future__ import annotations
import os
import sys
import cv2
import glob
import math
import shutil
import random
import hashlib
import pathlib
import traceback
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from dataclasses import dataclass
from PIL import Image
from pose_backend_yolov8 import estimate_keypoints

# Local SSD adapter
from face_detect_ssd import detect_face_box_bgr


# =========================
# ====== CONFIG BLOCK =====
# =========================

CONFIG = {
    "ENABLE_POSE": False,
    "ENABLE_OUTFIT": False,

    # Hair band sampling (fractions of face box height/width)
    "HAIR_BAND_TOP": 0.40,         # farthest above face top
    "HAIR_BAND_BOTTOM": 0.12,      # nearest above face top
    "HAIR_BAND_PAD_X": 0.04,       # widen band horizontally by % face width

    "HAIR_SAMPLES": 800,           # deterministic pseudo-random samples in band
    "BG_DELTA_RGB": 10,             # drop samples too close to background (per-channel)
    "DROP_NEAR_WHITE": 245,        # drop highlights (all channels >= this)
    
    # character post-merge
    "CHAR_MIN_CLUSTER": 5,        # any character cluster smaller than this...
    "CHAR_ABSORB_DELTAE": 24.0,   # ...gets absorbed into nearest big one if ΔE <= this
    
    # hair sampling refinement
    "HAIR_MIN_SAT": 12,           # drop low-saturation samples (0..255 scale)

    # Character clustering (simple RGB distance)
    "CHAR_DELTAE_THRESH": 12.0,     # ≈ gentle tolerance in Lab
    "CHAR_SMALL_JOIN": 16.5,        # allow tiny clusters to merge
    "HAIR_QUANT": 8,                # quantize Lab to 6 steps per channel for stability


    # Pose clustering (fallback silhouette-PCA angles; replace later with keypoints)
    "POSE_EPS": 0.32,    # start here; if under-clustering increase slightly, if over-clustering decrease
    "POSE_MIN_SAMPLES": 3,
    "POSE_MIN_CLUSTER": 6,      # NEW: enforce minimum cluster size
    "POSE_ABSORB_EPS": 0.42,    # NEW: small clusters merge into nearest within this

    # Outfit clustering via silhouette ray signature
    "RAYS_HORIZONTAL": 64,         # per side (left & right)
    "RAYS_VERTICAL": 32,           # per side (top & bottom). Set 0 to disable vertical rays
    "OUTFIT_EPS": 0.025,           # threshold for outfit clustering (L1/cosine-like)
    "OUTFIT_MIN_SAMPLES": 3,

    # Foreground mask thresholds
    "MASK_BG_DELTA": 12,           # pixel is foreground if any |channel diff| >= this

    # File types considered images
    "IMAGE_EXTS": {".png", ".jpg", ".jpeg", ".webp", ".bmp"},

    # Debug helpers
    "DEBUG_HAIR_PER_IMAGE": True,   # one line per image with hair Lab & label
    "DEBUG_HAIR_SAMPLER": False,     # one liner from inside _sample_hair_rgb
}


# =========================
# ====== UTILITIES ========
# =========================

def _iter_images(root: pathlib.Path) -> List[pathlib.Path]:
    """Return a flat list of image files inside root (non-recursive initially)."""
    files = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in CONFIG["IMAGE_EXTS"]:
            files.append(p)
    # If the folder already has subfolders from a previous run, include loose files only.
    return files


def _read_bgr(path: pathlib.Path) -> Optional[np.ndarray]:
    """
    Read image with cv2 (unchanged). Can return 1c (gray), 3c (BGR), or 4c (BGRA).
    Returns None on failure.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img

def _split_channels(img: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Ensure we have a 3-channel BGR image and optional alpha.
    Returns (bgr, alpha) where:
      - bgr is HxWx3 uint8
      - alpha is HxW uint8 or None
    """
    if img is None:
        return None, None  # type: ignore

    if img.ndim == 2:
        # grayscale -> BGR
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return bgr, None
    if img.shape[2] == 3:
        return img, None
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        return bgr, alpha
    # Unexpected channel count — coerce to BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) if img.shape[2] > 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return bgr, None


def _corner_patch_means(bgr: np.ndarray, k: int = 8) -> List[Tuple[int, int, int]]:
    """
    Return rounded mean BGR of the four corner kxk patches that exist.
    """
    h, w = bgr.shape[:2]
    patches = []
    ys = [0, max(0, h - k)]
    xs = [0, max(0, w - k)]
    for yy in ys:
        for xx in xs:
            patch = bgr[yy:yy+k, xx:xx+k, :]
            if patch.size == 0:
                continue
            mean_bgr = np.mean(patch.reshape(-1, 3), axis=0)
            patches.append(tuple(int(round(v)) for v in mean_bgr))
    return patches

def _ab_chroma(lab: Tuple[float,float,float]) -> float:
    _, a, b = lab
    return float(math.hypot(a, b))

def _is_whiteish_lab(lab: Tuple[float, float, float]) -> bool:
    L, a, b = lab
    # stricter: whites are very bright and extremely low chroma
    return (_ab_chroma(lab) < 10.0) and (L > 72.0)

def _is_blondeish_lab(lab: Tuple[float, float, float]) -> bool:
    L, a, b = lab
    c = _ab_chroma(lab)
    # modest chroma, yellowish (b positive), red/green near zero
    return (15.0 <= c <= 45.0) and (b > 8.0) and (abs(a) < 15.0) and (L > 50.0)

def _label_hair_lab(lab: Tuple[float, float, float]) -> str:
    """
    Coarse hair label from a single Lab value:
    - 'white'  : very bright and extremely low chroma
    - 'blonde' : light, yellow-tilted, low/medium chroma
    - 'other'  : everything else
    """
    L, a, b = lab
    C = _ab_chroma(lab)
    if (C < 10.0) and (L > 72.0):
        return "white"
    if (L > 50.0) and (b > 6.0) and (C >= 8.0) and (C <= 60.0) and (a > -6.0) and (a <= 36.0):
        return "blonde"

    return "other"



def _char_merge_thresh(lab1, lab2):
    c1 = _ab_chroma(lab1); c2 = _ab_chroma(lab2)
    cmin = min(c1, c2)

    # Harder barrier between blonde-ish and white-ish during greedy/merge
    is_w1, is_w2 = _is_whiteish_lab(lab1), _is_whiteish_lab(lab2)
    is_b1, is_b2 = _is_blondeish_lab(lab1), _is_blondeish_lab(lab2)
    if (is_w1 and is_b2) or (is_w2 and is_b1):
        # only allow if they are *very* close; otherwise keep them separate
        return 9.5

    if cmin >= 40.0:     # vivid colors
        return 10.5
    if cmin >= 25.0:     # medium chroma
        return 12.5
    if cmin >= 15.0:     # soft/pastel
        return 14.5
    # very low chroma (blonde/white/silver/gray)
    return 18.8


def _estimate_background_rgb(img_any: np.ndarray) -> Tuple[int, int, int]:
    """
    Estimate background color.
    - If alpha channel exists and has transparent pixels in corners, use the mean BGR of those transparent corner pixels.
    - Else, use the mode of the four corner means (kxk) from the BGR image.
    """
    bgr, alpha = _split_channels(img_any)
    h, w = bgr.shape[:2]

    # If alpha exists, try to grab transparent corner pixels
    if alpha is not None:
        k = 12
        vals = []
        corners = [(0, 0), (0, max(0, w - k)), (max(0, h - k), 0), (max(0, h - k), max(0, w - k))]
        for (yy, xx) in corners:
            a = alpha[yy:yy+k, xx:xx+k]
            if a.size == 0:
                continue
            mask = (a == 0)
            if np.any(mask):
                patch = bgr[yy:yy+k, xx:xx+k, :]
                sel = patch[mask]
                if sel.size > 0:
                    vals.append(np.mean(sel.reshape(-1, 3), axis=0))
        if vals:
            m = np.mean(np.stack(vals, axis=0), axis=0)
            return (int(round(m[0])), int(round(m[1])), int(round(m[2])))

    # Fallback: mode of corner means
    means = _corner_patch_means(bgr, k=8)
    if not means:
        return (0, 0, 0)
    counts: Dict[Tuple[int, int, int], int] = {}
    for t in means:
        counts[t] = counts.get(t, 0) + 1
    bg = max(counts.items(), key=lambda kv: kv[1])[0]
    return (int(bg[0]), int(bg[1]), int(bg[2]))

def _rgb_to_lab(rgb: Tuple[int,int,int]) -> Tuple[float,float,float]:
    bgr = np.array([[ [rgb[2], rgb[1], rgb[0]] ]], dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0,0]
    # OpenCV L in [0..255], a/b in [0..255] with 128 as zero; normalize lightly
    L, a, b = lab[0] * (100.0/255.0), (lab[1]-128.0), (lab[2]-128.0)
    return (float(L), float(a), float(b))

def _delta_e_abL(lab1: Tuple[float,float,float], lab2: Tuple[float,float,float],
                 w_ab: float = 1.0, w_L: float = 0.35) -> float:
    # Weight a/b (chromaticity) more than L (lightness). Great for white/very light hair.
    dL = (lab1[0]-lab2[0]) * w_L
    da = (lab1[1]-lab2[1]) * w_ab
    db = (lab1[2]-lab2[2]) * w_ab
    return float(math.sqrt(dL*dL + da*da + db*db))


def _delta_e(lab1: Tuple[float,float,float], lab2: Tuple[float,float,float]) -> float:
    # Simple ΔE*ab; good enough for clustering thresholds
    dL = lab1[0]-lab2[0]; da = lab1[1]-lab2[1]; db = lab1[2]-lab2[2]
    return float(math.sqrt(dL*dL + da*da + db*db))

def _quantize_lab(lab: Tuple[float,float,float], steps: int) -> Tuple[float,float,float]:
    # Uniform quantization to reduce flicker under lighting
    qL = round(lab[0] / (100.0/steps)) * (100.0/steps)
    qa = round(lab[1] / (256.0/steps)) * (256.0/steps)
    qb = round(lab[2] / (256.0/steps)) * (256.0/steps)
    return (qL, qa, qb)


def _seed_from_path(path: pathlib.Path) -> int:
    """Stable pseudo-random seed derived from file path."""
    h = hashlib.blake2b(str(path).encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFF

def _is_skin_bgr(b: int, g: int, r: int) -> bool:
    # Quick YCrCb skin test
    ycrcb = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2YCrCb)[0,0]
    Y, Cr, Cb = int(ycrcb[0]), int(ycrcb[1]), int(ycrcb[2])
    return (Cr > 135 and Cb > 85 and Y > 80 and Cr <= (1.5862*Cb + 20) and
            Cr >= (0.3448*Cb + 76.2069) and Cr >= (-4.5652*Cb + 234.5652) and
            Cr <= (-1.15*Cb + 301.75))

def _is_skin_consensus(b: int, g: int, r: int) -> bool:
    """
    Stricter (consensus) skin detection:
    - Classic YCrCb rule + a loose HSV gate.
    This avoids discarding pale-blonde yellows as 'skin'.
    """
    ycrcb = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2YCrCb)[0,0]
    Y, Cr, Cb = int(ycrcb[0]), int(ycrcb[1]), int(ycrcb[2])
    ycrcb_skin = (Cr > 135 and Cb > 85 and Y > 80 and
                  Cr <= (1.5862*Cb + 20) and
                  Cr >= (0.3448*Cb + 76.2069) and
                  Cr >= (-4.5652*Cb + 234.5652) and
                  Cr <= (-1.15*Cb + 301.75))

    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0,0]
    H, S, V = int(hsv[0]), int(hsv[1]), int(hsv[2])
    hsv_skin = (10 <= H <= 25) and (30 <= S <= 150) and (V >= 60)

    return ycrcb_skin and hsv_skin


def _sample_hair_rgb(img_bgr: np.ndarray, face: Tuple[int,int,int,int], bg_rgb: Tuple[int,int,int],
                     n_samples: int, band_top: float, band_bottom: float, pad_x: float,
                     seed: int, drop_near_white: int, bg_delta: int,
                     fg_mask: Optional[np.ndarray] = None) -> Optional[Tuple[int,int,int]]:
    """
    Robust hair color sampler:
    - Samples a top band above the face + narrow temple strips.
    - Filters background, skin (consensus), and very bright highlights.
    - Fuses samples in Lab with a trimmed median.
    - If clear 'yellowish blonde' evidence exists, bias to those pixels and
      prevent collapsing to 'white'.
    """
    x, y, w, h = face
    H, W = img_bgr.shape[:2]
    rng = random.Random(seed)

    def _collect_from_rect(x0, y0, x1, y1, want_sat_thresh: Optional[int], n_try: int) -> List[Tuple[int,int,int]]:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return []
        vals: List[Tuple[int,int,int]] = []
        for _ in range(n_try):
            yy = rng.randint(y0, y1 - 1)
            xx = rng.randint(x0, x1 - 1)
            if fg_mask is not None and fg_mask[yy, xx] == 0:
                continue
            b, g, r = img_bgr[yy, xx][:3]

            # reject background-like pixels (with a tiny blonde carve-out)
            bg_b, bg_g, bg_r = bg_rgb
            yellow_bias = (int(r) - int(bg_r)) * 0.5 + (int(g) - int(bg_g)) * 0.5 - (int(b) - int(bg_b))
            if (abs(int(b) - bg_b) < bg_delta and
                abs(int(g) - bg_g) < bg_delta and
                abs(int(r) - bg_r) < bg_delta) and yellow_bias <= 1:
                continue

            # reject obvious skin (consensus gate)
            if _is_skin_consensus(b, g, r):
                continue

            # reject extreme highlights; allow near-white so white hair still works
            if b >= drop_near_white and g >= drop_near_white and r >= drop_near_white:
                continue

            if want_sat_thresh is not None:
                mx, mn = max(r,g,b), min(r,g,b)
                if (mx - mn) < want_sat_thresh:
                    continue

            vals.append((int(r), int(g), int(b)))
        return vals

    samples: List[Tuple[int,int,int]] = []

    # Zone A: top band over the face
    y_top = max(0, int(round(y - band_top * h)))
    y_bot = max(0, int(round(y - band_bottom * h)))
    x0 = max(0, int(round(x - pad_x * w)))
    x1 = min(W, int(round(x + (1.0 + pad_x) * w)))
    samples += _collect_from_rect(x0, y_top, x1, y_bot, CONFIG["HAIR_MIN_SAT"], n_samples)

    # Zones B/C: temple strips (slightly above mid-face)
    ty0 = max(0, int(round(y - 0.10 * h)))
    ty1 = max(0, int(round(y + 0.10 * h)))
    temple_w = max(1, int(round(0.10 * w)))
    lx0 = max(0, x - int(round(0.15 * w))); lx1 = min(W, lx0 + temple_w)
    rx1 = min(W, x + w + int(round(0.15 * w))); rx0 = max(0, rx1 - temple_w)
    samples += _collect_from_rect(lx0, ty0, lx1, ty1, None, n_samples // 4)
    samples += _collect_from_rect(rx0, ty0, rx1, ty1, None, n_samples // 4)

    if not samples:
        return None

    # Build Lab arrays BEFORE trimming so we can inspect blonde evidence
    labs_all = np.array([_rgb_to_lab((r,g,b)) for (r,g,b) in samples], dtype=np.float32)

    # Trim by chroma to drop extreme highlights/shadows
    labs = labs_all.copy()
    
    # Additional guard: drop ultra-low-chroma (C*ab < 6) samples that often come from halos
    C_full = np.hypot(labs[:,1], labs[:,2])
    keep_c = C_full >= 6.0
    if keep_c.any():
        labs = labs[keep_c]

    chroma = np.hypot(labs[:,1], labs[:,2])
    if labs.shape[0] >= 20:
        lo = np.percentile(chroma, 15.0)
        hi = np.percentile(chroma, 85.0)
        keep = (chroma >= lo) & (chroma <= hi)
        labs = labs[keep] if keep.any() else labs

    # Detect “yellowish blonde” on the FULL set (pre-trim)
    La, aa, bb = labs_all[:,0], labs_all[:,1], labs_all[:,2]
    Ca = np.hypot(aa, bb)
    yellowish_mask_all = (
        (La >= 50.0) &
        (bb >= 6.0) &
        (np.abs(aa) <= 22.0) &
        (Ca >= 6.0) & (Ca <= 48.0)
    )
    n_yellow_all = int(yellowish_mask_all.sum())

    # If there’s a meaningful blonde signal, bias to it
    if n_yellow_all >= max(12, int(0.06 * labs_all.shape[0])):  # ≥12 px or ≥6%
        labs = labs_all[yellowish_mask_all]
        if labs.shape[0] >= 20:
            Cb = np.hypot(labs[:,1], labs[:,2])
            lo = np.percentile(Cb, 10.0); hi = np.percentile(Cb, 90.0)
            keep = (Cb >= lo) & (Cb <= hi)
            labs = labs[keep] if keep.any() else labs

    med = np.median(labs, axis=0)

    # Anti-white override if blonde evidence exists
    def _is_whiteish_lab_strict(Lab):
        L, a, b = float(Lab[0]), float(Lab[1]), float(Lab[2])
        return (np.hypot(a, b) < 10.0) and (L > 72.0)

    if _is_whiteish_lab_strict(med) and n_yellow_all >= max(12, int(0.06 * labs_all.shape[0])):
        cand = labs_all[yellowish_mask_all]
        if cand.size:
            med = np.median(cand, axis=0)

    # Convert back to RGB and return
    L, a, b = med
    lab_img = np.array([[[L * (255.0/100.0), a + 128.0, b + 128.0]]], dtype=np.float32)
    bgr = cv2.cvtColor(lab_img.astype(np.uint8), cv2.COLOR_Lab2BGR)[0,0]
    
    if CONFIG.get("DEBUG_HAIR_SAMPLER", False):
        # counts on sampled pixels before/after blonde bias
        C_all = np.hypot(labs_all[:,1], labs_all[:,2])
        L_all = labs_all[:,0]
        pct = lambda arr,p: float(np.percentile(arr, p)) if arr.size else float('nan')
        print(
            "HAUD",
            f"n_all={labs_all.shape[0]}",
            f"L50={pct(L_all,50):.1f}",
            f"C50={pct(C_all,50):.1f}",
            f"yellowish={int(yellowish_mask_all.sum())}",
            f"picked_n={labs.shape[0]}",
        )

    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))




def _lab_hue_chroma_dist(lab1: Tuple[float,float,float], lab2: Tuple[float,float,float]) -> float:
    """
    Hybrid distance that separates hair colors better than plain ΔE:
    - Strong weight on a/b (chromaticity) vector difference
    - Add angular (hue) difference in ab-plane
    - Add chroma (C*ab) gap
    - Low weight on L (lightness)
    Returns a scalar "distance" (lower is more similar).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    # ab vector distance
    d_ab = math.hypot(a1 - a2, b1 - b2)  # ~0..180ish
    # hue angle (in radians) on ab plane
    h1 = math.atan2(b1, a1)
    h2 = math.atan2(b2, a2)
    d_h = abs(h1 - h2)
    if d_h > math.pi:
        d_h = 2 * math.pi - d_h
    d_h_deg = d_h * (180.0 / math.pi)     # 0..180
    # chroma difference
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    d_c = abs(c1 - c2)
    # slight L weight (to avoid merging radically different brightness whites/grays)
    d_L = abs(L1 - L2)

    # Weighted sum (tuned for hair): ab dominates, then hue angle, then chroma, then L
    return 0.70 * d_ab + 0.20 * d_h_deg + 0.08 * d_c + 0.02 * d_L

def _rgb_dist(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    """Euclidean distance in RGB."""
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _binary_mask(img_any: np.ndarray, bg_rgb: Tuple[int, int, int], thresh: int) -> np.ndarray:
    """
    Foreground mask:
    - If alpha channel exists, use alpha > 0.
    - Else, difference from estimated background color.
    Returns uint8 mask {0,255}.
    """
    bgr, alpha = _split_channels(img_any)
    if alpha is not None:
        mask = (alpha > 0).astype(np.uint8) * 255
        return mask

    diff = np.max(np.abs(bgr.astype(np.int16) - np.array(bg_rgb, dtype=np.int16).reshape(1,1,3)), axis=2)
    mask = (diff >= thresh).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return mask

def _bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Tight bbox of non-zero mask; returns (x, y, w, h) or None if empty."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)

# --- begin keypoint pose descriptor helpers ---
def _normalize_keypoints(kps: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints for translation and scale, and roughly for rotation:
    - Translate so mid-hip is at origin (if hips present; else mean).
    - Scale so shoulder-to-hip distance is ~1 (fallback to torso bbox size).
    - Rotate so shoulders are approximately horizontal if both visible.
    """
    pts = kps.copy().astype(np.float32)
    valid = ~np.isnan(pts[:, 0])
    if not np.any(valid):
        return pts * 0.0

    # Translation
    center = np.nanmean(pts, axis=0)
    pts -= center

    # Scale (shoulder-hip if possible)
    # YOLOv8-Pose order commonly: 0:nose, 5:right_shoulder,6:right_elbow,7:right_wrist, 11:right_hip,12:right_knee...
    # But models vary; we’ll approximate using left/right shoulders (5,6 in some layouts) and hips (11,12).
    def _safe_pair_mean(a, b):
        if a < len(pts) and b < len(pts):
            pa, pb = pts[a], pts[b]
            if not (np.any(np.isnan(pa)) or np.any(np.isnan(pb))):
                return (pa + pb) / 2.0
        return np.array([np.nan, np.nan], dtype=np.float32)

    # mid-shoulder and mid-hip
    mid_sh = _safe_pair_mean(5, 6)
    mid_hp = _safe_pair_mean(11, 12)

    torso = mid_sh - mid_hp if not (np.any(np.isnan(mid_sh)) or np.any(np.isnan(mid_hp))) else None
    scale = np.linalg.norm(torso) if torso is not None else np.nan
    if not (np.isfinite(scale) and scale > 1e-3):
        # fallback: overall std
        scale = np.nanstd(pts)
    if not (np.isfinite(scale) and scale > 1e-3):
        scale = 1.0
    pts /= scale

    # Rotation: make shoulders roughly horizontal if both shoulders exist
    if not (np.any(np.isnan(mid_sh)) or np.any(np.isnan(mid_hp))):
        v = mid_sh - mid_hp
        ang = np.arctan2(v[1], v[0])
        c, s = np.cos(-ang), np.sin(-ang)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = pts @ R.T

    return pts

def _pose_descriptor_from_keypoints(kps_xy: np.ndarray) -> np.ndarray:
    """
    Robust pose descriptor:
    - normalize (translate/scale/roughly rotate)
    - limb directions as coarse bins (quantized angles)
    - limb lengths lightly clipped and quantized
    """
    pts = _normalize_keypoints(kps_xy)

    def v(a,b):
        if a >= len(pts) or b >= len(pts):
            return np.array([np.nan, np.nan], dtype=np.float32)
        return pts[b] - pts[a]

    limbs = [
        (5, 7), (6, 8),      # upper limbs to wrists
        (11, 13), (12, 14),  # legs to ankles
        (5, 6), (11, 12),    # shoulder/hip baseline
        (0, 5), (0, 6),      # nose->shoulders
    ]
    feat = []
    for a,b in limbs:
        d = v(a,b)
        n = np.linalg.norm(d)
        if not (np.isfinite(n) and n > 1e-6):
            ang_bin = 0
            len_bin = 0
        else:
            ang = np.arctan2(d[1], d[0])  # [-pi, pi]
            # quantize angle to 12 bins
            ang_bin = int(np.floor(((ang + np.pi) / (2*np.pi)) * 12.0 + 0.5)) % 12
            # quantize length to 8 bins after mild cap
            n = min(1.8, n)
            len_bin = int(np.floor((n / 1.8) * 8.0 + 0.5))
        feat.append(ang_bin); feat.append(len_bin)
    return np.array(feat, dtype=np.float32)  # 16-dim



def _ray_signature(mask: np.ndarray, bbox: Tuple[int,int,int,int], rays_h: int, rays_v: int) -> np.ndarray:
    """
    Compute silhouette ray signature:
    - From left and right edges (rays_h each), record normalized first-hit distances.
    - Optionally from top and bottom (rays_v each).
    Returns a 1D float32 vector.
    """
    x, y, w, h = bbox
    eps = 1e-6
    sig: List[float] = []

    def first_hit_row(row_y: int, from_left: bool) -> float:
        row = mask[row_y, x:x+w]
        if from_left:
            idx = np.argmax(row > 0)
            if row[idx] == 0:
                return 1.0
            return (idx + 1) / (w + eps)
        else:
            row_rev = row[::-1]
            idx = np.argmax(row_rev > 0)
            if row_rev[idx] == 0:
                return 1.0
            return (idx + 1) / (w + eps)

    def first_hit_col(col_x: int, from_top: bool) -> float:
        col = mask[y:y+h, col_x]
        if from_top:
            idx = np.argmax(col > 0)
            if col[idx] == 0:
                return 1.0
            return (idx + 1) / (h + eps)
        else:
            col_rev = col[::-1]
            idx = np.argmax(col_rev > 0)
            if col_rev[idx] == 0:
                return 1.0
            return (idx + 1) / (h + eps)

    # Horizontal rays (left/right)
    if rays_h > 0:
        for i in range(rays_h):
            ry = y + int(round((i + 0.5) * h / rays_h))
            sig.append(first_hit_row(ry, from_left=True))
        for i in range(rays_h):
            ry = y + int(round((i + 0.5) * h / rays_h))
            sig.append(first_hit_row(ry, from_left=False))

    # Vertical rays (top/bottom)
    if rays_v > 0:
        for i in range(rays_v):
            rx = x + int(round((i + 0.5) * w / rays_v))
            sig.append(first_hit_col(rx, from_top=True))
        for i in range(rays_v):
            rx = x + int(round((i + 0.5) * w / rays_v))
            sig.append(first_hit_col(rx, from_top=False))

    # Small median smoothing to reduce single-pixel noise
    arr = np.array(sig, dtype=np.float32)
    if arr.size >= 3:
        arr = cv2.medianBlur((arr * 10000).astype(np.uint16), 3).astype(np.float32) / 10000.0
    return arr


def _l1_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference; handles different lengths defensively (use min length)."""
    L = min(a.size, b.size)
    if L == 0:
        return 1e9
    return float(np.mean(np.abs(a[:L] - b[:L])))

def _maybe_split_bimodal_ab(group: List[ImgInfo],
                            split_thresh: float = 12.0,
                            min_size: int = 6) -> List[List[ImgInfo]]:
    """
    If a group's (a,b) values look bimodal, split it into 2 using tiny k-means.
    - split_thresh: minimum centroid separation in (a,b) to allow a split
    - min_size: each side must have at least this many items
    Returns 1 or 2 groups.
    """
    labs = [ _rgb_to_lab(g.hair_rgb) for g in group if g.hair_rgb is not None ]
    
    # Avoid splitting low-chroma (blonde-ish) groups: they vary mostly by lighting
    if labs:
        centroid = tuple(np.mean(np.array(labs, dtype=np.float32), axis=0))
        C = math.hypot(centroid[1], centroid[2])
        if C < 26.0:                 # low/medium chroma -> don't split
            return [group]

    if len(labs) < 2 * min_size:
        return [group]
    ab = np.array([[a, b] for (_, a, b) in labs], dtype=np.float32)

    # k=2 means (init by extremes)
    i0 = int(np.argmin(ab[:,0]**2 + ab[:,1]**2))  # lowest chroma
    i1 = int(np.argmax(ab[:,0]**2 + ab[:,1]**2))  # highest chroma
    c0, c1 = ab[i0].copy(), ab[i1].copy()
    for _ in range(10):
        d0 = np.linalg.norm(ab - c0, axis=1)
        d1 = np.linalg.norm(ab - c1, axis=1)
        m0 = d0 <= d1
        m1 = ~m0
        if m0.sum() == 0 or m1.sum() == 0:
            return [group]
        c0_new = ab[m0].mean(axis=0)
        c1_new = ab[m1].mean(axis=0)
        if np.allclose(c0, c0_new) and np.allclose(c1, c1_new):
            break
        c0, c1 = c0_new, c1_new

    sep = float(np.linalg.norm(c0 - c1))
    if sep < split_thresh:
        return [group]
    # build the two groups in original item order
    g0, g1 = [], []
    # re-eval assignment to match labs ordering
    for idx, info in enumerate(group):
        if info.hair_rgb is None:
            # keep low-info items with the larger side
            (g0 if len(g0) >= len(g1) else g1).append(info)
            continue
        a, b = _rgb_to_lab(info.hair_rgb)[1:]
        if np.linalg.norm(np.array([a, b]) - c0) <= np.linalg.norm(np.array([a, b]) - c1):
            g0.append(info)
        else:
            g1.append(info)

    if len(g0) >= min_size and len(g1) >= min_size:
        return [g0, g1]
    return [group]

def _split_white_clusters_to_pale_blonde(
    clusters: List[Tuple[Tuple[float,float,float], List["ImgInfo"]]],
    *, min_size: int = 6, min_fraction: float = 0.10
) -> List[Tuple[Tuple[float,float,float], List["ImgInfo"]]]:
    """
    For any cluster whose centroid is white-ish, split out a 'pale blonde' subgroup
    if there are enough yellowish members to be a real cluster.
    """
    if not clusters:
        return clusters

    new_clusters: List[Tuple[Tuple[float,float,float], List["ImgInfo"]]] = []
    for c, g in clusters:
        if not _is_whiteish_lab(c) or len(g) < (min_size * 2):
            new_clusters.append((c, g))
            continue

        # classify members inside this white-ish cluster
        yellowish_members, whiteish_members = [], []
        for inf in g:
            if inf.hair_rgb is None:
                whiteish_members.append(inf); continue
            lab = _rgb_to_lab(inf.hair_rgb)
            # broader, per-pixel blonde candidate test
            L, a, b = lab
            C = math.hypot(a, b)
            is_yellowish = (L >= 55.0) and (b >= 8.0) and (abs(a) <= 18.0) and (8.0 <= C <= 38.0)
            if is_yellowish:
                yellowish_members.append(inf)
            else:
                whiteish_members.append(inf)

        if (len(yellowish_members) >= min_size and
            len(yellowish_members) >= int(min_fraction * len(g))):
            # split: keep the white remainder as one cluster, add a new pale-blonde cluster
            def _centroid(members):
                arr = np.array([_rgb_to_lab(x.hair_rgb) for x in members if x.hair_rgb is not None], dtype=np.float32)
                return tuple(np.mean(arr, axis=0)) if arr.size else c

            c_white = _centroid(whiteish_members)
            c_blond = _centroid(yellowish_members)
            # Guard: if the two centroids are nearly identical, don't split
            if _lab_hue_chroma_dist(c_white, c_blond) >= 6.0:
                if whiteish_members:
                    new_clusters.append((c_white, whiteish_members))
                if yellowish_members:
                    new_clusters.append((c_blond, yellowish_members))
                continue  # handled this cluster

        # no meaningful blonde subgroup -> keep as-is
        new_clusters.append((c, g))

    return new_clusters


def _final_low_chroma_merge(clusters):
    if len(clusters) <= 1:
        return clusters
    changed = True
    while changed:
        changed = False
        best = None
        best_d = 1e9
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci, gi = clusters[i]
                cj, gj = clusters[j]
                # Only merge when BOTH are truly white-ish
                if not (_is_whiteish_lab(ci) and _is_whiteish_lab(cj)):
                    continue
                d = _lab_hue_chroma_dist(ci, cj)
                # Allow fairly tight merge for whites (they often split on lighting)
                if d < best_d and d <= 12.0:  # try 12.0 first
                    best = (i, j)
                    best_d = d
        if best is not None:
            i, j = best
            ci, gi = clusters[i]
            cj, gj = clusters[j]
            merged = gi + gj
            arr = np.array([_rgb_to_lab(g.hair_rgb) for g in merged if g.hair_rgb is not None], dtype=np.float32)
            mean_lab = tuple(np.mean(arr, axis=0))
            clusters[i] = (mean_lab, merged)
            del clusters[j]
            changed = True
    return clusters

def _final_blonde_merge(clusters):
    if len(clusters) <= 1:
        return clusters

    def chroma(c): 
        return float(math.hypot(c[1], c[2]))

    # Wide net for blondes (includes warm/strawberry but excludes vivid red/orange)
    def is_blonde_centroid(c):
        L, a, b = c[0], c[1], c[2]
        C = chroma(c)
        # Heuristics:
        # - light-ish
        # - yellowish tilt (b positive), allow warm a (up to ~30)
        # - not white (C >= ~6) and not vivid (C <= ~55)
        return (L >= 48.0) and (b > 5.0) and (-5.0 <= a <= 36.0) and (6.0 <= C <= 60.0)
    # Step A: pairwise cleanup of very close blondes
    changed = True
    while changed:
        changed = False
        best = None
        best_d = 1e9
        for i in range(len(clusters)):
            ci, gi = clusters[i]
            if not is_blonde_centroid(ci):
                continue
            for j in range(i+1, len(clusters)):
                cj, gj = clusters[j]
                if not is_blonde_centroid(cj):
                    continue
                d = _lab_hue_chroma_dist(ci, cj)
                if d < best_d and d <= 17.0:   # slightly looser
                    best = (i, j); best_d = d
        if best is not None:
            i, j = best
            ci, gi = clusters[i]; cj, gj = clusters[j]
            merged = gi + gj
            arr = np.array([_rgb_to_lab(g.hair_rgb) for g in merged if g.hair_rgb is not None], dtype=np.float32)
            mean_lab = tuple(np.mean(arr, axis=0))
            clusters[i] = (mean_lab, merged)
            del clusters[j]
            changed = True

    # Step B: dominant blonde absorbs ALL other blondes until only one remains
    def recompute_centroid(group):
        arr = np.array([_rgb_to_lab(g.hair_rgb) for g in group if g.hair_rgb is not None], dtype=np.float32)
        return tuple(np.mean(arr, axis=0))

    while True:
        blonde_idxs = [k for k,(c,g) in enumerate(clusters) if is_blonde_centroid(c)]
        if len(blonde_idxs) <= 1:
            break
        dom = max(blonde_idxs, key=lambda k: len(clusters[k][1]))
        dom_c, dom_g = clusters[dom]
        absorbed_any = False
        for k in sorted(blonde_idxs, reverse=True):
            if k == dom:
                continue
            ck, gk = clusters[k]
            d = _lab_hue_chroma_dist(dom_c, ck)
            if d <= 22.0:  # decisive final absorb window
                dom_g += gk
                del clusters[k]
                dom_c = recompute_centroid(dom_g)
                clusters[dom] = (dom_c, dom_g)
                absorbed_any = True
        if not absorbed_any:
            break

    return clusters


def _expel_nonwhite_from_white_clusters(
    clusters: List[Tuple[Tuple[float,float,float], List["ImgInfo"]]],
    margin: float = 0.0
):
    """
    For any cluster whose centroid is white-ish, expel members that are not white-ish
    to the nearest non-white cluster if they fit notably better (by 'margin')
    under the _lab_hue_chroma_dist metric.
    """
    if len(clusters) <= 1:
        return clusters

    # snapshot centroids
    cents = [c for (c, _) in clusters]
    white_idxs = [i for i, c in enumerate(cents) if _is_whiteish_lab(c)]
    nonwhite_idxs = [i for i, c in enumerate(cents) if not _is_whiteish_lab(c)]
    if not white_idxs or not nonwhite_idxs:
        return clusters

    def best_nonwhite_idx(lab):
        best_j, best_d = None, 1e9
        for j in nonwhite_idxs:
            d = _lab_hue_chroma_dist(lab, cents[j])
            if d < best_d - 1e-6:
                best_d, best_j = d, j
            elif abs(d - best_d) <= 1e-6:
                # tie-breaker: prefer yellow-tilted centroid for blondes
                if lab[2] > 6.0:  # b positive -> blondeish
                    if cents[j][2] > (cents[best_j][2] if best_j is not None else -1e9):
                        best_d, best_j = d, j
        return best_j, best_d


    changed = False
    for wi in white_idxs:
        c_white, g_white = clusters[wi]
        keep: List[ImgInfo] = []
        reassign: Dict[int, List[ImgInfo]] = defaultdict(list)

        for inf in g_white:
            if inf.hair_rgb is None:
                keep.append(inf)
                continue
            lab = _rgb_to_lab(inf.hair_rgb)

            if _is_whiteish_lab(lab):
                keep.append(inf)
                continue

            d_white = _lab_hue_chroma_dist(lab, c_white)
            j, d_best = best_nonwhite_idx(lab)

            # Only expel if a non-white centroid is clearly better
            if j is not None and (d_best + margin) < d_white:
                reassign[j].append(inf)
                changed = True
            else:
                keep.append(inf)

        # write back
        clusters[wi] = (c_white, keep)
        for j, moved in reassign.items():
            clusters[j] = (cents[j], clusters[j][1] + moved)

    # Recompute centroids after moves
    new_clusters: List[Tuple[Tuple[float,float,float], List[ImgInfo]]] = []
    for _, grp in clusters:
        if not grp:
            continue
        arr = np.array([_rgb_to_lab(g.hair_rgb) for g in grp if g.hair_rgb is not None], dtype=np.float32)
        new_clusters.append((tuple(np.mean(arr, axis=0)), grp))

    return new_clusters

def _is_blonde_candidate_pixel_lab(lab: Tuple[float,float,float]) -> bool:
    L, a, b = lab
    C = _ab_chroma(lab)
    # broader than centroid test: catches individual pale-blonde samples
    return (55.0 <= L <= 82.0) and (b > 6.0) and (C >= 8.0) and (C <= 38.0) and (a > -6.0)

def _rescue_blondeish_from_white(
    clusters: List[Tuple[Tuple[float,float,float], List['ImgInfo']]]
) -> List[Tuple[Tuple[float,float,float], List['ImgInfo']]]:
    """
    Safely rescue blown-out blonde items from bright/low-chroma 'white-like' clusters
    without stealing from true white/silver characters.

    Rules (per item):
      - Item itself must NOT be white-ish.
      - Item must look blonde-tilted (per-pixel test).
      - Destination centroid must be blonde-ish.
      - Destination must beat white-like by a margin (>= 2.5) and be absolutely close (<= 22.0).
    """
    if not clusters:
        return clusters

    # Treat very bright, low-chroma centroids as 'white-like' for this rescue pass only.
    def _is_white_like_centroid(c: Tuple[float,float,float]) -> bool:
        L, a, b = c
        C = math.hypot(a, b)
        return (L > 72.0) and (C < 14.0)  # broader than _is_whiteish_lab(C<10), but only for this pass

    white_idxs = [i for i, (c, _) in enumerate(clusters) if _is_white_like_centroid(c)]
    nonwhite_idxs = [i for i in range(len(clusters)) if i not in white_idxs]
    if not white_idxs or not nonwhite_idxs:
        return clusters

    # Precompute non-white centroids
    nonwhite_cents: Dict[int, Tuple[float,float,float]] = {i: clusters[i][0] for i in nonwhite_idxs}

    moved_any = False
    for wi in white_idxs:
        c_white, g_white = clusters[wi]
        keep_list: List[ImgInfo] = []
        move_list: List[Tuple[ImgInfo, int]] = []

        for inf in g_white:
            if inf.hair_rgb is None:
                keep_list.append(inf)
                continue

            lab = _rgb_to_lab(inf.hair_rgb)

            # HARD GUARDS: don't move true white/silver; only rescue blonde-tilted items
            if _is_whiteish_lab(lab):                 # item itself looks white/silver
                keep_list.append(inf)
                continue
            if not _is_blonde_candidate_pixel_lab(lab):  # not yellow-tilted blonde-ish
                keep_list.append(inf)
                continue

            # Distances: white-like centroid vs best non-white centroid
            d_white = _lab_hue_chroma_dist(lab, c_white)
            best_j, best_d = None, 1e9
            for j, c in nonwhite_cents.items():
                d = _lab_hue_chroma_dist(lab, c)
                if d < best_d:
                    best_d, best_j = d, j

            # Destination must be blonde-ish too (prevents jumping into green/purple, etc.)
            if best_j is None or not _is_blondeish_lab(nonwhite_cents[best_j]):
                keep_list.append(inf)
                continue

            # Require clear margin and a sane absolute distance
            if (d_white - best_d) >= 2.5 and best_d <= 22.0:
                move_list.append((inf, best_j))
            else:
                keep_list.append(inf)

        # Apply moves (if any) and update the white-like cluster in place
        if move_list:
            moved_any = True
            clusters[wi] = (c_white, keep_list)  # write back remaining members
            for inf, j in move_list:
                clusters[j][1].append(inf)

    if moved_any:
        # Recompute centroids and drop empties
        new_clusters: List[Tuple[Tuple[float,float,float], List['ImgInfo']]] = []
        for c, g in clusters:
            if not g:
                continue
            arr = np.array([_rgb_to_lab(x.hair_rgb) for x in g if x.hair_rgb is not None], dtype=np.float32)
            new_c = tuple(np.mean(arr, axis=0)) if arr.size else c
            new_clusters.append((new_c, g))
        return new_clusters

    return clusters


def _final_hue_prox_merge(
    clusters,
    *,
    hue_tol_deg: float = 24.0,  # max centroid hue gap to consider “same family”
    dist_tol: float = 16.0,     # overall Lab/hue/chroma distance to allow merge
    min_chroma: float = 10.0,   # below this, hue is unreliable
):
    """
    Generic merge for color-families:
    - Never merges a white-ish cluster with a non-white cluster.
    - For chromatic clusters (C*ab >= min_chroma):
        merge closest pairs whose centroid hue differs by <= hue_tol_deg AND
        the overall hair-distance <= dist_tol.
    - For achromatic-but-not-white (grays/silvers with L <= ~72):
        merge by distance only (hue is unstable).
    Repeats greedily until no eligible pair remains.
    """
    if len(clusters) <= 1:
        return clusters

    def chroma(c): return float(math.hypot(c[1], c[2]))
    def hue_deg(c):
        # return 0..180 minimal hue angle vs another using helper below
        return math.degrees(math.atan2(c[2], c[1])) % 360.0

    def circ_delta_deg(a, b):
        d = abs(a - b) % 360.0
        return d if d <= 180.0 else 360.0 - d

    # Work on a mutable list
    cl = list(clusters)

    changed = True
    while changed and len(cl) > 1:
        changed = False
        best_i = best_j = -1
        best_d = 1e9

        # Find best eligible pair to merge
        for i in range(len(cl)):
            ci, gi = cl[i]
            if _is_whiteish_lab(ci):
                continue  # do not merge white with anything here
            Ci = chroma(ci)
            Hi = hue_deg(ci)

            for j in range(i + 1, len(cl)):
                cj, gj = cl[j]
                # never merge white with non-white
                if _is_whiteish_lab(cj):
                    continue

                Cj = chroma(cj)
                Hj = hue_deg(cj)
                d = _lab_hue_chroma_dist(ci, cj)

                # Case A: both chromatic enough -> require hue proximity
                if Ci >= min_chroma and Cj >= min_chroma:
                    if circ_delta_deg(Hi, Hj) > hue_tol_deg:
                        continue
                    if d > dist_tol:
                        continue
                else:
                    # Case B: at least one is low-chroma (grayish) but not white
                    # rely on overall distance only, but still be conservative
                    if d > (dist_tol * 0.9):
                        continue

                if d < best_d:
                    best_d, best_i, best_j = d, i, j

        # Merge the best pair if any
        if best_i >= 0:
            ci, gi = cl[best_i]
            cj, gj = cl[best_j]
            merged = gi + gj
            arr = np.array([_rgb_to_lab(x.hair_rgb) for x in merged if x.hair_rgb is not None], dtype=np.float32)
            mean_lab = tuple(np.mean(arr, axis=0)) if arr.size else ci
            cl[best_i] = (mean_lab, merged)
            del cl[best_j]
            changed = True

    return cl



# =========================
# ==== MAIN PIPELINE ======
# =========================

@dataclass
class ImgInfo:
    path: pathlib.Path
    face_box: Optional[Tuple[int,int,int,int]] = None
    bg_rgb: Optional[Tuple[int,int,int]] = None
    hair_rgb: Optional[Tuple[int,int,int]] = None
    pose_desc: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    bbox: Optional[Tuple[int,int,int,int]] = None
    outfit_sig: Optional[np.ndarray] = None


def _cluster_by_character(infos: List[ImgInfo]) -> Dict[int, List[ImgInfo]]:
    """
    Cluster by hair color using ΔE with low L-weight (abL), with light Lab quantization,
    a simple greedy build, a merge pass, and a final 2-iteration reassignment refine.
    """
    steps = CONFIG.get("HAIR_QUANT", 6)
    thr   = CONFIG.get("CHAR_DELTAE_THRESH", 10.0)
    join  = CONFIG.get("CHAR_SMALL_JOIN", 17.0)

    # Build items with quantized Lab (skip images without a hair estimate)
    items: List[Tuple[ImgInfo, Tuple[float,float,float]]] = []
    for info in infos:
        if info.hair_rgb is None:
            continue
        lab = _rgb_to_lab(info.hair_rgb)
        lab = _quantize_lab(lab, steps)
        items.append((info, lab))

    if not items:
        return {}

    # 1) Greedy clustering
    clusters: List[Tuple[Tuple[float,float,float], List[ImgInfo]]] = []
    for info, lab in items:
        placed = False
        for idx, (centroid, group) in enumerate(clusters):
            # HARD GUARD: keep blondes out of white unless virtually identical
            if _label_hair_lab(lab) == "blonde" and _is_whiteish_lab(centroid):
                if _lab_hue_chroma_dist(lab, centroid) > 7.0:
                    continue  # don't consider this white cluster
            if _lab_hue_chroma_dist(lab, centroid) <= _char_merge_thresh(lab, centroid):
                group.append(info)
                # update centroid (mean in true Lab)
                arr = np.array([_rgb_to_lab(g.hair_rgb) for g in group if g.hair_rgb is not None], dtype=np.float32)
                mean_lab = tuple(np.mean(arr, axis=0))
                clusters[idx] = (mean_lab, group)
                placed = True
                break
        if not placed:
            clusters.append((lab, [info]))

    if not clusters:
        return {}

    # 2) Merge closest pairs while within adaptive 'join'
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        best_i = best_j = -1
        best_d = 1e9
        best_th = 0.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                di = _lab_hue_chroma_dist(clusters[i][0], clusters[j][0])
                th_adapt = _char_merge_thresh(clusters[i][0], clusters[j][0])

                # HARD STOP: don't merge white and blonde clusters
                if (_is_whiteish_lab(clusters[i][0]) and _is_blondeish_lab(clusters[j][0])) or \
                (_is_whiteish_lab(clusters[j][0]) and _is_blondeish_lab(clusters[i][0])):
                    continue

                if di < best_d:
                    best_d = di
                    best_i, best_j = i, j
                    best_th = th_adapt
        if best_i >= 0 and best_d <= min(join, best_th):
            ci, gi = clusters[best_i]
            cj, gj = clusters[best_j]
            merged = gi + gj
            arr = np.array([_rgb_to_lab(g.hair_rgb) for g in merged if g.hair_rgb is not None], dtype=np.float32)
            mean_lab = tuple(np.mean(arr, axis=0))
            clusters[best_i] = (mean_lab, merged)
            del clusters[best_j]
            changed = True

    # 3) Reassignment refine (like k-means “E-step/M-step”), 2 iterations
    for _ in range(2):
        if not clusters:
            break
        # current centroids
        cents = [c for (c, _) in clusters]
        # empty new groups
        new_groups: List[List[ImgInfo]] = [[] for _ in range(len(clusters))]
        # assign every image to nearest centroid (with white/blonde penalties)
        for _, grp in clusters:
            for info in grp:
                if info.hair_rgb is None:
                    continue
                lab = _rgb_to_lab(info.hair_rgb)

                base = [_lab_hue_chroma_dist(lab, cents[j]) for j in range(len(cents))]

                src_white  = _is_whiteish_lab(lab)
                src_blonde = _is_blondeish_lab(lab)

                adj = []
                for j, d in enumerate(base):
                    tgt = cents[j]
                    tgt_white  = _is_whiteish_lab(tgt)
                    tgt_blonde = _is_blondeish_lab(tgt)

                    penalty = 0.0
                    # discourage crossing white <-> non-white
                    if src_white ^ tgt_white:
                        penalty += 10.0

                    # make blonde -> white and white -> blonde effectively forbidden
                    if (src_blonde and tgt_white) or (src_white and tgt_blonde):
                        penalty += 1e6


                    adj.append(d + penalty)

                best_i = int(np.argmin(adj))
                new_groups[best_i].append(info)

        # rebuild clusters from new assignments (skip empties)
        new_clusters: List[Tuple[Tuple[float,float,float], List[ImgInfo]]] = []
        for grp in new_groups:
            if not grp:
                continue
            arr = np.array([_rgb_to_lab(g.hair_rgb) for g in grp if g.hair_rgb is not None], dtype=np.float32)
            mean_lab = tuple(np.mean(arr, axis=0))
            new_clusters.append((mean_lab, grp))
        clusters = new_clusters

        if len(clusters) <= 1:
            break

    # 4) Optional split (already in your file)
    final_groups = []
    for _, grp in clusters:
        final_groups.extend(_maybe_split_bimodal_ab(grp, split_thresh=12.0, min_size=CONFIG["CHAR_MIN_CLUSTER"]))

    # Rebuild clusters from final_groups (compute new centroids)
    clusters = []
    for grp in final_groups:
        arr = np.array([_rgb_to_lab(g.hair_rgb) for g in grp if g.hair_rgb is not None], dtype=np.float32)
        clusters.append((tuple(np.mean(arr, axis=0)), grp))

    # 5) Merge whites split by lighting
    clusters = _final_low_chroma_merge(clusters)

    # 5.5) Split pale blonde out of “white”, if present
    clusters = _split_white_clusters_to_pale_blonde(
        clusters, min_size=CONFIG["CHAR_MIN_CLUSTER"], min_fraction=0.08
    )

    # 6) Merge close blondes
    clusters = _final_blonde_merge(clusters)

    # 6.1) NEW: generic hue-aware merge for any chromatic hair family
    clusters = _final_hue_prox_merge(
        clusters,
        hue_tol_deg=24.0,
        dist_tol=16.0,
        min_chroma=10.0,
    )

    # 7) Rescue blondes from white; then expel non-white from white
    clusters = _rescue_blondeish_from_white(clusters)
    clusters = _expel_nonwhite_from_white_clusters(clusters)


    # DEBUG once
    for idx, (c, g) in enumerate(clusters):
        L, a, b = c
        C = math.hypot(a, b)
        lbl_counts = {"white": 0, "blonde": 0, "other": 0}
        for inf in g:
            if inf.hair_rgb is None:
                continue
            lab = _rgb_to_lab(inf.hair_rgb)
            lbl_counts[_label_hair_lab(lab)] += 1
        
        hue = (math.degrees(math.atan2(b, a)) + 360.0) % 360.0
        print(
            f"[char cluster {idx}] n={len(g)}  L={L:.1f} a={a:.1f} b={b:.1f} hue={hue:.1f}° "
            f"C={C:.1f}  white={lbl_counts['white']} blonde={lbl_counts['blonde']} other={lbl_counts['other']}"
        )


    out: Dict[int, List[ImgInfo]] = {}
    for i, (_, group) in enumerate(clusters):
        out[i] = group
    return out





def _pose_descriptor_fallback(mask: np.ndarray, bbox: Tuple[int,int,int,int],
                              face: Optional[Tuple[int,int,int,int]] = None) -> np.ndarray:
    """
    Pose-ish descriptor using silhouette shape + head position:
    - Principal axis angle (cos,sin)
    - Hu moments (log)
    - Horizontal & vertical projection profiles (downsampled)
    - Head center relative to body bbox (if face box available)
    """
    x, y, w, h = bbox
    crop = (mask[y:y+h, x:x+w] > 0).astype(np.uint8)

    # Principal axis
    ys, xs = np.where(crop > 0)
    if xs.size == 0:
        return np.zeros(32, dtype=np.float32)
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    v0 = eigvecs[:, order[0]]
    angle = math.atan2(float(v0[1]), float(v0[0]))
    ang = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

    # Hu moments (log)
    m = cv2.moments(crop)
    hu = cv2.HuMoments(m).flatten().astype(np.float32)
    hu = np.sign(hu) * np.log1p(np.abs(hu))
    hu = hu[:5]  # first five are usually enough

    # Projection profiles (normalize and downsample to 16 each)
    hp = crop.sum(axis=0).astype(np.float32)
    vp = crop.sum(axis=1).astype(np.float32)
    if hp.max() > 0: hp /= hp.max()
    if vp.max() > 0: vp /= vp.max()
    # downsample by linear interpolation
    def _resize_1d(arr, L=16):
        if arr.size == 0:
            return np.zeros(L, dtype=np.float32)
        xs = np.linspace(0, arr.size-1, num=L, dtype=np.float32)
        return np.interp(xs, np.arange(arr.size, dtype=np.float32), arr).astype(np.float32)
    hp16 = _resize_1d(hp, 16)
    vp16 = _resize_1d(vp, 16)

    # Head center relative to body bbox (helps split “look left/right/up/down”)
    head_rel = np.zeros(2, dtype=np.float32)
    if face is not None:
        fx, fy, fw, fh = face
        cx = (fx + fw/2) - x
        cy = (fy + fh/2) - y
        head_rel = np.array([cx / max(1, w), cy / max(1, h)], dtype=np.float32)

    return np.concatenate([ang, hu, hp16, vp16, head_rel], axis=0)

def _absorb_small_pose_clusters(pose_map: Dict[int, List[ImgInfo]]) -> Dict[int, List[ImgInfo]]:
    """
    Post-process pose clusters:
    - Any cluster with < POSE_MIN_CLUSTER images is absorbed into the nearest
      larger cluster if its centroid distance <= POSE_ABSORB_EPS.
    - Returns a compactly reindexed dict {0..N-1: [ImgInfo,...]}.
    """
    if not pose_map:
        return pose_map

    # Build centroids
    cent: Dict[int, np.ndarray] = {}
    for pid, items in pose_map.items():
        vecs = [inf.pose_desc for inf in items if inf.pose_desc is not None]
        if vecs:
            cent[pid] = np.mean(np.stack(vecs, axis=0), axis=0)

    small = [pid for pid, items in pose_map.items() if len(items) < CONFIG["POSE_MIN_CLUSTER"]]
    big   = [pid for pid, items in pose_map.items() if len(items) >= CONFIG["POSE_MIN_CLUSTER"]]
    if not small or not big:
        # Reindex and return
        out: Dict[int, List[ImgInfo]] = {}
        k = 0
        for pid, items in pose_map.items():
            if items:
                out[k] = items
                k += 1
        return out

    def dist(a: np.ndarray, b: np.ndarray) -> float:
        L = min(a.size, b.size)
        return float(np.linalg.norm(a[:L] - b[:L]))

    for s in small:
        if s not in cent:
            continue
        best, best_d = None, 1e9
        for b in big:
            if b not in cent:
                continue
            d = dist(cent[s], cent[b])
            if d < best_d:
                best, best_d = b, d
        if best is not None and best_d <= CONFIG["POSE_ABSORB_EPS"]:
            pose_map[best].extend(pose_map[s])
            pose_map[s] = []

    # Compact reindex
    out: Dict[int, List[ImgInfo]] = {}
    k = 0
    for pid, items in pose_map.items():
        if items:
            out[k] = items
            k += 1
    return out


def _cluster_pose(descriptors: List[Tuple[ImgInfo, np.ndarray]]) -> Dict[int, List[ImgInfo]]:
    """
    Minimal DBSCAN-like clustering on small vectors (pose descriptors).
    We implement a simple agglomerative threshold-based clustering to avoid sklearn deps.
    """
    eps = CONFIG["POSE_EPS"]
    min_pts = CONFIG["POSE_MIN_SAMPLES"]
    clusters: List[List[Tuple[ImgInfo, np.ndarray]]] = []

    def d(a: np.ndarray, b: np.ndarray) -> float:
        L = min(a.size, b.size)
        if L == 0:
            return 1e9
        return float(np.linalg.norm(a[:L] - b[:L]))

    for info, vec in descriptors:
        placed = False
        for cl in clusters:
            # compare to cluster centroid (mean)
            centroid = np.mean([v for _, v in cl], axis=0)
            if d(vec, centroid) <= eps:
                cl.append((info, vec))
                placed = True
                break
        if not placed:
            clusters.append([(info, vec)])

    # build output
    pose_map: Dict[int, List[ImgInfo]] = {}
    pid = 0
    for cl in clusters:
        if not cl:
            continue
        pose_map[pid] = [info for (info, _) in cl]
        pid += 1
    return pose_map


def _cluster_outfits(pose_infos: List[ImgInfo]) -> Dict[int, List[ImgInfo]]:
    """
    Cluster by L1 distance of silhouette ray signatures.
    """
    eps = CONFIG["OUTFIT_EPS"]
    min_pts = CONFIG["OUTFIT_MIN_SAMPLES"]
    clusters: List[List[ImgInfo]] = []

    def dist(a: ImgInfo, b: ImgInfo) -> float:
        return _l1_dist(a.outfit_sig, b.outfit_sig)

    for info in pose_infos:
        if info.outfit_sig is None:
            continue
        placed = False
        for cl in clusters:
            # compare to medoid (the member minimizing total distance)
            medoid = min(cl, key=lambda k: sum(dist(k, x) for x in cl))
            if dist(info, medoid) <= eps:
                cl.append(info)
                placed = True
                break
        if not placed:
            clusters.append([info])

    # merge small clusters if very close
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if not clusters[i] or len(clusters[i]) >= min_pts:
                continue
            best_j, best_d = None, 1e9
            for j in range(len(clusters)):
                if i == j or not clusters[j]:
                    continue
                # medoid-to-medoid
                mi = min(clusters[i], key=lambda k: sum(dist(k, x) for x in clusters[i]))
                mj = min(clusters[j], key=lambda k: sum(dist(k, x) for x in clusters[j]))
                dd = dist(mi, mj)
                if dd < best_d:
                    best_d, best_j = dd, j
            if best_j is not None and best_d <= eps * 0.6:
                clusters[best_j].extend(clusters[i])
                clusters[i] = []
                changed = True

    out: Dict[int, List[ImgInfo]] = {}
    oid = 0
    for cl in clusters:
        if not cl:
            continue
        out[oid] = cl
        oid += 1
    return out


def main():
    if len(sys.argv) != 2:
        print("Usage: python auto_sort_helper.py /path/to/unsorted_folder", file=sys.stderr)
        sys.exit(2)

    root = pathlib.Path(sys.argv[1]).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: '{root}' is not a valid folder.", file=sys.stderr)
        sys.exit(2)

    # 1) Gather candidate images (loose files only; we won't descend into existing subfolders)
    files = _iter_images(root)
    infos: List[ImgInfo] = [ImgInfo(path=f) for f in files]

    # 2) Per-image: face, background, hair color, mask/bbox, outfit signature, pose-fallback
    for info in infos:
        img_any = _read_bgr(info.path)
        if img_any is None:
            continue
        
        # Face box needs BGR, so split:
        img_bgr, _ = _split_channels(img_any)

        # Face box
        face = detect_face_box_bgr(img_bgr)
        info.face_box = face

        # Background color (alpha-aware)
        bg = _estimate_background_rgb(img_any)
        info.bg_rgb = (int(bg[0]), int(bg[1]), int(bg[2]))

        # Foreground mask (alpha preferred)
        mask = _binary_mask(img_any, info.bg_rgb, CONFIG["MASK_BG_DELTA"])
        info.mask = mask
        bbox = _bounding_box(mask)
        info.bbox = bbox

        # Hair color (sample on BGR image)  <-- move this AFTER mask is set
        if face is not None:
            seed = _seed_from_path(info.path)
            hair = _sample_hair_rgb(
                img_bgr, face, info.bg_rgb,
                n_samples=CONFIG["HAIR_SAMPLES"],
                band_top=CONFIG["HAIR_BAND_TOP"],
                band_bottom=CONFIG["HAIR_BAND_BOTTOM"],
                pad_x=CONFIG["HAIR_BAND_PAD_X"],
                seed=seed,
                drop_near_white=CONFIG["DROP_NEAR_WHITE"],
                bg_delta=CONFIG["BG_DELTA_RGB"],
                fg_mask=mask,  # << important
            )
            info.hair_rgb = hair

            if CONFIG["DEBUG_HAIR_PER_IMAGE"]:
                if hair is None:
                    print(f"[hair] {info.path.name}: NONE")
                else:
                    L, a, b = _rgb_to_lab(hair)
                    C = math.hypot(a, b)
                    print(
                        f"[hair] {info.path.name}: RGB={hair}  L={L:.1f} a={a:.1f} b={b:.1f} C={C:.1f}  label={_label_hair_lab((L,a,b))}"
                    )


        # Outfit ray signature (only if bbox exists)
        if bbox is not None:
            sig = _ray_signature(mask, bbox, CONFIG["RAYS_HORIZONTAL"], CONFIG["RAYS_VERTICAL"])
            info.outfit_sig = sig

        # Pose descriptor (prefer keypoints; fallback to silhouette if no detection)
        kps = estimate_keypoints(img_bgr)
        if kps is not None:
            info.pose_desc = _pose_descriptor_from_keypoints(kps)
        elif bbox is not None:
            info.pose_desc = _pose_descriptor_fallback(mask, bbox, face)

    # 3) Character clustering
    char_clusters = _cluster_by_character(infos)  # id -> [ImgInfo]

    # absorb tiny character clusters into nearest larger one
    def _absorb_small_char_clusters(char_clusters: Dict[int, List[ImgInfo]]) -> Dict[int, List[ImgInfo]]:
        # build centroids in Lab
        centroids = {}
        sizes = {}
        for cid, items in char_clusters.items():
            labs = [ _rgb_to_lab(i.hair_rgb) for i in items if i.hair_rgb is not None ]
            if labs:
                centroids[cid] = tuple(np.mean(np.array(labs, dtype=np.float32), axis=0))
                sizes[cid] = len(items)
        small = [cid for cid,s in sizes.items() if s < CONFIG["CHAR_MIN_CLUSTER"]]
        big   = [cid for cid,s in sizes.items() if s >= CONFIG["CHAR_MIN_CLUSTER"]]
        if not small or not big:
            return char_clusters

        for sid in small:
            best, best_d = None, 1e9
            for bid in big:
                d = _delta_e(centroids[sid], centroids[bid])
                if d < best_d:
                    best, best_d = bid, d
            if best is not None and best_d <= CONFIG["CHAR_ABSORB_DELTAE"]:
                # move members
                char_clusters[best].extend(char_clusters[sid])
                char_clusters[sid] = []
        # reindex compactly
        out = {}
        k = 0
        for cid, items in char_clusters.items():
            if items:
                out[k] = items
                k += 1
        return out

    char_clusters = _absorb_small_char_clusters(char_clusters)


    # 4) For each character, cluster poses, then outfits, then move files in place
    for char_id, char_infos in char_clusters.items():
        if not CONFIG.get("ENABLE_POSE", False):
            # Move directly into char_{id} folder (no pose/outfit)
            dest_char = root / f"char_{char_id:03d}"
            dest_char.mkdir(parents=True, exist_ok=True)
            for inf in char_infos:
                src = inf.path
                if src.parent == dest_char:
                    continue
                new_path = dest_char / src.name
                if new_path.exists():
                    stem, ext = src.stem, src.suffix
                    k = 1
                    while True:
                        cand = dest_char / f"{stem}_{k}{ext}"
                        if not cand.exists():
                            new_path = cand
                            break
                        k += 1
                try:
                    shutil.move(str(src), str(new_path))
                    inf.path = new_path
                except Exception:
                    traceback.print_exc()
            continue


        for pose_id, pose_infos in pose_map.items():
            # Outfit clustering
            outfit_map = _cluster_outfits(pose_infos)  # outfit_id -> [ImgInfo]

            # Create folders and move files
            for outfit_id, group in outfit_map.items():
                dest = root / f"char_{char_id:03d}" / f"pose_{pose_id:03d}" / f"outfit_{outfit_id:03d}"
                dest.mkdir(parents=True, exist_ok=True)
                for inf in group:
                    src = inf.path
                    # If already in place from a previous run, skip moving
                    if src.parent == dest:
                        continue
                    new_path = dest / src.name
                    # Handle name collisions by adding a counter
                    if new_path.exists():
                        stem, ext = src.stem, src.suffix
                        k = 1
                        while True:
                            cand = dest / f"{stem}_{k}{ext}"
                            if not cand.exists():
                                new_path = cand
                                break
                            k += 1
                    try:
                        shutil.move(str(src), str(new_path))
                        inf.path = new_path
                    except Exception:
                        # If move fails, continue with others
                        traceback.print_exc()

    # 5) Print the same folder path for Step 3
    print(str(root))


if __name__ == "__main__":
    main()
