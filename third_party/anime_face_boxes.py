# third_party/anime_face_boxes.py
from __future__ import annotations
import os
from itertools import product as product
from math import ceil
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# ====== CONFIGURE ========
# =========================
# You can tune these defaults from face_detect_ssd.py if desired.
DEFAULT_CONF_THRESH = 0.10   # confidence threshold
DEFAULT_NMS_THRESH  = 0.30   # IoU for NMS
DEFAULT_KEEP_TOPK   = 750    # limit after NMS
DEFAULT_TOP_K       = 5000   # limit before NMS
SHIFT_YMIN_FRAC     = 0.20   # matches the sample code's "ymin += 0.2 * (ymax - ymin + 1)"


# =========================
# ====== BUILD MODEL ======
# =========================

class BasicConv2d(nn.Module):
    """Conv2d + BN + ReLU block used by FaceBoxes."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    """Inception-like block used by FaceBoxes."""
    def __init__(self):
        super().__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)
        branch3x3 = self.branch3x3(self.branch3x3_reduce(x))
        branch3x3_3 = self.branch3x3_3(self.branch3x3_2(self.branch3x3_reduce_2(x)))
        return torch.cat([branch1x1, branch1x1_2, branch3x3, branch3x3_3], 1)


class CRelu(nn.Module):
    """CReLU + BN used by FaceBoxes."""
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        return F.relu(x, inplace=True)


class FaceBoxes(nn.Module):
    """FaceBoxes backbone + detection heads (loc/conf)."""
    def __init__(self, phase: str, size, num_classes: int):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        detection_sources = []
        loc, conf = [], []

        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x); x = self.inception2(x); x = self.inception3(x)
        detection_sources.append(x)

        x = self.conv3_1(x); x = self.conv3_2(x)
        detection_sources.append(x)

        x = self.conv4_1(x); x = self.conv4_2(x)
        detection_sources.append(x)

        for (x_fm, l, c) in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x_fm).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x_fm).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            return (loc.view(loc.size(0), -1, 4),
                    self.softmax(conf.view(-1, self.num_classes)))
        else:
            return (loc.view(loc.size(0), -1, 4),
                    conf.view(conf.size(0), -1, self.num_classes))


class PriorBox(object):
    """Generate SSD prior boxes for FaceBoxes."""
    def __init__(self, cfg, image_size=None, phase='train'):
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [
            [ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# =========================
# ====== UTILITIES ========
# =========================

def _mymax(a, b): return a if a >= b else b
def _mymin(a, b): return b if a >= b else a

def _cpu_nms(dets: np.ndarray, thresh: float) -> List[int]:
    """Standard NMS over [x1,y1,x2,y2,score] on CPU."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets,), dtype=int)
    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1, iy1, ix2, iy2, iarea = x1[i], y1[i], x2[i], y2[i], areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = _mymax(ix1, x1[j]); yy1 = _mymax(iy1, y1[j])
            xx2 = _mymin(ix2, x2[j]); yy2 = _mymin(iy2, y2[j])
            w = _mymax(0.0, xx2 - xx1 + 1)
            h = _mymax(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1
    return keep

def nms(dets: np.ndarray, thresh: float, force_cpu: bool = True) -> List[int]:
    """Expose NMS (CPU only is fine)."""
    if dets.shape[0] == 0:
        return []
    return _cpu_nms(dets, thresh)

def decode(loc: torch.Tensor, priors: torch.Tensor, variances) -> torch.Tensor:
    """
    Decode SSD locations back to [x1,y1,x2,y2] in relative coords [0..1].
    loc: [num_priors, 4], priors: [num_priors, 4], variances: [2]
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, 2:]
    return boxes


# =========================
# ====== INFERENCE ========
# =========================

_FACEBOXES_CFG = {
    'name': 'FaceBoxes',
    'min_sizes': [[32, 64, 128], [256], [512]],
    'steps': [32, 64, 128],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}

class _FaceBoxesRunner:
    """Small helper that owns the net and priors and runs detection."""
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.net = FaceBoxes(phase='test', size=None, num_classes=2)
        self.net = load_model(self.net, weights_path, load_to_cpu=(device == "cpu"))
        self.net.eval().to(self.device)

    @torch.no_grad()
    def detect_bgr(self, image_bgr: np.ndarray,
                   conf_thresh: float = DEFAULT_CONF_THRESH,
                   nms_thresh: float = DEFAULT_NMS_THRESH,
                   keep_topk: int = DEFAULT_KEEP_TOPK,
                   top_k: int = DEFAULT_TOP_K) -> List[Tuple[float, float, float, float, float]]:
        """
        Run detection on a BGR image. Returns a list of (x1, y1, x2, y2, score) in ABSOLUTE pixels.
        """
        img = np.float32(image_bgr)
        H, W, _ = img.shape
        scale = torch.tensor([W, H, W, H], dtype=torch.float32, device=self.device)

        # Classic FaceBoxes preprocessing
        img -= (104, 117, 123)
        img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        loc, conf = self.net(img)
        priorbox = PriorBox(_FACEBOXES_CFG, image_size=(H, W))
        priors = priorbox.forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, _FACEBOXES_CFG['variance'])
        boxes = (boxes * scale).cpu().numpy()  # absolute pixels
        scores = conf.data.cpu().numpy()[:, 1]  # face class

        # Filter low scores
        inds = np.where(scores > conf_thresh)[0]
        boxes = boxes[inds]; scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]; scores = scores[order]

        # NMS on [x1,y1,x2,y2,score]
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
        keep = nms(dets, nms_thresh, force_cpu=True)
        dets = dets[keep, :]

        # Keep top-K after NMS
        dets = dets[:keep_topk, :]

        # Optional ymin shift exactly like the example code
        for k in range(dets.shape[0]):
            xmin, ymin, xmax, ymax, sc = dets[k]
            ymin = ymin + SHIFT_YMIN_FRAC * (ymax - ymin + 1.0)
            dets[k] = (xmin, ymin, xmax, ymax, sc)

        return [(float(x1), float(y1), float(x2), float(y2), float(s)) for (x1, y1, x2, y2, s) in dets]


# Lazy singleton to avoid re-loading weights
_RUNNER: Optional[_FaceBoxesRunner] = None

def load_model(model: nn.Module, pretrained_path: str, load_to_cpu: bool):
    """Load FaceBoxes weights with legacy 'module.' prefix handling."""
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location="cpu")
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict:
        state = pretrained_dict["state_dict"]
    else:
        state = pretrained_dict
    # drop "module." prefix if present
    state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }
    missing = set(model.state_dict().keys()) - set(state.keys())
    if len(state) == 0:
        raise RuntimeError("No matching weights in checkpoint.")
    model.load_state_dict(state, strict=False)
    return model

def get_runner(weights_path: str, device: str = "cpu") -> _FaceBoxesRunner:
    """Return a cached runner so we only load weights once."""
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = _FaceBoxesRunner(weights_path=weights_path, device=device)
    return _RUNNER

def detect_boxes_bgr(image_bgr: np.ndarray,
                     weights_path: str,
                     conf_thresh: float = DEFAULT_CONF_THRESH,
                     nms_thresh: float  = DEFAULT_NMS_THRESH) -> List[Tuple[float, float, float, float, float]]:
    """
    Convenience function: run detector and return a list of (x1, y1, x2, y2, score).
    """
    runner = get_runner(weights_path, device="cpu")
    return runner.detect_bgr(image_bgr, conf_thresh=conf_thresh, nms_thresh=nms_thresh)
