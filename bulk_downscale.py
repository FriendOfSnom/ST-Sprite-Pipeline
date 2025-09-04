#!/usr/bin/env python3
"""
bulk_downscale.py

Bulk-downscale finalized Student Transfer sprites to their in-game size,
using a high-quality, gamma-aware LANCZOS pipeline.

Usage:
    python bulk_downscale.py /path/to/finalized_root [--dest-root /path/to/output]
                             [--allow-upscale] [--quality 95] [--lossless]
                             [--backup-originals] [--dry-run]

Behavior:
- For each character folder:
    - Read character.yml -> scale (float).
    - For every image in poses/*/(outfits/*.png,*.webp,*.jpg) and faces/face/*:
        - Resize to (w*scale, h*scale), skipping upscales unless --allow-upscale.
        - Gamma-aware (linear-light) downscale with LANCZOS; mild unsharp mask.
    - Write resized images in place (or under --dest-root).
    - Update character.yml:
        - original_scale: <previous value>
        - scale: 1.0
        - downscaled: true
Notes:
- If you will keep running expression_sheet_maker.py afterwards, this avoids
  double-scaling because the assets are already physically at their in-game size.
"""

from __future__ import annotations
import argparse
import os
import sys
import math
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict

import yaml
from PIL import Image, ImageFilter

# -----------------------
# Constants & Helpers
# -----------------------

VALID_EXTS = (".png", ".webp", ".jpg", ".jpeg")
DEFAULT_WEBP_QUALITY = 95

def _iter_character_dirs(root: Path) -> Iterable[Path]:
    """
    Yield character directories under the given root (directories only).
    """
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p

def _find_pose_image_paths(char_dir: Path) -> Iterable[Path]:
    """
    Yield all image paths in:
      <char>/<pose>/outfits/*.{ext}
      <char>/<pose>/faces/face/*.{ext}
    """
    for pose_dir in sorted([d for d in char_dir.iterdir() if d.is_dir()]):
        outfits = pose_dir / "outfits"
        faces   = pose_dir / "faces" / "face"
        for folder in (outfits, faces):
            if folder.is_dir():
                for f in sorted(folder.iterdir()):
                    if f.is_file() and f.suffix.lower() in VALID_EXTS:
                        yield f

def _srgb_gamma_lut(gamma: float) -> list[int]:
    """
    Build a 256-entry LUT mapping sRGB<->linear approximately via power law.
    For sRGB -> linear use gamma=2.2 ; for linear -> sRGB use gamma=1/2.2.
    """
    lut = []
    for i in range(256):
        v = (i / 255.0) ** gamma
        lut.append(int(round(v * 255.0)))
    return lut

_LUT_SRGB_TO_LINEAR = _srgb_gamma_lut(2.2)
_LUT_LINEAR_TO_SRGB = _srgb_gamma_lut(1/2.2)

def _to_linear_rgb(img: Image.Image) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    Convert an RGBA or RGB PIL image from sRGB to linear RGB using LUTs.
    Alpha is kept separate and NOT gamma-corrected.
    Returns (linear_rgb_image, alpha_or_None).
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        rgb = img.convert("RGB")
        a   = img.getchannel("A")
    else:
        rgb = img
        a = None
    # Apply per-channel LUT in RGB
    r, g, b = rgb.split()
    r = r.point(_LUT_SRGB_TO_LINEAR)
    g = g.point(_LUT_SRGB_TO_LINEAR)
    b = b.point(_LUT_SRGB_TO_LINEAR)
    lin = Image.merge("RGB", (r, g, b))
    return lin, a

def _to_srgb(rgb_linear: Image.Image, alpha: Optional[Image.Image]) -> Image.Image:
    """
    Convert linear RGB back to sRGB and reattach alpha if present.
    """
    r, g, b = rgb_linear.split()
    r = r.point(_LUT_LINEAR_TO_SRGB)
    g = g.point(_LUT_LINEAR_TO_SRGB)
    b = b.point(_LUT_LINEAR_TO_SRGB)
    srgb = Image.merge("RGB", (r, g, b))
    if alpha is not None:
        return Image.merge("RGBA", (*srgb.split(), alpha))
    return srgb

def _multi_step_sizes(src_w: int, src_h: int, tgt_w: int, tgt_h: int) -> Iterable[Tuple[int,int]]:
    """
    Yield intermediate sizes for large downscales to improve quality.
    We downscale by ~50% repeatedly until within ~1.5x of target.
    """
    w, h = src_w, src_h
    while w > tgt_w * 1.5 and h > tgt_h * 1.5:
        w = max(tgt_w, int(w * 0.5))
        h = max(tgt_h, int(h * 0.5))
        yield w, h
    yield tgt_w, tgt_h

def _resize_linear_light(im: Image.Image, new_w: int, new_h: int) -> Image.Image:
    """
    High-quality, gamma-aware LANCZOS resize with multi-step for large reductions.
    Resizes alpha in lockstep (using BOX to avoid ringing halos on transparency),
    then applies a mild unsharp mask to restore line crispness.

    Returns:
        PIL.Image in RGBA or RGB, matching the input's transparency.
    """
    # 1) Split into linear-RGB + alpha (alpha not gamma-corrected)
    lin_rgb, alpha = _to_linear_rgb(im)

    # 2) Multi-step downscale in linear light (RGB) AND alpha in lockstep
    curr_rgb = lin_rgb
    curr_a = alpha
    src_w, src_h = curr_rgb.size

    for step_w, step_h in _multi_step_sizes(src_w, src_h, new_w, new_h):
        # RGB: LANCZOS for high quality
        curr_rgb = curr_rgb.resize((max(1, step_w), max(1, step_h)), Image.LANCZOS)
        # Alpha: BOX is gentler and avoids ringing on hard edges
        if curr_a is not None:
            curr_a = curr_a.resize((max(1, step_w), max(1, step_h)), Image.BOX)
        src_w, src_h = curr_rgb.size  # advance for next iteration

    # 3) Convert back to sRGB and reattach alpha
    out = _to_srgb(curr_rgb, curr_a)

    # 4) Mild unsharp mask on color only (keep alpha untouched)
    if out.mode == "RGBA":
        rgb = out.convert("RGB").filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=2))
        out = Image.merge("RGBA", (*rgb.split(), out.getchannel("A")))
    else:
        out = out.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=2))

    return out


def _target_size(w: int, h: int, scale: float) -> Tuple[int,int]:
    """
    Compute target integer size from scale; at least 1x1 px.
    """
    tw = max(1, int(round(w * scale)))
    th = max(1, int(round(h * scale)))
    return tw, th

def _save_image(path_out: Path, img: Image.Image, *, lossless: bool, quality: int) -> None:
    """
    Save using settings that suit sprite art. Preserves extension.
    """
    ext = path_out.suffix.lower()
    path_out.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".webp":
        # For crisp line art: lossy quality 95 is very good; or True lossless.
        img.save(
            path_out,
            "WEBP",
            lossless=bool(lossless),
            quality=int(quality),
            method=6,
            exact=True,
        )
    elif ext in (".png",):
        # PNG is lossless by default; Pillow chooses good filters.
        img.save(path_out, "PNG", optimize=True)
    elif ext in (".jpg", ".jpeg"):
        # JPG doesn't support alpha; if we have RGBA, flatten on white.
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        img.save(path_out, "JPEG", quality=int(quality), subsampling="4:4:4", optimize=True)
    else:
        # Fallback: let Pillow guess
        img.save(path_out)

def _load_yaml(path: Path) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _dump_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

# -----------------------
# Main worker
# -----------------------

def process_character(char_dir: Path, dest_root: Optional[Path], *, allow_upscale: bool,
                      lossless: bool, quality: int, backup: bool, dry_run: bool) -> None:
    """
    Downscale all poses for one character folder based on character.yml scale.
    Updates YAML to scale=1.0 afterward to prevent double-scaling.
    """
    yml_path = char_dir / "character.yml"
    meta = _load_yaml(yml_path)
    scale = float(meta.get("scale", 1.0))

    if scale <= 0:
        print(f"[WARN] {char_dir.name}: invalid scale {scale}; skipping.")
        return

    if not allow_upscale and scale > 1.0:
        print(f"[INFO] {char_dir.name}: scale={scale:.3f} (>1). Skipping (no upscaling).")
        return

    # Destination base
    out_base = dest_root / char_dir.name if dest_root else char_dir

    if backup and not dest_root and not dry_run:
        backup_dir = char_dir / "_original_fullsize_backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"[INFO] Backup originals will be placed under: {backup_dir}")

    # Process images
    for img_path in _find_pose_image_paths(char_dir):
        rel = img_path.relative_to(char_dir)
        out_path = out_base / rel

        # Read & compute target size
        try:
            im = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            continue

        tgt_w, tgt_h = _target_size(im.width, im.height, scale)
        if not allow_upscale and (tgt_w >= im.width or tgt_h >= im.height):
            # No-op size; still copy if writing to --dest-root
            if dest_root and not dry_run:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                im.save(out_path)
            print(f"[SKIP] {rel} — no downscale needed.")
            continue

        print(f"[DO ] {char_dir.name}/{rel} -> {tgt_w}x{tgt_h} (scale {scale:.3f})")

        if dry_run:
            continue

        if backup and not dest_root:
            # Copy original before overwriting (keep tree structure)
            bpath = (char_dir / "_original_fullsize_backup" / rel)
            bpath.parent.mkdir(parents=True, exist_ok=True)
            if not bpath.exists():
                try:
                    im.save(bpath)
                except Exception:
                    pass

        # Resize (gamma-aware LANCZOS)
        out_img = _resize_linear_light(im, tgt_w, tgt_h)

        # Save
        _save_image(out_path, out_img, lossless=lossless, quality=quality)

    # Update YAML: mark downscaled and pin scale to 1.0
    if not dry_run:
        meta["original_scale"] = scale
        meta["scale"] = 1.0
        meta["downscaled"] = True
        _dump_yaml(out_base / "character.yml", meta)
        print(f"[INFO] Updated {out_base / 'character.yml'} (scale -> 1.0, original_scale={scale:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Bulk downscale sprites to in-game size.")
    parser.add_argument("root", type=str, help="Path to finalized output root (characters here).")
    parser.add_argument("--dest-root", type=str, default=None,
                        help="Optional separate output root. If omitted, operates in-place.")
    parser.add_argument("--allow-upscale", action="store_true",
                        help="Allow scaling >1.0 (default: skip).")
    parser.add_argument("--lossless", action="store_true",
                        help="Use lossless WebP when saving .webp (default: lossy).")
    parser.add_argument("--quality", type=int, default=DEFAULT_WEBP_QUALITY,
                        help="Quality for lossy WebP/JPEG (default: 95).")
    parser.add_argument("--backup-originals", action="store_true",
                        help="Keep originals in _original_fullsize_backup (in-place only).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] '{root}' is not a folder.")
        sys.exit(1)

    dest_root = Path(args.dest_root).resolve() if args.dest_root else None
    if dest_root:
        dest_root.mkdir(parents=True, exist_ok=True)

    chars = list(_iter_character_dirs(root))
    if not chars:
        print("[WARN] No character folders found.")
        sys.exit(0)

    print(f"[INFO] Characters found: {len(chars)}")
    for cdir in chars:
        process_character(
            cdir, dest_root,
            allow_upscale=args.allow_upscale,
            lossless=args.lossless,
            quality=args.quality,
            backup=args.backup_originals,
            dry_run=args.dry_run,
        )

    print("\n[INFO] Bulk downscale complete.")

if __name__ == "__main__":
    main()
