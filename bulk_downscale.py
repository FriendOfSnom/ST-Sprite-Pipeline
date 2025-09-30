#!/usr/bin/env python3
"""
bulk_downscale.py

Strictly downscale Student Transfer sprites to their exact in-game size using a
gamma-aware LANCZOS pipeline (historically "the good one"):
- RGB is resized in linear light with LANCZOS (multi-step).
- Alpha is resized with BOX (matches your original behavior).

Behavior:
- For each character folder (root/<character>):
  - Read character.yml -> `scale` (float).
  - If scale < 1.0: downscale *all* images under each pose's `outfits/` and `faces/`
    subfolders to the exact target size.
  - If scale == 1.0: copy-through if `--dest-root` is provided; otherwise skip.
  - If scale > 1.0: SKIP entirely (no upscaling in this tool).
  - Update character.yml *textually* so we preserve duplicates, order, and comments:
      * Rewrite every line `image_height: <int>` -> round(<int> * scale).
      * Rewrite every line `eye_line: <int|float>` -> round(value * scale).
      * Set `scale: 1.0`, add/update `original_scale: <old>`, `downscaled: true`.

Notes:
- No backups are written inside character folders.
- Supports PNG/WEBP/JPEG in nested folders.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

VALID_EXTS = (".png", ".webp", ".jpg", ".jpeg")
DEFAULT_WEBP_QUALITY = 95

# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------

def _iter_character_dirs(root: Path) -> Iterable[Path]:
    """
    Yield immediate subdirectories of `root` as character directories.
    """
    for p in sorted(root.iterdir()):
        if p.is_dir():
            yield p

def _find_pose_image_paths(char_dir: Path) -> Iterable[Path]:
    """
    Yield all image paths under each pose directory, recursively:

      <char>/<pose>/outfits/**/*.{png,webp,jpg,jpeg}
      <char>/<pose>/faces/**/*.{png,webp,jpg,jpeg}

    We intentionally search only 'outfits' and 'faces' subtrees.
    """
    for pose_dir in sorted([d for d in char_dir.iterdir() if d.is_dir()]):
        for subdir_name in ("outfits", "faces"):
            base = pose_dir / subdir_name
            if not base.is_dir():
                continue
            for f in base.rglob("*"):
                if f.is_file() and f.suffix.lower() in VALID_EXTS:
                    yield f

def _load_yaml(path: Path) -> Dict:
    """
    Load YAML as a dict for reading the current `scale`. This loses duplicate keys,
    so we DO NOT use it to write back. For writing, we do a textual pass.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def _save_image(path_out: Path, img: Image.Image, *, lossless: bool, quality: int) -> None:
    """
    Save an image with sane defaults for WEBP/PNG/JPEG.
    - WEBP honors lossless flag and 'quality'
    - PNG is always lossless
    - JPEG converts RGBA to RGB against white
    """
    ext = path_out.suffix.lower()
    path_out.parent.mkdir(parents=True, exist_ok=True)
    if ext == ".webp":
        img.save(path_out, "WEBP", lossless=bool(lossless), quality=int(quality), method=6, exact=True)
    elif ext == ".png":
        img.save(path_out, "PNG", optimize=True)
    elif ext in (".jpg", ".jpeg"):
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg
        img.save(path_out, "JPEG", quality=int(quality), subsampling="4:4:4", optimize=True)
    else:
        img.save(path_out)

# ---------------------------------------------------------------------
# Gamma-aware resizing (historical behavior)
# ---------------------------------------------------------------------

def _srgb_gamma_lut(gamma: float) -> list[int]:
    """
    Build a 256-entry LUT for sRGB <-> linear conversions.
    """
    lut = []
    for i in range(256):
        v = (i / 255.0) ** gamma
        lut.append(int(round(v * 255.0)))
    return lut

_LUT_SRGB_TO_LINEAR = _srgb_gamma_lut(2.2)
_LUT_LINEAR_TO_SRGB = _srgb_gamma_lut(1/2.2)

def _to_linear_rgb(img: Image.Image):
    """
    Convert sRGB image to linear-light RGB using LUTs. Keep alpha separate.
    Returns (linear_rgb, alpha_or_None).
    """
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        rgb = img.convert("RGB")
        a   = img.getchannel("A")
    else:
        rgb = img
        a = None
    r, g, b = rgb.split()
    r = r.point(_LUT_SRGB_TO_LINEAR)
    g = g.point(_LUT_SRGB_TO_LINEAR)
    b = b.point(_LUT_SRGB_TO_LINEAR)
    lin = Image.merge("RGB", (r, g, b))
    return lin, a

def _to_srgb(rgb_linear: Image.Image, alpha: Optional[Image.Image]) -> Image.Image:
    """
    Convert linear-light RGB back to sRGB and reattach alpha if present.
    """
    r, g, b = rgb_linear.split()
    r = r.point(_LUT_LINEAR_TO_SRGB)
    g = g.point(_LUT_LINEAR_TO_SRGB)
    b = b.point(_LUT_LINEAR_TO_SRGB)
    srgb = Image.merge("RGB", (r, g, b))
    if alpha is not None:
        return Image.merge("RGBA", (*srgb.split(), alpha))
    return srgb

def _multi_step_sizes(src_w: int, src_h: int, tgt_w: int, tgt_h: int):
    """
    Yield intermediate downscale sizes before the final size to avoid aliasing.
    We keep stepping by ~0.5 until we're close to the target, then land exactly.
    """
    w, h = src_w, src_h
    while w > tgt_w * 1.5 and h > tgt_h * 1.5:
        w = max(tgt_w, int(w * 0.5))
        h = max(tgt_h, int(h * 0.5))
        yield w, h
    yield tgt_w, tgt_h

def _resize_linear_light(im: Image.Image, new_w: int, new_h: int) -> Image.Image:
    """
    High-quality downscale:
      - Convert to linear RGB; keep alpha separate.
      - Multi-step resize RGB with LANCZOS (anti-ringing).
      - Multi-step resize alpha with BOX (historical choice; keeps crisp masks).
      - Convert back to sRGB; mild UnsharpMask on color only.
    """
    lin_rgb, alpha = _to_linear_rgb(im)

    curr_rgb = lin_rgb
    curr_a = alpha
    src_w, src_h = curr_rgb.size

    for step_w, step_h in _multi_step_sizes(src_w, src_h, new_w, new_h):
        curr_rgb = curr_rgb.resize((max(1, step_w), max(1, step_h)), Image.LANCZOS)
        if curr_a is not None:
            curr_a = curr_a.resize((max(1, step_w), max(1, step_h)), Image.BOX)
        src_w, src_h = curr_rgb.size

    out = _to_srgb(curr_rgb, curr_a)

    if out.mode == "RGBA":
        rgb = out.convert("RGB").filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=2))
        out = Image.merge("RGBA", (*rgb.split(), out.getchannel("A")))
    else:
        out = out.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=2))

    return out

# ---------------------------------------------------------------------
# YAML textual updater (preserve duplicates, order, comments)
# ---------------------------------------------------------------------

def _format_float_short(x: float) -> str:
    """
    Format a float for YAML values with minimal trailing zeros.
    '1.0' stays '1.0' for clarity.
    """
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if "." in s else f"{s}.0"

def _update_character_yml_textual_for_downscale(
    yml_path: Path,
    applied_scale: float,
    *,
    dry_run: bool = False,
) -> None:
    """
    Update character.yml in-place by text so we can:
      - Rewrite ALL `image_height: <int>` lines to round(<int> * scale).
      - Rewrite ALL `eye_line: <number>` lines to round(<number> * scale).
      - Set `scale: 1.0`.
      - Add/Update `original_scale: <old_scale>` and `downscaled: true`.
    We *only* run this when applied_scale < 1.0.
    """
    if not yml_path.exists():
        return

    try:
        text = yml_path.read_text(encoding="utf-8")
    except Exception:
        return

    changed = False

    # Scale all image_height ints
    ih_pat = re.compile(r"^(\s*image_height\s*:\s*)(\d+)(\s*)$", flags=re.MULTILINE)
    def ih_repl(m: re.Match) -> str:
        nonlocal changed
        old_val = int(m.group(2))
        new_val = max(1, int(round(old_val * applied_scale)))
        if new_val != old_val:
            changed = True
        return f"{m.group(1)}{new_val}{m.group(3)}"
    new_text = ih_pat.sub(ih_repl, text)

    # Scale all eye_line numbers (int or float). We write back as INT (rounded).
    el_pat = re.compile(r"^(\s*eye_line\s*:\s*)(-?\d+(?:\.\d+)?)(\s*)$", flags=re.MULTILINE)
    def el_repl(m: re.Match) -> str:
        nonlocal changed
        old_raw = m.group(2)
        try:
            old_val = float(old_raw)
        except ValueError:
            return m.group(0)  # leave as-is if not numeric
        new_val = int(round(old_val * applied_scale))
        if float(new_val) != old_val:
            changed = True
        return f"{m.group(1)}{new_val}{m.group(3)}"
    new_text = el_pat.sub(el_repl, new_text)

    # Pin scale: 1.0 (or add it)
    scale_pat = re.compile(r"^(\s*scale\s*:\s*)([0-9.]+)(\s*)$", flags=re.MULTILINE)
    if scale_pat.search(new_text):
        def scale_repl(m: re.Match) -> str:
            nonlocal changed
            if m.group(2) != "1.0":
                changed = True
            return f"{m.group(1)}1.0{m.group(3)}"
        new_text = scale_pat.sub(scale_repl, new_text, count=1)
    else:
        new_text = f"scale: 1.0\n{new_text}"
        changed = True

    # Add/Update original_scale
    os_pat = re.compile(r"^(\s*original_scale\s*:\s*)([0-9.]+)(\s*)$", flags=re.MULTILINE)
    os_value = _format_float_short(applied_scale)
    if os_pat.search(new_text):
        def os_repl(m: re.Match) -> str:
            nonlocal changed
            if m.group(2) != os_value:
                changed = True
            return f"{m.group(1)}{os_value}{m.group(3)}"
        new_text = os_pat.sub(os_repl, new_text, count=1)
    else:
        m = scale_pat.search(new_text)
        if m:
            insert_pos = m.end()
            new_text = new_text[:insert_pos] + f"\noriginal_scale: {os_value}" + new_text[insert_pos:]
        else:
            new_text = f"original_scale: {os_value}\n{new_text}"
        changed = True

    # Add/Update downscaled: true
    ds_pat = re.compile(r"^(\s*downscaled\s*:\s*)(\w+)(\s*)$", flags=re.MULTILINE)
    if ds_pat.search(new_text):
        def ds_repl(m: re.Match) -> str:
            nonlocal changed
            if m.group(2).lower() != "true":
                changed = True
            return f"{m.group(1)}true{m.group(3)}"
        new_text = ds_pat.sub(ds_repl, new_text, count=1)
    else:
        # Insert right after original_scale if present; else after scale; else at top.
        m_os = re.search(r"^(\s*original_scale\s*:\s*[0-9.]+\s*)$", new_text, flags=re.MULTILINE)
        if m_os:
            insert_pos = m_os.end()
            new_text = new_text[:insert_pos] + "\ndownscaled: true" + new_text[insert_pos:]
        else:
            m_scale = scale_pat.search(new_text)
            if m_scale:
                insert_pos = m_scale.end()
                new_text = new_text[:insert_pos] + "\ndownscaled: true" + new_text[insert_pos:]
            else:
                new_text = "downscaled: true\n" + new_text
        changed = True

    if dry_run:
        if text != new_text:
            print(f"[DRY ] Would update {yml_path.name}: image_height & eye_line × {applied_scale:.3f}, set scale=1.0, original_scale={os_value}, downscaled=true")
        return

    if changed:
        try:
            yml_path.write_text(new_text, encoding="utf-8")
            print(f"[INFO] Updated {yml_path} (image_height & eye_line × {applied_scale:.3f}, scale -> 1.0, original_scale, downscaled=true).")
        except Exception as e:
            print(f"[WARN] Failed to write {yml_path}: {e}")

def _append_original_scale_line(yml_path: Path, prev_scale: float, *, dry_run: bool = False) -> None:
    """
    If 'original_scale:' is not already present, append one line with the previous scale.
    Do not modify any other keys.
    """
    if not yml_path.exists():
        return
    try:
        text = yml_path.read_text(encoding="utf-8")
    except Exception:
        return

    if re.search(r"^\s*original_scale\s*:", text, flags=re.MULTILINE):
        return  # already present

    line = f"\noriginal_scale: {_format_float_short(prev_scale)}\n"
    if dry_run:
        print(f"[DRY ] Would append original_scale: {prev_scale} to {yml_path.name}")
        return
    try:
        yml_path.write_text(text + line, encoding="utf-8")
        print(f"[INFO] Added original_scale: {_format_float_short(prev_scale)} to {yml_path}")
    except Exception as e:
        print(f"[WARN] Failed to update {yml_path}: {e}")


# ---------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------

def _target_size(w: int, h: int, scale: float) -> Tuple[int, int]:
    """
    Compute target size via rounding to the nearest integer.
    """
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))

def process_character(
    char_dir: Path,
    *,
    lossless: bool,
    quality: int,
    dry_run: bool,
) -> None:
    yml_path = char_dir / "character.yml"
    meta = _load_yaml(yml_path)
    scale = float(meta.get("scale", 1.0))

    print(f"[INFO] {char_dir.name}: scale={scale:.3f}")

    for img_path in _find_pose_image_paths(char_dir):
        try:
            im = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            continue

        if scale == 1.0:
            print(f"[SKIP] {img_path.relative_to(char_dir)} — scale=1.0")
            continue
        if scale > 1.0:
            print(f"[SKIP] {img_path.relative_to(char_dir)} — scale>1.0 (not supported)")
            continue

        # Downscale to target size and overwrite in place
        tgt_w, tgt_h = _target_size(im.width, im.height, scale)
        print(f"[DO  ] {img_path.relative_to(char_dir)} -> {tgt_w}x{tgt_h} (downscale {scale:.3f}×)")
        if dry_run:
            continue
        out_img = _resize_linear_light(im, tgt_w, tgt_h)
        _save_image(img_path, out_img, lossless=lossless, quality=quality)

    # Only annotation: remember the original scale if we downscaled
    if scale < 1.0:
        _append_original_scale_line(yml_path, scale, dry_run=dry_run)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Downscale sprites to in-game size (downscale-only, in-place).")
    parser.add_argument("root", type=str, help="Path to finalized output root (characters here).")
    parser.add_argument("--lossless", action="store_true", help="Use lossless WebP when saving .webp (default: lossy).")
    parser.add_argument("--quality", type=int, default=DEFAULT_WEBP_QUALITY, help="Quality for lossy WebP/JPEG (default: 95).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] '{root}' is not a folder.")
        sys.exit(1)

    chars = list(_iter_character_dirs(root))
    if not chars:
        print("[WARN] No character folders found.")
        sys.exit(0)

    print(f"[INFO] Characters found: {len(chars)}")
    for cdir in chars:
        process_character(
            cdir,
            lossless=args.lossless,
            quality=args.quality,
            dry_run=args.dry_run,
        )
    print("\n[INFO] Downscale pass complete.")


if __name__ == "__main__":
    main()
