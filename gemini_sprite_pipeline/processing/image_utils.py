"""
Image utility functions for saving, loading, and manipulating sprite images.
"""

from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image

from gsp_constants import REF_SPRITES_DIR


def save_img_webp_or_png(img: Image.Image, dest_stem: Path) -> Path:
    """
    Save image as WEBP lossless, falling back to PNG if needed.

    Args:
        img: PIL Image to save.
        dest_stem: Destination path without extension.

    Returns:
        Path to saved file (with appropriate extension).
    """
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    safe_img = img.convert("RGBA")

    try:
        out_path = dest_stem.with_suffix(".webp")
        safe_img.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
        return out_path
    except Exception as e:
        print(f"[WARN] WEBP save failed for {dest_stem.name}: {e}. Falling back to PNG.")
        out_path = dest_stem.with_suffix(".png")
        safe_img.save(out_path, format="PNG")
        return out_path


def save_image_bytes_as_png(image_bytes: bytes, dest_stem: Path) -> Path:
    """
    Save raw image bytes as PNG to dest_stem.png.

    Args:
        image_bytes: Raw image data.
        dest_stem: Destination path without extension.

    Returns:
        Path to saved PNG file.
    """
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    out_path = dest_stem.with_suffix(".png")
    img.save(out_path, format="PNG")
    return out_path


def get_unique_folder_name(base_path: Path, desired_name: str) -> str:
    """
    Ensure folder name is unique within base_path by appending a counter.

    Args:
        base_path: Parent directory.
        desired_name: Desired folder name.

    Returns:
        Unique folder name (may have _2, _3, etc. appended).
    """
    candidate = desired_name
    counter = 1
    while (base_path / candidate).exists():
        counter += 1
        candidate = f"{desired_name}_{counter}"
    return candidate


def pick_representative_outfit(char_dir: Path) -> Path:
    """
    Choose a full-body outfit image to use for eye-line and scale selection.

    Preference:
        a/outfits/Base|Formal|Casual.(webp|png)
    Fallback:
        a/base.* or any image under char_dir.

    Args:
        char_dir: Character directory.

    Returns:
        Path to representative outfit image.

    Raises:
        RuntimeError: If no suitable image found.
    """
    a_dir = char_dir / "a"
    outfits_dir = a_dir / "outfits"
    preferred_names = [
        "Base.webp", "Formal.webp", "Casual.webp",
        "Base.png", "Formal.png", "Casual.png",
    ]

    if outfits_dir.is_dir():
        for name in preferred_names:
            candidate = outfits_dir / name
            if candidate.is_file():
                return candidate
        for p in sorted(outfits_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp"):
                return p

    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = (a_dir / "base").with_suffix(ext)
        if candidate.is_file():
            return candidate

    for p in char_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
            return p

    raise RuntimeError(f"No representative outfit image found in {char_dir}")


def get_reference_images_for_archetype(archetype_label: str) -> List[Path]:
    """
    Return all reference sprites for this archetype so Gemini can lock onto the style.

    Preference:
      1) reference_sprites/<archetype_label>/
      2) PNGs directly under reference_sprites/

    Args:
        archetype_label: Character archetype (e.g., "young woman").

    Returns:
        List of paths to reference images.
    """
    paths: List[Path] = []

    arch_dir = REF_SPRITES_DIR / archetype_label
    if arch_dir.is_dir():
        for p in sorted(arch_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                paths.append(p)

    # Fallback: generic refs at top level if the specific folder is empty
    if not paths and REF_SPRITES_DIR.is_dir():
        for p in sorted(REF_SPRITES_DIR.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                paths.append(p)

    return paths


def get_standard_uniform_reference_images(
    gender_style: str,
    max_images: int = 5,
) -> List[Path]:
    """
    Return a small set of reference images for the standardized school uniform.

    For girls (gender_style='f'), looks in: reference_sprites/young_woman_uniform
    For boys (gender_style='m'), looks in: reference_sprites/young_man_uniform

    Args:
        gender_style: 'f' or 'm'.
        max_images: Maximum number of reference images to return.

    Returns:
        List of paths to uniform reference images.
    """
    if gender_style == "m":
        folder_name = "young_man_uniform"
    else:
        folder_name = "young_woman_uniform"

    uniform_dir = REF_SPRITES_DIR / folder_name
    refs: List[Path] = []

    if uniform_dir.is_dir():
        for p in sorted(uniform_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                refs.append(p)
                if len(refs) >= max_images:
                    break

    return refs
