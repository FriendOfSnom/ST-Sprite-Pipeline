#!/usr/bin/env python3
"""
mid_thigh_cropper.py

Batch tool that reuses the existing crop UI from organize_sprites.py
to set a mid-thigh cutoff for each pose of each character.

For every character folder under a root, and every pose folder under that
character, it:

  - Finds the outfit image in:  <char>/<pose>/outfits/
  - Auto-crops padding using compute_bbox / crop_to_bbox.
  - Shows the familiar Tk crop UI asking you to click at mid-thigh.
  - Crops the outfit image down to that height and overwrites it in place.

Faces and YAML files are left untouched – this only modifies the outfit
image in each pose.
"""

from pathlib import Path
import sys

from PIL import Image

# Reuse your existing UI + helpers from organize_sprites.py
from organize_sprites import (
    prompt_for_crop,
    compute_bbox,
    crop_to_bbox,
    make_thumbnail_of_crop,
)

# Allowed image extensions for outfit files
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def find_outfit_image(outfits_dir: Path) -> Path | None:
    """
    Find the first outfit image in a pose's outfits/ directory.

    Args:
        outfits_dir: Path to the outfits/ directory inside a pose.

    Returns:
        Path to the first usable image, or None if none found.
    """
    if not outfits_dir.is_dir():
        return None

    files = sorted(
        f for f in outfits_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        return None

    if len(files) > 1:
        print(f"[WARN] Multiple outfit images in {outfits_dir}, using first: {files[0].name}")

    return files[0]


def crop_pose_outfit_to_mid_thigh(
    pose_dir: Path,
    previous_leg_crops: list[tuple[int, "Image.Image"]],
) -> None:
    """
    For a single pose directory:

      - Find the outfit image.
      - Auto-crop padding.
      - Ask the user (via prompt_for_crop) for a mid-thigh cutoff.
      - Crop the outfit to that height and overwrite it in place.

    Args:
        pose_dir: Path to the pose directory (e.g. sakural/a).
        previous_leg_crops: List of (y_cut, thumbnail_image) pairs used by
                            the crop UI to show previous choices.
                            This list is modified in-place.
    """
    outfits_dir = pose_dir / "outfits"
    outfit_img_path = find_outfit_image(outfits_dir)
    if outfit_img_path is None:
        print(f"  [POSE {pose_dir.name}] No outfit image found, skipping.")
        return

    print(f"  [POSE {pose_dir.name}] Cropping outfit: {outfit_img_path.name}")

    # Load and auto-crop padding once
    original = Image.open(outfit_img_path).convert("RGBA")
    bbox = compute_bbox(original)
    cropped_base = crop_to_bbox(original, bbox)

    # Ask the user for the mid-thigh cutoff
    prompt = (
        f"[{pose_dir.parent.name}/{pose_dir.name}] "
        "Click at the height of the character's mid-thigh.\n"
        "Everything below this line will be removed."
    )
    leg_cut, used_gallery = prompt_for_crop(cropped_base, prompt, previous_leg_crops)

    # Clamp for safety
    leg_cut = max(1, min(int(leg_cut), cropped_base.height))

    # If this was a new crop, remember it for the gallery
    if not used_gallery:
        thumb = make_thumbnail_of_crop(cropped_base, leg_cut)
        previous_leg_crops.append((leg_cut, thumb))

    # Apply the same auto-crop + leg-cut to the actual file and overwrite it
    final_img = crop_to_bbox(original, bbox)
    final_img = final_img.crop((0, 0, final_img.width, leg_cut))
    try:
        final_img.save(outfit_img_path)
        print(f"    [OK] Saved cropped outfit to {outfit_img_path}")
    except Exception as e:
        print(f"    [ERROR] Failed to save cropped outfit {outfit_img_path}: {e}")


def process_character_folder(char_dir: Path) -> None:
    """
    Process all pose folders under a single character directory.

    Args:
        char_dir: Path to a character folder (contains a, b, c... pose folders).
    """
    print(f"\n[CHAR] {char_dir.name}")

    # Shared gallery list for this character so you can reuse recent crops
    previous_leg_crops: list[tuple[int, "Image.Image"]] = []

    for pose_dir in sorted(p for p in char_dir.iterdir() if p.is_dir()):
        # Skip non-pose directories if any
        if pose_dir.name.lower() in {"faces", "outfits"}:
            continue
        crop_pose_outfit_to_mid_thigh(pose_dir, previous_leg_crops)


def process_root(root_dir: Path) -> None:
    """
    Process every character folder directly under the given root directory.

    Args:
        root_dir: Path to a folder that contains character folders.
    """
    if not root_dir.is_dir():
        print(f"[ERROR] {root_dir} is not a directory.")
        return

    char_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not char_dirs:
        print(f"[WARN] No character folders found under {root_dir}.")
        return

    print(f"[INFO] Found {len(char_dirs)} character folder(s) under {root_dir}.")

    for char_dir in sorted(char_dirs):
        process_character_folder(char_dir)

    print("\n[INFO] Mid-thigh cropping complete for all characters/poses.")


def main() -> None:
    """
    CLI entry point.

    Usage:
        python mid_thigh_cropper.py /path/to/root_characters_folder

    If no path is provided as an argument, the script prompts for one.
    """
    if len(sys.argv) >= 2:
        root_path_str = sys.argv[1]
    else:
        root_path_str = input("Enter the path to the root folder of characters:\n> ").strip()

    if not root_path_str:
        print("[ERROR] No path provided.")
        sys.exit(1)

    root_dir = Path(root_path_str).expanduser().resolve()
    process_root(root_dir)


if __name__ == "__main__":
    main()
