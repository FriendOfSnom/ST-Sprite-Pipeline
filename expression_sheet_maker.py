#!/usr/bin/env python3

"""
expression_sheet_maker.py

Generates expression sheet PNGs from organized sprite folders.

Usage:
    python expression_sheet_maker.py /path/to/output_folder

For each character, reads its scale value from character.yml and scales all
expressions accordingly.

Saves each sheet to the *pose folder* (alongside faces/ and outfits/), e.g.:

    <root>/<character>/<pose>/<pose>_sheet.png
"""

import os
import sys
import math
import yaml
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# CONFIGURATION
# -----------------------
GRID_COLUMNS = 3
MIN_GRID_ROWS = 5
PADDING = 20
LABEL_HEIGHT = 24
FONT_SIZE = 32

try:
    DEFAULT_FONT = ImageFont.truetype("arial.ttf", size=FONT_SIZE)
except Exception:
    DEFAULT_FONT = ImageFont.load_default()

# -----------------------
# Directory Traversal
# -----------------------
def get_all_pose_paths(root_folder):
    """
    Find all (character, pose, face_images_path) tuples under the given root.

    Returns:
        list[tuple[str, str, str]]
        character: folder name of the character
        pose:      folder name of the pose (e.g., 'a')
        face_images_path: full path to <pose>/faces/face
    """
    pose_paths = []
    for character in sorted(os.listdir(root_folder)):
        char_path = os.path.join(root_folder, character)
        if not os.path.isdir(char_path):
            continue

        for pose in sorted(os.listdir(char_path)):
            pose_path = os.path.join(char_path, pose)
            if not os.path.isdir(pose_path):
                continue

            face_images_path = os.path.join(pose_path, "faces", "face")
            if os.path.isdir(face_images_path):
                pose_paths.append((character, pose, face_images_path))

    return pose_paths

# -----------------------
# Image Loading with Numeric Sort
# -----------------------
def load_expression_images(face_images_path):
    """
    Load all expression images in a pose's 'faces/face' folder.
    Returns list of (label, PIL.Image).
    """
    def numeric_key(fname):
        base = os.path.splitext(fname)[0]
        try:
            return int(base)
        except ValueError:
            return float("inf")

    images = []
    for fname in sorted(os.listdir(face_images_path), key=numeric_key):
        if fname.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
            path = os.path.join(face_images_path, fname)
            try:
                img = Image.open(path).convert("RGBA")
                label = os.path.splitext(fname)[0]
                images.append((label, img))
            except Exception as e:
                print(f"[WARN] Couldn't load image {path}: {e}")
    return images

# -----------------------
# Canvas Sizing
# -----------------------
def calculate_sheet_size(image_size, count):
    """
    Given one sample image size and number of images,
    compute total (width, height) for the grid canvas.

    Args:
        image_size: (w, h) of a single expression image
        count:      number of expressions
    Returns:
        (total_width, total_height, rows)
    """
    img_w, img_h = image_size
    cols = GRID_COLUMNS
    rows = max(MIN_GRID_ROWS, math.ceil(count / cols))

    total_width = PADDING + cols * (img_w + PADDING)
    total_height = PADDING + rows * (img_h + LABEL_HEIGHT + PADDING)

    return total_width, total_height, rows

# -----------------------
# Draw Sheet
# -----------------------
def draw_expression_sheet(character, pose, images, output_path):
    """
    Generate a single expression sheet PNG for a pose and save to output_path.

    Args:
        character: character folder name (for logging only)
        pose:      pose folder name (e.g., 'a')
        images:    list[(label, PIL.Image)]
        output_path: full path to write the sheet PNG
    """
    if not images:
        print(f"[WARN] Skipping {character}/{pose}: no images found.")
        return

    sample_w, sample_h = images[0][1].size
    sheet_w, sheet_h, _ = calculate_sheet_size((sample_w, sample_h), len(images))

    sheet = Image.new("RGBA", (sheet_w, sheet_h), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, (label, img) in enumerate(images):
        row = idx // GRID_COLUMNS
        col = idx % GRID_COLUMNS

        x = PADDING + col * (sample_w + PADDING)
        y = PADDING + row * (sample_h + LABEL_HEIGHT + PADDING)

        sheet.paste(img, (x, y))

        # Center the label under the image
        text = f"{pose}_{label}"
        text_y = y + sample_h
        text_x_center = x + sample_w // 2
        if DEFAULT_FONT:
            bbox = draw.textbbox((0, 0), text, font=DEFAULT_FONT)
            text_w = bbox[2] - bbox[0]
            draw.text((text_x_center - text_w // 2, text_y), text, fill="black", font=DEFAULT_FONT)
        else:
            draw.text((text_x_center, text_y), text, fill="black")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)
    print(f"[INFO] Saved: {output_path}")

# -----------------------
# Main Function
# -----------------------
def main():
    """
    Walk the output root, read per-character scale, build a sheet for each pose,
    and save the sheet next to that pose's faces/ and outfits/ folders.
    """
    if len(sys.argv) < 2:
        print("Usage: python expression_sheet_maker.py /path/to/output_folder")
        sys.exit(1)

    root_folder = sys.argv[1]
    if not os.path.isdir(root_folder):
        print(f"[ERROR] '{root_folder}' is not a valid folder.")
        sys.exit(1)

    pose_entries = get_all_pose_paths(root_folder)
    if not pose_entries:
        print("[WARN] No poses found in the specified folder.")
        sys.exit(0)

    # Group by character
    characters = {}
    for character, pose, face_path in pose_entries:
        characters.setdefault(character, []).append((pose, face_path))

    for character, poses in characters.items():
        # Load scale from character.yml (default 1.0)
        char_folder = os.path.join(root_folder, character)
        yml_path = os.path.join(char_folder, "character.yml")
        scale = 1.0
        if os.path.isfile(yml_path):
            try:
                with open(yml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                scale = float(data.get("scale", 1.0))
                # If the downscaler already normalized images in place,
                # we avoid scaling a second time on the sheet.
                if "original_scale" in (data or {}):
                    scale = 1.0
                print(f"[INFO] Character {character} using sheet scale: {scale}")
            except Exception as e:
                print(f"[WARN] Could not read scale from {yml_path}: {e}")


        # Build each pose's sheet and save it *in that pose's folder*
        for pose, face_path in poses:
            images = load_expression_images(face_path)
            if not images:
                continue

            # Apply scale to all images
            scaled_images = []
            for label, img in images:
                if abs(scale - 1.0) > 0.01:
                    new_w = int(img.width * scale)
                    new_h = int(img.height * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                scaled_images.append((label, img))

            # face_path is .../<character>/<pose>/faces/face
            # Pose folder is two levels up from that
            pose_folder = os.path.dirname(os.path.dirname(face_path))
            out_path = os.path.join(pose_folder, f"{pose}_sheet.png")

            draw_expression_sheet(character, pose, scaled_images, out_path)

    print("\n[INFO] All expression sheets generated successfully.")

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    main()
