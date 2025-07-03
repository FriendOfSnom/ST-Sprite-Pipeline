#!/usr/bin/env python3

"""
expression_sheet_maker.py

Generates expression sheet PNGs from organized sprite folders.

Usage:
    python expression_sheet_maker.py /path/to/output_folder

It scans a structure like:

    <root>/<character>/<pose>/faces/face/<expressions>

and creates for each pose a single PNG sheet image with all expressions
laid out in a grid (3 columns wide, 5 rows minimum, expanding as needed),
with labels under each image (e.g., "a_24").

All sheets are saved into:

    <root>/expression_sheets/

Example:
    python expression_sheet_maker.py ./final_sprites_output
"""

import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont

# -----------------------
# CONFIGURATION
# -----------------------
GRID_COLUMNS = 4
MIN_GRID_ROWS = 5
PADDING = 20
LABEL_HEIGHT = 30
OUTPUT_FOLDER_NAME = "expression_sheets"

try:
    DEFAULT_FONT = ImageFont.load_default()
except:
    DEFAULT_FONT = None

# -----------------------
# Directory Traversal
# -----------------------
def get_all_pose_paths(root_folder):
    """
    Traverse the root folder to find all (character, pose, face_images_path) tuples.
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

            face_images_path = os.path.join(pose_path, 'faces', 'face')
            if os.path.isdir(face_images_path):
                pose_paths.append( (character, pose, face_images_path) )

    return pose_paths

# -----------------------
# Image Loading
# -----------------------
def load_expression_images(face_images_path):
    """
    Load all expression images in a pose's 'faces/face' folder.
    Returns list of (label, PIL.Image).
    """
    images = []
    for fname in sorted(os.listdir(face_images_path)):
        if fname.lower().endswith(('.png', '.webp', '.jpg', '.jpeg')):
            path = os.path.join(face_images_path, fname)
            try:
                img = Image.open(path).convert('RGBA')
                label = os.path.splitext(fname)[0]
                images.append( (label, img) )
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
    Generate a single expression sheet PNG for a pose.
    """
    if not images:
        print(f"[WARN] Skipping {character}/{pose}: no images found.")
        return

    sample_w, sample_h = images[0][1].size
    sheet_w, sheet_h, grid_rows = calculate_sheet_size((sample_w, sample_h), len(images))

    sheet = Image.new('RGBA', (sheet_w, sheet_h), color=(255, 255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for idx, (label, img) in enumerate(images):
        row = idx // GRID_COLUMNS
        col = idx % GRID_COLUMNS

        x = PADDING + col * (sample_w + PADDING)
        y = PADDING + row * (sample_h + LABEL_HEIGHT + PADDING)

        sheet.paste(img, (x, y))

        text_y = y + sample_h
        text_x_center = x + sample_w // 2
        text = f"{pose}_{label}"

        if DEFAULT_FONT:
            text_w, text_h = draw.textsize(text, font=DEFAULT_FONT)
            draw.text((text_x_center - text_w // 2, text_y), text, fill='black', font=DEFAULT_FONT)
        else:
            draw.text((text_x_center, text_y), text, fill='black')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path)
    print(f"[INFO] Saved: {output_path}")

# -----------------------
# Main Function
# -----------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python expression_sheet_maker.py /path/to/output_folder")
        sys.exit(1)

    root_folder = sys.argv[1]
    if not os.path.isdir(root_folder):
        print(f"[ERROR] '{root_folder}' is not a valid folder.")
        sys.exit(1)

    out_root = os.path.join(root_folder, OUTPUT_FOLDER_NAME)
    os.makedirs(out_root, exist_ok=True)

    pose_entries = get_all_pose_paths(root_folder)
    if not pose_entries:
        print("[WARN] No poses found in the specified folder.")
        sys.exit(0)

    for character, pose, face_path in pose_entries:
        images = load_expression_images(face_path)
        if not images:
            continue

        out_name = f"{character}_{pose}_sheet.png"
        out_path = os.path.join(out_root, out_name)
        draw_expression_sheet(character, pose, images, out_path)

    print("\n[INFO] All expression sheets generated successfully.")

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    main()
