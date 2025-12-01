#!/usr/bin/env python3
"""
set_global_scale_from_sample.py

Utility to unify the 'scale' value across a set of characters based on a
single "average" cropped sprite image.

Workflow:

  1. You provide the root folder containing character subfolders.
  2. A file picker opens; you choose one cropped sprite image from that set
     that you consider the "average" size.
  3. The script calls prompt_for_scale(image_path) from organize_sprites.py,
     so you can match it against your reference sprites and pick a scale.
  4. That chosen scale value is then written into every character.yml found
     directly under the root folder.

Only the 'scale' field is changed; all other YAML data is preserved.
"""

from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog

import yaml

# Reuse your scale UI
from organize_sprites import prompt_for_scale


def choose_reference_image(initial_dir: Path) -> Path | None:
    """
    Open a native file picker so the user can choose the "average" sprite image.

    Args:
        initial_dir: Directory to open the dialog in initially.

    Returns:
        Path to the selected image file, or None if the user cancels.
    """
    root = tk.Tk()
    root.withdraw()  # do not show a blank root window

    filetypes = [
        ("Image files", "*.png *.jpg *.jpeg *.webp"),
        ("All files", "*.*"),
    ]

    filename = filedialog.askopenfilename(
        title="Choose the average character image to base scale on",
        initialdir=str(initial_dir),
        filetypes=filetypes,
    )
    root.destroy()

    if not filename:
        return None

    return Path(filename).expanduser().resolve()


def update_character_scale(char_dir: Path, new_scale: float) -> None:
    """
    Load the character.yml in a character folder, update its 'scale' field,
    and write it back.

    Args:
        char_dir: Path to a character folder (contains character.yml).
        new_scale: The scale value to set for this character.
    """
    yml_path = char_dir / "character.yml"
    if not yml_path.is_file():
        print(f"  [WARN] No character.yml in {char_dir}, skipping.")
        return

    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"  [ERROR] Failed to read {yml_path}: {e}")
        return

    data["scale"] = float(new_scale)

    try:
        with open(yml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        print(f"  [OK] Updated scale in {yml_path} to {new_scale:.3f}")
    except Exception as e:
        print(f"  [ERROR] Failed to write {yml_path}: {e}")


def process_root(root_dir: Path) -> None:
    """
    Top-level workflow for setting a global scale.

    Args:
        root_dir: Folder containing character subfolders with character.yml.
    """
    if not root_dir.is_dir():
        print(f"[ERROR] {root_dir} is not a directory.")
        return

    # Step 1: let the user pick the reference image from this tree
    ref_image = choose_reference_image(root_dir)
    if ref_image is None:
        print("[ERROR] No reference image selected; aborting.")
        return

    print(f"[INFO] Selected reference image: {ref_image}")

    # Step 2: use your existing scale UI to choose the scale for this image
    # We do not have an eye-line ratio here, so we pass None.
    scale_value = prompt_for_scale(str(ref_image), user_eye_line_ratio=None)
    print(f"[INFO] Chosen global scale value: {scale_value:.3f}")

    # Step 3: update every character.yml directly under the root
    char_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    if not char_dirs:
        print(f"[WARN] No character folders found under {root_dir}.")
        return

    print(f"[INFO] Updating scale in {len(char_dirs)} character(s).")

    for char_dir in sorted(char_dirs):
        update_character_scale(char_dir, scale_value)

    print("\n[INFO] All character.yml files updated with the new scale.")


def main() -> None:
    """
    CLI entry point.

    Usage:
        python set_global_scale_from_sample.py /path/to/root_characters_folder

    If no path is given on the command line, the script will prompt for one.
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
