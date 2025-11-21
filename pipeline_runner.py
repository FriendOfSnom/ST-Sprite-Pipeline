#!/usr/bin/env python3
"""
pipeline_runner.py

Simple hub script for the sprite tools.

Menu:
  1. Run Gemini Sprite Character Creator
     - Asks for an output folder using a system folder picker.
     - Calls gemini_sprite_pipeline.run_pipeline() with that folder.

  2. Generate Expression Sheets for Existing Sprites
     - Asks for a root sprite folder using a system folder picker.
     - Calls expression_sheet_maker.py <root folder> so that expression
       sheets are written into each pose folder under that root.

  Q. Quit
"""

import os
import sys
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

from gemini_sprite_pipeline import run_pipeline  # core character creator


def pick_directory(title: str) -> str | None:
    """
    Try to open a native folder-chooser dialog (Explorer/Finder/etc.) and
    return the chosen directory as a string.

    If the dialog fails or the user cancels, fall back to asking for a
    path via stdin. Returns None if the user declines both.
    """
    print("[INFO] Opening a folder picker window. If you do not see it,")
    print("       check your taskbar or behind other windows.")

    folder: str | None = None

    try:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception as e:
        print(f"[WARN] Tk folder dialog failed: {e}")
        folder = None

    if folder:
        return os.path.abspath(os.path.expanduser(folder))

    # At this point either:
    #  - user clicked Cancel in the dialog, or
    #  - the dialog failed to appear / threw an exception.
    print("[INFO] No folder selected from the GUI dialog.")
    resp = input("Would you like to type a folder path manually? [y/N]: ").strip().lower()
    if resp != "y":
        return None

    typed = input("Enter full folder path (or leave empty to abort): ").strip()
    if not typed:
        return None

    return os.path.abspath(os.path.expanduser(typed))

def script_root() -> Path:
    """
    Return the directory where this script lives.
    Used to locate expression_sheet_maker.py reliably.
    """
    return Path(__file__).resolve().parent


def run_character_creator() -> None:
    """
    Ask the user where the new character sprite folder(s) should be created,
    then call the Gemini sprite pipeline directly as a function.
    """
    print("\n[Character Creator] Choose where to place the new character folder(s).")
    out_dir_str = pick_directory("Choose output folder for new character sprites")
    if out_dir_str is None:
        print("[INFO] No folder selected; returning to menu.")
        return

    out_dir = Path(out_dir_str)
    print(f"[INFO] Output folder selected: {out_dir}")

    # Call into the pipeline directly (no subprocess), so Tk windows appear
    # in the same process and we avoid argument/path issues.
    try:
        run_pipeline(out_dir, game_name=None)
    except SystemExit as e:
        # If the pipeline exits via SystemExit (e.g., user cancels), we catch it
        # so that the hub can continue running.
        print(f"[INFO] Character pipeline exited (code={e.code}). Returning to menu.")


def run_expression_sheet_generator() -> None:
    """
    Ask the user for a sprite root folder, then call expression_sheet_maker.py
    on that folder so that expression sheets are generated for all characters
    under it.
    """
    print("\n[Expression Sheets] Choose the root folder containing your character folders.")
    root_dir_str = pick_directory("Choose root folder for existing character sprites")
    if root_dir_str is None:
        print("[INFO] No folder selected; returning to menu.")
        return

    if not os.path.isdir(root_dir_str):
        print(f"[ERROR] '{root_dir_str}' is not a valid folder.")
        return

    print(f"[INFO] Generating expression sheets under: {root_dir_str}")

    root = script_root()
    script_path = root / "expression_sheet_maker.py"
    if not script_path.is_file():
        print(f"[ERROR] Could not find expression_sheet_maker.py at: {script_path}")
        return

    cmd = [sys.executable, str(script_path), root_dir_str]
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] expression_sheet_maker.py failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[ERROR] Could not run expression_sheet_maker.py: {e}")


def main() -> None:
    """
    Display a simple numeric menu that lets the user choose whether to:
      1) Run the Gemini sprite character creator, or
      2) Generate expression sheets for existing sprites.
    """
    while True:
        print("\n" + "=" * 60)
        print(" SPRITE TOOL HUB")
        print("=" * 60)
        print("1. Create a new character (Gemini Sprite Pipeline)")
        print("2. Generate expression sheets for existing character sprites")
        print("Q. Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "1":
            run_character_creator()
        elif choice == "2":
            run_expression_sheet_generator()
        elif choice == "q":
            print("\nExiting.")
            break
        else:
            print("\n[WARN] Invalid choice; please enter 1, 2, or Q.")


if __name__ == "__main__":
    main()
