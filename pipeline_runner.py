#!/usr/bin/env python3
"""
pipeline_runner.py

Simple hub script for the visual novel development tools.

Menu:
  1. Create New Ren'Py Project (Tool 1)
     - Creates a new Ren'Py visual novel project with character support
     - Optionally imports existing characters

  2. Create a New Character (Tool 2)
     - Run Gemini Sprite Character Creator
     - Asks for an output folder using a system folder picker
     - Calls gemini_sprite_pipeline.run_pipeline() with that folder

  3. Write Scenes Visually (Tool 3)
     - Launches the Visual Scene Editor
     - GUI-based scene creation with drag-drop characters
     - Automatically generates Ren'Py code

  4. Generate Expression Sheets (Utility)
     - Asks for a root sprite folder using a system folder picker
     - Calls expression_sheet_maker.py <root folder> so that expression
       sheets are written into each pose folder under that root

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


def run_project_scaffolder() -> None:
    """
    Create a new Ren'Py project with the character system pre-configured.
    """
    print("\n[Project Scaffolder] Creating a new Ren'Py visual novel project...")

    root = script_root()
    script_path = root / "renpy_project_scaffolder.py"
    if not script_path.is_file():
        print(f"[ERROR] Could not find renpy_project_scaffolder.py at: {script_path}")
        return

    cmd = [sys.executable, str(script_path)]
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Project scaffolder failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[ERROR] Could not run project scaffolder: {e}")


def run_visual_scene_editor() -> None:
    """
    Launch the Visual Scene Editor (Tool 3) for GUI-based scene creation.
    """
    print("\n[Visual Scene Editor] Launching Tool 3...")

    root = script_root()
    script_path = root / "visual_scene_editor.py"
    if not script_path.is_file():
        print(f"[ERROR] Could not find visual_scene_editor.py at: {script_path}")
        return

    cmd = [sys.executable, str(script_path)]
    print(f"[DEBUG] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Visual Scene Editor failed with exit code {e.returncode}")
    except Exception as e:
        print(f"[ERROR] Could not run visual scene editor: {e}")


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
    Display a simple numeric menu that lets the user choose:
      1) Create new Ren'Py project
      2) Run the Gemini sprite character creator
      3) Visual Scene Editor (GUI-based scene creation)
      4) Generate expression sheets for existing sprites
    """
    while True:
        print("\n" + "=" * 70)
        print(" VISUAL NOVEL DEVELOPMENT TOOLKIT")
        print("=" * 70)
        print("1. Create new Ren'Py project (Tool 1 - Project Scaffolder)")
        print("2. Create new character sprites (Tool 2 - Gemini Character Creator)")
        print("3. Write scenes visually (Tool 3 - Visual Scene Editor)")
        print("4. Generate expression sheets (Utility)")
        print("Q. Quit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "1":
            run_project_scaffolder()
        elif choice == "2":
            run_character_creator()
        elif choice == "3":
            run_visual_scene_editor()
        elif choice == "4":
            run_expression_sheet_generator()
        elif choice == "q":
            print("\nExiting.")
            break
        else:
            print("\n[WARN] Invalid choice; please enter 1, 2, 3, 4, or Q.")


if __name__ == "__main__":
    main()
