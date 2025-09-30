#!/usr/bin/env python3
"""
pipeline_runner.py

Main pipeline controller for the sprite processing tool.
Lets the user choose which step to start from, and passes folder paths forward.

Steps:
1. Downloader
2. Manual Sorting Helper
3. Organizer and Finalizer
4. Downscale Sprites (optional)
5. Generate Expression Sheets
6. Organize Character Folders (Sprite Library)
Q. Quit

Design:
- Each step can run standalone on a chosen folder.
- When run in sequence, we pass the most-recent path forward automatically.
- Step 4 runs in-place downscaling.
"""

import os
import sys
import subprocess

from downloader import run_downloader_interactive
from manual_sort_helper import run_manual_sort
from organize_sprites import run_organizer_interactive

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _abs(path: str | None) -> str | None:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))

def _unique_dir_sibling(base_dir: str, desired_name: str) -> str:
    parent = os.path.dirname(_abs(base_dir))
    candidate = os.path.join(parent, desired_name)
    if not os.path.exists(candidate):
        return candidate
    i = 2
    while True:
        alt = f"{candidate}_{i}"
        if not os.path.exists(alt):
            return alt
        i += 1

def _default_finalized_dir(sorted_path: str) -> str:
    sorted_path = _abs(sorted_path)
    base = os.path.basename(sorted_path.rstrip("/\\"))
    return _unique_dir_sibling(sorted_path, f"{base}_finalized")

def ask_user_continue(prompt: str) -> bool:
    answer = input(f"\n{prompt} (Y/N): ").strip().lower()
    return answer == 'y'

def run_expression_sheets(root_path: str) -> None:
    print("\n[INFO] Running Expression Sheet Generator...")
    subprocess.run([sys.executable, "expression_sheet_maker.py", _abs(root_path)], check=True)
    print("\n[INFO] Expression sheets generated successfully!")

def run_bulk_downscale_interactive(default_root: str | None = None) -> str:
    print("=" * 60)
    print(" Sprite Bulk Downscaler (Step 4)")
    print("=" * 60)

    default_root = _abs(default_root)
    if default_root and os.path.isdir(default_root):
        root = default_root
        print(f"\n[INFO] Using folder from the previous step:\n  {root}")
    else:
        root_in = input("\nEnter the folder that contains your character folders:\n> ").strip()
        root = _abs(root_in)
        if not root or not os.path.isdir(root):
            print("\n[ERROR] The specified folder does not exist. Exiting.")
            sys.exit(1)

    cmd = [sys.executable, "bulk_downscale.py", root]

    print("\n[INFO] Running Sprite Bulk Downscaler (in-place)...")
    try:
        subprocess.run(cmd, check=True)
        print("\n[INFO] Downscale pass finished.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run bulk_downscale.py: {e}")
        sys.exit(1)

    return root

def _documents_folder() -> str:
    return _abs(os.path.join(os.path.expanduser("~"), "Documents"))

def run_sprite_library_interactive(default_root: str | None = None) -> None:
    """
    Step 6 interactive wrapper. Calls sprite_library_organizer.py.
    """
    print("=" * 60)
    print(" Organize Character Folders (Step 6)")
    print("=" * 60)

    # Source
    default_root = _abs(default_root)
    if default_root and os.path.isdir(default_root):
        source_root = default_root
        print(f"\n[INFO] Using folder from previous step:\n  {source_root}")
    else:
        source_root_in = input("\nEnter the path to the folder containing your ST-compatible character folder(s):\n> ").strip()
        source_root = _abs(source_root_in)
        if not source_root or not os.path.isdir(source_root):
            print("\n[ERROR] Folder path is required. Exiting.")
            sys.exit(1)

    # Library destination
    lib_default = os.path.join(_documents_folder() or "", "ST Sprite Library")
    lib_in = input(f"\nWhere should the Sprite Library live?\n"
                   f"[Press ENTER for default]\n  {lib_default}\n> ").strip()
    library_root = _abs(lib_in) if lib_in else _abs(lib_default)

    try:
        subprocess.run(
            [sys.executable, "sprite_library_organizer.py", source_root, "--library-root", library_root],
            check=True
        )
        print("\n[INFO] Sprite Library updated.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run sprite_library_organizer.py: {e}")
        sys.exit(1)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("=" * 60)
    print(" SPRITE PIPELINE CONTROLLER")
    print("=" * 60)
    print("\nWhich step do you want to start with?")
    print("1. Downloader")
    print("2. Manual Sorting Helper")
    print("3. Organizer and Finalizer")
    print("4. Downscale Sprites")
    print("5. Generate Expression Sheets")
    print("6. Organize Character Folders (Sprite Library)")
    print("Q. Quit")

    choice = input("\nEnter your choice: ").strip().lower()

    if choice == 'q':
        print("\nExiting.")
        sys.exit(0)

    downloads_path = None
    sorted_path = None
    organizer_output = None
    downscaled_root = None

    if choice == '1':
        downloads_path = _abs(run_downloader_interactive())

        if ask_user_continue("Continue to Step 2 (Manual Sorting)?"):
            sorted_path = _abs(run_manual_sort(downloads_path))

            if ask_user_continue("Continue to Step 3 (Organizer)?"):
                organizer_output = _default_finalized_dir(sorted_path)
                organizer_output = _abs(run_organizer_interactive(sorted_path, organizer_output))

                if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
                    downscaled_root = _abs(run_bulk_downscale_interactive(organizer_output))
                else:
                    downscaled_root = organizer_output

                if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
                    try:
                        run_expression_sheets(downscaled_root)
                    except Exception as e:
                        print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

                if ask_user_continue("Continue to Step 6 (Organize Character Folders)?"):
                    run_sprite_library_interactive(downscaled_root)

            else:
                print("\nDone! You can run Step 3 later with that same folder.")
        else:
            print("\nDone! You can run Step 2 later with that same folder.")

    elif choice == '2':
        downloads_path_in = input("\nEnter the path to your downloaded images folder:\n> ").strip()
        downloads_path = _abs(downloads_path_in)
        if not downloads_path or not os.path.isdir(downloads_path):
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        sorted_path = _abs(run_manual_sort(downloads_path))

        if ask_user_continue("Continue to Step 3 (Organizer)?"):
            organizer_output = _default_finalized_dir(sorted_path)
            organizer_output = _abs(run_organizer_interactive(sorted_path, organizer_output))

            if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
                downscaled_root = _abs(run_bulk_downscale_interactive(organizer_output))
            else:
                downscaled_root = organizer_output

            if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
                try:
                    run_expression_sheets(downscaled_root)
                except Exception as e:
                    print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

            if ask_user_continue("Continue to Step 6 (Organize Character Folders)?"):
                run_sprite_library_interactive(downscaled_root)
        else:
            print("\nDone! You can run Step 3 later with that same folder.")

    elif choice == '3':
        sorted_path_in = input("\nEnter the path to your manually sorted sprite folder:\n> ").strip()
        sorted_path = _abs(sorted_path_in)
        if not sorted_path or not os.path.isdir(sorted_path):
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        organizer_output = _default_finalized_dir(sorted_path)
        organizer_output = _abs(run_organizer_interactive(sorted_path, organizer_output))

        if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
            downscaled_root = _abs(run_bulk_downscale_interactive(organizer_output))
        else:
            downscaled_root = organizer_output

        if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
            try:
                run_expression_sheets(downscaled_root)
            except Exception as e:
                print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

        if ask_user_continue("Continue to Step 6 (Organize Character Folders)?"):
            run_sprite_library_interactive(downscaled_root)

    elif choice == '4':
        default_root_in = input("\nEnter the path to the folder containing your ST compatible character folder(s):\n> ").strip() or None
        scaled_root = _abs(run_bulk_downscale_interactive(default_root_in))

        if ask_user_continue("Downscaling complete. Continue to Step 5 (Expression Sheets)?"):
            try:
                run_expression_sheets(scaled_root)
            except Exception as e:
                print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

        if ask_user_continue("Continue to Step 6 (Organize Character Folders)?"):
            run_sprite_library_interactive(scaled_root)

    elif choice == '5':
        sheets_path_in = input("\nEnter the path to the folder containing your ST compatible character folder(s):\n> ").strip()
        sheets_path = _abs(sheets_path_in)
        if not sheets_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        try:
            run_expression_sheets(sheets_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

        if ask_user_continue("Continue to Step 6 (Organize Character Folders)?"):
            run_sprite_library_interactive(sheets_path)

    elif choice == '6':
        # Step 6 standalone
        run_sprite_library_interactive(None)

    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nPipeline complete. Exiting.")

if __name__ == "__main__":
    main()
