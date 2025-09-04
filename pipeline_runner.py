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
Q. Quit

Design:
- Each step can run standalone on a chosen folder.
- When run in sequence, we pass the most-recent path forward automatically.
- Step 4 uses a small interactive wrapper that hides CLI flags from end users.
"""

import os
import sys
import subprocess

from downloader import run_downloader_interactive
from manual_sort_helper import run_manual_sort
from organize_sprites import run_organizer_interactive


def ask_user_continue(prompt: str) -> bool:
    """
    Ask the user a yes/no question on the console.
    Returns True if the user answers 'y' (case-insensitive).
    """
    answer = input(f"\n{prompt} (Y/N): ").strip().lower()
    return answer == 'y'


def run_expression_sheets(root_path: str) -> None:
    """
    Launch the expression sheet generator as a separate process.

    Args:
        root_path: Path containing character folders to process.
    """
    print("\n[INFO] Running Expression Sheet Generator...")
    subprocess.run([sys.executable, "expression_sheet_maker.py", root_path], check=True)
    print("\n[INFO] Expression sheets generated successfully!")


def run_downscaler_interactive(default_root: str | None = None) -> str:
    """
    Interactive wrapper for the bulk downscaler (Step 4).

    Behavior changes:
      - If default_root is provided and exists, we use it directly with no path prompt.
      - Choice: In-place overwrite OR write to a separate copy.
      - No backups are ever created (per your requirement).
      - Returns the path that now contains the downscaled sprites.
    """
    print("=" * 60)
    print(" Sprite Bulk Downscaler (Step 4)")
    print("=" * 60)

    # Use previous step's folder automatically if provided
    if default_root and os.path.isdir(default_root):
        root = default_root
        print(f"\n[INFO] Using folder from the previous step:\n  {root}")
    else:
        # Standalone usage: ask for a folder (no example paths)
        root = input("\nEnter the folder that contains your character folders:\n> ").strip()
        if not root or not os.path.isdir(root):
            print("\n[ERROR] The specified folder does not exist. Exiting.")
            sys.exit(1)

    # Simple choice: in-place overwrite or write to a separate copy
    print("\nDownscale destination:")
    print("1) In-place overwrite (recommended)")
    print("2) Write to a new copy (choose a destination folder)")
    dest_choice = input("\nEnter your choice [1/2]: ").strip()

    cmd = [sys.executable, "bulk_downscale.py", root]

    if dest_choice == "2":
        dest_path = input("\nEnter destination folder for the downscaled copy:\n> ").strip()
        if not dest_path:
            print("\n[ERROR] Destination path cannot be empty. Exiting.")
            sys.exit(1)
        # Write only the downscaled assets + updated YAML into the copy
        cmd += ["--dest-root", dest_path]
        out_path = dest_path
    else:
        # In-place overwrite: do not keep backups inside character folders
        # (no flags added; bulk_downscale.py overwrites by default)
        out_path = root

    print("\n[INFO] Running Sprite Bulk Downscaler...")
    try:
        subprocess.run(cmd, check=True)
        print("\n[INFO] Downscale pass finished.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run bulk_downscale.py: {e}")
        sys.exit(1)

    return out_path



def main():
    print("=" * 60)
    print(" SPRITE PIPELINE CONTROLLER")
    print("=" * 60)
    print("\nWhich step do you want to start with?")
    print("1. Downloader")
    print("2. Manual Sorting Helper")
    print("3. Organizer and Finalizer")
    print("4. Downscale Sprites (optional)")
    print("5. Generate Expression Sheets")
    print("Q. Quit")

    choice = input("\nEnter your choice: ").strip().lower()

    if choice == 'q':
        print("\nExiting.")
        sys.exit(0)

    downloads_path = None
    sorted_path = None

    if choice == '1':
        # Step 1: Downloader
        downloads_path = run_downloader_interactive()

        if ask_user_continue("Continue to Step 2 (Manual Sorting)?"):
            sorted_path = run_manual_sort(downloads_path)

            if ask_user_continue("Continue to Step 3 (Organizer)?"):
                organizer_path = run_organizer_interactive(sorted_path)

                # Step 4 (optional)
                if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
                    downscaled_root = run_downscaler_interactive(organizer_path)
                else:
                    downscaled_root = organizer_path

                # Step 5 always offered next
                if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
                    try:
                        run_expression_sheets(downscaled_root)
                    except Exception as e:
                        print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")
            else:
                print("\nDone! You can run Step 3 later with that same folder.")
        else:
            print("\nDone! You can run Step 2 later with that same folder.")

    elif choice == '2':
        # Step 2: Manual Sorting
        downloads_path = input("\nEnter the path to your downloaded images folder:\n> ").strip()
        if not downloads_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        sorted_path = run_manual_sort(downloads_path)

        if ask_user_continue("Continue to Step 3 (Organizer)?"):
            organizer_path = run_organizer_interactive(sorted_path)

            # Step 4 (optional)
            if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
                downscaled_root = run_downscaler_interactive(organizer_path)
            else:
                downscaled_root = organizer_path

            # Step 5 always offered
            if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
                try:
                    run_expression_sheets(downscaled_root)
                except Exception as e:
                    print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")
        else:
            print("\nDone! You can run Step 3 later with that same folder.")

    elif choice == '3':
        # Step 3: Organizer and Finalizer
        sorted_path = input("\nEnter the path to your manually sorted sprite folder:\n> ").strip()
        if not sorted_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        organizer_path = run_organizer_interactive(sorted_path)

        # Step 4 (optional)
        if ask_user_continue("Continue to Step 4 (Downscale Sprites)?"):
            downscaled_root = run_downscaler_interactive(organizer_path)
        else:
            downscaled_root = organizer_path

        # Step 5 always offered
        if ask_user_continue("Continue to Step 5 (Expression Sheets)?"):
            try:
                run_expression_sheets(downscaled_root)
            except Exception as e:
                print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

    elif choice == '4':
        # Step 4: Downscale Sprites (standalone)
        default_root = input("\nOptionally enter a default path to offer (or leave blank):\n> ").strip() or None
        downscaled_root = run_downscaler_interactive(default_root)

        # After Step 4, offer Step 5
        if ask_user_continue("Downscale complete. Continue to Step 5 (Expression Sheets)?"):
            try:
                run_expression_sheets(downscaled_root)
            except Exception as e:
                print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

    elif choice == '5':
        # Step 5: Generate Expression Sheets (standalone)
        sheets_path = input("\nEnter the path to the folder containing your character folders:\n> ").strip()
        if not sheets_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        try:
            run_expression_sheets(sheets_path)
        except Exception as e:
            print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nPipeline complete. Exiting.")


if __name__ == "__main__":
    main()
