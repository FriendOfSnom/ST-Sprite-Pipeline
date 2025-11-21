#!/usr/bin/env python3

"""
manual_sort_helper.py

This script guides the user through the manual sorting step
of the sprite processing pipeline.

It does not modify files itself—it simply helps the user open
the correct folder and gives clear instructions on how
to organize the sprites.

It can be run standalone (interactive) or imported and called
from another script (e.g., pipeline_runner) with an existing path.
"""

import os
import sys
import platform
import subprocess


def open_file_explorer(path):
    """
    Opens the given folder in the OS's default file explorer.

    Supports Windows, macOS, and Linux.
    """
    system_name = platform.system()
    if system_name == "Windows":
        os.startfile(path)
    elif system_name == "Darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


def run_manual_sort(input_path=None) -> str:
    """
    Guides the user through the manual sprite sorting step.

    - If input_path is provided, uses it directly.
    - Otherwise, prompts the user to enter a folder path.

    Returns:
        The path to the folder containing the manually sorted sprites.
    """
    print()
    print("=" * 60)
    print(" MANUAL SORTING HELPER")
    print("=" * 60)

    # Prompt for path if not provided
    if input_path is None:
        input_path = input(
            "\nPlease enter the full path to your folder of downloaded sprites:\n> "
        ).strip()

    # Verify the path exists
    if not os.path.exists(input_path):
        print("\nERROR: The specified path does not exist. Please check it and try again.")
        sys.exit(1)

    print(f"\nUsing folder: {input_path}")

    # Provide sorting instructions
    print("\nInstructions for Organizing:")
    print("- Organize the sprites in this folder into the following structure (folder names are flexible):")
    print("  CharacterFolder/")
    print("      PoseOrOutfit/")
    print("          sprite1.png, sprite2.png, ...")
    print("- Create one folder per character.")
    print("- Inside each character folder, create folders for each pose or outfit.")
    print("- Place the appropriate images in each pose or outfit folder.")
    print("\nWhen you are finished organizing, return here and press ENTER to continue.")

    # Offer to open folder in file explorer
    open_choice = input("\nWould you like to open this folder in your file explorer now? (Y/N): ").strip().lower()
    if open_choice == 'y':
        open_file_explorer(input_path)
        print("\nThe folder has been opened. Please organize your sprites in the required structure before continuing.")

    input("\nPress ENTER when you have finished organizing your sprites...")

    print("\nManual sorting step complete.")
    return str(input_path)


if __name__ == "__main__":
    run_manual_sort()
