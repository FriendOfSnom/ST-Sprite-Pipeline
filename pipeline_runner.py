#!/usr/bin/env python3

"""
pipeline_runner.py

Main pipeline controller for the sprite processing tool.
This script allows the user to choose which step of the pipeline to start from,
and manages folder paths between steps to avoid redundant prompts.

Steps:
1. Downloader
2. Manual Sorting Helper
3. Organizer and Finalizer
"""

import sys
from downloader import run_downloader_interactive
from manual_sort_helper import run_manual_sort
from organize_sprites import run_organizer_interactive

def ask_user_continue(prompt):
    """
    Ask the user a yes/no question.
    Returns True if the user answers 'y' (case-insensitive).
    """
    answer = input(f"\n{prompt} (Y/N): ").strip().lower()
    return answer == 'y'

def main():
    print("=" * 60)
    print(" SPRITE PIPELINE CONTROLLER")
    print("=" * 60)
    print("\nWhich step do you want to start with?")
    print("1. Downloader")
    print("2. Manual Sorting Helper")
    print("3. Organizer and Finalizer")
    print("Q. Quit")

    choice = input("\nEnter your choice: ").strip().lower()

    if choice == 'q':
        print("\nExiting.")
        sys.exit(0)

    downloads_path = None
    sorted_path = None

    if choice == '1':
        # Start with Downloader
        downloads_path = run_downloader_interactive()

        if ask_user_continue("Continue to Step 2 (Manual Sorting)?"):
            sorted_path = run_manual_sort(downloads_path)
            if ask_user_continue("Continue to Step 3 (Organizer)?"):
                run_organizer_interactive(sorted_path)
            else:
                print("\nDone! You can run Step 3 later with that same folder.")
        else:
            print("\nDone! You can run Step 2 later with that same folder.")

    elif choice == '2':
        # Start with Manual Sorting directly
        downloads_path = input("\nEnter the path to your downloaded images folder:\n> ").strip()
        if not downloads_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        sorted_path = run_manual_sort(downloads_path)
        if ask_user_continue("Continue to Step 3 (Organizer)?"):
            run_organizer_interactive(sorted_path)
        else:
            print("\nDone! You can run Step 3 later with that same folder.")

    elif choice == '3':
        # Start with Organizer directly
        sorted_path = input("\nEnter the path to your manually sorted sprite folder:\n> ").strip()
        if not sorted_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        run_organizer_interactive(sorted_path)
    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nPipeline complete. Exiting.")

if __name__ == "__main__":
    main()
