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
4. Generate Expression Sheets
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
    print("4. Generate Expression Sheets")
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
                if ask_user_continue("Continue to Step 4 (Expression Sheets)?"):
                    import subprocess
                    try:
                        print("\n[INFO] Running Expression Sheet Generator...")
                        subprocess.run([sys.executable, "expression_sheet_maker.py", organizer_path], check=True)
                        print("\n[INFO] Expression sheets generated successfully!")
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
            if ask_user_continue("Continue to Step 4 (Expression Sheets)?"):
                import subprocess
                try:
                    print("\n[INFO] Running Expression Sheet Generator...")
                    subprocess.run([sys.executable, "expression_sheet_maker.py", organizer_path], check=True)
                    print("\n[INFO] Expression sheets generated successfully!")
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
        if ask_user_continue("Continue to Step 4 (Expression Sheets)?"):
            import subprocess
            try:
                print("\n[INFO] Running Expression Sheet Generator...")
                subprocess.run([sys.executable, "expression_sheet_maker.py", organizer_path], check=True)
                print("\n[INFO] Expression sheets generated successfully!")
            except Exception as e:
                print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

    elif choice == '4':
        # Step 4: Generate Expression Sheets
        sheets_path = input("\nEnter the path to the folder containing your character folders:\n> ").strip()
        if not sheets_path:
            print("\nERROR: Folder path is required. Exiting.")
            sys.exit(1)

        import subprocess
        try:
            print("\n[INFO] Running Expression Sheet Generator...")
            subprocess.run([sys.executable, "expression_sheet_maker.py", sheets_path], check=True)
            print("\n[INFO] Expression sheets generated successfully!")
        except Exception as e:
            print(f"\n[ERROR] Failed to run expression_sheet_maker.py: {e}")

    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nPipeline complete. Exiting.")

if __name__ == "__main__":
    main()
