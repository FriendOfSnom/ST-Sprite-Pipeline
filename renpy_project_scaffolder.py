#!/usr/bin/env python3
"""
renpy_project_scaffolder.py

Tool 1: Ren'Py Project Scaffolder

Creates a new Ren'Py visual novel project using the official Ren'Py SDK,
then injects the custom character system for compatibility with Tool 2.

This tool:
- Launches the official Ren'Py launcher
- Lets user create a project through the official interface
- Injects custom character loading system files
- Optionally imports existing character folders
- Moves project to user's chosen location
"""

import os
import platform
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


# Ren'Py SDK configuration
SDK_VERSION = "8.5.0"
SDK_FOLDER_NAME = f"renpy-{SDK_VERSION}-sdk"
SDK_URLS = {
    "Windows": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.zip",
    "Darwin": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.tar.bz2",
    "Linux": "https://www.renpy.org/dl/8.5.0/renpy-8.5.0-sdk.tar.bz2",
}


def show_progress(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)

        size_mb = total_size / (1024 * 1024)
        downloaded_mb = downloaded / (1024 * 1024)

        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)')
        sys.stdout.flush()


def download_and_extract_sdk(script_dir):
    """Download and extract the Ren'Py SDK."""
    system = platform.system()

    if system not in SDK_URLS:
        print(f"[ERROR] Unsupported platform: {system}")
        return None

    print("\n" + "=" * 70)
    print(" Downloading Ren'Py SDK")
    print("=" * 70)
    print(f"Platform: {system}")
    print(f"Version: {SDK_VERSION}")

    download_url = SDK_URLS[system]
    file_extension = ".zip" if system == "Windows" else ".tar.bz2"
    download_filename = f"renpy-{SDK_VERSION}-sdk{file_extension}"
    download_path = script_dir / download_filename

    # Download
    print(f"\nDownloading from: {download_url}")
    print("This may take a few minutes (~150 MB)...\n")

    try:
        urlretrieve(download_url, download_path, reporthook=show_progress)
        print("\n[INFO] Download complete!")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return None

    # Extract
    print("\n[INFO] Extracting SDK...")
    try:
        if file_extension == ".zip":
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                total = len(members)
                for i, member in enumerate(members):
                    zip_ref.extract(member, script_dir)
                    percent = (i + 1) * 100 / total
                    sys.stdout.write(f'\rExtracting: {percent:.1f}%')
                    sys.stdout.flush()
        else:
            with tarfile.open(download_path, 'r:bz2') as tar_ref:
                members = tar_ref.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    tar_ref.extract(member, script_dir)
                    percent = (i + 1) * 100 / total
                    sys.stdout.write(f'\rExtracting: {percent:.1f}%')
                    sys.stdout.flush()

        print("\n[INFO] Extraction complete!")

        # Cleanup
        download_path.unlink()
        print(f"[INFO] Cleaned up download file")

        sdk_path = script_dir / SDK_FOLDER_NAME
        if sdk_path.exists():
            return sdk_path
        else:
            print(f"[ERROR] SDK folder not found after extraction: {sdk_path}")
            return None

    except Exception as e:
        print(f"\n[ERROR] Extraction failed: {e}")
        return None


def find_renpy_sdk():
    """Find the Ren'Py SDK folder."""
    script_dir = Path(__file__).parent
    sdk_path = script_dir / SDK_FOLDER_NAME

    if sdk_path.exists() and sdk_path.is_dir():
        return sdk_path

    # Try to find any renpy SDK folder
    for item in script_dir.iterdir():
        if item.is_dir() and item.name.startswith("renpy") and "sdk" in item.name.lower():
            return item

    return None


def get_renpy_executable(sdk_path):
    """Get the appropriate Ren'Py executable for this platform."""
    system = platform.system()

    if system == "Windows":
        exe = sdk_path / "renpy.exe"
        if exe.exists():
            return exe
    elif system == "Darwin":  # macOS
        exe = sdk_path / "renpy.sh"
        if exe.exists():
            return exe
    else:  # Linux
        exe = sdk_path / "renpy.sh"
        if exe.exists():
            return exe

    return None


def launch_renpy_launcher(sdk_path, renpy_exe):
    """Launch the Ren'Py launcher and wait for it to close."""
    print("\n" + "=" * 70)
    print(" Launching Ren'Py")
    print("=" * 70)
    print(f"SDK Location: {sdk_path}")
    print(f"Executable: {renpy_exe}")
    print("\nInstructions:")
    print("1. Click 'Create New Project' in the Ren'Py launcher")
    print("2. Enter your project name")
    print("3. Choose resolution (1280x720 recommended)")
    print("4. Pick a color scheme")
    print("5. Wait for project creation to complete")
    print("6. Close the Ren'Py launcher when done")
    print("\nLaunching Ren'Py now...")
    print("=" * 70)

    try:
        # Launch Ren'Py
        subprocess.run([str(renpy_exe)], cwd=str(sdk_path))
        print("\n[INFO] Ren'Py launcher closed")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to launch Ren'Py: {e}")
        return False


def find_projects(sdk_path):
    """Find all projects in the SDK's projects folder."""
    projects_dir = sdk_path / "projects"

    if not projects_dir.exists():
        return []

    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir():
            # Check if it's a valid Ren'Py project (has a game folder)
            game_dir = item / "game"
            if game_dir.exists() and game_dir.is_dir():
                projects.append(item)

    return projects


def select_project(projects):
    """Let user select which project to configure."""
    if not projects:
        print("[ERROR] No projects found in SDK/projects/")
        return None

    if len(projects) == 1:
        # Only one project, ask for confirmation
        root = tk.Tk()
        root.withdraw()
        confirm = messagebox.askyesno(
            "Configure Project",
            f"Found project: {projects[0].name}\n\nConfigure this project with character system?"
        )
        root.destroy()
        return projects[0] if confirm else None

    # Multiple projects, let user choose
    root = tk.Tk()
    root.title("Select Project to Configure")
    root.geometry("500x400")

    selected = [None]

    tk.Label(
        root,
        text="Select which project to configure:",
        font=("Arial", 12, "bold")
    ).pack(pady=20)

    listbox = tk.Listbox(root, font=("Arial", 11), height=15)
    listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    for proj in projects:
        listbox.insert(tk.END, proj.name)

    def on_ok():
        selection = listbox.curselection()
        if selection:
            selected[0] = projects[selection[0]]
        root.quit()
        root.destroy()

    def on_cancel():
        root.quit()
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)
    tk.Button(btn_frame, text="Configure", width=12, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Cancel", width=12, command=on_cancel).pack(side=tk.LEFT, padx=10)

    root.mainloop()

    return selected[0]


def inject_character_system(project_path, template_dir):
    """Inject custom character system files into the project."""
    print("\n[INFO] Injecting character system files...")

    game_dir = project_path / "game"
    template_game = template_dir / "game"

    if not template_game.exists():
        print(f"[ERROR] Template directory not found: {template_game}")
        return False

    # Files to copy
    files_to_inject = [
        "character.py",
        "body.py",
        "char_sprites.py",
        "pymage_size.py",
        "effects.rpy",
    ]

    for filename in files_to_inject:
        src = template_game / filename
        dst = game_dir / filename

        if src.exists():
            shutil.copy2(src, dst)
            print(f"[INFO] Injected: {filename}")
        else:
            print(f"[WARN] Template file not found: {filename}")

    # Modify script.rpy to add character loading
    script_file = game_dir / "script.rpy"
    if script_file.exists():
        inject_character_loading_code(script_file)
    else:
        print("[WARN] script.rpy not found, skipping character loading injection")

    # Create character folder structure
    (game_dir / "images" / "characters").mkdir(parents=True, exist_ok=True)

    print("[INFO] Character system injected successfully")
    return True


def inject_character_loading_code(script_file):
    """Add character loading code to script.rpy."""
    print("[INFO] Adding character loading code to script.rpy...")

    # Read existing script
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Character loading code to inject
    character_init_code = '''
# Initialize character system
init -60 python hide:
    import os
    import yaml
    from collections import defaultdict

    def create_nested_dict():
        return defaultdict(create_nested_dict)

    renpy.store.tree_map = create_nested_dict()

    # Process all character files
    for directory, filename in renpy.loader.listdirfiles(True):
        if filename.startswith('images/characters/'):
            from character import parse_character_structure
            parse_character_structure(renpy.store.tree_map, filename, native=False)

# Initialize character data stores
init -50 python:
    # Define valid file formats
    allowed_image_formats_character = ('.png', '.jpg', '.jpeg', '.webp')
    valid_image_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')

    # Character storage
    bodies = {}
    characters = {}
    all_emotions = set()
    all_outfits = set()

    # Helper function for image paths
    def image_path(path):
        """Convert path to proper image path"""
        return path.replace("\\\\", "/")

    # Load all characters from images/characters/
    from character import add_local_character, list_dirs

    for char_name in list_dirs(renpy.store.tree_map.get("images", {}).get("characters", {})):
        try:
            add_local_character(
                renpy.store.tree_map["images"]["characters"][char_name],
                char_name,
                "images/characters",
                ""
            )
        except Exception as e:
            renpy.store.logger.error(f"Failed to load character '{char_name}': {e}")

# Create Character objects from loaded character data
init -40 python:
    from char_sprites import Person, Ghost

    # Create Person objects for each loaded character
    for person_name, display_name in characters.items():
        person = Person(person_name, display_name, person_name)
        setattr(renpy.store, person_name, person)

        # Also create a Ghost version (for special effects)
        ghost_name = person_name + 'Ghost'
        ghost = Ghost(ghost_name, display_name, person_name, person_name)
        setattr(renpy.store, ghost_name, ghost)

# Special characters for narration
define narrator = Character(None, kind=nvl)
define think = Character(None, what_italic=True)

'''

    # Insert after the first line (usually a comment or define statement)
    lines = content.split('\n')

    # Find a good insertion point (after initial comments/defines, before label start)
    insert_index = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('label start:'):
            insert_index = i
            break
        elif not line.strip().startswith('#') and not line.strip().startswith('define') and line.strip():
            insert_index = i
            break

    # Insert the character loading code
    lines.insert(insert_index, character_init_code)

    # Write back
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print("[INFO] Character loading code added to script.rpy")


def import_character_folders(project_path, characters_source):
    """Copy character folders from source to project."""
    if not characters_source:
        return

    chars_src = Path(characters_source)
    if not chars_src.exists() or not chars_src.is_dir():
        print(f"[WARN] Character source not found: {chars_src}")
        return

    game_chars_dir = project_path / "game" / "images" / "characters"

    print(f"\n[INFO] Importing characters from: {chars_src}")

    # Find all character folders (folders containing character.yml)
    copied_count = 0
    for item in chars_src.iterdir():
        if item.is_dir():
            char_yml = item / "character.yml"
            if char_yml.exists():
                char_name = item.name
                dest = game_chars_dir / char_name

                if dest.exists():
                    print(f"[WARN] Character '{char_name}' already exists, skipping")
                    continue

                shutil.copytree(item, dest)
                print(f"[INFO] Imported character: {char_name}")
                copied_count += 1

    print(f"[INFO] Successfully imported {copied_count} character(s)")


def move_project_to_destination(project_path, sdk_path):
    """Move project from SDK to user's chosen location."""
    root = tk.Tk()
    root.withdraw()

    move = messagebox.askyesno(
        "Move Project",
        f"Project is currently in:\n{project_path}\n\n" +
        "Do you want to move it to a different location?\n\n" +
        "(Select 'No' to keep it in the SDK's projects folder)"
    )
    root.destroy()

    if not move:
        return project_path

    print("\n[INFO] Choose destination for the project...")

    root = tk.Tk()
    root.withdraw()
    dest_parent = filedialog.askdirectory(
        title="Choose location to move project to"
    )
    root.destroy()

    if not dest_parent:
        print("[INFO] No destination selected, keeping project in SDK")
        return project_path

    dest_path = Path(dest_parent) / project_path.name

    if dest_path.exists():
        root = tk.Tk()
        root.withdraw()
        overwrite = messagebox.askyesno(
            "Folder Exists",
            f"A folder named '{project_path.name}' already exists at the destination.\n\nOverwrite?"
        )
        root.destroy()

        if overwrite:
            shutil.rmtree(dest_path)
        else:
            print("[INFO] Cancelled due to existing folder")
            return project_path

    print(f"\n[INFO] Moving project to: {dest_path}")
    shutil.move(str(project_path), str(dest_path))
    print("[INFO] Project moved successfully")

    return dest_path


def main():
    """Main scaffolder function."""
    print("\n" + "=" * 70)
    print(" Ren'Py Project Scaffolder (Tool 1)")
    print("=" * 70)
    print("Create a new visual novel project with character support\n")

    # Find Ren'Py SDK
    print("[INFO] Looking for Ren'Py SDK...")
    sdk_path = find_renpy_sdk()

    if not sdk_path:
        print("[WARN] Ren'Py SDK not found!")
        print(f"Expected location: {Path(__file__).parent / SDK_FOLDER_NAME}")
        print("\nThe SDK is required to create Ren'Py projects.")

        # Ask if user wants to download it
        response = input("\nWould you like to download it automatically? (~150 MB) [Y/n]: ").strip().lower()

        if response in ('', 'y', 'yes'):
            sdk_path = download_and_extract_sdk(Path(__file__).parent)
            if not sdk_path:
                print("\n[ERROR] Failed to download SDK.")
                print("You can download it manually from: https://www.renpy.org/latest.html")
                input("\nPress Enter to exit...")
                return
        else:
            print("\n[INFO] Download cancelled.")
            print("You can download the SDK manually from: https://www.renpy.org/latest.html")
            print(f"Extract it to: {Path(__file__).parent}")
            input("\nPress Enter to exit...")
            return

    print(f"[INFO] Found SDK at: {sdk_path}")

    # Find executable
    renpy_exe = get_renpy_executable(sdk_path)
    if not renpy_exe:
        print("[ERROR] Could not find Ren'Py executable!")
        print(f"Looked for renpy.exe or renpy.sh in: {sdk_path}")
        input("\nPress Enter to exit...")
        return

    print(f"[INFO] Found executable: {renpy_exe.name}")

    # Launch Ren'Py launcher
    if not launch_renpy_launcher(sdk_path, renpy_exe):
        return

    # Find projects
    print("\n[INFO] Scanning for projects...")
    projects = find_projects(sdk_path)

    project = None

    if projects:
        print(f"[INFO] Found {len(projects)} project(s) in SDK/projects/")
        # Select project
        project = select_project(projects)
    else:
        print("[INFO] No projects found in SDK/projects/")
        print("(You may have created the project in a different location)")

    # If no project selected yet, ask user to browse to it
    if not project:
        print("\n[INFO] Please locate your newly created project...")
        root = tk.Tk()
        root.withdraw()
        browse = messagebox.askyesno(
            "Locate Project",
            "Would you like to browse to your project folder?\n\n" +
            "Select the main project folder (the one containing the 'game' subfolder)."
        )
        root.destroy()

        if not browse:
            print("[INFO] Project location not provided, exiting")
            return

        root = tk.Tk()
        root.withdraw()
        project_path_str = filedialog.askdirectory(
            title="Select your Ren'Py project folder"
        )
        root.destroy()

        if not project_path_str:
            print("[INFO] No project selected, exiting")
            return

        project = Path(project_path_str)

        # Verify it's a valid Ren'Py project
        game_dir = project / "game"
        if not game_dir.exists() or not game_dir.is_dir():
            print(f"[ERROR] Invalid project: 'game' folder not found in {project}")
            print("[ERROR] Please select the main project folder (the one containing 'game' subfolder)")
            input("\nPress Enter to exit...")
            return

    print(f"\n[INFO] Selected project: {project.name}")

    # Find template
    template_dir = Path(__file__).parent / "renpy_template"
    if not template_dir.exists():
        print(f"[ERROR] Template directory not found: {template_dir}")
        return

    # Inject character system
    if not inject_character_system(project, template_dir):
        print("[ERROR] Failed to inject character system")
        return

    # Ask about importing characters
    root = tk.Tk()
    root.withdraw()
    import_chars = messagebox.askyesno(
        "Import Characters",
        "Do you want to import existing character folders?\n\n" +
        "Select 'Yes' to choose a folder containing characters created with Tool 2."
    )
    root.destroy()

    characters_source = None
    if import_chars:
        print("\n[INFO] Choose folder containing character folders...")
        root = tk.Tk()
        root.withdraw()
        characters_source = filedialog.askdirectory(
            title="Choose characters folder"
        )
        root.destroy()

        if characters_source:
            import_character_folders(project, characters_source)

    # Move project to final location
    final_path = move_project_to_destination(project, sdk_path)

    # Success!
    print("\n" + "=" * 70)
    print(" ✓ Project Configured Successfully!")
    print("=" * 70)
    print(f"Project: {final_path.name}")
    print(f"Location: {final_path}")
    print("\nNext steps:")
    print(f"1. Launch Ren'Py: {renpy_exe}")
    print("2. Select your project from the list")
    print("3. Click 'Launch Project' to test it")
    print("\nOr use Tool 3 (Visual Scene Editor) to start writing your story!")
    print("=" * 70)

    # Show success dialog
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Success!",
        f"Project '{final_path.name}' configured successfully!\n\n" +
        f"Location: {final_path}\n\n" +
        "You can now launch it in Ren'Py or use Tool 3 to write scenes."
    )
    root.destroy()


if __name__ == "__main__":
    main()
