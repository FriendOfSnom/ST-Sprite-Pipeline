#!/usr/bin/env python3
"""
visual_scene_editor.py

Tool 3: Visual Scene Editor
A complete GUI-based scene editor for writing visual novels without code.

Features:
- Project and script selection
- Live preview window
- Drag-and-drop character positioning
- Visual expression/outfit/background selectors
- Dialogue editor
- Timeline navigation (undo/redo)
- Automatic Ren'Py code generation
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, List, Dict


class ProjectSelector:
    """Handles Ren'Py project selection and validation."""

    @staticmethod
    def select_project() -> Optional[Path]:
        """
        Prompt user to select a Ren'Py project folder.
        Returns the project path or None if cancelled.
        """
        root = tk.Tk()
        root.withdraw()

        project_path = filedialog.askdirectory(
            title="Select your Ren'Py project folder",
            mustexist=True
        )
        root.destroy()

        if not project_path:
            return None

        project = Path(project_path)

        # Validate it's a Ren'Py project
        game_dir = project / "game"
        if not game_dir.exists() or not game_dir.is_dir():
            messagebox.showerror(
                "Invalid Project",
                f"The selected folder is not a valid Ren'Py project.\n\n"
                f"Could not find 'game' subfolder in:\n{project}"
            )
            return None

        return project


class ScriptSelector:
    """Handles script file (.rpy) selection within a project."""

    @staticmethod
    def find_scripts(project_path: Path) -> List[Path]:
        """
        Find all .rpy script files in the project/game/ folder.
        Returns list of script file paths relative to project root.
        """
        game_dir = project_path / "game"
        scripts = []

        for rpy_file in game_dir.rglob("*.rpy"):
            # Exclude certain system files
            if rpy_file.name in ['options.rpy', 'gui.rpy', 'screens.rpy']:
                continue
            scripts.append(rpy_file)

        return sorted(scripts)

    @staticmethod
    def select_script(project_path: Path) -> Optional[Path]:
        """
        Display a dialog for the user to select which script to edit.
        Returns the selected script path or None if cancelled.
        """
        scripts = ScriptSelector.find_scripts(project_path)

        if not scripts:
            messagebox.showerror(
                "No Scripts Found",
                "No editable .rpy script files found in the project.\n\n"
                "Make sure your project has at least a script.rpy file."
            )
            return None

        # Create selection dialog
        dialog = tk.Toplevel()
        dialog.title("Select Script to Edit")
        dialog.geometry("600x400")

        selected = [None]  # Use list to capture value from nested function

        # Header
        header = tk.Label(
            dialog,
            text="Select the script file you want to edit:",
            font=("Arial", 12, "bold"),
            pady=10
        )
        header.pack()

        # Listbox with scrollbar
        frame = tk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            font=("Courier", 10)
        )
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        # Populate listbox with relative paths
        game_dir = project_path / "game"
        for script in scripts:
            try:
                relative = script.relative_to(game_dir)
                listbox.insert(tk.END, str(relative))
            except ValueError:
                listbox.insert(tk.END, script.name)

        # Select first item by default
        if scripts:
            listbox.selection_set(0)

        # Buttons
        def on_ok():
            selection = listbox.curselection()
            if selection:
                selected[0] = scripts[selection[0]]
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Open", width=12, command=on_ok).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", width=12, command=on_cancel).pack(side=tk.LEFT, padx=5)

        # Make dialog modal
        dialog.transient()
        dialog.grab_set()
        dialog.wait_window()

        return selected[0]


class VisualSceneEditor:
    """Main visual scene editor application."""

    def __init__(self):
        self.project_path: Optional[Path] = None
        self.script_path: Optional[Path] = None
        self.root: Optional[tk.Tk] = None

        # Scene state
        self.scene_timeline = []  # List of scene states
        self.current_index = 0  # Current position in timeline
        self.characters_on_screen = {}  # character_name -> {position, expression, outfit}
        self.current_background = None

    def start(self):
        """Start the editor application."""
        # Step 1: Select project
        print("\n=== Visual Scene Editor ===")
        print("Step 1: Select your Ren'Py project...")

        self.project_path = ProjectSelector.select_project()
        if not self.project_path:
            print("[INFO] Project selection cancelled")
            return

        print(f"[INFO] Selected project: {self.project_path.name}")

        # Step 2: Select script
        print("\nStep 2: Select the script file to edit...")

        self.script_path = ScriptSelector.select_script(self.project_path)
        if not self.script_path:
            print("[INFO] Script selection cancelled")
            return

        print(f"[INFO] Selected script: {self.script_path.name}")

        # Step 3: Launch main editor window
        print("\nStep 3: Launching editor...")
        self.launch_editor()

    def launch_editor(self):
        """Launch the main editor GUI window."""
        self.root = tk.Tk()
        self.root.title(f"Visual Scene Editor - {self.script_path.name}")
        self.root.geometry("1400x900")

        # Create main layout
        self.create_layout()

        # Load existing script content
        self.load_script()

        # Start GUI loop
        self.root.mainloop()

    def create_layout(self):
        """Create the main editor layout."""
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top menu bar
        self.create_menu_bar()

        # Split into 3 columns: Left panel | Center preview | Right panel
        left_panel = tk.Frame(main_frame, width=300, bg="#f0f0f0")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH)
        left_panel.pack_propagate(False)

        center_panel = tk.Frame(main_frame, bg="#2b2b2b")
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_panel = tk.Frame(main_frame, width=300, bg="#f0f0f0")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH)
        right_panel.pack_propagate(False)

        # Populate panels
        self.create_left_panel(left_panel)
        self.create_center_panel(center_panel)
        self.create_right_panel(right_panel)

        # Bottom control bar
        self.create_bottom_bar()

    def create_menu_bar(self):
        """Create the top menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save_script)
        file_menu.add_command(label="Save As...", command=self.save_script_as)
        file_menu.add_separator()
        file_menu.add_command(label="Close", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_left_panel(self, parent):
        """Create left panel with character management."""
        tk.Label(parent, text="Characters", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # Placeholder for character list
        frame = tk.Frame(parent, bg="white", relief=tk.SUNKEN, borderwidth=1)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Label(frame, text="Character list will go here", bg="white").pack(pady=20)

    def create_center_panel(self, parent):
        """Create center panel with live preview."""
        # Title
        tk.Label(parent, text="Live Preview", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)

        # Preview canvas (1920x1080 scaled to fit)
        self.preview_canvas = tk.Canvas(parent, bg="#1a1a1a", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Placeholder text
        self.preview_canvas.create_text(
            400, 300,
            text="Scene preview will appear here",
            fill="white",
            font=("Arial", 16)
        )

    def create_right_panel(self, parent):
        """Create right panel with dialogue editor and controls."""
        tk.Label(parent, text="Scene Editor", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # Dialogue section
        tk.Label(parent, text="Dialogue:", bg="#f0f0f0").pack(anchor=tk.W, padx=10)

        self.dialogue_text = tk.Text(parent, height=4, wrap=tk.WORD)
        self.dialogue_text.pack(fill=tk.X, padx=10, pady=5)

        # Character selector
        tk.Label(parent, text="Speaking Character:", bg="#f0f0f0").pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.character_var = tk.StringVar()
        self.character_dropdown = ttk.Combobox(parent, textvariable=self.character_var, state="readonly")
        self.character_dropdown['values'] = ["Narrator", "Character 1", "Character 2"]
        self.character_dropdown.pack(fill=tk.X, padx=10, pady=5)

        # Expression/Outfit section
        tk.Label(parent, text="Expression:", bg="#f0f0f0").pack(anchor=tk.W, padx=10, pady=(10, 0))

        self.expression_var = tk.StringVar()
        self.expression_dropdown = ttk.Combobox(parent, textvariable=self.expression_var, state="readonly")
        self.expression_dropdown['values'] = ["happy", "sad", "angry", "neutral"]
        self.expression_dropdown.pack(fill=tk.X, padx=10, pady=5)

        # Buttons
        tk.Button(parent, text="Add Character to Scene", command=self.add_character).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(parent, text="Change Background", command=self.change_background).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(parent, text="Add Sound/Music", command=self.add_sound).pack(fill=tk.X, padx=10, pady=5)

    def create_bottom_bar(self):
        """Create bottom control bar with navigation."""
        bottom = tk.Frame(self.root, bg="#e0e0e0", height=60)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # Navigation buttons
        nav_frame = tk.Frame(bottom, bg="#e0e0e0")
        nav_frame.pack(pady=10)

        tk.Button(nav_frame, text="← Back", width=12, command=self.go_back).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next →", width=12, command=self.go_next, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Generate Code", width=15, command=self.generate_code, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=15)

    def load_script(self):
        """Load and parse the selected script file."""
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"[INFO] Loaded {len(content)} characters from {self.script_path.name}")
            # TODO: Parse existing Ren'Py script content
        except Exception as e:
            messagebox.showerror("Error Loading Script", f"Failed to load script:\n{e}")

    def save_script(self):
        """Save changes to the script file."""
        # TODO: Implement save logic
        messagebox.showinfo("Save", "Save functionality coming soon!")

    def save_script_as(self):
        """Save script to a new file."""
        # TODO: Implement save as logic
        pass

    def undo(self):
        """Undo last action (go back in timeline)."""
        self.go_back()

    def redo(self):
        """Redo action (go forward in timeline)."""
        self.go_next()

    def go_back(self):
        """Navigate to previous scene in timeline."""
        if self.current_index > 0:
            self.current_index -= 1
            print(f"[INFO] Moved back to scene {self.current_index}")
            # TODO: Restore scene state
        else:
            print("[INFO] Already at beginning")

    def go_next(self):
        """Navigate to next scene or create new one."""
        if self.current_index < len(self.scene_timeline) - 1:
            self.current_index += 1
            print(f"[INFO] Moved forward to scene {self.current_index}")
            # TODO: Restore scene state
        else:
            # Create new scene
            print("[INFO] Creating new scene")
            # TODO: Create new scene state

    def generate_code(self):
        """Generate Ren'Py code from current scene and append to script."""
        # TODO: Implement code generation
        messagebox.showinfo("Generate Code", "Code generation functionality coming soon!")

    def add_character(self):
        """Add a character to the scene."""
        # TODO: Implement character addition
        messagebox.showinfo("Add Character", "Character addition coming soon!")

    def change_background(self):
        """Change the scene background."""
        # TODO: Implement background selection
        messagebox.showinfo("Change Background", "Background selection coming soon!")

    def add_sound(self):
        """Add sound or music to the scene."""
        # TODO: Implement sound/music selection
        messagebox.showinfo("Add Sound", "Sound selection coming soon!")

    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Visual Scene Editor",
            "Visual Scene Editor (Tool 3)\n"
            "Version 2.0.0\n\n"
            "A GUI-based scene editor for creating\n"
            "visual novels without writing code.\n\n"
            "Part of the Visual Novel Development Toolkit"
        )


def main():
    """Entry point for the visual scene editor."""
    editor = VisualSceneEditor()
    editor.start()


if __name__ == "__main__":
    main()
