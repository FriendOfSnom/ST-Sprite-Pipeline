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
import yaml
from PIL import Image, ImageTk


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
    def select_script(project_path: Path, parent=None) -> Optional[Path]:
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

        # Create selection dialog with proper parent
        dialog = tk.Toplevel(parent)
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


class CharacterLoader:
    """Loads characters and their metadata from a Ren'Py project."""

    @staticmethod
    def find_characters(project_path: Path) -> List[Dict]:
        """
        Find all character folders in game/images/characters/.
        Returns list of character dicts with metadata.
        """
        characters_dir = project_path / "game" / "images" / "characters"

        if not characters_dir.exists():
            print(f"[WARN] No characters folder found at: {characters_dir}")
            return []

        characters = []

        for char_folder in characters_dir.iterdir():
            if not char_folder.is_dir():
                continue

            # Check for character.yml
            char_yml = char_folder / "character.yml"
            if not char_yml.exists():
                print(f"[WARN] Skipping {char_folder.name}: no character.yml")
                continue

            # Load character metadata
            char_data = CharacterLoader.load_character_metadata(char_yml)
            if char_data:
                char_data['folder_name'] = char_folder.name
                char_data['folder_path'] = char_folder

                # Find available poses
                char_data['poses'] = CharacterLoader.find_poses(char_folder)

                characters.append(char_data)
                print(f"[INFO] Loaded character: {char_data.get('display_name', char_folder.name)}")

        return characters

    @staticmethod
    def load_character_metadata(yml_path: Path) -> Optional[Dict]:
        """Load and parse character.yml file."""
        try:
            with open(yml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            return data or {}
        except Exception as e:
            print(f"[ERROR] Failed to load {yml_path.name}: {e}")
            return None

    @staticmethod
    def find_poses(char_folder: Path) -> List[str]:
        """
        Find all pose folders (a, b, c, etc.) for a character.
        Returns list of pose names.
        """
        poses = []
        for item in char_folder.iterdir():
            if item.is_dir() and len(item.name) == 1 and item.name.isalpha():
                poses.append(item.name)
        return sorted(poses)

    @staticmethod
    def find_outfits(char_folder: Path, pose: str) -> List[str]:
        """
        Find all outfits for a character pose.
        Returns list of outfit names (without extension).
        """
        outfits_dir = char_folder / pose / "outfits"
        if not outfits_dir.exists():
            return []

        outfits = []
        for outfit_file in outfits_dir.glob("*.png"):
            outfits.append(outfit_file.stem)
        return sorted(outfits)

    @staticmethod
    def find_expressions(char_folder: Path, pose: str, face_folder: str = "face") -> List[str]:
        """
        Find all expressions for a character pose.
        Returns list of expression names (without extension).
        """
        faces_dir = char_folder / pose / "faces" / face_folder
        if not faces_dir.exists():
            return []

        expressions = []
        for expr_file in faces_dir.glob("*.png"):
            expressions.append(expr_file.stem)
        return sorted(expressions)

    @staticmethod
    def load_thumbnail(char_folder: Path, size=(100, 100)) -> Optional[ImageTk.PhotoImage]:
        """
        Load a thumbnail image for the character.
        Tries to find first available outfit+expression combo.
        """
        # Find first pose
        poses = CharacterLoader.find_poses(char_folder)
        if not poses:
            return None

        pose = poses[0]  # Use first pose

        # Find first outfit
        outfits = CharacterLoader.find_outfits(char_folder, pose)
        if not outfits:
            return None

        outfit_path = char_folder / pose / "outfits" / f"{outfits[0]}.png"

        try:
            img = Image.open(outfit_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception as e:
            print(f"[ERROR] Failed to load thumbnail for {char_folder.name}: {e}")
            return None

    @staticmethod
    def composite_character_sprite(char_folder: Path, pose: str, outfit: str, expression: str, scale: float = 1.0) -> Optional[Image.Image]:
        """
        Composite a character sprite from outfit and expression.
        Returns PIL Image ready for display.
        """
        # Load outfit
        outfit_path = char_folder / pose / "outfits" / f"{outfit}.png"
        if not outfit_path.exists():
            print(f"[ERROR] Outfit not found: {outfit_path}")
            return None

        # Load expression
        expression_path = char_folder / pose / "faces" / "face" / f"{expression}.png"
        if not expression_path.exists():
            print(f"[ERROR] Expression not found: {expression_path}")
            return None

        try:
            # Load images
            outfit_img = Image.open(outfit_path).convert("RGBA")
            expression_img = Image.open(expression_path).convert("RGBA")

            # Composite expression on top of outfit
            outfit_img.paste(expression_img, (0, 0), expression_img)

            # Scale if needed
            if scale != 1.0:
                new_width = int(outfit_img.width * scale)
                new_height = int(outfit_img.height * scale)
                outfit_img = outfit_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            return outfit_img
        except Exception as e:
            print(f"[ERROR] Failed to composite sprite: {e}")
            return None


class VisualSceneEditor:
    """Main visual scene editor application."""

    def __init__(self):
        self.project_path: Optional[Path] = None
        self.script_path: Optional[Path] = None
        self.root: Optional[tk.Tk] = None

        # Character data
        self.available_characters = []  # List of character dicts
        self.character_thumbnails = {}  # character_name -> PhotoImage

        # Preview canvas
        self.preview_width = 1920
        self.preview_height = 1080
        self.preview_scale = 0.5  # Scale down for display
        self.rendered_sprites = {}  # character_name -> PhotoImage

        # Drag and drop state
        self.dragging_char = None
        self.drag_start_x = 0
        self.drag_start_y = 0

        # Scene state
        self.scene_timeline = []  # List of scene states
        self.current_index = 0  # Current position in timeline
        self.characters_on_screen = {}  # character_name -> {position, expression, outfit}
        self.current_background = None

    def start(self):
        """Start the editor application."""
        # Create root window first (hidden during setup)
        self.root = tk.Tk()
        self.root.withdraw()  # Hide during setup

        # Step 1: Select project
        print("\n=== Visual Scene Editor ===")
        print("Step 1: Select your Ren'Py project...")

        self.project_path = ProjectSelector.select_project()
        if not self.project_path:
            print("[INFO] Project selection cancelled")
            self.root.destroy()
            return

        print(f"[INFO] Selected project: {self.project_path.name}")

        # Step 2: Select script
        print("\nStep 2: Select the script file to edit...")

        self.script_path = ScriptSelector.select_script(self.project_path, self.root)
        if not self.script_path:
            print("[INFO] Script selection cancelled")
            self.root.destroy()
            return

        print(f"[INFO] Selected script: {self.script_path.name}")

        # Step 3: Launch main editor window
        print("\nStep 3: Launching editor...")
        self.launch_editor()

    def launch_editor(self):
        """Launch the main editor GUI window."""
        # Show the root window (was hidden during setup)
        self.root.deiconify()
        self.root.title(f"Visual Scene Editor - {self.script_path.name}")
        self.root.geometry("1400x900")

        # Create main layout
        self.create_layout()

        # Load characters from project
        self.load_characters()

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

        # Scrollable character list
        container = tk.Frame(parent, bg="white", relief=tk.SUNKEN, borderwidth=1)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Canvas with scrollbar
        canvas = tk.Canvas(container, bg="white", highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)

        self.char_list_frame = tk.Frame(canvas, bg="white")

        self.char_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.char_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def create_center_panel(self, parent):
        """Create center panel with live preview."""
        # Title
        tk.Label(parent, text="Live Preview", font=("Arial", 14, "bold"), bg="#2b2b2b", fg="white").pack(pady=10)

        # Calculate canvas size
        canvas_width = int(self.preview_width * self.preview_scale)
        canvas_height = int(self.preview_height * self.preview_scale)

        # Preview canvas (1920x1080 scaled to fit)
        self.preview_canvas = tk.Canvas(
            parent,
            width=canvas_width,
            height=canvas_height,
            bg="#1a1a1a",
            highlightthickness=1,
            highlightbackground="#666"
        )
        self.preview_canvas.pack(padx=20, pady=10)

        # Bind mouse events for drag and drop
        self.preview_canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.preview_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Initial render (empty scene)
        self.render_preview()

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

    def load_characters(self):
        """Load all characters from the project."""
        print("[INFO] Loading characters...")

        self.available_characters = CharacterLoader.find_characters(self.project_path)

        if not self.available_characters:
            print("[WARN] No characters found in project")
            return

        # Load thumbnails for each character
        for char in self.available_characters:
            folder_name = char['folder_name']
            folder_path = char['folder_path']

            thumbnail = CharacterLoader.load_thumbnail(folder_path)
            if thumbnail:
                self.character_thumbnails[folder_name] = thumbnail

        print(f"[INFO] Loaded {len(self.available_characters)} character(s)")

        # Update the character list UI
        self.update_character_list()

    def update_character_list(self):
        """Update the left panel with character thumbnails."""
        if not hasattr(self, 'char_list_frame'):
            return

        # Clear existing widgets
        for widget in self.char_list_frame.winfo_children():
            widget.destroy()

        # Add character buttons with thumbnails
        for char in self.available_characters:
            folder_name = char['folder_name']
            display_name = char.get('display_name', folder_name)

            # Character frame
            char_frame = tk.Frame(self.char_list_frame, bg="white", relief=tk.RAISED, borderwidth=1)
            char_frame.pack(fill=tk.X, padx=5, pady=5)

            # Thumbnail (if available)
            if folder_name in self.character_thumbnails:
                img_label = tk.Label(char_frame, image=self.character_thumbnails[folder_name], bg="white")
                img_label.pack(pady=5)

            # Character name
            name_label = tk.Label(char_frame, text=display_name, bg="white", font=("Arial", 10))
            name_label.pack()

            # Add button
            add_btn = tk.Button(
                char_frame,
                text="Add to Scene",
                command=lambda c=char: self.add_character_to_scene(c)
            )
            add_btn.pack(pady=5)

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

    def add_character_to_scene(self, character):
        """Add a character to the scene."""
        folder_name = character['folder_name']
        display_name = character.get('display_name', folder_name)

        print(f"[INFO] Adding character '{display_name}' to scene")

        # Get default pose and outfit
        poses = character.get('poses', [])
        if not poses:
            messagebox.showerror("No Poses", f"Character '{display_name}' has no poses available")
            return

        pose = poses[0]  # Use first pose
        outfits = CharacterLoader.find_outfits(character['folder_path'], pose)

        if not outfits:
            messagebox.showerror("No Outfits", f"Character '{display_name}' has no outfits available")
            return

        outfit = outfits[0]  # Use first outfit

        # Get default expression
        expressions = CharacterLoader.find_expressions(character['folder_path'], pose)
        expression = expressions[0] if expressions else "neutral"

        # Add to scene state
        self.characters_on_screen[folder_name] = {
            'display_name': display_name,
            'pose': pose,
            'outfit': outfit,
            'expression': expression,
            'position': 'center',  # Default position
            'transform': None,
            'character_data': character
        }

        print(f"[INFO] Character added: {display_name} ({outfit} {expression})")

        # Update preview canvas
        self.render_preview()

        # TODO: Update character dropdown in right panel

        messagebox.showinfo("Character Added", f"{display_name} added to scene!")

    def get_position_x(self, position: str) -> int:
        """Get X coordinate for a position name."""
        positions = {
            'left': int(self.preview_width * 0.15),
            'centerleft': int(self.preview_width * 0.3),
            'center': int(self.preview_width * 0.5),
            'centerright': int(self.preview_width * 0.7),
            'right': int(self.preview_width * 0.85),
        }
        return positions.get(position, positions['center'])

    def render_preview(self):
        """Render the current scene to the preview canvas."""
        # Clear canvas
        self.preview_canvas.delete("all")

        # Get canvas dimensions
        canvas_width = int(self.preview_width * self.preview_scale)
        canvas_height = int(self.preview_height * self.preview_scale)

        # Draw background (placeholder for now)
        self.preview_canvas.create_rectangle(
            0, 0, canvas_width, canvas_height,
            fill="#4a4a4a", outline=""
        )

        # Render each character
        for char_name, char_state in self.characters_on_screen.items():
            self.render_character(char_name, char_state)

        print(f"[INFO] Rendered {len(self.characters_on_screen)} character(s) to preview")

    def render_character(self, char_name: str, char_state: Dict):
        """Render a single character to the preview canvas."""
        character_data = char_state['character_data']
        char_folder = character_data['folder_path']

        # Get character scale from metadata
        char_scale = character_data.get('scale', 1.0)

        # Composite the sprite
        sprite_img = CharacterLoader.composite_character_sprite(
            char_folder,
            char_state['pose'],
            char_state['outfit'],
            char_state['expression'],
            scale=char_scale
        )

        if not sprite_img:
            print(f"[ERROR] Failed to render character: {char_name}")
            return

        # Scale for preview
        preview_width = int(sprite_img.width * self.preview_scale)
        preview_height = int(sprite_img.height * self.preview_scale)
        sprite_img = sprite_img.resize((preview_width, preview_height), Image.Resampling.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(sprite_img)
        self.rendered_sprites[char_name] = photo  # Keep reference

        # Calculate position
        x = int(self.get_position_x(char_state['position']) * self.preview_scale)
        y = int(self.preview_height * self.preview_scale)  # Bottom of canvas

        # Draw on canvas (anchor at bottom center of sprite)
        self.preview_canvas.create_image(
            x, y,
            image=photo,
            anchor=tk.S,
            tags=f"char_{char_name}"
        )

    def on_canvas_click(self, event):
        """Handle mouse click on canvas - start dragging a character."""
        # Find which character was clicked
        x, y = event.x, event.y
        items = self.preview_canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)

        for item in items:
            tags = self.preview_canvas.gettags(item)
            for tag in tags:
                if tag.startswith("char_"):
                    char_name = tag[5:]  # Remove "char_" prefix
                    self.dragging_char = char_name
                    self.drag_start_x = x
                    self.drag_start_y = y
                    print(f"[INFO] Started dragging: {char_name}")
                    return

    def on_canvas_drag(self, event):
        """Handle mouse drag - move character."""
        if not self.dragging_char:
            return

        # Calculate new position in preview space
        x = event.x

        # Convert to full resolution position
        full_res_x = x / self.preview_scale

        # Determine which position zone this is
        if full_res_x < self.preview_width * 0.225:
            new_position = 'left'
        elif full_res_x < self.preview_width * 0.4:
            new_position = 'centerleft'
        elif full_res_x < self.preview_width * 0.6:
            new_position = 'center'
        elif full_res_x < self.preview_width * 0.775:
            new_position = 'centerright'
        else:
            new_position = 'right'

        # Update character position if changed
        if self.dragging_char in self.characters_on_screen:
            current_pos = self.characters_on_screen[self.dragging_char]['position']
            if current_pos != new_position:
                self.characters_on_screen[self.dragging_char]['position'] = new_position
                self.render_preview()
                print(f"[INFO] {self.dragging_char} moved to: {new_position}")

    def on_canvas_release(self, event):
        """Handle mouse release - stop dragging."""
        if self.dragging_char:
            print(f"[INFO] Stopped dragging: {self.dragging_char}")
            self.dragging_char = None

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
