#!/usr/bin/env python3

"""
organize_sprites.py

Interactive pipeline step for cropping, naming, and organizing
Ren'Py-ready sprite assets with user input.

Features:
- Auto-crops padding.
- Prompts user to crop legs and head for sprites.
- Captures metadata like eye line, hair color, voice.
- Ensures consistent naming and folder structure.
"""

import os
import sys
import shutil
import string
import random
import csv
from pathlib import Path
import yaml
import tkinter as tk
from PIL import Image, ImageTk

# -----------------------
# UI Font and Style Settings
# -----------------------
BG_COLOR = "lightgray"
TITLE_FONT = ("Arial", 16, "bold")
INSTRUCTION_FONT = ("Arial", 12)

# -----------------------
# Load name list from CSV
# -----------------------
def load_name_pool(csv_path):
    """
    Loads a pool of boy and girl names from a CSV file.

    Returns:
        (girl_names, boy_names): Lists of names.
    """
    girl_names = []
    boy_names = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gender = row['gender'].strip().lower()
                name = row['name'].strip()
                if gender == 'girl':
                    girl_names.append(name)
                elif gender == 'boy':
                    boy_names.append(name)
    except FileNotFoundError:
        print(f"[WARN] Could not find {csv_path}. Using fallback hardcoded names.")
        girl_names = ["Sakura", "Emily", "Yuki", "Hannah", "Aiko", "Madison", "Kana", "Sara"]
        boy_names = ["Takashi", "Ethan", "Yuto", "Liam", "Kenta", "Jacob", "Hiro", "Alex"]
    return girl_names, boy_names

# -----------------------
# Pick Random Name
# -----------------------
def pick_random_name(voice, girl_names, boy_names):
    """
    Picks a random name from the appropriate gender list.
    """
    return random.choice(girl_names if voice == "girl" else boy_names)

# -----------------------
# Ensure Unique Folder Names
# -----------------------
def get_unique_folder_name(base_path, desired_name):
    """
    Ensures that a new folder name is unique in the given base path.
    Appends a counter if needed.
    """
    counter = 1
    candidate = desired_name
    while (base_path / candidate).exists():
        counter += 1
        candidate = f"{desired_name}_{counter}"
    return candidate

# -----------------------
# Compute and Apply BBox
# -----------------------
def compute_bbox(img):
    """
    Returns the bounding box of non-transparent content.
    """
    return img.getbbox()

def crop_to_bbox(img, bbox):
    """
    Crops the image to the given bounding box.
    """
    return img.crop(bbox) if bbox else img

# -----------------------
# Thumbnail Helper
# -----------------------
def make_thumbnail_of_crop(image, y_cut):
    """
    Returns a small thumbnail of the top portion of the crop,
    helping the user remember their previous crop choice.
    """
    cropped = image.crop((0, 0, image.width, y_cut))
    thumb_height = 150
    aspect_ratio = image.width / y_cut
    thumb_width = int(thumb_height * aspect_ratio)
    return cropped.resize((thumb_width, thumb_height), Image.LANCZOS)

# -----------------------
# Shared Sizing Helper
# -----------------------
HEADER_HEIGHT = 50
FOOTER_HEIGHT = 300
DEFAULT_WINDOW_SCALE = 0.8

def compute_square_window_geometry(screen_w, screen_h, img_w, img_h):
    """
    Computes a consistent window size and placement for cropping UIs.
    Ensures the window is square, fits on screen, with consistent header/footer space.

    Returns:
        (scale_factor, display_w, display_h, window_size, center_x, center_y)
    """
    MIN_WINDOW_WIDTH = 800

    usable_height = screen_h - HEADER_HEIGHT - FOOTER_HEIGHT - 50
    usable_width = screen_w * DEFAULT_WINDOW_SCALE
    usable_dim = min(usable_height, usable_width)

    scale_factor = min(usable_dim / img_h, usable_dim / img_w, 1.0)
    display_w = int(img_w * scale_factor)
    display_h = int(img_h * scale_factor)

    window_size = max(display_w, display_h) + HEADER_HEIGHT + FOOTER_HEIGHT
    window_size = max(window_size, MIN_WINDOW_WIDTH)
    window_size = min(window_size, screen_h - 50)

    center_x = max((screen_w - window_size) // 2, 0)
    center_y = 0

    return scale_factor, display_w, display_h, window_size, center_x, center_y

# -----------------------
# Prompt for Legs or Head Crop
# -----------------------
def prompt_for_crop(image, prompt_text, previous_crops):
    """
    Shows a Tkinter UI to:
    - Display the image and allow the user to click to set the Y crop line.
    - Offer thumbnails of previous crops for quick reuse.
    
    Returns:
        (chosen_y_cut, used_gallery)
    """
    result = {"y_cut": None, "used_gallery": False}

    original_width, original_height = image.size
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.update_idletasks()

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    scale_factor, display_width, display_height, window_size, center_x, center_y = compute_square_window_geometry(
        screen_width, screen_height, original_width, original_height
    )

    display_img = image.resize((display_width, display_height), Image.LANCZOS)
    scale_y = original_height / display_height

    root.title(f"Sprite Crop Tool - {prompt_text}")
    root.geometry(f"{window_size}x{window_size}+{center_x}+{center_y}")

    # Instructions
    instructions = tk.Label(
        root,
        text=f"{prompt_text}\nClick on the image to set your horizontal crop line.",
        font=TITLE_FONT, bg=BG_COLOR
    )
    instructions.pack(pady=5)

    # Image Canvas
    tk_img = ImageTk.PhotoImage(display_img)
    canvas = tk.Canvas(root, width=display_width, height=display_height)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.pack()

    def on_click(event):
        real_y = int(event.y * scale_y)
        result["y_cut"] = real_y
        result["used_gallery"] = False
        root.after(100, root.destroy)

    canvas.bind("<Button-1>", on_click)

    # Previous crops gallery
    if previous_crops:
        history_label = tk.Label(
            root,
            text="Previous crops for this character:",
            font=TITLE_FONT,
            bg=BG_COLOR
        )
        history_label.pack(pady=5)

        thumbs_frame = tk.Frame(root, bg=BG_COLOR)
        thumbs_frame.pack()

        for (y_value, thumb_img) in previous_crops:
            tk_thumb = ImageTk.PhotoImage(thumb_img)

            def use_this_crop(y=y_value):
                result["y_cut"] = y
                result["used_gallery"] = True
                root.destroy()

            btn = tk.Button(thumbs_frame, image=tk_thumb, command=use_this_crop)
            btn.image = tk_thumb  # Keep reference
            btn.pack(side=tk.LEFT, padx=5, pady=5)

    exit_button = tk.Button(
        root,
        text="Cancel and Exit",
        command=lambda: sys.exit(0),
        font=INSTRUCTION_FONT
    )
    exit_button.pack(pady=10)

    root.mainloop()
    return result["y_cut"], result["used_gallery"]


# -----------------------
# Prompt for Eye/Hair/Voice
# -----------------------
def prompt_for_character_data(image_path):
    """
    Pops up a Tkinter UI to collect:
    - Eye line ratio
    - Hair color (by clicking pixel)
    - Voice choice (girl/boy)
    
    Returns:
        (eye_line_ratio, hair_color_hex, voice)
    """
    result = {"eye_line": None, "hair_color": None, "voice": None}
    state = {"step": 1}

    original_img = Image.open(image_path).convert("RGBA")
    original_width, original_height = original_img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.update_idletasks()
    root.title("Eye & Hair Picker")

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    scale_factor, display_w, display_h, window_size, center_x, center_y = compute_square_window_geometry(
        screen_w, screen_h, original_width, original_height
    )

    display_img = original_img.resize((display_w, display_h), Image.LANCZOS)
    scale_x = original_width / display_w
    scale_y = original_height / display_h

    root.geometry(f"{window_size}x{window_size}+{center_x}+{center_y}")

    instructions = tk.Label(
        root,
        text="Step 1: Click to mark the eye line (relative head height).",
        font=TITLE_FONT, bg=BG_COLOR
    )
    instructions.pack(pady=10)

    # Image Canvas
    tk_img = ImageTk.PhotoImage(display_img)
    canvas = tk.Canvas(root, width=display_w, height=display_h)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.pack(pady=5)

    # Bottom frame
    bottom_frame = tk.Frame(root, bg=BG_COLOR)
    bottom_frame.pack(pady=10, fill=tk.X)

    # Voice buttons (hidden initially)
    button_frame = tk.Frame(bottom_frame, bg=BG_COLOR)
    def choose_voice(v):
        result["voice"] = v
        root.destroy()
    tk.Button(button_frame, text="Girl", width=12, command=lambda: choose_voice("girl")).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Boy", width=12, command=lambda: choose_voice("boy")).pack(side=tk.RIGHT, padx=10)

    exit_button = tk.Button(
        bottom_frame,
        text="Cancel and Exit",
        command=lambda: sys.exit(0),
        font=INSTRUCTION_FONT
    )
    exit_button.pack(pady=5)

    def on_click(event):
        if state["step"] == 1:
            real_y = event.y * scale_y
            result["eye_line"] = real_y / original_height
            instructions.config(text="Eye line recorded.\nStep 2: Click on the hair color.")
            state["step"] = 2

        elif state["step"] == 2:
            real_x = int(event.x * scale_x)
            real_y = int(event.y * scale_y)
            real_x = min(max(real_x, 0), original_width - 1)
            real_y = min(max(real_y, 0), original_height - 1)
            pixel = original_img.getpixel((real_x, real_y))
            if len(pixel) == 4 and pixel[3] < 10:
                color = "#915f40"
            else:
                color = '#{:02x}{:02x}{:02x}'.format(*pixel[:3])
            result["hair_color"] = color

            instructions.config(text="Hair color recorded.\nStep 3: Select the character's voice (Girl or Boy).")
            state["step"] = 3
            button_frame.pack(pady=5)

    canvas.bind("<Button-1>", on_click)
    root.mainloop()

    print(f"[INFO] Eye line: {result['eye_line']:.3f}")
    print(f"[INFO] Hair color: {result['hair_color']}")
    print(f"[INFO] Voice: {result['voice']}")
    return result["eye_line"], result["hair_color"], result["voice"]


# -----------------------
# Prompt for Scale
# -----------------------
def prompt_for_scale(image_path):
    """
    Opens a Tkinter UI to help the user adjust the sprite's scale
    by visually comparing it to reference sprites.

    Returns:
        Chosen scale as a float.
    """
    import yaml

    ##################################
    # SETTINGS
    ##################################
    REF_SPRITES_DIR = "reference_sprites"

    ##################################
    # Load all reference data
    ##################################
    ref_data = {}
    for file in os.listdir(REF_SPRITES_DIR):
        if file.lower().endswith(".png"):
            name = os.path.splitext(file)[0]
            img_path = os.path.join(REF_SPRITES_DIR, file)
            yml_path = os.path.join(REF_SPRITES_DIR, name + ".yml")
            if not os.path.exists(yml_path):
                continue
            with open(yml_path, "r", encoding="utf-8") as f:
                yml = yaml.safe_load(f)
            ref_data[name] = {
                "image": Image.open(img_path).convert("RGBA"),
                "eye_line": yml.get("eye_line", 0.1),
                "scale": yml.get("scale", 1.0)
            }

    if not ref_data:
        print(f"[ERROR] No reference sprites found in '{REF_SPRITES_DIR}/'.")
        sys.exit(1)

    ref_names = sorted(ref_data.keys())

    ##################################
    # Load user image
    ##################################
    user_img = Image.open(image_path).convert("RGBA")
    user_orig_w, user_orig_h = user_img.size
    init_scale = 1.0

    ##################################
    # Create root window
    ##################################
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Adjust Sprite Scale Compared to Reference")
    root.update_idletasks()

    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Compute dynamic canvas size
    RESERVED_HEADER = 50
    RESERVED_INSTRUCTIONS = 100
    RESERVED_SLIDER = 100
    RESERVED_BUTTONS = 80
    RESERVED_TOTAL = RESERVED_HEADER + RESERVED_INSTRUCTIONS + RESERVED_SLIDER + RESERVED_BUTTONS + 50

    usable_height = max(screen_h - RESERVED_TOTAL, 400)
    usable_width = max(int(screen_w * 0.4), 300)

    CANVAS_H = usable_height
    CANVAS_W = usable_width

    window_w = CANVAS_W * 2 + 200
    window_h = RESERVED_TOTAL + CANVAS_H

    center_x = max((screen_w - window_w) // 2, 50)
    center_y = max((screen_h - window_h) // 2, 50)
    root.geometry(f"{window_w}x{window_h}+{center_x}+{center_y}")

    ##################################
    # Instructions
    ##################################
    instructions = tk.Label(
        root,
        text=(
            "Step 1: Choose a reference character.\n"
            "Step 2: Adjust the scale slider so your sprite matches the reference proportions.\n"
            "Step 3: Click 'Done' when finished."
        ),
        font=INSTRUCTION_FONT,
        bg=BG_COLOR
    )
    instructions.pack(pady=10)

    ##################################
    # Reference Picker
    ##################################
    selected_ref_name = tk.StringVar(value=ref_names[0])
    ref_selector = tk.OptionMenu(root, selected_ref_name, *ref_names)
    ref_selector.pack(pady=5)

    ##################################
    # Scrollable Canvases
    ##################################
    canvas_frame = tk.Frame(root, bg=BG_COLOR)
    canvas_frame.pack(pady=10, expand=True, fill=tk.BOTH)

    def make_scrollable_canvas(parent):
        outer = tk.Frame(parent)
        canvas = tk.Canvas(outer, width=CANVAS_W, height=CANVAS_H, bg="lightgray")
        vsb = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side=tk.LEFT, fill="both", expand=True)
        vsb.pack(side=tk.RIGHT, fill="y")
        outer.pack(side=tk.LEFT, fill="both", expand=True)
        return canvas

    ref_canvas = make_scrollable_canvas(canvas_frame)
    user_canvas = make_scrollable_canvas(canvas_frame)

    ##################################
    # Scale Slider
    ##################################
    scale_value = tk.DoubleVar(value=round(init_scale, 3))
    slider = tk.Scale(
        root,
        from_=0.1,
        to=2.5,
        resolution=0.05,
        orient=tk.HORIZONTAL,
        label="Adjust User Scale",
        variable=scale_value,
        length=window_w - 100,
        tickinterval=0.1
    )
    slider.pack(padx=50, pady=10, fill=tk.X, expand=True)

    ##################################
    # Redraw Function
    ##################################
    def redraw():
        ref_canvas.delete("all")
        user_canvas.delete("all")

        user_scale = scale_value.get()
        selected_ref = ref_data[selected_ref_name.get()]

        # Reference sprite
        ref_image = selected_ref["image"]
        ref_scale_factor = selected_ref["scale"]
        ref_resized = ref_image.resize(
            (int(ref_image.width * ref_scale_factor), int(ref_image.height * ref_scale_factor)),
            Image.LANCZOS
        )
        ref_tk_img = ImageTk.PhotoImage(ref_resized)
        ref_canvas.create_image(CANVAS_W // 2, CANVAS_H, anchor="s", image=ref_tk_img)
        ref_canvas.image = ref_tk_img
        ref_canvas.config(scrollregion=(0, 0, CANVAS_W, ref_resized.height))

        # User sprite
        user_resized = user_img.resize(
            (int(user_orig_w * user_scale), int(user_orig_h * user_scale)),
            Image.LANCZOS
        )
        user_tk_img = ImageTk.PhotoImage(user_resized)
        user_canvas.create_image(CANVAS_W // 2, CANVAS_H, anchor="s", image=user_tk_img)
        user_canvas.image = user_tk_img
        user_canvas.config(scrollregion=(0, 0, CANVAS_W, user_resized.height))

    # Bind redraw triggers
    slider.config(command=lambda v: redraw())
    selected_ref_name.trace("w", lambda *args: redraw())

    ##################################
    # Initial Draw
    ##################################
    redraw()

    ##################################
    # Action Buttons
    ##################################
    close_button = tk.Button(root, text="Done - Use This Scale", command=root.destroy)
    close_button.pack(pady=10)

    exit_button = tk.Button(
        root,
        text="Cancel and Exit",
        command=lambda: sys.exit(0),
        font=INSTRUCTION_FONT
    )
    exit_button.pack(pady=10)

    ##################################
    # Start UI Loop
    ##################################
    root.mainloop()

    ##################################
    # Return
    ##################################
    chosen_scale = scale_value.get()
    print(f"[INFO] User-picked scale: {chosen_scale:.3f}")
    return chosen_scale



# -----------------------
# Confirm Character (all poses)
# -----------------------
def confirm_character(image_paths):
    """
    Opens a Tkinter UI showing all final outfit images (one per pose).

    Lets the user review the crops for the entire character.
    User can choose to continue or redo.

    Returns:
        True if accepted, False if redo.
    """
    decision = {"proceed": None}

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Confirm Final Character Poses")
    root.update_idletasks()

    # Measure screen size
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()

    # Reserve space for header, instructions, buttons
    RESERVED_HEADER = 50
    RESERVED_INSTRUCTIONS = 100
    RESERVED_BUTTONS = 80
    RESERVED_TOTAL = RESERVED_HEADER + RESERVED_INSTRUCTIONS + RESERVED_BUTTONS + 50

    # Dynamically size window
    usable_height = max(screen_h - RESERVED_TOTAL, 400)
    usable_width = max(int(screen_w * 0.8), 600)

    window_w = usable_width
    window_h = usable_height + RESERVED_TOTAL

    center_x = max((screen_w - window_w) // 2, 50)
    center_y = max((screen_h - window_h) // 2, 50)
    root.geometry(f"{window_w}x{window_h}+{center_x}+{center_y}")

    # Instructions
    instructions = tk.Label(
        root,
        text="Review all pose/outfit crops for this character below.\n"
             "Click 'Continue' to accept or 'Redo' to start over.",
        font=INSTRUCTION_FONT,
        bg=BG_COLOR
    )
    instructions.pack(pady=10)

    # Scrollable frame for images
    canvas_frame = tk.Frame(root, bg=BG_COLOR)
    canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    canvas = tk.Canvas(canvas_frame, bg="black")
    v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=v_scroll.set)
    v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner_frame = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    def update_scrollregion(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner_frame.bind("<Configure>", update_scrollregion)

    # Load and show images in grid
    thumbs = []
    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path)
            aspect = img.width / img.height
            thumb_height = 200
            thumb_width = int(thumb_height * aspect)
            img_thumb = img.resize((thumb_width, thumb_height), Image.LANCZOS)
            tk_img = ImageTk.PhotoImage(img_thumb)
            thumbs.append(tk_img)

            img_label = tk.Label(inner_frame, image=tk_img, bg=BG_COLOR)
            img_label.grid(row=i // 3, column=i % 3, padx=10, pady=10)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")

    if not thumbs:
        tk.Label(
            inner_frame,
            text="No images available for confirmation!",
            font=TITLE_FONT,
            fg="red",
            bg=BG_COLOR
        ).pack(pady=20)

    # Action buttons
    btn_frame = tk.Frame(root, bg=BG_COLOR)
    btn_frame.pack(pady=15)

    def accept():
        decision["proceed"] = True
        root.destroy()

    def redo():
        decision["proceed"] = False
        root.destroy()

    tk.Button(btn_frame, text="Continue", width=20, command=accept).pack(side=tk.LEFT, padx=20)
    tk.Button(btn_frame, text="Redo Character", width=20, command=redo).pack(side=tk.RIGHT, padx=20)

    root.mainloop()
    return decision["proceed"]
    

# -----------------------
# YAML Writer
# -----------------------
def write_character_yml(path, display_name, voice, eye_line, hair_color, scale, poses):
    """
    Writes the final character metadata to a YAML file.

    Args:
        path (Path): Path to write the YAML file.
        display_name (str): Display name of the character.
        voice (str): 'girl' or 'boy'.
        eye_line (float): Relative eye height ratio.
        hair_color (str): Color hex string.
        scale (float): User-picked scale factor.
        poses (dict): Dict of poses and their metadata.
    """
    data = {
        'display_name': display_name,
        'eye_line': round(eye_line, 4),
        'name_color': hair_color,
        'poses': poses,
        'scale': scale,
        'voice': voice
    }

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Wrote character YAML to: {path}")

# -----------------------
# Main Organizer
# -----------------------
def run_organizer(workspace_input, output_input):
    print("=" * 60)
    print(" Ren'Py Sprite Organizer - Cropping Pipeline")
    print("=" * 60)

    workspace_dir = Path(workspace_input)
    output_dir = Path(output_input)
    output_dir.mkdir(parents=True, exist_ok=True)

    girl_names, boy_names = load_name_pool("names.csv")
    letter_labels = list(string.ascii_lowercase)
    character_folders = sorted([f for f in workspace_dir.iterdir() if f.is_dir()])

    print(f"[INFO] Found {len(character_folders)} character folders to process.")

    for char_index, char_folder in enumerate(character_folders):
        while True:
            temp_char_name = f"char_{char_index + 1}"
            char_output = output_dir / temp_char_name
            print(f"\n[INFO] Processing '{char_folder.name}' -> '{temp_char_name}'")
            char_output.mkdir(parents=True, exist_ok=True)

            pose_folders = sorted([p for p in char_folder.iterdir() if p.is_dir()])
            poses_yaml = {}
            all_face_images = []
            previous_leg_crops = []

            for pose_index, pose_folder in enumerate(pose_folders):
                letter = letter_labels[pose_index]
                print(f"[INFO] Pose: '{pose_folder.name}' -> '{letter}'")

                pose_output = char_output / letter
                faces_face_dir = pose_output / "faces" / "face"
                outfits_dir = pose_output / "outfits"
                faces_face_dir.mkdir(parents=True, exist_ok=True)
                outfits_dir.mkdir(parents=True, exist_ok=True)

                image_files = sorted([
                    f for f in pose_folder.iterdir()
                    if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
                ])
                if not image_files:
                    continue

                # === 1. Auto-Crop Padding ===
                print("[INFO] Auto-cropping padding")
                first_img = Image.open(image_files[0]).convert("RGBA")
                bbox = compute_bbox(first_img)
                first_img_cropped = crop_to_bbox(first_img, bbox)

                # === 2. Legs Crop ===
                leg_cut, used_gallery = prompt_for_crop(
                    first_img_cropped,
                    "Click at the height of the character's mid-thigh. This removes the lower part of the image.",
                    previous_leg_crops
                )

                if not used_gallery:
                    thumb = make_thumbnail_of_crop(first_img_cropped, leg_cut)
                    previous_leg_crops.append((leg_cut, thumb))

                cropped_leg_images = []
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGBA")
                    img = crop_to_bbox(img, bbox)
                    img = img.crop((0, 0, img.width, leg_cut))
                    cropped_leg_images.append(img)

                # === 3. Outfit Copy ===
                print("[INFO] Saving outfit copy")
                outfit_name = f"outfit_{pose_index}.webp"
                cropped_leg_images[0].save(outfits_dir / outfit_name, "WEBP")
                all_face_images.append(outfits_dir / outfit_name)

                # === 4. Head Crop ===
                print("[INFO] Prompting for expression crop")
                head_cut, _ = prompt_for_crop(
                    cropped_leg_images[0],
                    "Click at the height of the character's chin. This will set the bottom of the characters expression sheet.",
                    []
                )

                for i, img in enumerate(cropped_leg_images):
                    face_img = img.crop((0, 0, img.width, head_cut))
                    new_name = f"{i}.webp"
                    face_img.save(faces_face_dir / new_name, "WEBP")

                poses_yaml[letter] = {"facing": "right"}

            # === Confirm Step ===
            if all_face_images:
                print("\n[INFO] Collecting metadata for character")
                eye_line, hair_color, voice = prompt_for_character_data(all_face_images[0])
                scale = prompt_for_scale(all_face_images[0])

                proceed = confirm_character(all_face_images)
                if not proceed:
                    print("[WARN] User chose to redo character. Deleting output and restarting...")
                    shutil.rmtree(char_output)
                    continue

                display_name = pick_random_name(voice, girl_names, boy_names)
                unique_name = get_unique_folder_name(output_dir, display_name)
                new_char_output = output_dir / unique_name
                char_output.rename(new_char_output)
                char_output = new_char_output
            else:
                eye_line = 0.195
                hair_color = "#ffffff"
                voice = "girl"
                display_name = temp_char_name
                scale = 1.0

            yml_path = char_output / "character.yml"
            write_character_yml(yml_path, display_name, voice, eye_line, hair_color, scale, poses_yaml)
            print(f"[INFO] Created metadata: {yml_path}")

            break

    print("\n[INFO] All done! Your Student Transfer sprite folders are saved to:")
    print(f"       {output_input}")

# -----------------------
# Organizer Interactive Entry
# -----------------------
def run_organizer_interactive(workspace_input=None, output_input=None):
    """
    Interactive wrapper for the organizer step.
    Works both standalone and when called from pipeline_runner.
    """
    print("=" * 60)
    print(" Student Transfer Sprite Organizer - Finalizer")
    print("=" * 60)

    print("\nThis tool finalizes your manually sorted sprite folders")
    print("by guiding you through cropping and metadata collection.\n")

    # Prompt for workspace folder if missing
    if not workspace_input:
        workspace_input = input("\nEnter the full path to your manually sorted sprite folder:\n> ").strip()
    if not workspace_input or not os.path.isdir(workspace_input):
        print("\n[ERROR] The specified input folder does not exist. Exiting.")
        sys.exit(1)

    # Prompt for output folder if missing
    if not output_input:
        output_input = input("\nEnter the full path to the output folder for finalized sprites:\n> ").strip()
    if not output_input:
        print("\n[ERROR] Output path cannot be empty. Exiting.")
        sys.exit(1)

    os.makedirs(output_input, exist_ok=True)

    run_organizer(workspace_input, output_input)

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    run_organizer_interactive()
