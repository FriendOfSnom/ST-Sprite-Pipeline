#!/usr/bin/env python3

"""
organize_sprites.py

Interactive pipeline step for cropping, naming, and organizing
Ren'Py-ready sprite assets with user input.

Features:
- Auto-crops padding.
- Prompts user to crop legs and head for sprites with live Y readout + guide line.
- Captures metadata like eye line, hair color, voice.
- Ensures consistent naming and folder structure.
- Reuse last crop for both legs and head per character.
- Safer image handling (verification, corrupt-skip), WEBP->PNG fallback.
"""

import os
import sys
import shutil
import string
import random
import csv
import json
from pathlib import Path
import yaml
import tkinter as tk
from PIL import Image, ImageTk

# =========================
# PATCH: UI/UX CONSOLIDATION
# =========================

# ---- UI constants (add/replace) ----
BG_COLOR = "lightgray"
TITLE_FONT = ("Arial", 16, "bold")
INSTRUCTION_FONT = ("Arial", 12)
LINE_COLOR = "#00E5FF"  # high-contrast cyan for the crop guide line
WINDOW_MARGIN = 10      # top/side margin from screen edges
WRAP_PADDING = 40       # label wrap padding so long titles wrap nicely

HEADER_HEIGHT = 50
FOOTER_HEIGHT = 300
DEFAULT_WINDOW_SCALE = 0.8


# =========================
# PATCH: missing core helpers
# Paste this block once below the imports.
# =========================

def _compute_display_size(screen_w, screen_h, img_w, img_h, *, max_w_ratio=0.90, max_h_ratio=0.55):
    """
    Compute an image display size that leaves vertical room for text/buttons.
    Returns (disp_w, disp_h). We do NOT set the window size here.
    """
    max_w = int(screen_w * max_w_ratio) - 2 * WINDOW_MARGIN
    max_h = int(screen_h * max_h_ratio) - 2 * WINDOW_MARGIN
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))

def _center_and_clamp(root: tk.Tk):
    """
    After widgets are laid out, measure requested size and clamp to screen.
    Keeps a small top/bottom margin and positions near the top (no off-screen footers).
    """
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    w = min(req_w + WINDOW_MARGIN, sw - 2 * WINDOW_MARGIN)
    h = min(req_h + WINDOW_MARGIN, sh - 2 * WINDOW_MARGIN)
    x = max((sw - w) // 2, WINDOW_MARGIN)
    y = WINDOW_MARGIN  # pin near top instead of vertical centering

    root.geometry(f"{w}x{h}+{x}+{y}")


def load_name_pool(csv_path: str):
    """
    Load girl/boy name pools from a CSV file with columns: name, gender.
    Falls back to a small hardcoded list if the CSV is missing.
    Returns:
        (girl_names, boy_names)
    """
    girl_names, boy_names = [], []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                gender = (row.get('gender') or '').strip().lower()
                name = (row.get('name') or '').strip()
                if not name:
                    continue
                if gender == 'girl':
                    girl_names.append(name)
                elif gender == 'boy':
                    boy_names.append(name)
    except FileNotFoundError:
        print(f"[WARN] Could not find {csv_path}. Using fallback hardcoded names.")
        girl_names = ["Sakura", "Emily", "Yuki", "Hannah", "Aiko", "Madison", "Kana", "Sara"]
        boy_names  = ["Takashi", "Ethan", "Yuto", "Liam", "Kenta", "Jacob", "Hiro", "Alex"]
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}. Using fallback names.")
        girl_names = ["Sakura", "Emily", "Yuki", "Hannah", "Aiko", "Madison", "Kana", "Sara"]
        boy_names  = ["Takashi", "Ethan", "Yuto", "Liam", "Kenta", "Jacob", "Hiro", "Alex"]
    return girl_names, boy_names


def pick_random_name(voice: str, girl_names, boy_names) -> str:
    """
    Pick a random name from the appropriate list based on the selected voice.
    """
    pool = girl_names if (voice or "").lower() == "girl" else boy_names
    # Fallback safety if pool is empty
    if not pool:
        pool = ["Alex", "Riley", "Taylor", "Jordan"]
    return random.choice(pool)


def get_unique_folder_name(base_path: Path, desired_name: str) -> str:
    """
    Ensure the folder name is unique within base_path by appending a counter.
    Example: "Hannah", "Hannah_2", "Hannah_3", ...
    """
    candidate = desired_name
    counter = 1
    while (base_path / candidate).exists():
        counter += 1
        candidate = f"{desired_name}_{counter}"
    return candidate


def compute_bbox(img: Image.Image):
    """
    Compute the bounding box of non-transparent (or non-empty) content.
    For RGBA images this respects alpha; for others it uses overall content.
    Returns:
        bbox tuple (left, upper, right, lower) or None if no content.
    """
    try:
        return img.getbbox()
    except Exception:
        # Very old Pillow or unusual modes: convert as a fallback
        return img.convert("RGBA").getbbox()


def crop_to_bbox(img: Image.Image, bbox):
    """
    Crop the image to the given bounding box if it exists, otherwise return the original.
    """
    return img.crop(bbox) if bbox else img


def save_img_webp_or_png(img: Image.Image, dest_stem: Path) -> Path:
    """
    Save an image to disk, preferring WEBP and falling back to PNG if WEBP isn't available.
    Args:
        img: PIL Image (will be converted to RGBA to avoid palette/alpha issues).
        dest_stem: Path without extension (e.g., output_dir / "outfit_0")
    Returns:
        The full Path of the saved file (with extension).
    """
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)

    # Convert to a safe mode for web output
    safe = img.convert("RGBA")

    # Try WEBP first
    try:
        out_path = dest_stem.with_suffix(".webp")
        safe.save(out_path, "WEBP")
        return out_path
    except Exception as e:
        print(f"[WARN] WEBP save failed for {dest_stem.name}: {e}. Falling back to PNG.")
        out_path = dest_stem.with_suffix(".png")
        safe.save(out_path, "PNG")
        return out_path


def _wraplength_for(width_px: int) -> int:
    """Return a wraplength for labels that prevents edge collisions."""
    return max(200, width_px - WRAP_PADDING)


def compute_square_window_geometry(screen_w, screen_h, img_w, img_h):
    """
    Computes a consistent window size and placement for cropping/preview UIs.

    Returns:
        (scale_factor, display_w, display_h, window_size, center_x, center_y)

    Changes vs. previous:
    - Keeps the window entirely on-screen with a small top margin.
    - Uses DEFAULT_WINDOW_SCALE to size against screen width and height.
    """
    MIN_WINDOW_WIDTH = 900  # small bump for more breathing room

    usable_height = screen_h - HEADER_HEIGHT - FOOTER_HEIGHT - (WINDOW_MARGIN * 2)
    usable_width = int(screen_w * DEFAULT_WINDOW_SCALE)
    usable_dim = max(400, min(usable_height, usable_width))

    scale_factor = min(usable_dim / img_h, usable_dim / img_w, 1.0)
    display_w = int(img_w * scale_factor)
    display_h = int(img_h * scale_factor)

    window_size = max(display_w, display_h) + HEADER_HEIGHT + FOOTER_HEIGHT
    window_size = max(window_size, MIN_WINDOW_WIDTH)
    window_size = min(window_size, screen_h - (WINDOW_MARGIN * 2))

    center_x = max((screen_w - window_size) // 2, WINDOW_MARGIN)
    center_y = WINDOW_MARGIN  # keep near the top, not floating

    return scale_factor, display_w, display_h, window_size, center_x, center_y


# ---- safer thumbnail helper (replace your make_thumbnail_of_crop) ----
def make_thumbnail_of_crop(image, y_cut):
    """
    Returns a small thumbnail of the top portion of the crop,
    helping the user remember their previous crop choice.

    Notes:
        - Clamps y_cut to [1, image.height] to avoid zero-size crops.
    """
    y_cut = max(1, min(int(y_cut), image.height))
    cropped = image.crop((0, 0, image.width, y_cut))
    thumb_height = 150
    aspect_ratio = image.width / max(1, y_cut)
    thumb_width = max(1, int(thumb_height * aspect_ratio))
    return cropped.resize((thumb_width, thumb_height), Image.LANCZOS)


# -----------------------
# Crop UI (legs/head)
# -----------------------
def prompt_for_crop(image, prompt_text, previous_crops):
    """
    Crop UI with:
      - Fit-to-screen image sizing (reserves room for thumbs/buttons if present)
      - Live Y readout and cyan guide line
      - Scrollable previous-crop strip (always visible without resizing)
      - Clear tip explaining that previous crops are clickable to reuse
    Returns:
      (chosen_y_cut, used_gallery)
    """
    result = {"y_cut": None, "used_gallery": False}

    ow, oh = image.size
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Set Crop Line")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    # If we will render a thumb strip, slightly reduce the canvas height
    disp_w, disp_h = _compute_display_size(
        sw, sh, ow, oh,
        max_w_ratio=0.90,
        max_h_ratio=0.45 if previous_crops else 0.55
    )

    # Layout (no expanding rows -> no extra vertical slack)
    root.grid_columnconfigure(0, weight=0)

    wrap_len = _wraplength_for(int(sw * 0.9))
    tip = "\nTip: Click any thumbnail below to reuse that crop." if previous_crops else ""
    title = tk.Label(
        root,
        text=f"{prompt_text}\nClick on the image to set your horizontal crop line.{tip}",
        font=TITLE_FONT, bg=BG_COLOR, justify="center", wraplength=wrap_len
    )
    title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    disp_img = image.resize((disp_w, disp_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(disp_img)
    scale_y = oh / disp_h

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, highlightthickness=0, bg="black")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img
    canvas.grid(row=1, column=0, padx=10, pady=4, sticky="n")

    info = tk.Label(root, text="y: -", font=INSTRUCTION_FONT, bg=BG_COLOR)
    info.grid(row=2, column=0, pady=(2, 6))

    guide_id = None
    def _draw_guide(y_disp):
        nonlocal guide_id
        y_disp = max(0, min(int(y_disp), disp_h))
        if guide_id is None:
            guide_id = canvas.create_line(0, y_disp, disp_w, y_disp, fill=LINE_COLOR, width=3)
        else:
            canvas.coords(guide_id, 0, y_disp, disp_w, y_disp)
        info.config(text=f"y: {int(y_disp * scale_y)}")

    def _confirm(y_disp):
        y_disp = max(0, min(int(y_disp), disp_h))
        result["y_cut"] = int(y_disp * scale_y)
        result["used_gallery"] = False
        root.after(60, root.destroy)

    canvas.bind("<Motion>", lambda e: _draw_guide(e.y))
    canvas.bind("<Button-1>", lambda e: _confirm(e.y))
    _draw_guide(disp_h // 2)

    # Previous crops strip (horizontal scroll) — always visible when present
    if previous_crops:
        strip = tk.Frame(root, bg=BG_COLOR)
        strip.grid(row=3, column=0, padx=10, pady=(6, 4), sticky="we")

        strip_h = 160
        sc = tk.Canvas(strip, height=strip_h, bg=BG_COLOR, highlightthickness=0)
        hsb = tk.Scrollbar(strip, orient=tk.HORIZONTAL, command=sc.xview)
        sc.configure(xscrollcommand=hsb.set)
        inner = tk.Frame(sc, bg=BG_COLOR)
        sc.create_window((0, 0), window=inner, anchor="nw")

        # Keep references so Tk doesn't GC the images
        _thumb_refs = []
        for (y_val, thumb_img) in previous_crops:
            tki = ImageTk.PhotoImage(thumb_img)
            _thumb_refs.append(tki)
            def _use(y=y_val):
                result["y_cut"] = int(y)
                result["used_gallery"] = True
                root.destroy()
            tk.Button(inner, image=tki, command=_use).pack(side=tk.LEFT, padx=5, pady=5)

        def _upd(_=None):
            inner.update_idletasks()
            bbox = sc.bbox("all")
            if bbox:
                sc.config(scrollregion=bbox)
        inner.bind("<Configure>", _upd)
        _upd()

        sc.pack(side=tk.TOP, fill=tk.X, expand=True)
        hsb.pack(side=tk.TOP, fill=tk.X)

    # Buttons
    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=4, column=0, pady=(6, 10))
    tk.Button(btns, text="Cancel and Exit", command=lambda: sys.exit(0), font=INSTRUCTION_FONT).pack()

    # Fit window to content, then clamp + top-pin
    _center_and_clamp(root)
    root.mainloop()
    return result["y_cut"], result["used_gallery"]


# -----------------------
# Character data picker
# -----------------------
def prompt_for_character_data(image_path):
    """
    Eye line (click), hair color (click), then voice selection.
    Improvements:
      - Canvas stays horizontally centered across steps (no shifting).
      - Step 1 shows a cyan horizontal guide line under the cursor.
      - Step 2 shows a crosshair reticle at the cursor to aid color picking.
      - Girl/Boy row appears above Cancel and window re-clamps when shown.
    Returns:
      (eye_line_ratio, hair_color_hex, voice)
    """
    result = {"eye_line": None, "hair_color": None, "voice": None}
    state = {"step": 1}

    # Load and size
    img = Image.open(image_path).convert("RGBA")
    ow, oh = img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Pick Eye Line, Hair Color, Voice")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    # Keep the image short enough that buttons always fit
    dw, dh = _compute_display_size(
        sw, sh, ow, oh, max_w_ratio=0.90, max_h_ratio=0.44
    )
    sx, sy = ow / max(1, dw), oh / max(1, dh)  # display->original scale

    # Header/instructions (centered + wrapped)
    wrap_len = _wraplength_for(int(sw * 0.9))
    title = tk.Label(
        root,
        text="Step 1: Click to mark the eye line (relative head height).",
        font=TITLE_FONT, bg=BG_COLOR, wraplength=wrap_len, justify="center"
    )
    title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # ---- Centered canvas wrapper ----
    # We put the canvas in a fixed-size frame and center it with place()
    cwrap = tk.Frame(root, bg=BG_COLOR, width=dw, height=dh)
    cwrap.grid(row=1, column=0, padx=10, pady=4, sticky="n")
    cwrap.grid_propagate(False)  # keep wrapper size fixed (prevents sideways "shift")

    # Canvas with the resized sprite
    disp = img.resize((dw, dh), Image.LANCZOS)
    tki = ImageTk.PhotoImage(disp)
    cvs = tk.Canvas(cwrap, width=dw, height=dh, highlightthickness=0, bg="black")
    cvs.create_image(0, 0, anchor="nw", image=tki)
    cvs.image = tki
    cvs.place(relx=0.5, rely=0.0, anchor="n")  # horizontally centered in wrapper

    # Overlay elements for guidance
    guide_line_id = None     # horizontal line for eye-line stage
    reticle_h_id = None      # hair-pick crosshair (horizontal)
    reticle_v_id = None      # hair-pick crosshair (vertical)

    def draw_eyeline(y_disp: int):
        """Draw/move the horizontal eye-line guide while in step 1."""
        nonlocal guide_line_id
        y_disp = max(0, min(int(y_disp), dh))
        if guide_line_id is None:
            guide_line_id = cvs.create_line(0, y_disp, dw, y_disp, fill=LINE_COLOR, width=3)
        else:
            cvs.coords(guide_line_id, 0, y_disp, dw, y_disp)

    def clear_eyeline():
        """Remove the eye-line guide."""
        nonlocal guide_line_id
        if guide_line_id is not None:
            cvs.delete(guide_line_id)
            guide_line_id = None

    def draw_reticle(x_disp: int, y_disp: int, arm: int = 16):
        """Draw/move a crosshair reticle for hair color picking (step 2)."""
        nonlocal reticle_h_id, reticle_v_id
        x_disp = max(0, min(int(x_disp), dw))
        y_disp = max(0, min(int(y_disp), dh))
        if reticle_h_id is None:
            reticle_h_id = cvs.create_line(x_disp - arm, y_disp, x_disp + arm, y_disp, fill=LINE_COLOR, width=2)
            reticle_v_id = cvs.create_line(x_disp, y_disp - arm, x_disp, y_disp + arm, fill=LINE_COLOR, width=2)
        else:
            cvs.coords(reticle_h_id, x_disp - arm, y_disp, x_disp + arm, y_disp)
            cvs.coords(reticle_v_id, x_disp, y_disp - arm, x_disp, y_disp + arm)

    def clear_reticle():
        """Remove the hair-pick crosshair."""
        nonlocal reticle_h_id, reticle_v_id
        if reticle_h_id is not None:
            cvs.delete(reticle_h_id)
            cvs.delete(reticle_v_id)
            reticle_h_id = reticle_v_id = None

    # Bottom area (two rows: 0=voice row (hidden initially), 1=Cancel)
    bottom = tk.Frame(root, bg=BG_COLOR)
    bottom.grid(row=2, column=0, pady=(6, 10))
    bottom.grid_columnconfigure(0, weight=1)

    voice_row = tk.Frame(bottom, bg=BG_COLOR)  # gridded when we reach step 3

    def choose_voice(v):
        """Finalize the selection and close the window."""
        result["voice"] = v
        root.destroy()

    g_btn = tk.Button(voice_row, text="Girl", width=12, command=lambda: choose_voice("girl"))
    b_btn = tk.Button(voice_row, text="Boy",  width=12, command=lambda: choose_voice("boy"))
    g_btn.pack(side=tk.LEFT, padx=10)
    b_btn.pack(side=tk.LEFT, padx=10)

    cancel_btn = tk.Button(bottom, text="Cancel and Exit", command=lambda: sys.exit(0), font=INSTRUCTION_FONT)
    cancel_btn.grid(row=1, column=0, pady=(8, 0))

    # --- Mouse interactions ---
    def on_motion(e):
        """Update overlays depending on the current step."""
        if state["step"] == 1:
            draw_eyeline(e.y)
        elif state["step"] == 2:
            draw_reticle(e.x, e.y)

    def on_click(e):
        """Handle clicks for step 1 (eye line) and step 2 (hair color)."""
        nonlocal wrap_len
        if state["step"] == 1:
            # Record eye line as a ratio of original image height
            real_y = e.y * sy
            result["eye_line"] = real_y / oh
            clear_eyeline()
            title.config(
                text="Eye line recorded.\nStep 2: Click on the hair color.",
                wraplength=wrap_len
            )
            state["step"] = 2
            # Show an initial reticle at the cursor position for clarity
            draw_reticle(e.x, e.y)

        elif state["step"] == 2:
            # Sample color from original image coordinates
            rx = min(max(int(e.x * sx), 0), ow - 1)
            ry = min(max(int(e.y * sy), 0), oh - 1)
            px = img.getpixel((rx, ry))
            # Treat nearly-transparent pixels as a neutral fallback color
            color = "#915f40" if (len(px) == 4 and px[3] < 10) else f"#{px[0]:02x}{px[1]:02x}{px[2]:02x}"
            result["hair_color"] = color
            clear_reticle()
            title.config(
                text="Hair color recorded.\nStep 3: Select the character's voice.",
                wraplength=wrap_len
            )
            state["step"] = 3
            # Reveal voice row above Cancel and re-clamp to keep everything visible
            voice_row.grid(row=0, column=0, pady=(0, 6))
            _center_and_clamp(root)

    cvs.bind("<Motion>", on_motion)
    cvs.bind("<Button-1>", on_click)

    # Start with a centered eye-line guide
    draw_eyeline(dh // 2)

    # Initial size/clamp
    _center_and_clamp(root)
    root.mainloop()

    print(f"[INFO] Eye line: {result['eye_line']:.3f}")
    print(f"[INFO] Hair color: {result['hair_color']}")
    print(f"[INFO] Voice: {result['voice']}")
    return result["eye_line"], result["hair_color"], result["voice"]




# -----------------------
# Scale picker
# -----------------------
def prompt_for_scale(image_path, user_eye_line_ratio=None):
    """
    Side-by-side scaling UI.

    Goals:
      - Reference renders at its YAML "scale" (true in-game size).
      - User sprite renders at slider scale.
      - A *view_scale* is applied equally to both so they always fit inside
        their canvases (width & height), preserving their relative sizes.
      - Both canvases are bottom-aligned to a shared "floor" (no scrollbars).
      - Optional cyan eye-line guide is drawn ONLY for the user sprite.

    Args:
        image_path: Path to the user's (cropped) image to scale.
        user_eye_line_ratio: Optional float in [0..1] (from earlier UI) to draw
            the user's eye-line for easier proportion matching.

    Returns:
        float: The chosen scale that should be written to character.yml
    """
    import yaml

    # --- Load reference sprites & scales (eye_line ignored) ---
    REF_SPRITES_DIR = "reference_sprites"
    refs = {}
    for fn in os.listdir(REF_SPRITES_DIR):
        if not fn.lower().endswith(".png"):
            continue
        name = os.path.splitext(fn)[0]
        img_path = os.path.join(REF_SPRITES_DIR, fn)
        yml_path = os.path.join(REF_SPRITES_DIR, name + ".yml")

        # Default scale = 1.0 if YAML missing or unreadable
        ref_scale = 1.0
        if os.path.exists(yml_path):
            try:
                with open(yml_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
                ref_scale = float(meta.get("scale", 1.0))
            except Exception:
                pass

        try:
            img = Image.open(img_path).convert("RGBA")
            refs[name] = {"image": img, "scale": ref_scale}
        except Exception as e:
            print(f"[WARN] Skipping reference '{img_path}': {e}")

    if not refs:
        print(f"[ERROR] No usable reference sprites found in '{REF_SPRITES_DIR}/'.")
        sys.exit(1)

    names = sorted(refs.keys())

    # --- Load the user's sprite (cropped head/outfit image you just made) ---
    user_img = Image.open(image_path).convert("RGBA")
    uw, uh = user_img.size

    # --- Window / layout (no scrollbars by design) ---
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Adjust Scale vs Reference (in-game accurate + screen-fit)")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()

    # Two canvases side-by-side; keep room for header + dropdown + slider + buttons.
    CANV_H = max(int(sh * 0.52), 360)
    CANV_W = max(int((sw - 3 * WINDOW_MARGIN) // 2), 360)

    wrap_len = _wraplength_for(int(sw * 0.9))
    title = tk.Label(
        root,
        text="1) Choose a reference (left is true in-game size).  "
             "2) Adjust your scale (right) to match proportions.  3) Done.",
        font=INSTRUCTION_FONT, bg=BG_COLOR, wraplength=wrap_len, justify="center"
    )
    title.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 6), sticky="we")

    sel = tk.StringVar(value=names[0])
    tk.OptionMenu(root, sel, *names).grid(row=1, column=0, columnspan=2, pady=(0, 6))

    # Fixed-size canvases; no scrollbars so the "floor" never moves.
    ref_canvas = tk.Canvas(root, width=CANV_W, height=CANV_H, bg="black", highlightthickness=0)
    usr_canvas = tk.Canvas(root, width=CANV_W, height=CANV_H, bg="black", highlightthickness=0)
    ref_canvas.grid(row=2, column=0, padx=(10, 5), pady=6, sticky="n")
    usr_canvas.grid(row=2, column=1, padx=(5, 10), pady=6, sticky="n")

    # Slider controls only the user's in-game scale value (what we save)
    scale_val = tk.DoubleVar(value=1.0)
    slider = tk.Scale(
        root, from_=0.1, to=2.5, resolution=0.01, orient=tk.HORIZONTAL,
        label="Adjust Your Character's Scale (in-game)", variable=scale_val,
        length=int(sw * 0.8), tickinterval=0.05
    )
    slider.grid(row=3, column=0, columnspan=2, padx=10, pady=(4, 8), sticky="we")

    # References to Tk images to avoid garbage collection
    _img_refs = {"ref": None, "usr": None}

    def _redraw(*_):
        """
        Draw both images bottom-aligned, with a shared view_scale that ensures
        both fit in CANV_W x CANV_H (width & height). This keeps in-game sizes
        proportional while guaranteeing on-screen fit without scrolling.
        """
        ref_canvas.delete("all")
        usr_canvas.delete("all")

        # --- Reference sprite at YAML 'scale' (in-game size) ---
        r_meta = refs[sel.get()]
        rimg = r_meta["image"]
        r_scale = r_meta["scale"]
        r_engine_w = rimg.width * r_scale
        r_engine_h = rimg.height * r_scale

        # --- User sprite at chosen (in-game) scale ---
        u_scale = float(scale_val.get())
        u_engine_w = uw * u_scale
        u_engine_h = uh * u_scale

        # --- Compute a single view_scale for both (fit to panel width & height) ---
        max_w = max(r_engine_w, u_engine_w)
        max_h = max(r_engine_h, u_engine_h)
        view_scale = min(CANV_W / max_w, CANV_H / max_h, 1.0)

        # Effective display dimensions
        r_disp_w = max(1, int(r_engine_w * view_scale))
        r_disp_h = max(1, int(r_engine_h * view_scale))
        u_disp_w = max(1, int(u_engine_w * view_scale))
        u_disp_h = max(1, int(u_engine_h * view_scale))

        # --- Render bottom-aligned ("floor" at canvas bottom) ---
        r_resized = rimg.resize((r_disp_w, r_disp_h), Image.LANCZOS)
        _img_refs["ref"] = ImageTk.PhotoImage(r_resized)
        ref_canvas.create_image(CANV_W // 2, CANV_H, anchor="s", image=_img_refs["ref"])

        u_resized = user_img.resize((u_disp_w, u_disp_h), Image.LANCZOS)
        _img_refs["usr"] = ImageTk.PhotoImage(u_resized)
        usr_canvas.create_image(CANV_W // 2, CANV_H, anchor="s", image=_img_refs["usr"])

        # --- Optional eye-line guide for the user only ---
        if isinstance(user_eye_line_ratio, (int, float)) and 0.0 <= user_eye_line_ratio <= 1.0:
            img_top = CANV_H - u_disp_h
            y_inside = int(u_disp_h * float(user_eye_line_ratio))
            y_canvas = img_top + y_inside
            usr_canvas.create_line(0, y_canvas, CANV_W, y_canvas, fill=LINE_COLOR, width=2)

    # IMPORTANT CHANGE:
    #   - We do NOT redraw on every slider tick.
    #   - We only redraw (a) when the user releases the mouse, and (b) when the
    #     reference selection changes. This removes the jank during drags.

    # Redraw once when the user releases the slider
    slider.bind("<ButtonRelease-1>", lambda e: _redraw())

    # Also redraw on keyboard changes after key release (fine-grained adjustments)
    slider.bind("<KeyRelease>", lambda e: _redraw())

    # When the reference changes, repaint immediately
    sel.trace_add("write", lambda *_: _redraw())

    # Initial paint
    _redraw()

    # Buttons
    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=4, column=0, columnspan=2, pady=(6, 10))
    tk.Button(btns, text="Done - Use This Scale", command=root.destroy).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", command=lambda: sys.exit(0)).pack(side=tk.LEFT, padx=10)

    _center_and_clamp(root)
    root.mainloop()

    chosen = float(scale_val.get())
    print(f"[INFO] User-picked scale: {chosen:.3f}")
    return chosen



# -----------------------
# Confirm window
# -----------------------
def confirm_character(image_paths):
    """
    Scrollable review with larger thumbs; window fits content and clamps to screen.
    Uses responsive columns (2–4) based on screen width so we don't cut off a column.
    """
    decision = {"proceed": None}

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Review Poses: Continue or Redo")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="Review all pose/outfit crops below. Click Continue to accept or Redo to start over.",
        font=INSTRUCTION_FONT, bg=BG_COLOR, wraplength=wrap_len, justify="center"
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # Canvas sized to most of the screen width; we keep a single vertical scrollbar
    canvas_w = int(sw * 0.90) - 2 * WINDOW_MARGIN
    canvas_h = int(sh * 0.60)  # leave room for buttons
    outer = tk.Frame(root, bg=BG_COLOR)
    outer.grid(row=1, column=0, sticky="n", padx=10, pady=6)

    canvas = tk.Canvas(outer, width=canvas_w, height=canvas_h, bg="black", highlightthickness=0)
    v_scroll = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=v_scroll.set)
    canvas.grid(row=0, column=0, sticky="n")
    v_scroll.grid(row=0, column=1, sticky="ns")

    inner = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    # Choose column count based on available width
    #  <1400px -> 2 cols, <2200px -> 3 cols, else 4 cols
    cols = 2 if sw < 1400 else (3 if sw < 2200 else 4)
    pad = 12
    slot_w = max(220, (canvas_w - (cols + 1) * pad) // cols)  # column "slot" width

    TH = 280  # target max thumb height
    thumb_refs = []

    # Create a fixed-size slot for each image and center the thumb inside it.
    def add_thumb_slot(parent, r, c, pil_img):
        # Scale thumb to fit both height TH and slot width
        ar = pil_img.width / pil_img.height if pil_img.height else 1.0
        w_by_h = int(TH * ar)
        tw = min(w_by_h, slot_w)
        th = min(TH, max(1, int(tw / max(ar, 0.0001))))
        thumb = pil_img.resize((tw, th), Image.LANCZOS)
        tki = ImageTk.PhotoImage(thumb)
        thumb_refs.append(tki)

        slot = tk.Frame(parent, width=slot_w, height=TH, bg=BG_COLOR, highlightthickness=0)
        slot.grid(row=r, column=c, padx=pad, pady=pad)
        slot.grid_propagate(False)  # keep the slot fixed size
        lbl = tk.Label(slot, image=tki, bg=BG_COLOR)
        lbl.place(relx=0.5, rely=0.5, anchor="center")

    # Lay out thumbs row by row
    r = c = 0
    for p in image_paths:
        try:
            im = Image.open(p)
            add_thumb_slot(inner, r, c, im)
            c += 1
            if c >= cols:
                c = 0
                r += 1
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")

    def _upd(_=None):
        inner.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner.bind("<Configure>", _upd)
    _upd()

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(6, 10))

    def accept():
        decision["proceed"] = True
        root.destroy()
    def redo():
        decision["proceed"] = False
        root.destroy()

    tk.Button(btns, text="Continue", width=20, command=accept).pack(side=tk.LEFT, padx=20)
    tk.Button(btns, text="Redo Character", width=20, command=redo).pack(side=tk.LEFT, padx=20)

    _center_and_clamp(root)
    root.mainloop()
    return decision["proceed"]



# -----------------------
# YAML Writer
# -----------------------
def write_character_yml(path, display_name, voice, eye_line, hair_color, scale, poses, *, game=None):
    """
    Writes the final character metadata to a YAML file.
    Adds `game` without changing existing behavior.
    """
    v = (voice or "").lower()
    voice_out = "male" if v == "boy" else voice

    data = {
        "display_name": display_name,
        "eye_line": round(eye_line, 4),
        "name_color": hair_color,
        "poses": poses,
        "scale": scale,
        "voice": voice_out,
    }
    if game:
        data["game"] = game

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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

    # Determine source game from downloader metadata if available
    game_name = "_unknown_game"
    meta_path = workspace_dir / "download_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            g = (meta.get("source_game") or "").strip()
            if g:
                game_name = g
        except Exception as e:
            print(f"[WARN] Could not read download_meta.json: {e}")
    print(f"[INFO] Using game name for character.yml: {game_name}")

    girl_names, boy_names = load_name_pool("names.csv")
    letter_labels = list(string.ascii_lowercase)
    character_folders = sorted([f for f in workspace_dir.iterdir() if f.is_dir()])

    print(f"[INFO] Found {len(character_folders)} character folders to process.")

    produced = []  # collect final character folders we produced

    for char_index, char_folder in enumerate(character_folders):
        while True:
            temp_char_name = f"char_{char_index + 1}"
            char_output = output_dir / temp_char_name
            print(f"\n[INFO] Processing '{char_folder.name}' -> '{temp_char_name}'")
            char_output.mkdir(parents=True, exist_ok=True)

            pose_folders = sorted([p for p in char_folder.iterdir() if p.is_dir()])
            poses_yaml = {}
            all_face_images = []

            # Remember cuts across poses (per character)
            previous_leg_crops = []   # FIX: initialize legs gallery
            previous_head_crops = []  # remember head/chin cuts

            for pose_index, pose_folder in enumerate(pose_folders):
                # Label poses a..z then p00, p01, ...
                letter = letter_labels[pose_index] if pose_index < len(letter_labels) else f"p{pose_index:02d}"
                print(f"[INFO] Pose: '{pose_folder.name}' -> '{letter}'")

                pose_output = char_output / letter
                faces_face_dir = pose_output / "faces" / "face"
                outfits_dir = pose_output / "outfits"
                faces_face_dir.mkdir(parents=True, exist_ok=True)
                outfits_dir.mkdir(parents=True, exist_ok=True)

                # Collect images; verify and skip corrupt/unsupported
                raw_files = sorted([
                    f for f in pose_folder.iterdir()
                    if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
                ])
                if not raw_files:
                    print(f"[WARN] No images found in pose folder: {pose_folder}")
                    continue

                safe_files = []
                for f in raw_files:
                    try:
                        with Image.open(f) as im:
                            im.verify()  # lightweight integrity check
                        safe_files.append(f)
                    except Exception as e:
                        print(f"[WARN] Skipping corrupt or unsupported image: {f} ({e})")

                image_files = safe_files
                if not image_files:
                    print(f"[WARN] All images in '{pose_folder}' were unreadable. Skipping.")
                    continue

                # === 1. Auto-Crop Padding ===
                print("[INFO] Auto-cropping padding")
                first_img = Image.open(image_files[0]).convert("RGBA")
                bbox = compute_bbox(first_img)
                first_img_cropped = crop_to_bbox(first_img, bbox)

                # === 2. Legs Crop ===
                leg_cut, used_gallery = prompt_for_crop(
                    first_img_cropped,
                    f"[{pose_folder.name}] Click at the height of the character's mid-thigh. This removes the lower part of the image.",
                    previous_leg_crops
                )
                # Clamp leg cut for safety
                leg_cut = max(1, min(int(leg_cut), first_img_cropped.height))

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
                outfit_path = save_img_webp_or_png(cropped_leg_images[0], outfits_dir / f"outfit_{pose_index}")
                all_face_images.append(outfit_path)

                # === 4. Head Crop ===
                print("[INFO] Prompting for expression crop")
                head_cut, used_gallery_head = prompt_for_crop(
                    cropped_leg_images[0],
                    f"[{pose_folder.name}] Click at the height of the character's chin. This will set the bottom of the character's expression sheet.",
                    previous_head_crops
                )
                # Clamp head cut for safety
                head_cut = max(1, min(int(head_cut), cropped_leg_images[0].height))

                # Keep a small gallery of head crops too
                if not used_gallery_head:
                    thumb_head = make_thumbnail_of_crop(cropped_leg_images[0], head_cut)
                    previous_head_crops.append((head_cut, thumb_head))

                # Save faces
                for i, img in enumerate(cropped_leg_images):
                    face_img = img.crop((0, 0, img.width, head_cut))
                    _ = save_img_webp_or_png(face_img, faces_face_dir / f"{i}")

                poses_yaml[letter] = {"facing": "right"}

            # === Confirm Step ===
            if all_face_images:
                print("\n[INFO] Collecting metadata for character")
                eye_line, hair_color, voice = prompt_for_character_data(all_face_images[0])

                # Pass the user eye-line ratio into the scale UI so we can draw a guide
                scale = prompt_for_scale(all_face_images[0], user_eye_line_ratio=eye_line)


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
                produced.append(str(char_output))
            else:
                eye_line = 0.195
                hair_color = "#ffffff"
                voice = "girl"
                display_name = temp_char_name
                scale = 1.0

            yml_path = char_output / "character.yml"
            write_character_yml(
                yml_path,
                display_name,
                voice,
                eye_line,
                hair_color,
                scale,
                poses_yaml,
                game=game_name
            )
            print(f"[INFO] Created metadata: {yml_path}")

            break

    print("\n[INFO] All done! Your Student Transfer sprite folders are saved to:")
    print(f"       {output_input}")
    if produced:
        print("[INFO] Characters created:")
        for p in produced:
            print(f"  - {p}")

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

    print("\n[INFO] Organizer step complete.")
    return output_input

# -----------------------
# Entry Point
# -----------------------
if __name__ == "__main__":
    run_organizer_interactive()
