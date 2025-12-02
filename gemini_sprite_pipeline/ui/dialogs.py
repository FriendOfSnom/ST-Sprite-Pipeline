"""
Dialog windows for user input and configuration.

Provides interactive dialogs for character setup, outfit/expression selection,
cropping, eye line/color selection, and scaling.
"""

import csv
import os
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageTk

from gsp_constants import (
    NAMES_CSV_PATH,
    REF_SPRITES_DIR,
    GENDER_ARCHETYPES,
    ALL_OUTFIT_KEYS,
    OUTFIT_KEYS,
    EXPRESSIONS_SEQUENCE,
)

from .tk_common import (
    BG_COLOR,
    TITLE_FONT,
    INSTRUCTION_FONT,
    LINE_COLOR,
    center_and_clamp,
    compute_display_size,
    wraplength_for,
)


# =============================================================================
# Name Pool Utilities
# =============================================================================

def load_name_pool(csv_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load girl/boy name pools from CSV with columns: name, gender.

    Args:
        csv_path: Path to CSV file containing names.

    Returns:
        (girl_names, boy_names) tuple of name lists.
    """
    girl_names: List[str] = []
    boy_names: List[str] = []

    try:
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gender = (row.get("gender") or "").strip().lower()
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                if gender == "girl":
                    girl_names.append(name)
                elif gender == "boy":
                    boy_names.append(name)
    except FileNotFoundError:
        print(f"[WARN] Could not find {csv_path}. Using fallback names.")
        girl_names = ["Sakura", "Emily", "Yuki", "Hannah", "Aiko", "Madison", "Kana", "Sara"]
        boy_names = ["Takashi", "Ethan", "Yuto", "Liam", "Kenta", "Jacob", "Hiro", "Alex"]
    except Exception as e:
        print(f"[WARN] Failed to read {csv_path}: {e}. Using fallback names.")
        girl_names = ["Sakura", "Emily", "Yuki", "Hannah", "Aiko", "Madison", "Kana", "Sara"]
        boy_names = ["Takashi", "Ethan", "Yuto", "Liam", "Kenta", "Jacob", "Hiro", "Alex"]

    return girl_names, boy_names


def pick_random_name(voice: str, girl_names: List[str], boy_names: List[str]) -> str:
    """
    Pick a random name based on voice.

    Args:
        voice: "girl" or "boy".
        girl_names: List of girl names.
        boy_names: List of boy names.

    Returns:
        Random name from appropriate list.
    """
    pool = girl_names if (voice or "").lower() == "girl" else boy_names
    if not pool:
        pool = ["Alex", "Riley", "Taylor", "Jordan"]
    return random.choice(pool)


# =============================================================================
# Character Setup Dialogs
# =============================================================================

def prompt_voice_archetype_and_name(image_path: Path) -> Tuple[str, str, str, str]:
    """
    Show source image and prompt for voice, name, and archetype.

    Args:
        image_path: Path to source character image.

    Returns:
        (voice, display_name, archetype_label, gender_style)
    """
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)
    img = Image.open(image_path).convert("RGBA")
    original_w, original_h = img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Character Setup")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    # Title
    title = tk.Label(
        root,
        text="Choose this character's voice, name, and archetype.",
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg="black",
        wraplength=wrap_len,
        justify="center",
    )
    title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # Image preview
    disp_w, disp_h = compute_display_size(
        sw, sh, original_w, original_h,
        max_w_ratio=0.70, max_h_ratio=0.45
    )
    disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(disp_img)
    canvas = tk.Canvas(root, width=disp_w, height=disp_h, bg="black", highlightthickness=0)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img  # type: ignore
    canvas.grid(row=1, column=0, padx=10, pady=4, sticky="n")

    # State variables
    voice_var = tk.StringVar(value="")
    archetype_var = tk.StringVar(value="")
    gender_style_var = {"value": None}
    name_var = tk.StringVar(value="")

    # Name entry
    name_frame = tk.Frame(root, bg=BG_COLOR)
    name_frame.grid(row=2, column=0, pady=(4, 4))
    tk.Label(
        name_frame,
        text="Character Name:",
        font=INSTRUCTION_FONT,
        bg=BG_COLOR,
        fg="black",
    ).pack(side=tk.LEFT, padx=(0, 6))
    name_entry = tk.Entry(name_frame, textvariable=name_var, width=24)
    name_entry.pack(side=tk.LEFT)

    def update_archetype_menu():
        """Update archetype menu based on selected voice."""
        menu = arche_menu["menu"]
        menu.delete(0, "end")
        v = voice_var.get()
        if v == "girl":
            labels = [label for (label, g) in GENDER_ARCHETYPES if g == "f"]
            gstyle = "f"
        elif v == "boy":
            labels = [label for (label, g) in GENDER_ARCHETYPES if g == "m"]
            gstyle = "m"
        else:
            labels = []
            gstyle = None
        gender_style_var["value"] = gstyle
        archetype_var.set(labels[0] if labels else "")
        for lbl in labels:
            menu.add_command(label=lbl, command=lambda v=lbl: archetype_var.set(v))

    def choose_voice(v: str):
        """Handle voice button click."""
        voice_var.set(v)
        display_name = pick_random_name(v, girl_names, boy_names)
        if not name_var.get().strip():
            name_var.set(display_name)
        update_archetype_menu()
        name_entry.focus_set()
        name_entry.icursor(tk.END)

    # Voice buttons
    btn_row = tk.Frame(root, bg=BG_COLOR)
    btn_row.grid(row=3, column=0, pady=(4, 4))
    tk.Button(btn_row, text="Girl", width=12, command=lambda: choose_voice("girl")).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btn_row, text="Boy", width=12, command=lambda: choose_voice("boy")).pack(
        side=tk.LEFT, padx=10
    )

    # Archetype menu
    archetype_frame = tk.Frame(root, bg=BG_COLOR)
    archetype_frame.grid(row=4, column=0, pady=(4, 4))
    tk.Label(
        archetype_frame,
        text="Archetype:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).pack(side=tk.LEFT, padx=(0, 6))
    arche_menu = tk.OptionMenu(archetype_frame, archetype_var, "")
    arche_menu.config(width=20)
    arche_menu.pack(side=tk.LEFT)

    decision = {"done": False, "voice": None, "name": None, "arch": None, "gstyle": None}

    def on_ok():
        """Handle OK button click."""
        v = voice_var.get()
        nm = name_var.get().strip()
        arch = archetype_var.get()
        gs = gender_style_var["value"]
        if not v or not arch or not gs:
            messagebox.showerror("Missing data", "Please choose voice and archetype.")
            return
        if not nm:
            nm = pick_random_name(v, girl_names, boy_names)
        decision.update(done=True, voice=v, name=nm, arch=arch, gstyle=gs)
        root.destroy()

    def on_cancel():
        """Handle cancel button click."""
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    # Bottom buttons
    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=5, column=0, pady=(8, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(side=tk.LEFT, padx=10)

    center_and_clamp(root)
    root.mainloop()

    if not decision["done"]:
        sys.exit(0)

    return (
        decision["voice"],
        decision["name"],
        decision["arch"],
        decision["gstyle"],
    )


def prompt_source_mode() -> str:
    """
    Ask whether to generate from an image or from a text prompt.

    Returns:
        "image", "prompt", or exits if cancelled.
    """
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Sprite Source")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="How would you like to create this character?",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    mode_var = tk.StringVar(value="image")

    modes_frame = tk.Frame(root, bg=BG_COLOR)
    modes_frame.grid(row=1, column=0, pady=(4, 8))

    tk.Radiobutton(
        modes_frame,
        text="From an existing image (pick a file)",
        variable=mode_var,
        value="image",
        bg=BG_COLOR,
        anchor="w",
    ).pack(anchor="w", padx=10, pady=2)

    tk.Radiobutton(
        modes_frame,
        text="From a text prompt (Gemini designs a new character)",
        variable=mode_var,
        value="prompt",
        bg=BG_COLOR,
        anchor="w",
    ).pack(anchor="w", padx=10, pady=2)

    decision = {"mode": "image"}

    def on_ok():
        decision["mode"] = mode_var.get()
        root.destroy()

    def on_cancel():
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(6, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(side=tk.LEFT, padx=10)

    center_and_clamp(root)
    root.mainloop()

    return decision["mode"]


def prompt_character_idea_and_archetype() -> Tuple[str, str, str, str, str]:
    """
    Prompt for character concept text, voice, name, and archetype.

    Returns:
        (concept_text, archetype_label, voice, display_name, gender_style)
    """
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Character Concept")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="Describe the kind of character you want Gemini to design,\n"
        "then choose their voice, name, and archetype.",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # Text input area
    text_frame = tk.Frame(root, bg=BG_COLOR)
    text_frame.grid(row=1, column=0, padx=10, pady=(4, 4), sticky="nsew")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    txt = tk.Text(text_frame, width=60, height=8, wrap="word")
    txt.pack(fill="both", expand=True)

    # State variables
    voice_var = tk.StringVar(value="")
    name_var = tk.StringVar(value="")
    gender_style_var = {"value": None}

    # Voice and name controls
    vn_frame = tk.Frame(root, bg=BG_COLOR)
    vn_frame.grid(row=2, column=0, padx=10, pady=(4, 4), sticky="we")

    tk.Label(
        vn_frame,
        text="Voice:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).grid(row=0, column=0, padx=(0, 6), pady=2, sticky="w")

    def _pick_random_name_for_voice(v: str) -> str:
        return pick_random_name(v, girl_names, boy_names)

    def set_voice(v: str):
        voice_var.set(v)
        if not name_var.get().strip():
            name_var.set(_pick_random_name_for_voice(v))
        update_archetype_menu()

    tk.Button(
        vn_frame, text="Girl", width=10, command=lambda: set_voice("girl")
    ).grid(row=0, column=1, padx=4, pady=2, sticky="w")
    tk.Button(
        vn_frame, text="Boy", width=10, command=lambda: set_voice("boy")
    ).grid(row=0, column=2, padx=4, pady=2, sticky="w")

    tk.Label(
        vn_frame,
        text="Name:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).grid(row=1, column=0, padx=(0, 6), pady=2, sticky="w")

    name_entry = tk.Entry(vn_frame, textvariable=name_var, width=24)
    name_entry.grid(row=1, column=1, columnspan=2, padx=4, pady=2, sticky="w")

    # Archetype menu
    arch_frame = tk.Frame(root, bg=BG_COLOR)
    arch_frame.grid(row=3, column=0, pady=(4, 4))

    tk.Label(
        arch_frame,
        text="Archetype:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).pack(side=tk.LEFT, padx=(0, 6))

    arch_var = tk.StringVar(value="")
    arche_menu = tk.OptionMenu(arch_frame, arch_var, "")
    arche_menu.config(width=24)
    arche_menu.pack(side=tk.LEFT)

    def update_archetype_menu():
        menu = arche_menu["menu"]
        menu.delete(0, "end")
        v = voice_var.get()
        if v == "girl":
            labels = [label for (label, g) in GENDER_ARCHETYPES if g == "f"]
            gs = "f"
        elif v == "boy":
            labels = [label for (label, g) in GENDER_ARCHETYPES if g == "m"]
            gs = "m"
        else:
            labels = []
            gs = None

        gender_style_var["value"] = gs

        arch_var.set(labels[0] if labels else "")
        for lbl in labels:
            menu.add_command(label=lbl, command=lambda v=lbl: arch_var.set(v))

    decision = {
        "ok": False,
        "concept": "",
        "archetype": "",
        "voice": "",
        "name": "",
        "gstyle": None,
    }

    def on_ok():
        concept = txt.get("1.0", "end").strip()
        v = voice_var.get()
        nm = name_var.get().strip()
        arch = arch_var.get()
        gs = gender_style_var["value"]

        if not concept:
            messagebox.showerror("Missing description", "Please describe the character concept.")
            return
        if not v or not arch or not gs:
            messagebox.showerror("Missing data", "Please choose a voice and archetype.")
            return
        if not nm:
            nm = _pick_random_name_for_voice(v)

        decision.update(
            ok=True,
            concept=concept,
            archetype=arch,
            voice=v,
            name=nm,
            gstyle=gs,
        )
        root.destroy()

    def on_cancel():
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=4, column=0, pady=(6, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(side=tk.LEFT, padx=10)

    center_and_clamp(root)
    root.mainloop()

    if not decision["ok"]:
        sys.exit(0)

    return (
        decision["concept"],
        decision["archetype"],
        decision["voice"],
        decision["name"],
        decision["gstyle"],
    )
import yaml
from PIL import Image, ImageTk

from gsp_constants import (
    REF_SPRITES_DIR,
    ALL_OUTFIT_KEYS,
    OUTFIT_KEYS,
    EXPRESSIONS_SEQUENCE,
)

from .tk_common import (
    BG_COLOR,
    TITLE_FONT,
    INSTRUCTION_FONT,
    LINE_COLOR,
    center_and_clamp,
    compute_display_size,
    wraplength_for,
)


def prompt_outfits_and_expressions(
    archetype_label: str,
    gender_style: str,
) -> Tuple[
    List[str],
    List[Tuple[str, str]],
    Dict[str, Dict[str, Optional[str]]],
]:
    """
    Tk dialog asking which outfits and expressions to generate.

    For each outfit:
      - Checkbox: whether to generate that outfit.
      - Radio buttons:
          - Random: use a random CSV prompt (if available).
          - Custom: user types their own prompt.
          - Standard (uniform only, when archetype is young woman/man): use
            the standardized school uniform reference sprites.

    Returns:
        (selected_outfit_keys, expressions_sequence, outfit_prompt_config)

    Where outfit_prompt_config[key] contains:
        {
          "use_random": bool,
          "custom_prompt": Optional[str],
          "use_standard_uniform": bool  # only meaningful for 'uniform'
        }
    """
    # Only young woman/man archetypes get the standardized school uniform toggle
    arch_lower = (archetype_label or "").strip().lower()
    uniform_eligible = (
        (arch_lower == "young woman" and gender_style == "f")
        or (arch_lower == "young man" and gender_style == "m")
    )

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Generation Options")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text=(
            "Choose which outfits and expressions to generate for this sprite.\n\n"
            "Base outfit is always included.\n"
            "Neutral expression is always included."
        ),
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    body_frame = tk.Frame(root, bg=BG_COLOR)
    body_frame.grid(row=1, column=0, padx=10, pady=(4, 4), sticky="nsew")
    body_frame.grid_columnconfigure(0, weight=1)
    body_frame.grid_columnconfigure(1, weight=1)

    # =================================================================
    # LEFT COLUMN: Outfits
    # =================================================================
    outfit_frame = tk.LabelFrame(
        body_frame,
        text="Additional outfits (Base is always included):",
        bg=BG_COLOR,
    )
    outfit_frame.grid(row=0, column=0, padx=5, pady=4, sticky="nsew")
    outfit_frame.grid_columnconfigure(0, weight=0)
    outfit_frame.grid_columnconfigure(1, weight=0)
    outfit_frame.grid_columnconfigure(2, weight=0)
    outfit_frame.grid_columnconfigure(3, weight=1)

    outfit_selected_vars: Dict[str, tk.IntVar] = {}
    outfit_mode_vars: Dict[str, tk.StringVar] = {}
    outfit_prompt_entries: Dict[str, tk.Entry] = {}

    hint_text = (
        "Prompt hints:\n"
        "  - Describe the full outfit from the mid-thigh up.\n"
        "  - DO NOT mention shoes, boots, socks, or anything on the feet.\n"
        "  - Avoid describing anything below the mid-thigh, or Gemini may draw too low."
    )
    tk.Label(
        outfit_frame,
        text=hint_text,
        bg=BG_COLOR,
        justify="left",
        anchor="w",
        wraplength=wrap_len // 2,
    ).grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 6))

    row_idx = 1

    for key in ALL_OUTFIT_KEYS:
        # Whether this outfit is selected to be generated at all
        sel_var = tk.IntVar(value=1 if key in OUTFIT_KEYS else 0)
        outfit_selected_vars[key] = sel_var

        # One mode variable per outfit: "random", "custom", or "standard_uniform"
        mode_var = tk.StringVar(value="random")
        outfit_mode_vars[key] = mode_var

        row_frame = tk.Frame(outfit_frame, bg=BG_COLOR)
        row_frame.grid(row=row_idx, column=0, columnspan=4, sticky="we", pady=2)
        row_idx += 1
        row_frame.grid_columnconfigure(0, weight=0)
        row_frame.grid_columnconfigure(1, weight=0)
        row_frame.grid_columnconfigure(2, weight=0)
        row_frame.grid_columnconfigure(3, weight=1)

        # Checkbox to toggle this outfit on/off
        chk = tk.Checkbutton(
            row_frame,
            text=key.capitalize(),
            variable=sel_var,
            bg=BG_COLOR,
            anchor="w",
        )
        chk.grid(row=0, column=0, padx=(6, 4), sticky="w")

        # Radio: Random
        rb_random = tk.Radiobutton(
            row_frame,
            text="Random",
            variable=mode_var,
            value="random",
            bg=BG_COLOR,
            anchor="w",
        )
        rb_random.grid(row=0, column=1, padx=(0, 4), sticky="w")

        # Radio: Custom
        rb_custom = tk.Radiobutton(
            row_frame,
            text="Custom",
            variable=mode_var,
            value="custom",
            bg=BG_COLOR,
            anchor="w",
        )
        rb_custom.grid(row=0, column=2, padx=(0, 4), sticky="w")

        # Optional radio: Standard school uniform (only for uniform, and only
        # when archetype is eligible)
        rb_standard = None
        if key == "uniform" and uniform_eligible:
            rb_standard = tk.Radiobutton(
                row_frame,
                text="Standard school uniform",
                variable=mode_var,
                value="standard_uniform",
                bg=BG_COLOR,
                anchor="w",
            )
            rb_standard.grid(row=0, column=3, padx=(0, 4), sticky="w")

        # Custom prompt entry (only enabled when this outfit is selected
        # and mode is "custom")
        entry = tk.Entry(row_frame, width=60)
        entry.grid(row=1, column=0, columnspan=4, padx=(24, 6), pady=(1, 2), sticky="we")
        outfit_prompt_entries[key] = entry

        def make_update_fn(
            _sel_var=sel_var,
            _mode_var=mode_var,
            _entry=entry,
            _rb_random=rb_random,
            _rb_custom=rb_custom,
            _rb_standard=rb_standard,
        ):
            """
            Enable/disable controls based on:
              - whether the outfit is selected at all
              - which mode is chosen (random/custom/standard_uniform)
            """
            def _update(*_args):
                if _sel_var.get() == 0:
                    # Outfit is not selected: everything is disabled
                    _rb_random.config(state=tk.DISABLED)
                    _rb_custom.config(state=tk.DISABLED)
                    if _rb_standard is not None:
                        _rb_standard.config(state=tk.DISABLED)
                    _entry.config(state=tk.DISABLED)
                    return

                # Outfit selected: radios enabled
                _rb_random.config(state=tk.NORMAL)
                _rb_custom.config(state=tk.NORMAL)
                if _rb_standard is not None:
                    _rb_standard.config(state=tk.NORMAL)

                # Entry only enabled in custom mode
                if _mode_var.get() == "custom":
                    _entry.config(state=tk.NORMAL)
                else:
                    _entry.config(state=tk.DISABLED)

            return _update

        updater = make_update_fn()
        sel_var.trace_add("write", updater)
        mode_var.trace_add("write", updater)
        updater()

    # =================================================================
    # RIGHT COLUMN: Expressions (with vertical scrollbar)
    # =================================================================
    expr_frame = tk.LabelFrame(
        body_frame,
        text="Expressions (neutral is always included):",
        bg=BG_COLOR,
    )
    expr_frame.grid(row=0, column=1, padx=5, pady=4, sticky="nsew")
    expr_frame.grid_rowconfigure(0, weight=1)
    expr_frame.grid_columnconfigure(0, weight=1)

    # Canvas + scrollbar so a long list of expressions can be scrolled
    expr_canvas = tk.Canvas(
        expr_frame,
        bg=BG_COLOR,
        highlightthickness=0,
    )
    expr_canvas.grid(row=0, column=0, sticky="nsew")

    expr_scrollbar = tk.Scrollbar(
        expr_frame,
        orient=tk.VERTICAL,
        command=expr_canvas.yview,
    )
    expr_scrollbar.grid(row=0, column=1, sticky="ns")

    expr_canvas.configure(yscrollcommand=expr_scrollbar.set)

    # Inner frame that actually holds the labels and checkboxes
    expr_inner = tk.Frame(expr_canvas, bg=BG_COLOR)
    expr_canvas.create_window((0, 0), window=expr_inner, anchor="nw")

    def _update_expr_scrollregion(_event=None) -> None:
        """
        Update the scrollable region whenever the inner frame changes size.
        This keeps the scrollbar in sync with the content height.
        """
        expr_inner.update_idletasks()
        bbox = expr_canvas.bbox("all")
        if bbox:
            expr_canvas.configure(scrollregion=bbox)

    expr_inner.bind("<Configure>", _update_expr_scrollregion)

    # Now build the actual expression controls inside expr_inner
    expr_vars: Dict[str, tk.IntVar] = {}
    for key, desc in EXPRESSIONS_SEQUENCE:
        if key == "0":
            # Neutral expression is always generated; show as a label only
            tk.Label(
                expr_inner,
                text=f"0 – {desc} (always generated)",
                bg=BG_COLOR,
                anchor="w",
                justify="left",
                wraplength=wrap_len // 2,
            ).pack(anchor="w", padx=6, pady=2)
            continue

        var = tk.IntVar(value=1)
        chk_expr = tk.Checkbutton(
            expr_inner,
            text=f"{key} – {desc}",
            variable=var,
            bg=BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=wrap_len // 2,
        )
        chk_expr.pack(anchor="w", padx=6, pady=2)
        expr_vars[key] = var

    decision = {
        "ok": False,
        "outfits": [],
        "expr_seq": EXPRESSIONS_SEQUENCE,
        "config": {},
    }

    def on_ok():
        selected_outfits: List[str] = []
        cfg: Dict[str, Dict[str, Optional[str]]] = {}

        for key in ALL_OUTFIT_KEYS:
            if outfit_selected_vars[key].get() == 1:
                selected_outfits.append(key)
                mode = outfit_mode_vars[key].get()

                use_random = False
                custom_prompt_val: Optional[str] = None
                use_standard_uniform = False

                if key == "uniform" and uniform_eligible and mode == "standard_uniform":
                    # Standard uniform path: we ignore prompts and just mark it
                    use_standard_uniform = True
                    use_random = True
                elif mode == "random":
                    use_random = True
                elif mode == "custom":
                    txt_val = outfit_prompt_entries[key].get().strip()
                    if not txt_val:
                        messagebox.showerror(
                            "Missing custom prompt",
                            f"Please enter a custom prompt for {key.capitalize()}, "
                            f"or switch it back to Random, or uncheck it.",
                        )
                        return
                    custom_prompt_val = txt_val
                else:
                    # Should not happen, but default to random
                    use_random = True

                cfg[key] = {
                    "use_random": use_random,
                    "custom_prompt": custom_prompt_val,
                    "use_standard_uniform": use_standard_uniform,
                }

        # Build the expression sequence (always keep neutral '0')
        new_seq: List[Tuple[str, str]] = []
        for k, desc in EXPRESSIONS_SEQUENCE:
            if k == "0":
                new_seq.append((k, desc))
                break

        for k, desc in EXPRESSIONS_SEQUENCE:
            if k == "0":
                continue
            if expr_vars.get(k, tk.IntVar(value=0)).get() == 1:
                new_seq.append((k, desc))

        decision["ok"] = True
        decision["outfits"] = selected_outfits
        decision["expr_seq"] = new_seq
        decision["config"] = cfg
        root.destroy()

    def on_cancel():
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(6, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(
        side=tk.LEFT, padx=10
    )

    center_and_clamp(root)
    root.mainloop()

    if not decision["ok"]:
        sys.exit(0)

    return (
        decision["outfits"],
        decision["expr_seq"],
        decision["config"],
    )


# =============================================================================
# Cropping and Measurement Dialogs
# =============================================================================

def prompt_for_crop(
    img: Image.Image,
    instruction_text: str,
    previous_crops: list,
) -> Tuple[Optional[int], list]:
    """
    Tk UI that shows the image and lets the user click a horizontal crop line.

    Returns (y_cut, updated_previous_crops).

    If the user clicks at or below the current bottom of the image, we interpret
    that as "do not crop".
    """
    result = {"y": None}
    used_gallery = list(previous_crops)

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Thigh Crop Selection")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = max(200, int(sw * 0.8))

    tk.Label(
        root,
        text=instruction_text,
        font=INSTRUCTION_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).pack(pady=(10, 6))

    original_w, original_h = img.size
    disp_w, disp_h = compute_display_size(
        sw, sh, original_w, original_h,
        max_w_ratio=0.60, max_h_ratio=0.60
    )
    disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(disp_img)

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, bg="black", highlightthickness=0)
    canvas.pack(pady=6)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img  # type: ignore

    guide_line_id = None

    def draw_line(y):
        nonlocal guide_line_id
        y = max(0, min(int(y), disp_h))
        if guide_line_id is None:
            guide_line_id = canvas.create_line(0, y, disp_w, y, fill=LINE_COLOR, width=3)
        else:
            canvas.coords(guide_line_id, 0, y, disp_w, y)

    def on_motion(e):
        draw_line(e.y)

    def on_click(e):
        disp_y = max(0, min(e.y, disp_h))
        if guide_line_id is not None:
            canvas.coords(guide_line_id, 0, disp_y, disp_w, disp_y)
        real_y = int((disp_y / disp_h) * original_h)
        result["y"] = real_y
        root.destroy()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_click)

    center_and_clamp(root)
    root.mainloop()

    return result["y"], used_gallery


def prompt_for_eye_and_hair(image_path: Path) -> Tuple[float, str]:
    """
    Tk UI to choose:
      - eye line (click once, height ratio)
      - hair color (click once, used as name_color)
    """
    result = {"eye_line": None, "name_color": None}
    state = {"step": 1}

    img = Image.open(image_path).convert("RGBA")
    original_w, original_h = img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Eye Line and Name Color")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    disp_w, disp_h = compute_display_size(
        sw, sh, original_w, original_h,
        max_w_ratio=0.90, max_h_ratio=0.80
    )
    scale_x, scale_y = original_w / max(1, disp_w), original_h / max(1, disp_h)

    wrap_len = wraplength_for(int(sw * 0.9))
    title = tk.Label(
        root,
        text="Step 1: Click to mark the eye line (relative head height).",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    )
    title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    cwrap = tk.Frame(root, bg=BG_COLOR, width=disp_w, height=disp_h)
    cwrap.grid(row=1, column=0, padx=10, pady=4, sticky="n")
    cwrap.grid_propagate(False)

    disp_img = img.resize((disp_w, disp_h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(disp_img)
    cvs = tk.Canvas(cwrap, width=disp_w, height=disp_h, highlightthickness=0, bg="black")
    cvs.create_image(0, 0, anchor="nw", image=tk_img)
    cvs.image = tk_img  # type: ignore
    cvs.place(relx=0.5, rely=0.0, anchor="n")

    guide_line_id = None
    reticle_h_id = None
    reticle_v_id = None

    def draw_eyeline(y_disp: int):
        nonlocal guide_line_id
        y_disp = max(0, min(int(y_disp), disp_h))
        if guide_line_id is None:
            guide_line_id = cvs.create_line(0, y_disp, disp_w, y_disp, fill=LINE_COLOR, width=3)
        else:
            cvs.coords(guide_line_id, 0, y_disp, disp_w, y_disp)

    def clear_eyeline():
        nonlocal guide_line_id
        if guide_line_id is not None:
            cvs.delete(guide_line_id)
            guide_line_id = None

    def draw_reticle(x_disp: int, y_disp: int, arm: int = 16):
        nonlocal reticle_h_id, reticle_v_id
        x_disp = max(0, min(int(x_disp), disp_w))
        y_disp = max(0, min(int(y_disp), disp_h))
        if reticle_h_id is None:
            reticle_h_id = cvs.create_line(
                x_disp - arm, y_disp, x_disp + arm, y_disp, fill=LINE_COLOR, width=2
            )
            reticle_v_id = cvs.create_line(
                x_disp, y_disp - arm, x_disp, y_disp + arm, fill=LINE_COLOR, width=2
            )
        else:
            cvs.coords(reticle_h_id, x_disp - arm, y_disp, x_disp + arm, y_disp)
            cvs.coords(reticle_v_id, x_disp, y_disp - arm, x_disp, y_disp + arm)

    def clear_reticle():
        nonlocal reticle_h_id, reticle_v_id
        if reticle_h_id is not None:
            cvs.delete(reticle_h_id)
            cvs.delete(reticle_v_id)
            reticle_h_id = reticle_v_id = None

    def on_motion(e):
        if state["step"] == 1:
            draw_eyeline(e.y)
        elif state["step"] == 2:
            draw_reticle(e.x, e.y)

    def on_click(e):
        nonlocal wrap_len
        if state["step"] == 1:
            real_y = e.y * scale_y
            result["eye_line"] = real_y / original_h
            clear_eyeline()
            title.config(
                text="Eye line recorded.\nStep 2: Click on the hair color to use as name color.",
                wraplength=wrap_len,
            )
            state["step"] = 2
            draw_reticle(e.x, e.y)
        elif state["step"] == 2:
            rx = min(max(int(e.x * scale_x), 0), original_w - 1)
            ry = min(max(int(e.y * scale_y), 0), original_h - 1)
            px = img.getpixel((rx, ry))
            if len(px) == 4 and px[3] < 10:
                color = "#915f40"
            else:
                color = f"#{px[0]:02x}{px[1]:02x}{px[2]:02x}"
            result["name_color"] = color
            clear_reticle()
            root.destroy()

    cvs.bind("<Motion>", on_motion)
    cvs.bind("<Button-1>", on_click)
    draw_eyeline(disp_h // 2)

    center_and_clamp(root)
    root.mainloop()

    if result["eye_line"] is None or result["name_color"] is None:
        sys.exit(0)

    print(f"[INFO] Eye line: {result['eye_line']:.3f}")
    print(f"[INFO] Name color: {result['name_color']}")
    return float(result["eye_line"]), str(result["name_color"])


def prompt_for_scale(
    image_path: Path,
    user_eye_line_ratio: Optional[float] = None
) -> float:
    """
    Side-by-side scaling UI vs reference_sprites, returns chosen scale.
    """
    if not REF_SPRITES_DIR.is_dir():
        print(f"[ERROR] No 'reference_sprites' folder found at: {REF_SPRITES_DIR}")
        sys.exit(1)

    # Load reference sprites
    refs = {}
    for fn in os.listdir(REF_SPRITES_DIR):
        if not fn.lower().endswith(".png"):
            continue
        name = os.path.splitext(fn)[0]
        img_path = REF_SPRITES_DIR / fn
        yml_path = REF_SPRITES_DIR / (name + ".yml")

        ref_scale = 1.0
        if yml_path.exists():
            try:
                with yml_path.open("r", encoding="utf-8") as f:
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
        print("[ERROR] No usable reference sprites found.")
        sys.exit(1)

    names = sorted(refs.keys())
    user_img = Image.open(image_path).convert("RGBA")
    user_w, user_h = user_img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Adjust Scale vs Reference")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    CANV_H = max(int(sh * 0.52), 360)
    CANV_W = max(int((sw - 3 * WINDOW_MARGIN) // 2), 360)

    wrap_len = wraplength_for(int(sw * 0.9))
    tk.Label(
        root,
        text="1) Choose a reference (left).  2) Adjust your scale (right).  3) Click Done.",
        font=INSTRUCTION_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 6), sticky="we")

    sel = tk.StringVar(value=names[0])
    tk.OptionMenu(root, sel, *names).grid(row=1, column=0, columnspan=2, pady=(0, 6))

    ref_canvas = tk.Canvas(root, width=CANV_W, height=CANV_H, bg="black", highlightthickness=0)
    usr_canvas = tk.Canvas(root, width=CANV_W, height=CANV_H, bg="black", highlightthickness=0)
    ref_canvas.grid(row=2, column=0, padx=(10, 5), pady=6, sticky="n")
    usr_canvas.grid(row=2, column=1, padx=(5, 10), pady=6, sticky="n")

    scale_val = tk.DoubleVar(value=1.0)
    slider = tk.Scale(
        root,
        from_=0.1,
        to=2.5,
        resolution=0.01,
        orient=tk.HORIZONTAL,
        label="Adjust Your Character's Scale (in-game)",
        variable=scale_val,
        length=int(sw * 0.8),
        tickinterval=0.05,
    )
    slider.grid(row=3, column=0, columnspan=2, padx=10, pady=(4, 8), sticky="we")

    _img_refs = {"ref": None, "usr": None}

    def redraw(*_):
        ref_canvas.delete("all")
        usr_canvas.delete("all")

        r_meta = refs[sel.get()]
        rimg = r_meta["image"]
        r_scale = r_meta["scale"]
        r_engine_w = rimg.width * r_scale
        r_engine_h = rimg.height * r_scale

        u_scale = float(scale_val.get())
        u_engine_w = user_w * u_scale
        u_engine_h = user_h * u_scale

        max_w = max(r_engine_w, u_engine_w)
        max_h = max(r_engine_h, u_engine_h)
        view_scale = min(CANV_W / max_w, CANV_H / max_h, 1.0)

        r_disp_w = max(1, int(r_engine_w * view_scale))
        r_disp_h = max(1, int(r_engine_h * view_scale))
        u_disp_w = max(1, int(u_engine_w * view_scale))
        u_disp_h = max(1, int(u_engine_h * view_scale))

        r_resized = rimg.resize((r_disp_w, r_disp_h), Image.LANCZOS)
        _img_refs["ref"] = ImageTk.PhotoImage(r_resized)
        ref_canvas.create_image(CANV_W // 2, CANV_H, anchor="s", image=_img_refs["ref"])

        u_resized = user_img.resize((u_disp_w, u_disp_h), Image.LANCZOS)
        _img_refs["usr"] = ImageTk.PhotoImage(u_resized)
        usr_canvas.create_image(CANV_W // 2, CANV_H, anchor="s", image=_img_refs["usr"])

        if isinstance(user_eye_line_ratio, (int, float)) and 0.0 <= user_eye_line_ratio <= 1.0:
            img_top = CANV_H - u_disp_h
            y_inside = int(u_disp_h * float(user_eye_line_ratio))
            y_canvas = img_top + y_inside
            usr_canvas.create_line(0, y_canvas, CANV_W, y_canvas, fill=LINE_COLOR, width=2)

    slider.bind("<ButtonRelease-1>", lambda e: redraw())
    slider.bind("<KeyRelease>", lambda e: redraw())
    sel.trace_add("write", lambda *_: redraw())

    redraw()

    def done():
        root.destroy()

    def cancel():
        try:
            root.destroy()
        except Exception:
            pass
        sys.exit(0)

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=4, column=0, columnspan=2, pady=(6, 10))
    tk.Button(btns, text="Done - Use This Scale", command=done).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", command=cancel).pack(side=tk.LEFT, padx=10)

    center_and_clamp(root)
    root.mainloop()

    chosen = float(scale_val.get())
    print(f"[INFO] User-picked scale: {chosen:.3f}")
    return chosen
