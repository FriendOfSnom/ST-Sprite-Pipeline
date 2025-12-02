"""
Review windows for image approval and regeneration.

Provides UI for reviewing generated images with options to accept,
regenerate, or perform per-item actions.
"""

import sys
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk

from .tk_common import (
    BG_COLOR,
    TITLE_FONT,
    INSTRUCTION_FONT,
    WINDOW_MARGIN,
    center_and_clamp,
    wraplength_for,
)


def review_images_for_step(
    image_infos: List[Tuple[Path, str]],
    title_text: str,
    body_text: str,
    *,
    per_item_buttons: Optional[List[List[Tuple[str, str]]]] = None,
    show_global_regenerate: bool = True,
) -> Dict[str, Optional[object]]:
    """
    Show a scrollable strip of images and return a decision dictionary.

    Creates a horizontal scrolling window displaying multiple images with
    captions and optional per-item action buttons.

    Args:
        image_infos: List of (image_path, caption) pairs.
        title_text: Window title text.
        body_text: Instructional text.
        per_item_buttons: Optional list (same length as image_infos) where each entry
            is a list of (button_label, action_code) tuples. For each image card,
            those buttons are rendered under the caption. Pressing one closes the
            window and returns a decision with:
                {"choice": "per_item", "index": idx, "action": action_code}
        show_global_regenerate: If True, show a global "Regenerate" button at the
            bottom that behaves like the old "regenerate all" behavior and returns:
                {"choice": "regenerate_all", ...}

    Returns:
        A dict with at least:
            choice: "accept", "cancel", "regenerate_all", or "per_item"
            index: index of the image (only for per_item)
            action: action_code string (only for per_item)
    """
    decision: Dict[str, Optional[object]] = {
        "choice": "cancel",
        "index": None,
        "action": None,
    }

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Review Step")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    # Title label
    tk.Label(
        root,
        text=title_text,
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg="black",
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 4), sticky="we")

    # Instruction label
    tk.Label(
        root,
        text=body_text,
        font=INSTRUCTION_FONT,
        bg=BG_COLOR,
        fg="black",
        wraplength=wrap_len,
        justify="center",
    ).grid(row=1, column=0, padx=10, pady=(0, 6), sticky="we")

    # Scrollable canvas setup
    canvas_w = int(sw * 0.90) - 2 * WINDOW_MARGIN
    canvas_h = int(sh * 0.60)

    outer = tk.Frame(root, bg=BG_COLOR)
    outer.grid(row=2, column=0, padx=10, pady=6, sticky="nsew")
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    canvas = tk.Canvas(
        outer,
        width=canvas_w,
        height=canvas_h,
        bg="black",
        highlightthickness=0,
    )
    canvas.grid(row=0, column=0, sticky="nsew")

    h_scroll = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)
    h_scroll.grid(row=1, column=0, sticky="we")
    canvas.configure(xscrollcommand=h_scroll.set)

    inner = tk.Frame(canvas, bg=BG_COLOR)
    canvas.create_window((0, 0), window=inner, anchor="nw")

    thumb_refs: List[ImageTk.PhotoImage] = []
    max_thumb_height = min(600, canvas_h - 40)

    # Normalize per_item_buttons length if provided
    if per_item_buttons is not None:
        if len(per_item_buttons) < len(image_infos):
            per_item_buttons = per_item_buttons + [
                [] for _ in range(len(image_infos) - len(per_item_buttons))
            ]
    else:
        per_item_buttons = [[] for _ in image_infos]

    def make_item_handler(idx: int, action_code: str):
        """Create a button handler for per-item actions."""
        def _handler():
            decision["choice"] = "per_item"
            decision["index"] = idx
            decision["action"] = action_code
            root.destroy()
        return _handler

    # Create image cards
    for col_index, (img_path, caption) in enumerate(image_infos):
        try:
            img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Failed to load {img_path}: {e}")
            continue

        w, h = img.size
        scale = min(max_thumb_height / max(1, h), 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = img.resize((tw, th), Image.LANCZOS)
        tki = ImageTk.PhotoImage(thumb)
        thumb_refs.append(tki)

        card = tk.Frame(inner, bg=BG_COLOR)
        card.grid(row=0, column=col_index, padx=10, pady=6)

        # Image display
        tk.Label(card, image=tki, bg=BG_COLOR).pack()

        # Caption
        tk.Label(
            card,
            text=caption,
            font=INSTRUCTION_FONT,
            bg=BG_COLOR,
            fg="black",
            wraplength=tw + 40,
            justify="center",
        ).pack(pady=(2, 2))

        # Optional per-item buttons under each image
        btn_cfgs = per_item_buttons[col_index]
        if btn_cfgs:
            btn_row = tk.Frame(card, bg=BG_COLOR)
            btn_row.pack(pady=(0, 2))
            for label, action_code in btn_cfgs:
                tk.Button(
                    btn_row,
                    text=label,
                    width=20,
                    command=make_item_handler(col_index, action_code),
                ).pack(side=tk.TOP, pady=1)

    def _update_scrollregion(_event=None) -> None:
        """Update scrollable region when content changes."""
        inner.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    inner.bind("<Configure>", _update_scrollregion)
    _update_scrollregion()

    # Action button handlers
    def accept() -> None:
        decision["choice"] = "accept"
        root.destroy()

    def regenerate_all() -> None:
        decision["choice"] = "regenerate_all"
        root.destroy()

    def cancel():
        decision["choice"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    # Bottom button row
    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=3, column=0, pady=(6, 10))

    tk.Button(btns, text="Accept", width=20, command=accept).pack(side=tk.LEFT, padx=10)

    if show_global_regenerate:
        tk.Button(btns, text="Regenerate", width=20, command=regenerate_all).pack(
            side=tk.LEFT, padx=10
        )

    tk.Button(btns, text="Cancel and Exit", width=20, command=cancel).pack(
        side=tk.LEFT, padx=10
    )

    center_and_clamp(root)
    root.mainloop()
    return decision


def review_initial_base_pose(base_pose_path: Path) -> Tuple[str, bool]:
    """
    Review normalized base pose and decide whether to accept, regenerate, or cancel.

    Also allows user to choose whether to treat this base pose as a 'Base' outfit.

    Args:
        base_pose_path: Path to the base pose image to review.

    Returns:
        (choice, use_as_outfit) where:
            choice: "accept", "regenerate", or "cancel"
            use_as_outfit: Whether to keep this as a Base outfit
    """
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Review Normalized Base Pose")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = wraplength_for(int(sw * 0.9))

    # Instructions
    tk.Label(
        root,
        text=(
            "This is the normalized base pose Gemini created for this character.\n\n"
            "You can accept it, regenerate it, or cancel.\n"
            "You can also choose whether to keep this exact look as a 'Base' outfit\n"
            "in addition to any other outfits you generate later."
        ),
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # Image preview
    preview_frame = tk.Frame(root, bg=BG_COLOR)
    preview_frame.grid(row=1, column=0, padx=10, pady=(4, 4))

    img = Image.open(base_pose_path).convert("RGBA")
    max_size = int(min(sw, sh) * 0.4)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    root._base_preview_img = img_tk  # type: ignore[attr-defined]
    tk.Label(preview_frame, image=img_tk, bg=BG_COLOR).pack()

    # Checkbox for using as outfit
    use_as_outfit_var = tk.IntVar(value=1)  # Default: keep as Base

    chk_frame = tk.Frame(root, bg=BG_COLOR)
    chk_frame.grid(row=2, column=0, padx=10, pady=(4, 4), sticky="w")

    tk.Checkbutton(
        chk_frame,
        text="Use this normalized base as a 'Base' outfit",
        variable=use_as_outfit_var,
        bg=BG_COLOR,
        anchor="w",
    ).pack(anchor="w")

    decision = {"choice": "accept", "use_as_outfit": True}

    # Button handlers
    def on_accept():
        decision["choice"] = "accept"
        decision["use_as_outfit"] = bool(use_as_outfit_var.get())
        root.destroy()

    def on_regenerate():
        decision["choice"] = "regenerate"
        decision["use_as_outfit"] = bool(use_as_outfit_var.get())
        root.destroy()

    def on_cancel():
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    # Bottom buttons
    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=3, column=0, pady=(6, 10))

    tk.Button(btns, text="Accept", width=16, command=on_accept).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btns, text="Regenerate", width=16, command=on_regenerate).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(
        side=tk.LEFT, padx=10
    )

    center_and_clamp(root)
    root.mainloop()

    return decision["choice"], decision["use_as_outfit"]
