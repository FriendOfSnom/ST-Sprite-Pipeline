"""
Common Tkinter utilities and layout helpers.

Shared functions for window sizing, positioning, and UI calculations.
"""

import tkinter as tk
from typing import Tuple

from gsp_constants import (
    BG_COLOR,
    TITLE_FONT,
    INSTRUCTION_FONT,
    LINE_COLOR,
    WINDOW_MARGIN,
    WRAP_PADDING,
)


def compute_display_size(
    screen_w: int,
    screen_h: int,
    img_w: int,
    img_h: int,
    *,
    max_w_ratio: float = 0.90,
    max_h_ratio: float = 0.55,
) -> Tuple[int, int]:
    """
    Calculate display size for an image that fits within given screen ratios.

    Maintains aspect ratio while ensuring the image fits on screen with
    room for text and controls.

    Args:
        screen_w: Screen width in pixels.
        screen_h: Screen height in pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        max_w_ratio: Maximum width as fraction of screen width.
        max_h_ratio: Maximum height as fraction of screen height.

    Returns:
        (display_width, display_height) in pixels.
    """
    max_w = int(screen_w * max_w_ratio) - 2 * WINDOW_MARGIN
    max_h = int(screen_h * max_h_ratio) - 2 * WINDOW_MARGIN
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))


def center_and_clamp(root: tk.Tk) -> None:
    """
    Clamp window to screen bounds and center horizontally near top.

    Ensures window is fully visible with appropriate margins, positioned
    near the top of the screen for better visibility.

    Args:
        root: Tkinter root window to position.
    """
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    w = min(req_w + WINDOW_MARGIN, sw - 2 * WINDOW_MARGIN)
    h = min(req_h + WINDOW_MARGIN, sh - 2 * WINDOW_MARGIN)
    x = max((sw - w) // 2, WINDOW_MARGIN)
    y = WINDOW_MARGIN  # Pin near top instead of vertical centering

    root.geometry(f"{w}x{h}+{x}+{y}")


def wraplength_for(width_px: int) -> int:
    """
    Calculate appropriate wraplength for labels given a target width.

    Ensures text wraps properly without extending beyond window bounds.

    Args:
        width_px: Target width in pixels.

    Returns:
        Wraplength value for Tkinter labels.
    """
    return max(200, width_px - WRAP_PADDING)


# Re-export constants for convenience
__all__ = [
    "compute_display_size",
    "center_and_clamp",
    "wraplength_for",
    "BG_COLOR",
    "TITLE_FONT",
    "INSTRUCTION_FONT",
    "LINE_COLOR",
    "WINDOW_MARGIN",
]
