"""
UI module for Tkinter-based user interfaces.

Provides dialogs and windows for character setup, option selection,
and image review/regeneration workflows.
"""

from .tk_common import (
    compute_display_size,
    center_and_clamp,
    wraplength_for,
)

from .dialogs import (
    prompt_voice_archetype_and_name,
    prompt_source_mode,
    prompt_character_idea_and_archetype,
    prompt_outfits_and_expressions,
    prompt_for_crop,
    prompt_for_eye_and_hair,
    prompt_for_scale,
)

from .review_windows import (
    review_images_for_step,
    review_initial_base_pose,
)

__all__ = [
    # Common utilities
    "compute_display_size",
    "center_and_clamp",
    "wraplength_for",
    # Dialogs
    "prompt_voice_archetype_and_name",
    "prompt_source_mode",
    "prompt_character_idea_and_archetype",
    "prompt_outfits_and_expressions",
    "prompt_for_crop",
    "prompt_for_eye_and_hair",
    "prompt_for_scale",
    # Review windows
    "review_images_for_step",
    "review_initial_base_pose",
]
