"""
Processing module for image manipulation and generation workflows.

Handles image utilities, pose processing, expression generation,
and character finalization.
"""

from .image_utils import (
    save_img_webp_or_png,
    save_image_bytes_as_png,
    get_unique_folder_name,
    pick_representative_outfit,
    get_reference_images_for_archetype,
    get_standard_uniform_reference_images,
)

from .pose_processor import (
    generate_initial_pose_once,
    generate_single_outfit,
    generate_outfits_once,
    generate_standard_uniform_outfit,
    flatten_pose_outfits_to_letter_poses,
    write_character_yml,
)

from .expression_generator import (
    generate_expressions_for_single_outfit_once,
    regenerate_single_expression,
    generate_and_review_expressions_for_pose,
    generate_initial_character_from_prompt,
)

from .character_finalizer import (
    finalize_character,
    generate_expression_sheets_for_root,
)

__all__ = [
    # Image utilities
    "save_img_webp_or_png",
    "save_image_bytes_as_png",
    "get_unique_folder_name",
    "pick_representative_outfit",
    "get_reference_images_for_archetype",
    "get_standard_uniform_reference_images",
    # Pose processing
    "generate_initial_pose_once",
    "generate_single_outfit",
    "generate_outfits_once",
    "generate_standard_uniform_outfit",
    "flatten_pose_outfits_to_letter_poses",
    "write_character_yml",
    # Expression generation
    "generate_expressions_for_single_outfit_once",
    "regenerate_single_expression",
    "generate_and_review_expressions_for_pose",
    "generate_initial_character_from_prompt",
    # Character finalization
    "finalize_character",
    "generate_expression_sheets_for_root",
]
