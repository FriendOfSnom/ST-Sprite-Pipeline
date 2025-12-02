"""
API module for Gemini interactions.

Handles all communication with Google Gemini API including:
- Authentication and configuration
- Image generation and editing
- Retry logic and error handling
- Prompt building
"""

from .gemini_client import (
    get_api_key,
    call_gemini_image_edit,
    call_gemini_text_or_refs,
    load_config,
    save_config,
    interactive_api_key_setup,
    strip_background,
)

from .background_removal_legacy import strip_background_legacy

from .prompt_builders import (
    build_initial_pose_prompt,
    build_expression_prompt,
    build_outfit_prompt,
    build_standard_school_uniform_prompt,
    build_prompt_for_idea,
    archetype_to_gender_style,
    load_outfit_prompts,
    build_outfit_prompts_with_config,
    build_simple_outfit_description,
)

__all__ = [
    # Client functions
    "get_api_key",
    "call_gemini_image_edit",
    "call_gemini_text_or_refs",
    "load_config",
    "save_config",
    "interactive_api_key_setup",
    # Background removal
    "strip_background",
    "strip_background_legacy",
    # Prompt builders
    "build_initial_pose_prompt",
    "build_expression_prompt",
    "build_outfit_prompt",
    "build_standard_school_uniform_prompt",
    "build_prompt_for_idea",
    "archetype_to_gender_style",
    "load_outfit_prompts",
    "build_outfit_prompts_with_config",
    "build_simple_outfit_description",
]
