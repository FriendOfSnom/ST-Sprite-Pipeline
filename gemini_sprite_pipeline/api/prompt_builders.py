"""
Prompt builders for Gemini API requests.

All prompt generation logic for sprite creation, outfit changes, and expressions.
Prompts are preserved exactly as originally written per user request.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

from gsp_constants import OUTFIT_CSV_PATH, GENDER_ARCHETYPES


# =============================================================================
# Archetype and Gender Utilities
# =============================================================================

def archetype_to_gender_style(archetype_label: str) -> str:
    """
    Given an archetype label, return gender style code 'f' or 'm'.

    Args:
        archetype_label: Archetype name (e.g., "young woman", "adult man").

    Returns:
        'f' for female archetypes, 'm' for male archetypes, 'f' as default.
    """
    for label, gender in GENDER_ARCHETYPES:
        if label == archetype_label:
            return gender
    return "f"  # Default to female


# =============================================================================
# Core Prompt Builders (DO NOT MODIFY - per user request)
# =============================================================================

def build_initial_pose_prompt(gender_style: str) -> str:
    """
    Prompt to normalize the original sprite (mid-thigh, magenta background).

    NOTE: Prompt wording preserved exactly as-is per user request.

    Args:
        gender_style: 'f' or 'm' (currently unused but kept for signature consistency).

    Returns:
        Prompt string for initial pose normalization.
    """
    return (
        "Edit the image of the character, to give them a pure, flat, magenta (#FF00FF) background behind them."
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "Make sure that the character, outfit, or hair end up with none of the magenta background color on them. "
        "Make sure the head, arms, hair, hands, and clothes are all kept within the image."
        "Keep the crop the same from the mid-thigh on up."
        "Dont change the art style either, just edit the background that the character is on, to be that magenta color."
    )


def build_expression_prompt(expression_desc: str) -> str:
    """
    Prompt to change facial expression, keeping style and framing.

    NOTE: Prompt wording preserved exactly as-is per user request.

    Args:
        expression_desc: Description of the desired expression.

    Returns:
        Prompt string for expression generation.
    """
    return (
        "Edit the inputed visual novel sprite in the same art style. "
        f"Change the facial expression to match this description: {expression_desc}. "
        "Keep the hair volume, hair outlines, and the hair style all the exact same. "
        "Do not change the hairstyle, crop from the mid-thigh up, image size, lighting, or background. "
        "Change the pose of the character, based upon the expression we are giving them. "
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "Do not have the head, arms, hair, or hands extending outside the frame."
        "Do not crop off the head, and don't change the size or proportions of the character."
    )


def build_outfit_prompt(base_outfit_desc: str, gender_style: str) -> str:
    """
    Prompt to change clothing to base_outfit_desc on the given pose.

    NOTE: Prompt wording preserved exactly as-is per user request.

    Args:
        base_outfit_desc: Description of the outfit to generate.
        gender_style: 'f' or 'm' for gender-appropriate wording.

    Returns:
        Prompt string for outfit generation.
    """
    gender_clause = "girl" if gender_style == "f" else "boy"
    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Please change the clothing, pose, hair style, and outfit to match this description: {base_outfit_desc}. "
        "Do not change the body proportions, hair length, crop from the mid-thigh up, or image size. "
        "Do not change how long the character's hair is, but you can style the hair to fit the new outfit."
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "have none of the background color on them. "
        "Do not change the body, chest, and hip proportions to be different from the original."
        "Do not crop off the head, and don't change the size of the character."
    )


def build_standard_school_uniform_prompt(
    archetype_label: str,
    gender_style: str,
) -> str:
    """
    Build a standardized school-uniform prompt matching the rest of the outfit prompts.

    Uses archetype label and gender style to describe the character,
    then describes the school uniform in text. The cropped uniform reference
    image is used as visual backup.

    NOTE: Prompt wording preserved exactly as-is per user request.

    Args:
        archetype_label: Character archetype (e.g., "young woman").
        gender_style: 'f' or 'm'.

    Returns:
        Prompt string for standard school uniform generation.
    """
    # Base description that applies to both variants
    base_intro = (
        f"Edit the inputed {archetype_label} visual novel sprite, to give them the outfit we have also attached."
        "For redundency, I am going to also describe the outfit below, but using the reference image is your first priority when it comes to what this outfit needs to look like."
    )

    if gender_style == "f":
        # Female student uniform: blazer + bow + pleated skirt description
        uniform_desc = (
            "She should be wearing: A navy blue tailored sleeveless blazer hybrid, tightly fitted to the torso. The blazer has gold piping along all the outer edges. The front features a double-breasted design with two rows of two gold buttons. The vest dips into a sharp angled hem near the waist, creating a stylish contour. Underneath it is a white short-sleeved dress shirt. The sleeve has a school crest patch on the upper arm: gold/yellow with an emblem inside. Her arms are bare below the sleeves. She should have on a bright red necktie with white stripes near the bottom. A short, red, plaid, pleated skirt finishes out the outfit. No ribbons."
        )
    else:
        # Male student uniform: blazer + tie + slacks description
        uniform_desc = (
            "He should be wearing: A white short-sleeved dress shirt. The sleeve has a school crest patch on the upper arm: gold/yellow with an emblem inside. He should have on a  bright red necktie with white stripes near the bottom. A pair of dark-colored slacks with a belt, which the white shirt tucks into, completes the look."
        )

    # Shared constraints and ST-format requirements
    tail = (
        "Again, copy over the outfit from the image sent. The description above is just to help with consistency."
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "Do not change the art style, size, proportions, or hair length of the character, and keep their arms, hands, and hair all inside the image."
        "Thats all to say, the goal is to copy over the outfit from the reference, to the character we are editing, to replace their current outfit."
    )

    return base_intro + uniform_desc + tail


def build_prompt_for_idea(concept: str, archetype_label: str, gender_style: str) -> str:
    """
    Build text prompt used when generating a new character from a concept.

    NOTE: Prompt wording preserved exactly as-is per user request.

    Args:
        concept: User's character concept description.
        archetype_label: Character archetype.
        gender_style: 'f' or 'm'.

    Returns:
        Prompt string for new character generation from text.
    """
    gender_word = "girl" if gender_style == "f" else "boy"
    return (
        f"Create concept art for an original {archetype_label} {gender_word} character "
        f"for a Japanese-style visual novel. The character idea is:\n\n"
        f"{concept}\n\n"
        "Match the art style and rendering of the reference character images exactly so the new character looks "
        "like they come from the same artist as the others. The character should be cropped from the "
        "mid-thigh up, facing mostly toward the viewer in a friendly, neutral pose that "
        "would work as a base sprite. They should not be holding anything in their hands. "
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair "
        "have none of the background color on them. "
        "Use clean line art and vibrant but not overly saturated colors that match the reference style."
    )


# =============================================================================
# Outfit Prompt Loading and Configuration
# =============================================================================

def load_outfit_prompts(csv_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load outfit prompts from CSV: archetype, outfit_key, prompt.

    CSV format:
        archetype,outfit_key,prompt
        young woman,formal,"a dressy outfit..."

    Returns:
        {archetype: {outfit_key: [prompt, ...]}, ...}
    """
    database: Dict[str, Dict[str, List[str]]] = {}

    if not csv_path.is_file():
        print(f"[WARN] Outfit CSV not found at {csv_path}. Using generic prompts.")
        return database

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                archetype = (row.get("archetype") or "").strip()
                outfit_key = (row.get("outfit_key") or "").strip()
                prompt = (row.get("prompt") or "").strip()

                if not archetype or not outfit_key or not prompt:
                    continue

                database.setdefault(archetype, {}).setdefault(outfit_key, []).append(prompt)
    except Exception as e:
        print(f"[WARN] Failed to read outfit CSV {csv_path}: {e}")

    return database


def build_simple_outfit_description(outfit_key: str, gender_style: str) -> str:
    """
    Fallback generic outfit description if no CSV prompt is available.

    Args:
        outfit_key: Outfit type (formal, casual, uniform, athletic, swimsuit).
        gender_style: 'f' or 'm'.

    Returns:
        Generic outfit description string.
    """
    gender_word = "girl" if gender_style == "f" else "boy"

    if outfit_key == "formal":
        return (
            f"a slightly dressy outfit this {gender_word} would wear to a school dance "
            "or evening party, grounded and modern"
        )
    if outfit_key == "casual":
        return (
            f"a comfy everyday casual outfit this {gender_word} would wear to school "
            "or to hang out with friends"
        )
    if outfit_key == "uniform":
        return (
            f"a school or work uniform that fits this {gender_word}'s age and vibe, "
            "with clean lines and a coordinated look"
        )
    if outfit_key == "athletic":
        return (
            f"a sporty and practical athletic outfit this {gender_word} would wear "
            "for PE, training, or a casual game"
        )
    if outfit_key == "swimsuit":
        return (
            f"a modest but cute swimsuit this {gender_word} would wear to swim practice "
            "or a beach episode, nothing too revealing"
        )

    return f"a simple outfit that fits this {gender_word}'s personality"


def build_outfit_prompts_with_config(
    archetype_label: str,
    gender_style: str,
    selected_outfit_keys: List[str],
    outfit_database: Dict[str, Dict[str, List[str]]],
    outfit_prompt_config: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, str]:
    """
    Build one prompt per selected outfit_key, honoring per-outfit settings.

    For each key:
      - use_random=True: pick random CSV prompt if available; else fallback.
      - use_random=False: use custom_prompt (fallback if empty for safety).

    Args:
        archetype_label: Character archetype.
        gender_style: 'f' or 'm'.
        selected_outfit_keys: List of outfit keys to generate.
        outfit_database: Loaded CSV outfit prompts.
        outfit_prompt_config: Per-outfit configuration settings.

    Returns:
        {outfit_key: prompt_text}
    """
    prompts: Dict[str, str] = {}
    archetype_pool = outfit_database.get(archetype_label, {})

    for key in selected_outfit_keys:
        config = outfit_prompt_config.get(key, {})
        use_random = bool(config.get("use_random", True))
        custom_prompt = config.get("custom_prompt")

        if use_random:
            # Try to get random CSV prompt
            candidates = archetype_pool.get(key)
            if candidates:
                prompts[key] = random.choice(candidates)
            else:
                # Fallback to generic description
                prompts[key] = build_simple_outfit_description(key, gender_style)
        else:
            # Use custom prompt, or fallback if empty
            prompts[key] = custom_prompt or build_simple_outfit_description(key, gender_style)

    return prompts
