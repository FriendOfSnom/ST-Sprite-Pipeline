#!/usr/bin/env python3
"""
gemini_sprite_pipeline.py

Single-character Student Transfer-style sprite builder using Gemini.

This is the main orchestrator that coordinates the entire sprite generation workflow.
All heavy lifting has been refactored into the gemini_sprite_pipeline package modules.

Flow (per character):
  - Start from either an existing image or a prompt-generated base
  - Pick voice, name, archetype
  - Choose extra outfits + expressions
  - Normalize to pose A (mid-thigh, magenta bg) and review
  - Generate outfits and expressions, with review loops
  - Pick eye line, name color, and scale vs reference
  - Flatten pose/outfit combos into ST-style poses and write character.yml
"""

import argparse
import os
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Dict, List, Optional, Tuple

from PIL import Image

from gsp_constants import OUTFIT_CSV_PATH, EXPRESSIONS_SEQUENCE

# Import from refactored package modules
from gemini_sprite_pipeline.api import (
    get_api_key,
    load_outfit_prompts,
    build_outfit_prompts_with_config,
)
from gemini_sprite_pipeline.ui import (
    prompt_voice_archetype_and_name,
    prompt_source_mode,
    prompt_character_idea_and_archetype,
    prompt_outfits_and_expressions,
    prompt_for_crop,
    review_images_for_step,
    review_initial_base_pose,
)
from gemini_sprite_pipeline.processing import (
    generate_initial_pose_once,
    generate_single_outfit,
    generate_outfits_once,
    generate_and_review_expressions_for_pose,
    generate_initial_character_from_prompt,
    finalize_character,
    generate_expression_sheets_for_root,
    get_unique_folder_name,
)


# =============================================================================
# Main Character Processing Pipeline
# =============================================================================

def process_single_character(
    api_key: str,
    image_path: Path,
    output_root: Path,
    outfit_db: Dict[str, Dict[str, List[str]]],
    game_name: Optional[str] = None,
    preselected: Optional[Dict[str, str]] = None,
) -> None:
    """
    Run the pipeline for a single source image.

    Steps:
      - voice/name/archetype (or preselected for prompt mode)
      - choose outfits+expressions
      - normalize base pose with review
      - generate outfits + expressions
      - finalize (eye line, color, scale, flatten, yaml)

    Args:
        api_key: Gemini API key.
        image_path: Path to source character image.
        output_root: Root directory for output.
        outfit_db: Loaded outfit prompts database.
        game_name: Optional game name for metadata.
        preselected: Optional preselected character info (from prompt mode).
    """
    print(f"\n=== Processing source image: {image_path.name} ===")

    # Get character metadata
    if preselected is not None:
        voice = preselected["voice"]
        display_name = preselected["display_name"]
        archetype_label = preselected["archetype_label"]
        gender_style = preselected["gender_style"]
        print(
            f"[INFO] Using preselected voice/name/archetype for {display_name}: "
            f"voice={voice}, archetype={archetype_label}, gender_style={gender_style}"
        )
    else:
        voice, display_name, archetype_label, gender_style = \
            prompt_voice_archetype_and_name(image_path)

    # Get outfit and expression selections
    (
        selected_outfit_keys,
        expressions_sequence,
        outfit_prompt_config,
    ) = prompt_outfits_and_expressions(archetype_label, gender_style)

    print(f"[INFO] Selected outfits (Base always included): {selected_outfit_keys}")
    print(
        "[INFO] Selected expressions (including neutral): "
        f"{[key for key, _ in expressions_sequence]}"
    )
    print("[INFO] Per-outfit prompt config:")
    for key in selected_outfit_keys:
        cfg = outfit_prompt_config.get(key, {})
        if key == "uniform" and cfg.get("use_standard_uniform"):
            mode_str = "standard_uniform"
        elif cfg.get("use_random", True):
            mode_str = "random"
        else:
            mode_str = "custom"
        print(f"  - {key}: {mode_str}")

    # Create character folder
    char_folder_name = get_unique_folder_name(output_root, display_name)
    char_dir = output_root / char_folder_name
    char_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output character folder: {char_dir}")

    # Prepare pose A directory
    a_dir = char_dir / "a"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_base_stem = a_dir / "base"

    use_base_as_outfit = True

    # Generate and review base pose
    while True:
        a_base_path = generate_initial_pose_once(
            api_key,
            image_path,
            a_base_stem,
            gender_style,
        )

        choice, use_flag = review_initial_base_pose(a_base_path)
        use_base_as_outfit = use_flag

        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # Generate outfits
    print("[INFO] Generating outfits for pose A...")
    outfits_dir = a_dir / "outfits"
    outfits_dir.mkdir(parents=True, exist_ok=True)

    # Build outfit prompts with configuration
    current_outfit_prompts = build_outfit_prompts_with_config(
        archetype_label,
        gender_style,
        selected_outfit_keys,
        outfit_db,
        outfit_prompt_config,
    )

    # Initial generation: make Base (if requested) and all outfits once
    generate_outfits_once(
        api_key,
        a_base_path,
        outfits_dir,
        gender_style,
        current_outfit_prompts,
        outfit_prompt_config,
        archetype_label,
        include_base_outfit=use_base_as_outfit,
    )

    # Review loop: regenerate individual outfits as needed
    while True:
        a_out_paths: List[Path] = []
        per_buttons: List[List[Tuple[str, str]]] = []
        index_to_outfit_key: Dict[int, str] = {}

        # Collect current outfit images from disk
        for p in sorted(outfits_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".png", ".webp"):
                continue

            a_out_paths.append(p)
            stem_lower = p.stem.lower()

            # Try to match this file back to one of the logical outfit keys
            matched_key: Optional[str] = None
            for key in selected_outfit_keys:
                if key.lower() == stem_lower or key.capitalize().lower() == stem_lower:
                    matched_key = key
                    break

            # Decide which per-card buttons to show
            btn_list: List[Tuple[str, str]] = []
            if matched_key is not None:
                cfg = outfit_prompt_config.get(matched_key, {})
                # Every logical outfit can be regenerated with the same prompt
                btn_list.append(("Regenerate same outfit", "same"))
                # "New random outfit" button only for CSV/random system
                if not (matched_key == "uniform" and cfg.get("use_standard_uniform")):
                    btn_list.append(("New random outfit", "new"))
                index_to_outfit_key[len(a_out_paths) - 1] = matched_key

            per_buttons.append(btn_list)

        a_infos = [(p, f"Pose A – {p.name}") for p in a_out_paths]

        decision = review_images_for_step(
            a_infos,
            "Review Outfits for Pose A",
            (
                "Accept these outfits, regenerate individual outfits (random outfits will pick new "
                "CSV prompts when you choose 'New random outfit'; custom outfits and standard "
                "uniforms will keep the same prompt), or cancel."
            ),
            per_item_buttons=per_buttons,
            show_global_regenerate=False,
        )

        choice = decision.get("choice")

        if choice == "accept":
            break
        if choice == "cancel":
            sys.exit(0)

        if choice == "per_item":
            idx = decision.get("index")
            action = decision.get("action")
            if idx is None or action is None:
                continue

            outfit_key = index_to_outfit_key.get(int(idx))
            if not outfit_key:
                continue

            # If "new" random outfit, roll a fresh prompt
            if action == "new":
                new_prompt_dict = build_outfit_prompts_with_config(
                    archetype_label,
                    gender_style,
                    [outfit_key],
                    outfit_db,
                    outfit_prompt_config,
                )
                current_outfit_prompts[outfit_key] = new_prompt_dict[outfit_key]

            # Get description (reuse existing or generate new)
            desc = current_outfit_prompts.get(outfit_key)
            if not desc:
                fallback_prompt_dict = build_outfit_prompts_with_config(
                    archetype_label,
                    gender_style,
                    [outfit_key],
                    outfit_db,
                    outfit_prompt_config,
                )
                desc = fallback_prompt_dict[outfit_key]
                current_outfit_prompts[outfit_key] = desc

            # Regenerate just this one outfit image
            generate_single_outfit(
                api_key,
                a_base_path,
                outfits_dir,
                gender_style,
                outfit_key,
                desc,
                outfit_prompt_config,
                archetype_label,
            )
            continue

        # Safety: support "regenerate_all" if ever re-enabled
        if choice == "regenerate_all":
            current_outfit_prompts = build_outfit_prompts_with_config(
                archetype_label,
                gender_style,
                selected_outfit_keys,
                outfit_db,
                outfit_prompt_config,
            )
            generate_outfits_once(
                api_key,
                a_base_path,
                outfits_dir,
                gender_style,
                current_outfit_prompts,
                outfit_prompt_config,
                archetype_label,
                include_base_outfit=use_base_as_outfit,
            )
            continue

    # Generate expressions for all outfits
    print("[INFO] Generating expressions for pose A (per outfit)...")
    generate_and_review_expressions_for_pose(
        api_key,
        char_dir,
        a_dir,
        "A",
        expressions_sequence=expressions_sequence,
    )

    # Finalize character
    finalize_character(char_dir, display_name, voice, game_name)

    # Generate expression sheets
    generate_expression_sheets_for_root(output_root)


# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

def run_pipeline(output_root: Path, game_name: Optional[str] = None) -> None:
    """
    Run the interactive sprite pipeline for a single character.

    This is the core entry point used both by:
      - the command-line interface in main(), and
      - the external hub script (pipeline_runner.py).

    Args:
        output_root: Root folder where character sprite folders will be created.
        game_name: Optional game name to include in character.yml.
    """
    random.seed(int.from_bytes(os.urandom(16), "big"))
    api_key = get_api_key()

    outfit_db = load_outfit_prompts(OUTFIT_CSV_PATH)
    output_root.mkdir(parents=True, exist_ok=True)

    # Ask how we are creating the character: from an image, or from a text prompt
    mode = prompt_source_mode()

    if mode == "image":
        # Choose the source image via file dialog
        root = tk.Tk()
        root.withdraw()
        initialdir = str(Path.cwd())
        filename = filedialog.askopenfilename(
            title="Choose character source image",
            initialdir=initialdir,
            filetypes=[
                ("Images", "*.png;*.jpg;*.jpeg;*.webp"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("WEBP", "*.webp"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if not filename:
            raise SystemExit("No image selected. Exiting.")

        image_path = Path(filename)
        print(f"[INFO] Selected source image: {image_path}")

        # Optional thigh crop UI before normalization
        try:
            src_img = Image.open(image_path).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Could not open image for cropping ({e}); using original.")
            src_img = None

        if src_img is not None:
            prompt_text = (
                "If this sprite is not already cropped to mid-thigh, click where you "
                "want the lower thigh-level crop line.\n\n"
                "If it *is* already cropped the way you like, just click along the "
                "existing bottom edge of the character."
            )

            y_cut, used_gallery = prompt_for_crop(
                src_img,
                prompt_text,
                previous_crops=[],
            )

            # If the user clicked somewhere above the bottom edge, crop
            if y_cut is not None and 0 < y_cut < src_img.height:
                cropped = src_img.crop((0, 0, src_img.width, y_cut))
                crop_dir = output_root / "_cropped_sources"
                crop_dir.mkdir(parents=True, exist_ok=True)
                cropped_path = crop_dir / f"{image_path.stem}_cropped.png"
                cropped.save(cropped_path, format="PNG")
                print(f"[INFO] Saved thigh-cropped source image to: {cropped_path}")
                image_path = cropped_path
            else:
                print("[INFO] User kept original height; skipping pre-crop.")

        process_single_character(api_key, image_path, output_root, outfit_db, game_name)

    else:
        # Prompt-generated character path
        concept, arch_label, voice, display_name, gender_style = (
            prompt_character_idea_and_archetype()
        )

        # Generate and review the initial prompt-based sprite
        while True:
            src_path = generate_initial_character_from_prompt(
                api_key,
                concept,
                arch_label,
                output_root,
            )

            decision = review_images_for_step(
                [(src_path, f"Prompt-generated base: {src_path.name}")],
                "Review Prompt-Generated Base Sprite",
                "Accept this as the starting sprite, regenerate it, or cancel.",
            )

            choice = decision.get("choice")

            if choice == "accept":
                break
            if choice == "regenerate_all":
                continue
            if choice == "cancel":
                sys.exit(0)

        preselected = {
            "voice": voice,
            "display_name": display_name,
            "archetype_label": arch_label,
            "gender_style": gender_style,
        }
        process_single_character(
            api_key,
            src_path,
            output_root,
            outfit_db,
            game_name,
            preselected=preselected,
        )

    print("\nCharacter processed.")
    print(f"Final sprite folder(s) are in:\n  {output_root}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """
    Command-line entry point.

    Parses optional arguments, chooses an output folder if needed,
    and then runs the interactive pipeline.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Student Transfer sprite builder using Google Gemini:\n"
            "  - base pose\n"
            "  - outfits (Base + selected extras like Formal/Casual/Uniform/...)\n"
            "  - expressions per outfit (0 + selected non-neutral ones)\n"
            "  - eye line / name color / scale\n"
            "  - character.yml\n"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Root folder to write final character sprite folders. "
            "If omitted, you will be prompted to choose a folder."
        ),
    )
    parser.add_argument(
        "--game-name",
        type=str,
        default=None,
        help="Optional game name to write into character.yml (game field).",
    )

    args = parser.parse_args()

    output_root: Path | None = args.output_dir
    game_name: Optional[str] = args.game_name

    # If no output folder was provided, ask the user via a folder picker
    if output_root is None:
        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        chosen = filedialog.askdirectory(
            title="Choose output folder for character sprite(s)"
        )
        root.destroy()
        if not chosen:
            raise SystemExit("No output folder selected. Exiting.")
        output_root = Path(os.path.abspath(os.path.expanduser(chosen)))

    run_pipeline(output_root, game_name)


if __name__ == "__main__":
    main()
