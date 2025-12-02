"""
Expression generation and character creation workflows.

Handles expression generation for outfits, regeneration of individual expressions,
and prompt-based character generation.
"""

import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from gsp_constants import EXPRESSIONS_SEQUENCE, REF_SPRITES_DIR

from ..api.gemini_client import (
    call_gemini_image_edit,
    call_gemini_text_or_refs,
    load_image_as_base64,
)
from ..api.prompt_builders import (
    build_expression_prompt,
    build_prompt_for_idea,
)
from ..ui.review_windows import review_images_for_step
from .image_utils import (
    save_img_webp_or_png,
    save_image_bytes_as_png,
    get_reference_images_for_archetype,
)


def generate_expressions_for_single_outfit_once(
    api_key: str,
    pose_dir: Path,
    outfit_path: Path,
    faces_root: Path,
    expressions_sequence: Optional[List[Tuple[str, str]]] = None,
) -> List[Path]:
    """
    Generate a full expression set for a single outfit in a single pose.

    Layout (pose 'a', outfit 'Base'):
        a/outfits/Base.png
        a/faces/face/0.webp ... N.webp
    For non-base outfits (e.g. 'Formal'):
        a/faces/Formal/0.webp ... N.webp

    0.webp is always the neutral outfit image itself.

    Args:
        api_key: Gemini API key.
        pose_dir: Pose directory.
        outfit_path: Path to outfit image.
        faces_root: Root directory for face images.
        expressions_sequence: List of (key, description) tuples for expressions.

    Returns:
        List of paths to generated expression images.
    """
    faces_root.mkdir(parents=True, exist_ok=True)
    if expressions_sequence is None:
        expressions_sequence = EXPRESSIONS_SEQUENCE

    generated_paths: List[Path] = []

    if not outfit_path.is_file() or outfit_path.suffix.lower() not in (".png", ".webp"):
        return generated_paths

    outfit_name = outfit_path.stem
    if outfit_name.lower() == "base":
        out_dir = faces_root / "face"
    else:
        out_dir = faces_root / outfit_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] Generating expressions for pose '{pose_dir.name}', "
        f"outfit '{outfit_name}'"
    )

    # Clear existing expression files
    for f in list(out_dir.iterdir()):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    # Save neutral expression (0) as the outfit itself
    neutral_stem = out_dir / "0"
    outfit_img = Image.open(outfit_path).convert("RGBA")
    neutral_path = save_img_webp_or_png(outfit_img, neutral_stem)
    generated_paths.append(neutral_path)
    print(f"  [Expr] Using outfit as neutral '0' -> {neutral_path}")

    # Generate remaining expressions
    image_b64 = load_image_as_base64(outfit_path)

    for idx, (orig_key, desc) in enumerate(expressions_sequence[1:], start=1):
        out_stem = out_dir / str(idx)
        prompt = build_expression_prompt(desc)
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        final_path = save_image_bytes_as_png(img_bytes, out_stem)

        generated_paths.append(final_path)
        print(
            f"  [Expr] Saved {pose_dir.name}/{outfit_name} "
            f"expression '{orig_key}' as '{idx}' -> {final_path}"
        )

    return generated_paths


def regenerate_single_expression(
    api_key: str,
    outfit_path: Path,
    out_dir: Path,
    expressions_sequence: List[Tuple[str, str]],
    expr_index: int,
) -> Path:
    """
    Regenerate a single expression image for one outfit.

    expr_index is the numeric index into expressions_sequence and also
    the filename stem (0, 1, 2, ...).

    Args:
        api_key: Gemini API key.
        outfit_path: Path to outfit image.
        out_dir: Directory to save expression.
        expressions_sequence: List of expression definitions.
        expr_index: Index of expression to regenerate.

    Returns:
        Path to regenerated expression image.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Neutral 0 is always just the outfit itself; no need to call Gemini
    if expr_index == 0:
        outfit_img = Image.open(outfit_path).convert("RGBA")
        neutral_stem = out_dir / "0"
        neutral_path = save_img_webp_or_png(outfit_img, neutral_stem)
        print(f"  [Expr] Regenerated neutral expression 0 -> {neutral_path}")
        return neutral_path

    if expr_index < 0 or expr_index >= len(expressions_sequence):
        raise ValueError(f"Expression index {expr_index} out of range.")

    _, desc = expressions_sequence[expr_index]

    image_b64 = load_image_as_base64(outfit_path)
    out_stem = out_dir / str(expr_index)
    prompt = build_expression_prompt(desc)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(
        f"  [Expr] Regenerated expression index {expr_index} "
        f"for '{outfit_path.stem}' -> {final_path}"
    )
    return final_path


def generate_and_review_expressions_for_pose(
    api_key: str,
    char_dir: Path,
    pose_dir: Path,
    pose_label: str,
    expressions_sequence: List[Tuple[str, str]],
) -> None:
    """
    For a given pose directory (e.g., 'a'), iterate each outfit and:

      - Generate expression set (once per outfit).
      - Show review window.
      - Allow:
          * Accept (keep all as-is),
          * Regenerate all expressions for that outfit,
          * Regenerate a single expression via buttons under each image,
          * Cancel the whole pipeline.

    Args:
        api_key: Gemini API key.
        char_dir: Character directory.
        pose_dir: Pose directory.
        pose_label: Pose label for display.
        expressions_sequence: List of expression definitions.
    """
    outfits_dir = pose_dir / "outfits"
    faces_root = pose_dir / "faces"
    outfits_dir.mkdir(parents=True, exist_ok=True)
    faces_root.mkdir(parents=True, exist_ok=True)

    for outfit_path in sorted(outfits_dir.iterdir()):
        if not outfit_path.is_file():
            continue
        if outfit_path.suffix.lower() not in (".png", ".webp"):
            continue

        outfit_name = outfit_path.stem

        # First, build the full expression set once
        generate_expressions_for_single_outfit_once(
            api_key,
            pose_dir,
            outfit_path,
            faces_root,
            expressions_sequence=expressions_sequence,
        )

        # Determine the folder where the expression images for this outfit live
        if outfit_name.lower() == "base":
            out_dir = faces_root / "face"
        else:
            out_dir = faces_root / outfit_name

        while True:
            # Collect current expression images from disk, sorted by index
            expr_paths: List[Path] = []
            for p in sorted(out_dir.iterdir(), key=lambda q: q.stem):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".png", ".webp"):
                    continue
                expr_paths.append(p)

            # Ensure numeric ordering (0, 1, 2, ...)
            expr_paths.sort(key=lambda q: int(q.stem))

            infos = [
                (
                    p,
                    f"Pose {pose_label} – {outfit_name} – expression {p.stem} "
                    f"({expressions_sequence[int(p.stem)][0] if int(p.stem) < len(expressions_sequence) else '?'})",
                )
                for p in expr_paths
            ]

            # One "regenerate this expression" button under each expression
            per_buttons: List[List[Tuple[str, str]]] = [
                [("Regenerate this expression", "regen_expr")] for _ in expr_paths
            ]

            decision = review_images_for_step(
                infos,
                f"Review Expressions for Pose {pose_label} – {outfit_name}",
                (
                    "These expressions are generated for this single pose/outfit.\n"
                    "Accept them, regenerate all, regenerate a single expression, or cancel."
                ),
                per_item_buttons=per_buttons,
                show_global_regenerate=True,
            )

            choice = decision.get("choice")

            if choice == "accept":
                break
            if choice == "cancel":
                sys.exit(0)

            if choice == "regenerate_all":
                # Wipe and rebuild the whole expression set for this outfit
                generate_expressions_for_single_outfit_once(
                    api_key,
                    pose_dir,
                    outfit_path,
                    faces_root,
                    expressions_sequence=expressions_sequence,
                )
                continue

            if choice == "per_item":
                idx_obj = decision.get("index")
                if idx_obj is None:
                    continue
                idx = int(idx_obj)
                if idx < 0 or idx >= len(expr_paths):
                    continue

                # Card index -> expression index: the filename stem
                try:
                    expr_index = int(expr_paths[idx].stem)
                except ValueError:
                    continue

                if expr_index < 0 or expr_index >= len(expressions_sequence):
                    continue

                regenerate_single_expression(
                    api_key,
                    outfit_path,
                    out_dir,
                    expressions_sequence,
                    expr_index,
                )
                # Loop to show the updated images
                continue


def generate_initial_character_from_prompt(
    api_key: str,
    concept: str,
    archetype_label: str,
    output_root: Path,
) -> Path:
    """
    Use Gemini + reference sprites to generate a base character image
    from a text concept.

    Output path:
        <output_root>/_prompt_sources/<slug>.png

    Args:
        api_key: Gemini API key.
        concept: User's character concept description.
        archetype_label: Character archetype.
        output_root: Root output directory.

    Returns:
        Path to generated character image.
    """
    from ..api.prompt_builders import archetype_to_gender_style

    gender_style = archetype_to_gender_style(archetype_label)
    refs = get_reference_images_for_archetype(archetype_label)
    if refs:
        print(f"[INFO] Using {len(refs)} reference sprite(s) for archetype '{archetype_label}'.")
    else:
        print(
            f"[WARN] No reference sprites found for archetype '{archetype_label}'. "
            "Gemini will rely on the text prompt alone."
        )

    full_prompt = build_prompt_for_idea(concept, archetype_label, gender_style)
    print("[Gemini] Generating new character from text prompt...")
    img_bytes = call_gemini_text_or_refs(api_key, full_prompt, refs)

    rand_token = hex(random.getrandbits(32))[2:]
    slug = f"{archetype_label.replace(' ', '_')}_{rand_token}"
    prompt_src_dir = output_root / "_prompt_sources"
    prompt_src_dir.mkdir(parents=True, exist_ok=True)
    out_stem = prompt_src_dir / slug

    final_path = save_image_bytes_as_png(img_bytes, out_stem)

    print(f"[INFO] Saved prompt-generated source sprite to: {final_path}")
    return final_path
