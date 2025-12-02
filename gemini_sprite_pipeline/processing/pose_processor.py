"""
Pose processing and outfit generation.

Handles initial pose normalization, outfit generation, pose flattening,
and character.yml writing.
"""

import random
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from PIL import Image

from ..api.gemini_client import (
    call_gemini_image_edit,
    call_gemini_text_or_refs,
    load_image_as_base64,
)
from ..api.prompt_builders import (
    build_initial_pose_prompt,
    build_outfit_prompt,
    build_standard_school_uniform_prompt,
)
from .image_utils import (
    save_image_bytes_as_png,
    get_standard_uniform_reference_images,
)


def write_character_yml(
    path: Path,
    display_name: str,
    voice: str,
    eye_line: float,
    name_color: str,
    scale: float,
    poses: Dict[str, Dict[str, str]],
    *,
    game: Optional[str] = None,
) -> None:
    """
    Write final character metadata YAML in organizer format.

    Args:
        path: Path to character.yml file.
        display_name: Character's display name.
        voice: Character voice ("girl"/"boy", converted to "male" if "boy").
        eye_line: Eye line ratio (0.0-1.0).
        name_color: Hex color for name display.
        scale: Character scale multiplier.
        poses: Dict of pose letters to pose metadata.
        game: Optional game name to include in metadata.
    """
    v = (voice or "").lower()
    voice_out = "male" if v == "boy" else voice

    data = {
        "display_name": display_name,
        "eye_line": round(float(eye_line), 4),
        "name_color": name_color,
        "poses": poses,
        "scale": float(scale),
        "voice": voice_out,
    }
    if game:
        data["game"] = game

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Wrote character YAML to: {path}")


def generate_initial_pose_once(
    api_key: str,
    image_path: Path,
    out_stem: Path,
    gender_style: str,
) -> Path:
    """
    Normalize the original sprite into pose A with a flat magenta background.

    Args:
        api_key: Gemini API key.
        image_path: Source image path.
        out_stem: Output path stem (without extension).
        gender_style: 'f' or 'm'.

    Returns:
        Path to saved normalized pose image.
    """
    print("  [Gemini] Normalizing base pose...")
    image_b64 = load_image_as_base64(image_path)
    prompt = build_initial_pose_prompt(gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved base pose to: {final_path}")
    return final_path


def generate_single_outfit(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    outfit_key: str,
    outfit_desc: str,
    outfit_prompt_config: Dict[str, Dict[str, Optional[str]]],
    archetype_label: str,
) -> Path:
    """
    Generate or regenerate a single outfit image for the given key.

    This is used both by the bulk outfit generator and by the
    per-outfit "regenerate" buttons in the review window.

    Args:
        api_key: Gemini API key.
        base_pose_path: Path to base pose image.
        outfits_dir: Directory to save outfits.
        gender_style: 'f' or 'm'.
        outfit_key: Outfit identifier.
        outfit_desc: Outfit description/prompt.
        outfit_prompt_config: Per-outfit configuration.
        archetype_label: Character archetype.

    Returns:
        Path to saved outfit image.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    config = outfit_prompt_config.get(outfit_key, {})

    # Special handling for standardized school uniform
    if outfit_key == "uniform" and config.get("use_standard_uniform"):
        final_path = generate_standard_uniform_outfit(
            api_key,
            base_pose_path,
            outfits_dir,
            gender_style,
            archetype_label,
            outfit_desc,
        )
        print(f"  Saved standardized outfit '{outfit_key}' to: {final_path}")
        return final_path

    # Normal text-prompt-based outfit
    image_b64 = load_image_as_base64(base_pose_path)
    out_stem = outfits_dir / outfit_key.capitalize()
    prompt = build_outfit_prompt(outfit_desc, gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved outfit '{outfit_key}' to: {final_path}")
    return final_path


def generate_standard_uniform_outfit(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    archetype_label: str,
    outfit_desc: str,  # Kept for signature compatibility
) -> Path:
    """
    Generate the standardized school uniform outfit using reference images.

    Uses the base pose as the main character to keep, and a cropped uniform
    reference image (gender-specific) as visual guidance.

    Args:
        api_key: Gemini API key.
        base_pose_path: Path to base pose image.
        outfits_dir: Directory to save outfits.
        gender_style: 'f' or 'm'.
        archetype_label: Character archetype.
        outfit_desc: Not used directly, kept for compatibility.

    Returns:
        Path to saved uniform outfit image.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    # Collect the uniform reference image(s) for this gender
    uniform_refs = get_standard_uniform_reference_images(gender_style)
    if not uniform_refs:
        # Fallback to normal outfit prompt path
        print("[WARN] No uniform reference found, falling back to normal prompt-based uniform.")
        image_b64 = load_image_as_base64(base_pose_path)
        prompt = build_outfit_prompt(outfit_desc, gender_style)
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        out_stem = outfits_dir / "Uniform"
        final_path = save_image_bytes_as_png(img_bytes, out_stem)
        print(f"  Saved fallback prompt-based uniform to: {final_path}")
        return final_path

    # Use first uniform reference
    uniform_ref = uniform_refs[0]
    print(f"[INFO] Using uniform reference: {uniform_ref}")

    # Build unified, standardized uniform prompt
    uniform_prompt = build_standard_school_uniform_prompt(
        archetype_label,
        gender_style,
    )

    # Call Gemini with prompt and two reference images:
    #   1) the base pose (the character to keep)
    #   2) the cropped uniform example (clothes to copy)
    img_bytes = call_gemini_text_or_refs(
        api_key,
        uniform_prompt,
        ref_images=[base_pose_path, uniform_ref],
    )

    out_stem = outfits_dir / "Uniform"
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved standardized school uniform to: {final_path}")
    return final_path


def generate_outfits_once(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    outfit_descriptions: Dict[str, str],
    outfit_prompt_config: Dict[str, Dict[str, Optional[str]]],
    archetype_label: str,
    include_base_outfit: bool = True,
) -> List[Path]:
    """
    Generate outfits for a pose.

    Layout:
      - If include_base_outfit=True: copies base pose as Base.png.
      - For each outfit_descriptions[key], generate <Key>.png.

    Args:
        api_key: Gemini API key.
        base_pose_path: Path to base pose image.
        outfits_dir: Directory to save outfits.
        gender_style: 'f' or 'm'.
        outfit_descriptions: Dict of outfit keys to descriptions.
        outfit_prompt_config: Per-outfit configuration.
        archetype_label: Character archetype.
        include_base_outfit: Whether to include base pose as an outfit.

    Returns:
        List of paths to generated outfit images.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Optional: copy base pose as "Base.png" outfit
    if include_base_outfit:
        base_bytes = base_pose_path.read_bytes()
        base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
        base_out_path = (outfits_dir / "Base").with_suffix(".png")
        base_img.save(base_out_path, format="PNG")
        paths.append(base_out_path)

    # Generate each selected outfit key
    for key, desc in outfit_descriptions.items():
        final_path = generate_single_outfit(
            api_key,
            base_pose_path,
            outfits_dir,
            gender_style,
            key,
            desc,
            outfit_prompt_config,
            archetype_label,
        )
        paths.append(final_path)

    return paths


def flatten_pose_outfits_to_letter_poses(char_dir: Path) -> List[str]:
    """
    Flatten pose/outfit combos into separate ST poses with single outfits.

    Input:
        <char>/a/outfits/OutfitName.png
        <char>/a/faces/face/*.webp
        <char>/a/faces/OutfitName/*.webp

    Output:
        <char>/a, /b, /c, ... each with outfits/<OutfitName>.png (transparent base)
        and faces/face/0.webp..N.webp

    Args:
        char_dir: Character directory.

    Returns:
        List of final pose letters (sorted).
    """
    original_pose_dirs = [
        p for p in char_dir.iterdir()
        if p.is_dir() and len(p.name) == 1 and p.name.isalpha()
    ]
    original_pose_dirs.sort(key=lambda p: p.name)

    tmp_root = char_dir / "_tmp_pose_flattened"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    letters = [chr(ord("a") + i) for i in range(26)]
    next_index = 0

    def _next_letter() -> Optional[str]:
        nonlocal next_index
        if next_index >= len(letters):
            return None
        letter = letters[next_index]
        next_index += 1
        return letter

    final_pose_letters: List[str] = []

    for pose_dir in original_pose_dirs:
        outfits_dir = pose_dir / "outfits"
        faces_root = pose_dir / "faces"

        if not outfits_dir.is_dir() or not faces_root.is_dir():
            continue

        for outfit_path in sorted(outfits_dir.iterdir()):
            if not outfit_path.is_file():
                continue
            if outfit_path.suffix.lower() not in (".png", ".webp"):
                continue

            outfit_name = outfit_path.stem
            if outfit_name.lower() == "base":
                src_expr_dir = faces_root / "face"
            else:
                src_expr_dir = faces_root / outfit_name

            if not src_expr_dir.is_dir():
                print(
                    f"[WARN] No expression folder for pose '{pose_dir.name}', "
                    f"outfit '{outfit_name}' at {src_expr_dir}; skipping."
                )
                continue

            pose_letter = _next_letter()
            if pose_letter is None:
                print(
                    "[WARN] Ran out of pose letters (more than 26 combinations); "
                    "skipping remaining outfits."
                )
                break

            new_pose_dir = tmp_root / pose_letter
            new_faces_dir = new_pose_dir / "faces" / "face"
            new_outfits_dir = new_pose_dir / "outfits"
            new_faces_dir.mkdir(parents=True, exist_ok=True)
            new_outfits_dir.mkdir(parents=True, exist_ok=True)

            # Copy expression files
            for src in sorted(src_expr_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = new_faces_dir / src.name
                shutil.copy2(src, dest)

            # Create transparent outfit base
            try:
                outfit_img = Image.open(outfit_path).convert("RGBA")
                w, h = outfit_img.size
                transparent = Image.new("RGBA", (w, h), (0, 0, 0, 0))
                out_name = outfit_path.stem + outfit_path.suffix.lower()
                transparent.save(new_outfits_dir / out_name)
            except Exception as e:
                print(
                    f"[WARN] Failed to create transparent outfit for "
                    f"{pose_dir.name}/{outfit_name}: {e}"
                )
                continue

            final_pose_letters.append(pose_letter)
            print(
                f"[INFO] Created pose '{pose_letter}' from "
                f"orig pose '{pose_dir.name}', outfit '{outfit_name}'"
            )

    # Remove original pose directories
    for pose_dir in original_pose_dirs:
        try:
            shutil.rmtree(pose_dir)
        except Exception as e:
            print(f"[WARN] Failed to remove original pose folder {pose_dir}: {e}")

    # Move flattened poses to character directory
    for new_pose_dir in sorted(tmp_root.iterdir(), key=lambda p: p.name):
        if not new_pose_dir.is_dir():
            continue
        target = char_dir / new_pose_dir.name
        if target.exists():
            try:
                shutil.rmtree(target)
            except Exception:
                pass
        shutil.move(str(new_pose_dir), str(target))

    # Clean up temp directory
    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    final_pose_letters.sort()
    return final_pose_letters
