#!/usr/bin/env python3
"""
gemini_sprite_pipeline.py

Single-character Student Transfer-style sprite builder using Gemini.

Flow (per character):
  - Start from either an existing image or a prompt-generated base.
  - Pick voice, name, archetype.
  - Choose extra outfits + expressions.
  - Normalize to pose A (mid-thigh, magenta bg) and review.
  - Generate outfits and expressions, with review loops.
  - Pick eye line, name color, and scale vs reference.
  - Flatten pose/outfit combos into ST-style poses and write character.yml.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import yaml
import webbrowser

from gsp_constants import (
    CONFIG_PATH,
    OUTFIT_CSV_PATH,
    NAMES_CSV_PATH,
    REF_SPRITES_DIR,
    GEMINI_API_URL,
    GBG_COLOR,
    BG_COLOR,
    TITLE_FONT,
    INSTRUCTION_FONT,
    LINE_COLOR,
    WINDOW_MARGIN,
    WRAP_PADDING,
    ALL_OUTFIT_KEYS,
    OUTFIT_KEYS,
    EXPRESSIONS_SEQUENCE,
    GENDER_ARCHETYPES,
)

# Tk layout helpers

def _compute_display_size(
    screen_w: int,
    screen_h: int,
    img_w: int,
    img_h: int,
    *,
    max_w_ratio: float = 0.90,
    max_h_ratio: float = 0.55,
) -> Tuple[int, int]:
    """Return (disp_w, disp_h) that fits within given screen ratios."""
    max_w = int(screen_w * max_w_ratio) - 2 * WINDOW_MARGIN
    max_h = int(screen_h * max_h_ratio) - 2 * WINDOW_MARGIN
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))

def _center_and_clamp(root: tk.Tk) -> None:
    """Clamp window to screen and center near top."""
    root.update_idletasks()
    req_w = root.winfo_reqwidth()
    req_h = root.winfo_reqheight()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    w = min(req_w + WINDOW_MARGIN, sw - 2 * WINDOW_MARGIN)
    h = min(req_h + WINDOW_MARGIN, sh - 2 * WINDOW_MARGIN)
    x = max((sw - w) // 2, WINDOW_MARGIN)
    y = WINDOW_MARGIN

    root.geometry(f"{w}x{h}+{x}+{y}")

def _wraplength_for(width_px: int) -> int:
    """Wrap length for labels given a target width."""
    return max(200, width_px - WRAP_PADDING)

# Names and YAML helpers

def load_name_pool(csv_path: Path) -> Tuple[List[str], List[str]]:
    """Load girl/boy name pools from CSV with columns: name, gender."""
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
    """Pick a random name based on voice."""
    pool = girl_names if (voice or "").lower() == "girl" else boy_names
    if not pool:
        pool = ["Alex", "Riley", "Taylor", "Jordan"]
    return random.choice(pool)

def get_unique_folder_name(base_path: Path, desired_name: str) -> str:
    """Ensure folder name is unique within base_path by appending a counter."""
    candidate = desired_name
    counter = 1
    while (base_path / candidate).exists():
        counter += 1
        candidate = f"{desired_name}_{counter}"
    return candidate

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
    """Write final character metadata YAML in organizer format."""
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

# Image helpers / background stripping

def save_img_webp_or_png(img: Image.Image, dest_stem: Path) -> Path:
    """Save as WEBP lossless, falling back to PNG if needed."""
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    safe = img.convert("RGBA")

    try:
        out_path = dest_stem.with_suffix(".webp")
        safe.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
        return out_path
    except Exception as e:
        print(f"[WARN] WEBP save failed for {dest_stem.name}: {e}. Falling back to PNG.")
        out_path = dest_stem.with_suffix(".png")
        safe.save(out_path, format="PNG", lossless=True)
        return out_path


def save_image_bytes_as_png(image_bytes: bytes, dest_stem: Path) -> Path:
    """Save raw image bytes as PNG to dest_stem.png."""
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    out_path = dest_stem.with_suffix(".png")
    img.save(out_path, format="PNG", lossless=True)
    return out_path

def strip_background(image_bytes: bytes) -> bytes:
    """
    Strip a flat-ish magenta background.
    Strategy:
      1) Load RGBA.
      2) Collect all opaque border pixels.
      3) Estimate background color as the average of those border pixels.
      4) Clear any pixel sufficiently close to that background color.
    """
    BG_CLEAR_THRESH = 56  # tweak this if it's too aggressive or too gentle

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        w, h = img.size
        pixels = img.load()

        border_samples = []

        # top and bottom rows
        for x in range(w):
            for y in (0, h - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                border_samples.append((r, g, b))

        # left and right columns
        for y in range(h):
            for x in (0, w - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                border_samples.append((r, g, b))

        if not border_samples:
            # Nothing to go on; just return original
            return image_bytes

        n = float(len(border_samples))
        bg_r = sum(r for r, _, _ in border_samples) / n
        bg_g = sum(g for _, g, _ in border_samples) / n
        bg_b = sum(b for _, _, b in border_samples) / n

        bg_clear_thresh_sq = BG_CLEAR_THRESH * BG_CLEAR_THRESH

        def is_bg(r, g, b):
            dr = r - bg_r
            dg = g - bg_g
            db = b - bg_b
            return (dr * dr + dg * dg + db * db) <= bg_clear_thresh_sq

        out = Image.new("RGBA", (w, h))
        out_pixels = out.load()

        for y in range(h):
            for x in range(w):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    out_pixels[x, y] = (r, g, b, 0)
                elif is_bg(r, g, b):
                    out_pixels[x, y] = (r, g, b, 0)
                else:
                    out_pixels[x, y] = (r, g, b, a)

        buf = BytesIO()
        out.save(buf, format="PNG")
        return buf.getvalue()

    except Exception as e:
        print(f"  [WARN] strip_background failed, returning original bytes: {e}")
        return image_bytes

def load_image_as_base64(path: Path) -> str:
    """Load image, re-encode as PNG, return base64 string."""
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", lossless=True)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8")

# Config / Gemini HTTP

def load_config() -> dict:
    """Load ~/.st_gemini_config.json if present."""
    if CONFIG_PATH.is_file():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_config(cfg: dict) -> None:
    """Save config dictionary to CONFIG_PATH."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass

def interactive_api_key_setup() -> str:
    """Prompt user for Gemini API key and save it."""
    print("\nIt looks like you haven't configured a Gemini API key yet.")
    print("To use Google Gemini's image model, we need an API key.")
    print("I will open the Gemini API key page in your browser.")
    input("Press Enter to open the Gemini API key page in your browser...")

    key_page_url = "https://aistudio.google.com/app/apikey"
    try:
        webbrowser.open(key_page_url)
    except Exception as e:
        print(f"Warning: could not open browser automatically: {e}")
        print(f"Please open this URL manually in your browser: {key_page_url}")

    api_key = input("\nPaste your Gemini API key here and press Enter:\n> ").strip()
    if not api_key:
        raise SystemExit("No API key entered. Please rerun the script when you have a key.")

    cfg = load_config()
    cfg["api_key"] = api_key
    save_config(cfg)
    print(f"Saved API key to {CONFIG_PATH}.")
    return api_key

def get_api_key() -> str:
    """Return Gemini API key from env or config, prompting if needed."""
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    cfg = load_config()
    if cfg.get("api_key"):
        return cfg["api_key"]
    return interactive_api_key_setup()

def _extract_inline_image_from_response(data: dict) -> Optional[bytes]:
    """Pull the first inline image bytes from a Gemini JSON response."""
    candidates = data.get("candidates", [])
    for cand in candidates:
        content = cand.get("content", {})
        for part in content.get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and "data" in blob:
                return base64.b64decode(blob["data"])
    return None

def _call_gemini_with_parts(api_key: str, parts: List[dict], context: str) -> bytes:
    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            if not resp.ok:
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    print(f"[WARN] Gemini API error {resp.status_code} ({context}) attempt {attempt}; retrying...")
                    last_error = f"Gemini API error {resp.status_code}: {resp.text}"
                    continue
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_bytes = _extract_inline_image_from_response(data)
            if raw_bytes is not None:
                return strip_background(raw_bytes)

            last_error = f"No image data in Gemini response ({context})."
            if attempt < max_retries:
                print(f"[WARN] Gemini response missing image ({context}) attempt {attempt}; retrying...")
                continue
            raise RuntimeError(last_error)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"[WARN] Gemini call failed ({context}) attempt {attempt}; retrying: {e}")
                continue
            raise RuntimeError(f"Gemini call failed after {max_retries} attempts ({context}): {last_error}")

def call_gemini_image_edit(api_key: str, prompt: str, image_b64: str) -> bytes:
    """Call Gemini image model with an input image + text prompt."""
    parts: List[dict] = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
    ]
    return _call_gemini_with_parts(api_key, parts, "image_edit")


# Outfit prompts / expression descriptions

def archetype_to_gender_style(archetype_label: str) -> str:
    """Given an archetype label, return gender style code 'f' or 'm' (default 'f')."""
    for lbl, g in GENDER_ARCHETYPES:
        if lbl == archetype_label:
            return g
    return "f"

def load_outfit_prompts(csv_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load outfit prompts from CSV: archetype,outfit_key,prompt.

    Returns: {archetype: {outfit_key: [prompt, ...]}, ...}
    """
    db: Dict[str, Dict[str, List[str]]] = {}

    if not csv_path.is_file():
        print(f"[WARN] Outfit CSV not found at {csv_path}. Using generic prompts.")
        return db

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            archetype = (row.get("archetype") or "").strip()
            outfit_key = (row.get("outfit_key") or "").strip()
            prompt = (row.get("prompt") or "").strip()
            if not archetype or not outfit_key or not prompt:
                continue
            db.setdefault(archetype, {}).setdefault(outfit_key, []).append(prompt)

    return db

def build_simple_outfit_description(outfit_key: str, gender_style: str) -> str:
    """Fallback generic outfit description if no CSV prompt is available."""
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
    outfit_db: Dict[str, Dict[str, List[str]]],
    outfit_prompt_config: Dict[str, Dict[str, Optional[str]]],
) -> Dict[str, str]:
    """
    Build one prompt per selected outfit_key, honoring per-outfit settings.

    For each key:
      - use_random=True: pick random CSV prompt if available; else fallback.
      - use_random=False: use custom_prompt (fallback if empty for safety).
    """
    prompts: Dict[str, str] = {}
    arch_pool = outfit_db.get(archetype_label, {})

    for key in selected_outfit_keys:
        cfg = outfit_prompt_config.get(key, {})
        use_random = bool(cfg.get("use_random", True))
        custom_prompt = cfg.get("custom_prompt")

        if use_random:
            candidates = arch_pool.get(key)
            if candidates:
                prompts[key] = random.choice(candidates)
            else:
                prompts[key] = build_simple_outfit_description(key, gender_style)
        else:
            prompts[key] = custom_prompt or build_simple_outfit_description(key, gender_style)

    return prompts

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

            for src in sorted(src_expr_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = new_faces_dir / src.name
                shutil.copy2(src, dest)

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

    for pose_dir in original_pose_dirs:
        try:
            shutil.rmtree(pose_dir)
        except Exception as e:
            print(f"[WARN] Failed to remove original pose folder {pose_dir}: {e}")

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

    try:
        shutil.rmtree(tmp_root)
    except Exception:
        pass

    final_pose_letters.sort()
    return final_pose_letters

# Gemini prompt builders

def build_initial_pose_prompt(gender_style: str) -> str:
    """Prompt to normalize the original sprite (mid-thigh, magenta background)."""
    return (
        "Edit the image of the character, to give them a pure, flat, magenta (#FF00FF) background behind them."
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "Make sure that the character, outfit, or hair end up with none of the magenta background color on them. "
        "Make sure the head, arms, hair, hands, and clothes are all kept within the image."
        "Keep the crop the same from the mid-thigh on up."
        "Dont change the art style either, just edit the background that the character is on, to be that magenta color."
    )

def build_expression_prompt(expression_desc: str) -> str:
    """Prompt to change facial expression, keeping style and framing."""
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
    """Prompt to change clothing to base_outfit_desc on the given pose."""
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
    Build a standardized school-uniform prompt that matches the rest of the
    outfit prompts, but is specific to the canonical uniform.

    This uses the archetype label and the gender style ("f" or "m") to describe
    the character, and then describes the school uniform in text. The cropped
    uniform reference image is used as a visual backup, but this text gives
    Gemini a clear, redundant description.
    """
    gender_word = "girl" if gender_style == "f" else "boy"

    # Base description that applies to both variants.
    base_intro = (
        f"Edit the inputed {archetype_label} visual novel sprite, to give them the outfit we have also attached."
        "For redundency, I am going to also describe the outfit below, but using the reference image is your first priority when it comes to what this outfit needs to look like."
    )

    if gender_style == "f":
        # Female student uniform: blazer + bow + pleated skirt description.
        uniform_desc = (
            "She should be wearing: A navy blue tailored sleeveless blazer hybrid, tightly fitted to the torso. The blazer has gold piping along all the outer edges. The front features a double-breasted design with two rows of two gold buttons. The vest dips into a sharp angled hem near the waist, creating a stylish contour. Underneath it is a white short-sleeved dress shirt. The sleeve has a school crest patch on the upper arm: gold/yellow with an emblem inside. Her arms are bare below the sleeves. She should have on a bright red necktie with white stripes near the bottom. A short, red, plaid, pleated skirt finishes out the outfit. No ribbons."
        )
    else:
        # Male student uniform: blazer + tie + slacks description.
        uniform_desc = (
            "He should be wearing: A white short-sleeved dress shirt. The sleeve has a school crest patch on the upper arm: gold/yellow with an emblem inside. He should have on a  bright red necktie with white stripes near the bottom. A pair of dark-colored slacks with a belt, which the white shirt tucks into, completes the look."
        )

    # Shared constraints and ST-format requirements.
    tail = (
        "Again, copy over the outfit from the image sent. The description above is just to help with consistency."
        "Use a pure, single color, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them. If the character has magenta on them, slightly change those pixels to something farther away from the new background color, magenta."
        "Do not change the art style, size, proportions, or hair length of the character, and keep their arms, hands, and hair all inside the image."
        "Thats all to say, the goal is to copy over the outfit from the reference, to the character we are editing, to replace their current outfit."
    )

    return base_intro + uniform_desc + tail

# Tk: voice + archetype + name

def prompt_voice_archetype_and_name(image_path: Path) -> Tuple[str, str, str, str]:
    """
    Tk window to show source image and pick:
      - voice (Girl/Boy)
      - name (auto-filled, editable)
      - archetype (filtered by voice)

    Returns: (voice, display_name, archetype_label, gender_style)
    """
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)
    img = Image.open(image_path).convert("RGBA")
    ow, oh = img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Character Setup")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

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

    disp_w, disp_h = _compute_display_size(sw, sh, ow, oh, max_w_ratio=0.70, max_h_ratio=0.45)
    disp = img.resize((disp_w, disp_h), Image.LANCZOS)
    tki = ImageTk.PhotoImage(disp)
    canvas = tk.Canvas(root, width=disp_w, height=disp_h, bg="black", highlightthickness=0)
    canvas.create_image(0, 0, anchor="nw", image=tki)
    canvas.image = tki
    canvas.grid(row=1, column=0, padx=10, pady=4, sticky="n")

    voice_var = tk.StringVar(value="")
    archetype_var = tk.StringVar(value="")
    gender_style_var = {"value": None}
    name_var = tk.StringVar(value="")

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
        voice_var.set(v)
        display_name = pick_random_name(v, girl_names, boy_names)
        if not name_var.get().strip():
            name_var.set(display_name)
        update_archetype_menu()
        name_entry.focus_set()
        name_entry.icursor(tk.END)

    btn_row = tk.Frame(root, bg=BG_COLOR)
    btn_row.grid(row=3, column=0, pady=(4, 4))
    tk.Button(btn_row, text="Girl", width=12, command=lambda: choose_voice("girl")).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btn_row, text="Boy", width=12, command=lambda: choose_voice("boy")).pack(
        side=tk.LEFT, padx=10
    )

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
        decision["mode"] = "cancel"
        try:
            root.destroy()
        except Exception:
            pass

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=5, column=0, pady=(8, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(side=tk.LEFT, padx=10)

    _center_and_clamp(root)
    root.mainloop()

    if not decision["done"]:
        sys.exit(0)

    return (
        decision["voice"],
        decision["name"],
        decision["arch"],
        decision["gstyle"],
    )
# Tk: generic review window

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

    Args:
        image_infos:
            List of (image_path, caption) pairs.
        title_text:
            Window title text.
        body_text:
            Instructional text.
        per_item_buttons:
            Optional list (same length as image_infos) where each entry is a
            list of (button_label, action_code) tuples. For each image card,
            those buttons are rendered under the caption. Pressing one closes
            the window and returns a decision with:
                {"choice": "per_item", "index": idx, "action": action_code}
        show_global_regenerate:
            If True, show a global "Regenerate" button at the bottom that
            behaves like the old "regenerate all" behavior and returns:
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
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text=title_text,
        font=TITLE_FONT,
        bg=BG_COLOR,
        fg="black",
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 4), sticky="we")

    tk.Label(
        root,
        text=body_text,
        font=INSTRUCTION_FONT,
        bg=BG_COLOR,
        fg="black",
        wraplength=wrap_len,
        justify="center",
    ).grid(row=1, column=0, padx=10, pady=(0, 6), sticky="we")

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

    # Normalize per_item_buttons length if provided.
    if per_item_buttons is not None:
        if len(per_item_buttons) < len(image_infos):
            per_item_buttons = per_item_buttons + [[] for _ in range(len(image_infos) - len(per_item_buttons))]
    else:
        per_item_buttons = [[] for _ in image_infos]

    def make_item_handler(idx: int, action_code: str):
        def _handler():
            decision["choice"] = "per_item"
            decision["index"] = idx
            decision["action"] = action_code
            root.destroy()
        return _handler

    for col_index, (p, caption) in enumerate(image_infos):
        try:
            im = Image.open(p).convert("RGBA")
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
            continue

        w, h = im.size
        scale = min(max_thumb_height / max(1, h), 1.0)
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = im.resize((tw, th), Image.LANCZOS)
        tki = ImageTk.PhotoImage(thumb)
        thumb_refs.append(tki)

        card = tk.Frame(inner, bg=BG_COLOR)
        card.grid(row=0, column=col_index, padx=10, pady=6)

        tk.Label(card, image=tki, bg=BG_COLOR).pack()

        tk.Label(
            card,
            text=caption,
            font=INSTRUCTION_FONT,
            bg=BG_COLOR,
            fg="black",
            wraplength=tw + 40,
            justify="center",
        ).pack(pady=(2, 2))

        # Optional per-item buttons under each image.
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
        inner.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    inner.bind("<Configure>", _update_scrollregion)
    _update_scrollregion()

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

    _center_and_clamp(root)
    root.mainloop()
    return decision

def review_initial_base_pose(base_pose_path: Path) -> Tuple[str, bool]:
    """
    Review normalized base pose and decide:

      - Accept / Regenerate / Cancel
      - Whether to treat this base pose as a 'Base' outfit.

    Returns: (choice, use_as_outfit)
    """
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Review Normalized Base Pose")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

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

    preview_frame = tk.Frame(root, bg=BG_COLOR)
    preview_frame.grid(row=1, column=0, padx=10, pady=(4, 4))

    img = Image.open(base_pose_path).convert("RGBA")
    max_size = int(min(sw, sh) * 0.4)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    root._base_preview_img = img_tk  # type: ignore[attr-defined]
    tk.Label(preview_frame, image=img_tk, bg=BG_COLOR).pack()

    use_as_outfit_var = tk.IntVar(value=1)  # default: keep as Base

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

    _center_and_clamp(root)
    root.mainloop()

    return decision["choice"], decision["use_as_outfit"]

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

    ow, oh = img.size
    disp_w, disp_h = _compute_display_size(sw, sh, ow, oh, max_w_ratio=0.60, max_h_ratio=0.60)
    disp = img.resize((disp_w, disp_h), Image.LANCZOS)
    tki = ImageTk.PhotoImage(disp)

    canvas = tk.Canvas(root, width=disp_w, height=disp_h, bg="black", highlightthickness=0)
    canvas.pack(pady=6)
    canvas.create_image(0, 0, anchor="nw", image=tki)
    canvas.image = tki

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
        real_y = int((disp_y / disp_h) * oh)
        result["y"] = real_y
        root.destroy()

    canvas.bind("<Motion>", on_motion)
    canvas.bind("<Button-1>", on_click)

    _center_and_clamp(root)
    root.mainloop()

    return result["y"], used_gallery 

# Tk: eye line + name color

def prompt_for_eye_and_hair(image_path: Path) -> Tuple[float, str]:
    """
    Tk UI to choose:
      - eye line (click once, height ratio)
      - hair color (click once, used as name_color)
    """
    result = {"eye_line": None, "name_color": None}
    state = {"step": 1}

    img = Image.open(image_path).convert("RGBA")
    ow, oh = img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Eye Line and Name Color")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    dw, dh = _compute_display_size(sw, sh, ow, oh, max_w_ratio=0.90, max_h_ratio=0.80)
    sx, sy = ow / max(1, dw), oh / max(1, dh)

    wrap_len = _wraplength_for(int(sw * 0.9))
    title = tk.Label(
        root,
        text="Step 1: Click to mark the eye line (relative head height).",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    )
    title.grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    cwrap = tk.Frame(root, bg=BG_COLOR, width=dw, height=dh)
    cwrap.grid(row=1, column=0, padx=10, pady=4, sticky="n")
    cwrap.grid_propagate(False)

    disp = img.resize((dw, dh), Image.LANCZOS)
    tki = ImageTk.PhotoImage(disp)
    cvs = tk.Canvas(cwrap, width=dw, height=dh, highlightthickness=0, bg="black")
    cvs.create_image(0, 0, anchor="nw", image=tki)
    cvs.image = tki
    cvs.place(relx=0.5, rely=0.0, anchor="n")

    guide_line_id = None
    reticle_h_id = None
    reticle_v_id = None

    def draw_eyeline(y_disp: int):
        nonlocal guide_line_id
        y_disp = max(0, min(int(y_disp), dh))
        if guide_line_id is None:
            guide_line_id = cvs.create_line(0, y_disp, dw, y_disp, fill=LINE_COLOR, width=3)
        else:
            cvs.coords(guide_line_id, 0, y_disp, dw, y_disp)

    def clear_eyeline():
        nonlocal guide_line_id
        if guide_line_id is not None:
            cvs.delete(guide_line_id)
            guide_line_id = None

    def draw_reticle(x_disp: int, y_disp: int, arm: int = 16):
        nonlocal reticle_h_id, reticle_v_id
        x_disp = max(0, min(int(x_disp), dw))
        y_disp = max(0, min(int(y_disp), dh))
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
            real_y = e.y * sy
            result["eye_line"] = real_y / oh
            clear_eyeline()
            title.config(
                text="Eye line recorded.\nStep 2: Click on the hair color to use as name color.",
                wraplength=wrap_len,
            )
            state["step"] = 2
            draw_reticle(e.x, e.y)
        elif state["step"] == 2:
            rx = min(max(int(e.x * sx), 0), ow - 1)
            ry = min(max(int(e.y * sy), 0), oh - 1)
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
    draw_eyeline(dh // 2)

    _center_and_clamp(root)
    root.mainloop()

    if result["eye_line"] is None or result["name_color"] is None:
        sys.exit(0)

    print(f"[INFO] Eye line: {result['eye_line']:.3f}")
    print(f"[INFO] Name color: {result['name_color']}")
    return float(result["eye_line"]), str(result["name_color"])

def pick_representative_outfit(char_dir: Path) -> Path:
    """
    Choose a full-body outfit image to use for eye-line and scale selection.

    Preference:
        a/outfits/Base|Formal|Casual.(webp|png)
    Fallback:
        a/base.* or any image under char_dir.
    """
    a_dir = char_dir / "a"
    outfits_dir = a_dir / "outfits"
    preferred_names = [
        "Base.webp", "Formal.webp", "Casual.webp",
        "Base.png", "Formal.png", "Casual.png",
    ]

    if outfits_dir.is_dir():
        for name in preferred_names:
            candidate = outfits_dir / name
            if candidate.is_file():
                return candidate
        for p in sorted(outfits_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp"):
                return p

    for ext in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = (a_dir / "base").with_suffix(ext)
        if candidate.is_file():
            return candidate

    for p in char_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
            return p

    raise RuntimeError(f"No representative outfit image found in {char_dir}")

# Tk: scale vs reference

def prompt_for_scale(image_path: Path, user_eye_line_ratio: Optional[float] = None) -> float:
    """Side-by-side scaling UI vs reference_sprites, returns chosen scale."""
    if not REF_SPRITES_DIR.is_dir():
        print(f"[ERROR] No 'reference_sprites' folder found at: {REF_SPRITES_DIR}")
        sys.exit(1)

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
    uw, uh = user_img.size

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Adjust Scale vs Reference")
    root.update_idletasks()

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    CANV_H = max(int(sh * 0.52), 360)
    CANV_W = max(int((sw - 3 * WINDOW_MARGIN) // 2), 360)

    wrap_len = _wraplength_for(int(sw * 0.9))
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
        u_engine_w = uw * u_scale
        u_engine_h = uh * u_scale

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

    _center_and_clamp(root)
    root.mainloop()

    chosen = float(scale_val.get())
    print(f"[INFO] User-picked scale: {chosen:.3f}")
    return chosen

def generate_expression_sheets_for_root(root_folder: Path) -> None:
    """
    Run expression_sheet_maker.py on the given root folder so that expression
    sheets are generated for all characters under it.

    This is used at the end of the pipeline so that a newly created character
    immediately has expression sheets available for Ren'Py scripting.
    """
    if not root_folder.is_dir():
        print(f"[WARN] Not generating expression sheets; '{root_folder}' is not a directory.")
        return

    cmd = [sys.executable, "expression_sheet_maker.py", str(root_folder)]
    try:
        print(f"[INFO] Running expression_sheet_maker.py on: {root_folder}")
        subprocess.run(cmd, check=True)
        print("[INFO] Expression sheets generated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] expression_sheet_maker.py failed: {e}")
    except Exception as e:
        print(f"[WARN] Could not run expression_sheet_maker.py: {e}")

def finalize_character(
    char_dir: Path,
    display_name: str,
    voice: str,
    game_name: Optional[str],
) -> None:
    """Pick eye line, name color, scale, flatten poses, and write character.yml."""
    rep_outfit = pick_representative_outfit(char_dir)

    print("[INFO] Collecting eye line and name color...")
    eye_line, name_color = prompt_for_eye_and_hair(rep_outfit)

    print("[INFO] Collecting scale vs reference...")
    scale = prompt_for_scale(rep_outfit, user_eye_line_ratio=eye_line)

    print("[INFO] Flattening pose/outfit combinations into ST pose letters...")
    final_pose_letters = flatten_pose_outfits_to_letter_poses(char_dir)
    if not final_pose_letters:
        print("[WARN] Flattening produced no poses; using existing letter folders.")
        final_pose_letters = sorted(
            [
                p.name
                for p in char_dir.iterdir()
                if p.is_dir() and len(p.name) == 1 and p.name.isalpha()
            ]
        )

    poses_yaml = {letter: {"facing": "right"} for letter in final_pose_letters}

    yml_path = char_dir / "character.yml"
    write_character_yml(
        yml_path,
        display_name,
        voice,
        eye_line,
        name_color,
        scale,
        poses_yaml,
        game=game_name,
    )

    print(f"=== Finished character: {display_name} ({char_dir.name}) ===")

# Gemini generation helpers

def generate_initial_pose_once(
    api_key: str,
    image_path: Path,
    out_stem: Path,
    gender_style: str,
) -> Path:
    """Normalize the original sprite into pose A with a flat magenta background."""
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
    Generate or regenerate a *single* outfit image for the given key.

    This is used both by the bulk outfit generator and by the
    per-outfit "regenerate" buttons in the review window.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    cfg = outfit_prompt_config.get(outfit_key, {})

    # Special handling for standardized school uniform.
    if outfit_key == "uniform" and cfg.get("use_standard_uniform"):
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

    # Normal text-prompt-based outfit.
    image_b64 = load_image_as_base64(base_pose_path)
    out_stem = outfits_dir / outfit_key.capitalize()
    prompt = build_outfit_prompt(outfit_desc, gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved outfit '{outfit_key}' to: {final_path}")
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

    If outfit_prompt_config['uniform']['use_standard_uniform'] is True,
    the uniform outfit is generated using the standardized school uniform
    reference sprites instead of a CSV/custom prompt.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    # Optional: copy base pose as "Base.png" outfit.
    if include_base_outfit:
        base_bytes = base_pose_path.read_bytes()
        base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
        base_out_path = (outfits_dir / "Base").with_suffix(".png")
        base_img.save(base_out_path, format="PNG", lossless=True)
        paths.append(base_out_path)

    # Generate each selected outfit key.
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

    for f in list(out_dir.iterdir()):
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    neutral_stem = out_dir / "0"
    outfit_img = Image.open(outfit_path).convert("RGBA")
    neutral_path = save_img_webp_or_png(outfit_img, neutral_stem)
    generated_paths.append(neutral_path)
    print(f"  [Expr] Using outfit as neutral '0' -> {neutral_path}")

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
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Neutral 0 is always just the outfit itself; no need to call Gemini.
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

        # First, build the full expression set once.
        generate_expressions_for_single_outfit_once(
            api_key,
            pose_dir,
            outfit_path,
            faces_root,
            expressions_sequence=expressions_sequence,
        )

        # Determine the folder where the expression images for this outfit live.
        if outfit_name.lower() == "base":
            out_dir = faces_root / "face"
        else:
            out_dir = faces_root / outfit_name

        while True:
            # Collect current expression images from disk, sorted by index.
            expr_paths: List[Path] = []
            for p in sorted(out_dir.iterdir(), key=lambda q: q.stem):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".png", ".webp"):
                    continue
                expr_paths.append(p)

            # Ensure numeric ordering (0, 1, 2, ...).
            expr_paths.sort(key=lambda q: int(q.stem))

            infos = [
                (
                    p,
                    f"Pose {pose_label} – {outfit_name} – expression {p.stem} "
                    f"({expressions_sequence[int(p.stem)][0] if int(p.stem) < len(expressions_sequence) else '?'})",
                )
                for p in expr_paths
            ]

            # One "regenerate this expression" button under each expression.
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
                # Wipe and rebuild the whole expression set for this outfit.
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

                # Card index -> expression index: the filename stem.
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
                # Loop to show the updated images.
                continue

def get_reference_images_for_archetype(archetype_label: str) -> List[Path]:
    """
    Return *all* reference sprites for this archetype so Gemini can really
    lock onto the style.

    Preference:
      1) reference_sprites/<archetype_label>/
      2) PNGs directly under reference_sprites/
    """
    paths: List[Path] = []

    arch_dir = REF_SPRITES_DIR / archetype_label
    if arch_dir.is_dir():
        for p in sorted(arch_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                paths.append(p)

    # Fallback: generic refs at top level if the specific folder is empty
    if not paths and REF_SPRITES_DIR.is_dir():
        for p in sorted(REF_SPRITES_DIR.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                paths.append(p)

    return paths

def get_standard_uniform_reference_images(
    gender_style: str,
    max_images: int = 5,
) -> List[Path]:
    """
    Return a small set of reference images for the standardized school uniform.

    For girls (gender_style='f'), this looks in:
        reference_sprites/young_woman_uniform

    For boys (gender_style='m'), this looks in:
        reference_sprites/young_man_uniform
    """
    if gender_style == "m":
        folder_name = "young_man_uniform"
    else:
        folder_name = "young_woman_uniform"

    uniform_dir = REF_SPRITES_DIR / folder_name
    refs: List[Path] = []

    if uniform_dir.is_dir():
        for p in sorted(uniform_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                refs.append(p)
                if len(refs) >= max_images:
                    break

    return refs

def generate_standard_uniform_outfit(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    archetype_label: str,
    outfit_desc: str,  # kept for signature compatibility, not used directly now
) -> Path:
    """
    Generate the standardized school uniform outfit using:

      - The base pose as the main character to keep.
      - A cropped uniform reference image (gender-specific) as visual guidance.
      - A single, archetype- and gender-aware text prompt that describes the
        school uniform in words.

    This keeps the prompting style consistent with the rest of the pipeline
    and avoids the custom 'parts' scheme.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    # Collect the uniform reference image(s) for this gender.
    uniform_refs = get_standard_uniform_reference_images(gender_style)
    if not uniform_refs:
        # If we somehow do not have a uniform ref, fall back to the normal
        # outfit prompt path so 'uniform' still works instead of erroring.
        print("[WARN] No uniform reference found, falling back to normal prompt-based uniform.")
        image_b64 = load_image_as_base64(base_pose_path)
        # Use the normal outfit prompt builder as a last resort.
        prompt = build_outfit_prompt(outfit_desc, gender_style)
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        out_stem = outfits_dir / "Uniform"
        final_path = save_image_bytes_as_png(img_bytes, out_stem)
        print(f"  Saved fallback prompt-based uniform to: {final_path}")
        return final_path

    # Prefer the first uniform reference (you said you're cropping them already).
    uniform_ref = uniform_refs[0]
    print(f"[INFO] Using uniform reference: {uniform_ref}")

    # Build a unified, standardized uniform prompt that matches your style.
    uniform_prompt = build_standard_school_uniform_prompt(
        archetype_label,
        gender_style,
    )

    # Call Gemini with:
    #   - the prompt
    #   - two reference images:
    #       1) the base pose (the character to keep)
    #       2) the cropped uniform example (clothes to copy)
    #
    # The prompt text clearly describes which image is which conceptually.
    img_bytes = call_gemini_text_or_refs(
        api_key,
        uniform_prompt,
        ref_images=[base_pose_path, uniform_ref],
    )

    out_stem = outfits_dir / "Uniform"
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved standardized school uniform to: {final_path}")
    return final_path

def build_prompt_for_idea(concept: str, archetype_label: str, gender_style: str) -> str:
    """Build text prompt used when generating a new character from a concept."""
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
    """
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

# Tk: source mode + prompt entry

def prompt_source_mode() -> str:
    """Ask whether to generate from an image or from a text prompt."""
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Sprite Source")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

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

    _center_and_clamp(root)
    root.mainloop()

    return decision["mode"]

def prompt_character_idea_and_archetype() -> Tuple[str, str, str, str, str]:
    """
    Tk dialog asking for:
      - character concept text
      - voice
      - name
      - archetype (filtered by voice)

    Returns: (concept_text, archetype_label, voice, display_name, gender_style)
    """
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Character Concept")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="Describe the kind of character you want Gemini to design,\n"
        "then choose their voice, name, and archetype.",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    text_frame = tk.Frame(root, bg=BG_COLOR)
    text_frame.grid(row=1, column=0, padx=10, pady=(4, 4), sticky="nsew")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    txt = tk.Text(text_frame, width=60, height=8, wrap="word")
    txt.pack(fill="both", expand=True)

    voice_var = tk.StringVar(value="")
    name_var = tk.StringVar(value="")
    gender_style_var = {"value": None}

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

    _center_and_clamp(root)
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

# Tk: generation options (outfits + expressions)

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
    # Only young woman/man archetypes get the standardized school uniform toggle.
    arch_lower = (archetype_label or "").strip().lower()
    uniform_eligible = (
        (arch_lower == "young woman" and gender_style == "f")
        or (arch_lower == "young man" and gender_style == "m")
    )

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Generation Options")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

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

    # Outfits column
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
        "  - Do NOT mention shoes, boots, socks, or anything on the feet.\n"
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
        # Whether this outfit is selected to be generated at all.
        sel_var = tk.IntVar(value=1 if key in OUTFIT_KEYS else 0)
        outfit_selected_vars[key] = sel_var

        # One mode variable per outfit: "random", "custom", or "standard_uniform".
        mode_var = tk.StringVar(value="random")
        outfit_mode_vars[key] = mode_var

        row_frame = tk.Frame(outfit_frame, bg=BG_COLOR)
        row_frame.grid(row=row_idx, column=0, columnspan=4, sticky="we", pady=2)
        row_idx += 1
        row_frame.grid_columnconfigure(0, weight=0)
        row_frame.grid_columnconfigure(1, weight=0)
        row_frame.grid_columnconfigure(2, weight=0)
        row_frame.grid_columnconfigure(3, weight=1)

        # Checkbox to toggle this outfit on/off.
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
        # when archetype is eligible).
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
        # and mode is "custom").
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
                    # Outfit is not selected: everything is disabled.
                    _rb_random.config(state=tk.DISABLED)
                    _rb_custom.config(state=tk.DISABLED)
                    if _rb_standard is not None:
                        _rb_standard.config(state=tk.DISABLED)
                    _entry.config(state=tk.DISABLED)
                    return

                # Outfit selected: radios enabled.
                _rb_random.config(state=tk.NORMAL)
                _rb_custom.config(state=tk.NORMAL)
                if _rb_standard is not None:
                    _rb_standard.config(state=tk.NORMAL)

                # Entry only enabled in custom mode.
                if _mode_var.get() == "custom":
                    _entry.config(state=tk.NORMAL)
                else:
                    _entry.config(state=tk.DISABLED)

            return _update

        updater = make_update_fn()
        sel_var.trace_add("write", updater)
        mode_var.trace_add("write", updater)
        updater()

    # Expressions column (with vertical scrollbar)
    expr_frame = tk.LabelFrame(
        body_frame,
        text="Expressions (neutral is always included):",
        bg=BG_COLOR,
    )
    expr_frame.grid(row=0, column=1, padx=5, pady=4, sticky="nsew")
    expr_frame.grid_rowconfigure(0, weight=1)
    expr_frame.grid_columnconfigure(0, weight=1)

    # Canvas + scrollbar so a long list of expressions can be scrolled.
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

    # Inner frame that actually holds the labels and checkboxes.
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

    # Now build the actual expression controls inside expr_inner.
    expr_vars: Dict[str, tk.IntVar] = {}
    for key, desc in EXPRESSIONS_SEQUENCE:
        if key == "0":
            # Neutral expression is always generated; show as a label only.
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
                    # Standard uniform path: we ignore prompts and just mark it.
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
                    # Should not happen, but default to random.
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

    _center_and_clamp(root)
    root.mainloop()

    if not decision["ok"]:
        sys.exit(0)

    return (
        decision["outfits"],
        decision["expr_seq"],
        decision["config"],
    )
# Character pipeline (per source image)
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
    """
    print(f"\n=== Processing source image: {image_path.name} ===")

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

    char_folder_name = get_unique_folder_name(output_root, display_name)
    char_dir = output_root / char_folder_name
    char_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output character folder: {char_dir}")

    a_dir = char_dir / "a"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_base_stem = a_dir / "base"

    use_base_as_outfit = True

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

        print("[INFO] Generating outfits for pose A...")

    outfits_dir = a_dir / "outfits"
    outfits_dir.mkdir(parents=True, exist_ok=True)

    # Current text prompts per outfit key; this will be updated when the user
    # asks for a "new random outfit" for a given key.
    current_outfit_prompts = build_outfit_prompts_with_config(
        archetype_label,
        gender_style,
        selected_outfit_keys,
        outfit_db,
        outfit_prompt_config,
    )

    # Initial generation: make Base (if requested) and all outfits once.
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

    # Review loop: the user can regenerate individual outfits as many times
    # as they want without touching the others.
    while True:
        a_out_paths: List[Path] = []
        per_buttons: List[List[Tuple[str, str]]] = []
        index_to_outfit_key: Dict[int, str] = {}

        # Collect current outfit images from disk.
        for p in sorted(outfits_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".png", ".webp"):
                continue

            a_out_paths.append(p)
            stem_lower = p.stem.lower()

            # Try to match this file back to one of the logical outfit keys.
            matched_key: Optional[str] = None
            for key in selected_outfit_keys:
                if key.lower() == stem_lower or key.capitalize().lower() == stem_lower:
                    matched_key = key
                    break

            # Decide which per-card buttons to show.
            btn_list: List[Tuple[str, str]] = []
            if matched_key is not None:
                cfg = outfit_prompt_config.get(matched_key, {})
                # Every logical outfit can be regenerated with the same prompt.
                btn_list.append(("Regenerate same outfit", "same"))
                # The "new random outfit" button should only appear when we are
                # actually using the CSV/random system (and not the standardized
                # school uniform case).
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
            show_global_regenerate=False,  # we now regenerate outfits individually
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
                # The user clicked under something that is not a logical outfit
                # (for example, the copied Base.png). Ignore and redraw.
                continue

            # If they asked for a *new* random outfit, roll a fresh prompt
            # using the same CSV/random logic as the initial generation.
            if action == "new":
                new_prompt_dict = build_outfit_prompts_with_config(
                    archetype_label,
                    gender_style,
                    [outfit_key],
                    outfit_db,
                    outfit_prompt_config,
                )
                current_outfit_prompts[outfit_key] = new_prompt_dict[outfit_key]

            # If they asked for "same", we reuse the existing description in
            # current_outfit_prompts. If that somehow does not exist yet, we
            # fall back to a freshly built one.
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

            # Actually regenerate just this one outfit image.
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

            # Loop again to show the updated outfits.
            continue

        # Safety: if someone ever calls this with show_global_regenerate=True
        # again, we can still support "regenerate_all".
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

    print("[INFO] Generating expressions for pose A (per outfit)...")
    generate_and_review_expressions_for_pose(
        api_key,
        char_dir,
        a_dir,
        "A",
        expressions_sequence=expressions_sequence,
    )

    finalize_character(char_dir, display_name, voice, game_name)

    # After finishing this character, generate expression sheets for all
    # characters under the same output root so sheets are immediately usable.
    generate_expression_sheets_for_root(output_root)

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

    # Ask how we are creating the character: from an image, or from a text prompt.
    mode = prompt_source_mode()

    if mode == "image":
        # Choose the source image via file dialog.
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

        # NEW: optional thigh crop UI before normalization
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

            # We don't have previous crops yet, so pass an empty list for now.
            y_cut, used_gallery = prompt_for_crop(
                src_img,
                prompt_text,
                previous_crops=[],
            )

            # If the user clicked somewhere above the bottom edge, crop. If they
            # clicked at (or beyond) the current bottom, we treat that as "no crop".
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
        # Prompt-generated character path.
        concept, arch_label, voice, display_name, gender_style = (
            prompt_character_idea_and_archetype()
        )

        # Generate and review the initial prompt-based sprite.
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

# CLI entry point

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

    # If no output folder was provided, ask the user via a folder picker.
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