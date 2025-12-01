#!/usr/bin/env python3
"""
gemini_sprite_pipeline.py (refactored + prompt mode + generation options)

Key features:

- Generate character sprites either:
  * From an existing image (file picker), or
  * From a text prompt (Gemini designs a new base sprite).

- For each run, you can choose:
  * Whether to create a normal character sprite folder OR a gender-bent character sprite folder.
  * Which outfits to generate (Base is always included; you choose extra outfits).
  * Which expressions to generate (neutral is always included; you choose others).

- The pipeline for the chosen sprite type (normal or gender-bent) runs once per execution:
  * Pose A: normalized base.
  * Outfits for the pose.
  * Expressions per outfit (expressions are allowed to slightly adjust the pose).
  * Eye line, name color (hair), scale vs reference.
  * Flatten pose+outfit combinations.
  * character.yml.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import shutil
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from rembg import remove, new_session
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import yaml
import webbrowser

from gsp_constants import (
    SCRIPT_DIR,
    CONFIG_PATH,
    OUTFIT_CSV_PATH,
    NAMES_CSV_PATH,
    REF_SPRITES_DIR,
    GEMINI_IMAGE_MODEL,
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
    EXPRESSIONS,
    EXPRESSIONS_SEQUENCE,
    GENDER_ARCHETYPES,
)

# =========================
# Global paths/constants
# =========================
session = new_session("u2net_human_seg")

# =========================
# Helper functions: Tk layout
# =========================

def _compute_display_size(
    screen_w: int,
    screen_h: int,
    img_w: int,
    img_h: int,
    *,
    max_w_ratio: float = 0.90,
    max_h_ratio: float = 0.55,
) -> Tuple[int, int]:
    """
    Compute an image display size that leaves vertical room for labels/buttons.
    Returns (disp_w, disp_h).
    """
    max_w = int(screen_w * max_w_ratio) - 2 * WINDOW_MARGIN
    max_h = int(screen_h * max_h_ratio) - 2 * WINDOW_MARGIN
    scale = min(max_w / img_w, max_h / img_h, 1.0)
    return max(1, int(img_w * scale)), max(1, int(img_h * scale))


def _center_and_clamp(root: tk.Tk) -> None:
    """
    After widgets are laid out, measure requested size and clamp to screen.
    Keeps a small top margin and positions near the top.
    """
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
    """Compute a sensible wraplength for labels given a target width in pixels."""
    return max(200, width_px - WRAP_PADDING)


# =========================
# Name pool and YAML helpers
# =========================

def load_name_pool(csv_path: Path) -> Tuple[List[str], List[str]]:
    """
    Load girl/boy name pools from a CSV file with columns: name, gender.
    Returns: (girl_names, boy_names).
    """
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
    """Pick a random name from the appropriate list based on the selected voice."""
    pool = girl_names if (voice or "").lower() == "girl" else boy_names
    if not pool:
        pool = ["Alex", "Riley", "Taylor", "Jordan"]
    return random.choice(pool)


def get_unique_folder_name(base_path: Path, desired_name: str) -> str:
    """
    Ensure the folder name is unique within base_path by appending a counter:
    "Hannah", "Hannah_2", "Hannah_3", ...
    """
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
    """
    Write final character metadata YAML matching organizer format.
    name_color is the sampled hair color.

    Args:
        path: Output path for character.yml.
        display_name: Character's display name.
        voice: "girl" / "boy" / "male" / "female" string for the organizer.
        eye_line: Relative eye line ratio (0..1).
        name_color: Hex color string (e.g. "#aabbcc").
        scale: In-game scale factor.
        poses: Mapping of pose letter -> metadata dict.
        game: Optional game name.
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

# =========================
# Image helpers and background removal
# =========================

def save_img_webp_or_png(img: Image.Image, dest_stem: Path) -> Path:
    """
    Save an image to disk, preferring WEBP (lossless) and falling back to PNG if WEBP fails.

    Args:
        img: PIL Image to save.
        dest_stem: Path without extension (e.g., char_dir/'a'/'base').

    Returns:
        Actual output path (with .webp or .png).
    """
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
    """
    Save raw image bytes (e.g., from Gemini, already background-stripped)
    as a lossless PNG. This avoids any WebP color drift during the pipeline.

    Args:
        image_bytes: Raw PNG bytes from Gemini or another source.
        dest_stem: Output stem path without extension.

    Returns:
        Path to the saved .png file.
    """
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    out_path = dest_stem.with_suffix(".png")
    img.save(out_path, format="PNG", lossless=True)
    return out_path


def strip_background(image_bytes: bytes) -> bytes:
    """
    Strip a flat-ish magenta background that Gemini uses.

    1) Load the RGBA image.
    2) On the border, only keep pixels close to GBG_COLOR when estimating
       the actual background color.
    3) Treat pixels close to that estimated bg color as background and
       clear their alpha.
    """
    # How far from GBG_COLOR a border pixel can be to *count* as a bg sample.
    BG_SAMPLE_THRESH = 150    # distance in RGB space

    # How far from the *estimated* background color we treat as bg when clearing.
    BG_CLEAR_THRESH = 60

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        w, h = img.size
        pixels = img.load()

        def dist_sq_to_target(r, g, b):
            dr = r - GBG_COLOR[0]
            dg = g - GBG_COLOR[1]
            db = b - GBG_COLOR[2]
            return dr * dr + dg * dg + db * db

        bg_sample_thresh_sq = BG_SAMPLE_THRESH * BG_SAMPLE_THRESH
        bg_samples = []

        # top & bottom rows
        for x in range(w):
            for y in (0, h - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                if dist_sq_to_target(r, g, b) <= bg_sample_thresh_sq:
                    bg_samples.append((r, g, b))

        # left & right columns
        for y in range(h):
            for x in (0, w - 1):
                r, g, b, a = pixels[x, y]
                if a <= 0:
                    continue
                if dist_sq_to_target(r, g, b) <= bg_sample_thresh_sq:
                    bg_samples.append((r, g, b))

        if not bg_samples:
            # no magenta-ish pixels on the border, fall back to ideal magenta
            bg_r, bg_g, bg_b = GBG_COLOR
        else:
            sr = sum(r for r, _, _ in bg_samples)
            sg = sum(g for _, g, _ in bg_samples)
            sb = sum(b for _, _, b in bg_samples)
            n = float(len(bg_samples))
            bg_r = sr / n
            bg_g = sg / n
            bg_b = sb / n

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
                if is_bg(r, g, b):
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
    """
    Load an image from disk, re-encode it as PNG in memory, and return base64-encoded bytes.

    Args:
        path: File path to load.

    Returns:
        Base64-encoded PNG bytes as a string.
    """
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG", lossless=True)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8")


# =========================
# Gemini configuration and HTTP helper
# =========================

def load_config() -> dict:
    """Load ~/.st_gemini_config.json if present, else return an empty dict."""
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
    """Prompt the user to obtain and paste a Gemini API key, then save it."""
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
    """
    Fetch the Gemini API key from environment or config,
    or run interactive setup on first use.
    """
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    cfg = load_config()
    if cfg.get("api_key"):
        return cfg["api_key"]

    return interactive_api_key_setup()


def _extract_inline_image_from_response(data: dict) -> Optional[bytes]:
    """
    Internal helper to pull the first inline image bytes from a Gemini JSON response.

    Args:
        data: Parsed JSON from Gemini.

    Returns:
        Raw PNG bytes or None.
    """
    candidates = data.get("candidates", [])
    for cand in candidates:
        content = cand.get("content", {})
        for part in content.get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and "data" in blob:
                return base64.b64decode(blob["data"])
    return None


def call_gemini_image_edit(api_key: str, prompt: str, image_b64: str) -> bytes:
    """
    Call the Gemini image model with a text prompt and a single input image.

    Includes a small retry loop to handle transient failures.

    Args:
        api_key: Gemini API key.
        prompt: Text instructions describing the edit.
        image_b64: Base64-encoded PNG image to edit.

    Returns:
        Processed PNG bytes (with background stripped).
    """
    parts: List[dict] = [
        {"text": prompt},
        {
            "inline_data": {
                "mime_type": "image/png",
                "data": image_b64,
            }
        },
    ]

    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            if not resp.ok:
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    print(
                        f"[WARN] Gemini API error {resp.status_code} on attempt {attempt}; "
                        "retrying..."
                    )
                    last_error = f"Gemini API error {resp.status_code}: {resp.text}"
                    continue
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_bytes = _extract_inline_image_from_response(data)
            if raw_bytes is not None:
                return strip_background(raw_bytes)

            last_error = "No image data found in Gemini response."
            if attempt < max_retries:
                print(
                    f"[WARN] Gemini returned no image data on attempt {attempt}; "
                    "retrying..."
                )
                continue
            raise RuntimeError(last_error)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"[WARN] Gemini call failed on attempt {attempt}; retrying: {e}")
                continue
            raise RuntimeError(
                f"Gemini call failed after {max_retries} attempts: {last_error}"
            )


def call_gemini_text_or_refs(
    api_key: str,
    prompt: str,
    ref_images: Optional[List[Path]] = None,
) -> bytes:
    """
    Call Gemini with a pure text prompt plus optional reference images.

    Used for "generate from prompt" mode.

    Args:
        api_key: Gemini API key.
        prompt: Full text prompt for the character concept.
        ref_images: Optional list of image paths to use as style references.

    Returns:
        PNG bytes (with background stripped) from the first candidate.
    """
    parts: List[dict] = [{"text": prompt}]

    if ref_images:
        for path in ref_images:
            try:
                img = Image.open(path).convert("RGBA")
            except Exception as e:
                print(f"[WARN] Skipping reference sprite {path}: {e}")
                continue
            buf = BytesIO()
            img.save(buf, format="PNG", lossless=True)
            data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": data_b64,
                    }
                }
            )

    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            if not resp.ok:
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    print(
                        f"[WARN] Gemini API error {resp.status_code} on attempt {attempt}; "
                        "retrying..."
                    )
                    last_error = f"Gemini API error {resp.status_code}: {resp.text}"
                    continue
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_bytes = _extract_inline_image_from_response(data)
            if raw_bytes is not None:
                return strip_background(raw_bytes)

            last_error = "No image data found in Gemini response for text+refs."
            if attempt < max_retries:
                print(
                    f"[WARN] Gemini returned no image data on attempt {attempt}; "
                    "retrying..."
                )
                continue
            raise RuntimeError(last_error)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(f"[WARN] Gemini call failed on attempt {attempt}; retrying: {e}")
                continue
            raise RuntimeError(
                f"Gemini call failed after {max_retries} attempts: {last_error}"
            )


# =========================
# Outfit prompts and expression descriptions
# =========================


def archetype_to_gender_style(archetype_label: str) -> str:
    """
    Given an archetype label ("young woman", "adult man", etc.),
    return its gender style code "f" or "m". Defaults to "f".
    """
    for lbl, g in GENDER_ARCHETYPES:
        if lbl == archetype_label:
            return g
    return "f"


def load_outfit_prompts(csv_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load outfit prompts from CSV: archetype,outfit_key,prompt.

    Returns:
        {archetype: {outfit_key: [prompt, ...]}, ...}
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

    For each outfit_key in selected_outfit_keys:
      - If outfit_prompt_config[key]["use_random"] is True:
          * pick a random prompt from outfit_prompts.csv if available
          * otherwise use a generic fallback description.
      - Else:
          * use outfit_prompt_config[key]["custom_prompt"] directly.

    Args:
        archetype_label: Archetype string (e.g. "young woman").
        gender_style: "f" or "m".
        selected_outfit_keys: Which outfits to generate this run.
        outfit_db: Loaded CSV outfit prompt database.
        outfit_prompt_config:
            {
              "formal": {"use_random": True/False, "custom_prompt": str or None},
              ...
            }

    Returns:
        Dict of outfit_key -> prompt string.
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
            # Custom prompt must be non-empty by the time we get here.
            if not custom_prompt:
                prompts[key] = build_simple_outfit_description(key, gender_style)
            else:
                prompts[key] = custom_prompt

    return prompts


def choose_outfit_prompts_for_archetype(
    archetype_label: str,
    gender_style: str,
    outfit_keys: List[str],
    outfit_db: Dict[str, Dict[str, List[str]]],
) -> Dict[str, str]:
    """
    For a given archetype and gender, choose one prompt per outfit_key.
    If CSV has prompts, pick one at random. Else use a generic description.

    Args:
        archetype_label: Archetype string.
        gender_style: "f" or "m".
        outfit_keys: Outfit keys to generate (formal, casual, etc.).
        outfit_db: Loaded CSV database.

    Returns:
        Dict of outfit_key -> prompt string.
    """
    prompts: Dict[str, str] = {}
    archetype_pool = outfit_db.get(archetype_label, {})

    for key in outfit_keys:
        candidates = archetype_pool.get(key)
        if candidates:
            prompts[key] = random.choice(candidates)
        else:
            prompts[key] = build_simple_outfit_description(key, gender_style)

    return prompts


def flatten_pose_outfits_to_letter_poses(char_dir: Path) -> List[str]:
    """
    Flatten pose/outfit combinations into separate ST poses with single outfits.

    Input layout:
        <char>/a/outfits/Base.webp, Formal.webp, Casual.webp, ...
        <char>/a/faces/face/*.webp        (Base expressions)
        <char>/a/faces/Formal/*.webp      (Formal expressions)
        <char>/a/faces/Casual/*.webp      (Casual expressions)

    Output layout:
        <char>/a/
            outfits/Base.webp (transparent outfit)
            faces/face/0.webp ... N.webp

        <char>/b/
            outfits/Formal.webp
            faces/face/0.webp ... N.webp

        etc.

    Returns:
        Sorted list of final pose letters.
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


# =========================
# Gemini prompt builders
# =========================

def build_initial_pose_prompt(gender_style: str) -> str:
    """Prompt to normalize the original sprite (mid-thigh, magenta background)."""
    return (
        "Crop the image so we only see from the character's mid-thigh on up."
        "Do not change the style of the character."
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of the background color on them."
        "Crop the character at the midpoint of the thigh."
    )


def build_genderbend_pose_prompt(
    original_gender_style: str,
    target_gender_style: str,
    target_archetype: str,
) -> str:
    """Prompt to generate a gender-bent version of the character with a new archetype."""
    original_clause = "feminine" if original_gender_style == "f" else "masculine"
    target_clause = "feminine" if target_gender_style == "f" else "masculine"

    if target_gender_style == "f":
        pose_examples = (
            f"a clearly different cute or cool pose that suits a {target_archetype} "
            "and would fit well in a visual novel. Make sure the character is not holding anything."
        )
    else:
        pose_examples = (
            f"a clearly different relaxed or friendly pose that suits a {target_archetype} "
            "and would fit well in a visual novel. Make sure the character is not holding anything."
        )

    return (
        "Edit the input visual novel sprite, in the same art style. "
        f"Transform this {original_clause} character into a gender-bent {target_archetype} version of themselves, "
        f"clearly presenting in a {target_clause} way, while keeping their recognizable traits: similar face shape, "
        "hair color, eye color, and general vibe. "
        f"Put them in {pose_examples} "
        "Go ahead and change the hair style and hair length to better match the new pose and gender presentation, but keep the hair recognizably related to the original design. "
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair "
        "have none of the background color on them."
        "Do not change the crop from the mid-thigh up, or the image size."
    )


def build_expression_prompt(expression_desc: str) -> str:
    """Prompt to change only the facial expression, pixel-aligned with the input."""
    return (
        "Edit the input visual novel sprite in the same art style. "
        f"Change the facial expression to match this description: {expression_desc}. "
        "Keep the hair volume, hair outlines, and the hair style, all the exact same. "
        "Do not change the hairstyle, crop from the mid-thigh up, image size, lighting, or background. "
        "Change the pose of the character, based upon the expression we are making. "
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair "
        "have none of the background color on them."
        "do not have the arms, hair, or hands extending outside the frame."
    )


def build_outfit_prompt(base_outfit_desc: str, gender_style: str) -> str:
    """Prompt to change only the clothing to base_outfit_desc on the given pose."""
    gender_clause = "girl" if gender_style == "f" else "boy"
    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Please change the clothing, pose, hair style, and outfit to match this description: {base_outfit_desc}. "
        "Do not change the body proportions, hair length, crop from the mid-thigh up, or image size. "
        "Change the hair style to match the outfit, but do not change the hair length. "
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair "
        "have none of the background color on them. "
        "Do not change the body, chest, and hip proportions to be different from the original. "
    )


# =========================
# Tk UI: voice + archetype + name
# =========================

def prompt_voice_archetype_and_name(image_path: Path) -> Tuple[str, str, str, str]:
    """
    Tk window to:
      - Show the source character image.
      - Pick voice (Girl/Boy).
      - Auto-assign random name (editable).
      - Pick archetype from the subset matching that voice.

    Returns:
        (voice, display_name, archetype_label, gender_style).
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
        if labels:
            archetype_var.set(labels[0])
        else:
            archetype_var.set("")
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
        decision["done"] = True
        decision["voice"] = v
        decision["name"] = nm
        decision["arch"] = arch
        decision["gstyle"] = gs
        root.destroy()

    def on_cancel():
        sys.exit(0)

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


def prompt_sprite_type() -> str:
    """
    Tk dialog asking what type of sprite we are creating this run:

      - "normal":     normal character sprite folder
      - "genderbend": gender-bent character sprite folder only
      - "fusion":     fusion character (placeholder for now)

    Only one option can be selected at a time (radio buttons).

    Returns:
        One of: "normal", "genderbend", or "fusion".
    """
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Sprite Type")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text=(
            "What kind of sprite do you want to create this run?\n\n"
            "You can either:\n"
            "  • Create a normal character.\n"
            "  • Create a gender-bent version only.\n"
            "  • (Future) Create a fusion character, built from two inputs."
        ),
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    mode_var = tk.StringVar(value="normal")

    options_frame = tk.Frame(root, bg=BG_COLOR)
    options_frame.grid(row=1, column=0, padx=10, pady=(4, 8), sticky="w")

    # Normal sprite
    tk.Radiobutton(
        options_frame,
        text="Normal character",
        variable=mode_var,
        value="normal",
        bg=BG_COLOR,
        anchor="w",
        justify="left",
        wraplength=wrap_len,
    ).pack(anchor="w", padx=6, pady=2)

    # Gender-bent only
    tk.Radiobutton(
        options_frame,
        text="Gender-bent character only (no normal folder for this run)",
        variable=mode_var,
        value="genderbend",
        bg=BG_COLOR,
        anchor="w",
        justify="left",
        wraplength=wrap_len,
    ).pack(anchor="w", padx=6, pady=2)

    # Fusion placeholder
    tk.Radiobutton(
        options_frame,
        text="Fusion character (placeholder – combines two characters, not implemented yet)",
        variable=mode_var,
        value="fusion",
        bg=BG_COLOR,
        anchor="w",
        justify="left",
        wraplength=wrap_len,
    ).pack(anchor="w", padx=6, pady=2)

    decision = {"ok": False, "mode": "normal"}

    def on_ok():
        decision["ok"] = True
        decision["mode"] = mode_var.get()
        root.destroy()

    def on_cancel():
        sys.exit(0)

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

    return decision["mode"]


def prompt_genderbend_archetype(original_gender_style: str) -> Tuple[str, str]:
    """
    Tk dialog to choose a gender-bent archetype from the opposite gender group.

    Args:
        original_gender_style: "f" or "m".

    Returns:
        (gender_bent_archetype_label, gender_bent_gender_style).
    """
    target_style = "f" if original_gender_style == "m" else "m"
    options = [label for (label, g) in GENDER_ARCHETYPES if g == target_style]

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Gender-Bent Archetype")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="Choose how the gender-bent version of this character should present.",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    arch_var = tk.StringVar(value=options[0] if options else "")
    tk.OptionMenu(root, arch_var, *options).grid(row=1, column=0, pady=(4, 4))

    decision = {"ok": False, "arch": None}

    def on_ok():
        decision["ok"] = True
        decision["arch"] = arch_var.get()
        root.destroy()

    def on_cancel():
        sys.exit(0)

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=2, column=0, pady=(8, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(side=tk.LEFT, padx=10)

    _center_and_clamp(root)
    root.mainloop()

    if not decision["ok"] or not decision["arch"]:
        sys.exit(0)

    return decision["arch"], target_style


# =========================
# Tk UI: generic review window
# =========================

def review_images_for_step(
    image_infos: List[Tuple[Path, str]],
    title_text: str,
    body_text: str,
) -> str:
    """
    Show a horizontally scrollable strip of image thumbnails and ask the user to:
      - Accept
      - Regenerate
      - Cancel

    Args:
        image_infos: List of (Path, caption).
        title_text: Window title text.
        body_text: Instruction text.

    Returns:
        "accept", "regenerate", or "cancel".
    """
    decision = {"choice": "cancel"}

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
        ).pack(pady=(2, 0))

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

    def regenerate() -> None:
        decision["choice"] = "regenerate"
        root.destroy()

    def cancel() -> None:
        decision["choice"] = "cancel"
        root.destroy()

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=3, column=0, pady=(6, 10))
    tk.Button(btns, text="Accept", width=20, command=accept).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Regenerate", width=20, command=regenerate).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=20, command=cancel).pack(side=tk.LEFT, padx=10)

    _center_and_clamp(root)
    root.mainloop()
    return decision["choice"]


def review_initial_base_pose(
    base_pose_path: Path,
) -> Tuple[str, bool]:
    """
    Tk dialog to review the normalized base pose and decide:

      - Accept / Regenerate / Cancel
      - Whether to treat this base pose as a 'Base' outfit.

    Returns:
        (choice, use_as_outfit)

        choice in {"accept", "regenerate", "cancel"}
        use_as_outfit: True if user wants this saved as Base.png
    """
    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Review Normalized Base Pose")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    # Title / instructions
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
    # keep it modestly sized on screen
    max_size = int(min(sw, sh) * 0.4)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    # keep reference so Tk doesn't GC it
    root._base_preview_img = img_tk  # type: ignore[attr-defined]

    tk.Label(preview_frame, image=img_tk, bg=BG_COLOR).pack()

    # Checkbox: keep as Base outfit
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

    # Buttons
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
        decision["choice"] = "cancel"
        decision["use_as_outfit"] = bool(use_as_outfit_var.get())
        root.destroy()

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=3, column=0, pady=(6, 10))

    tk.Button(btns, text="Accept", width=16, command=on_accept).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btns, text="Regenerate", width=16, command=on_regenerate).pack(
        side=tk.LEFT, padx=10
    )
    tk.Button(btns, text="Cancel and Exit", width=16, command=lambda: sys.exit(0)).pack(
        side=tk.LEFT, padx=10
    )

    _center_and_clamp(root)
    root.mainloop()

    return decision["choice"], decision["use_as_outfit"]


# =========================
# Tk UI: eye line + name color (hair color)
# =========================

def prompt_for_eye_and_hair(image_path: Path) -> Tuple[float, str]:
    """
    Tk UI to choose:
      - Eye line (click once, as a height ratio).
      - Hair color (click once to sample RGB) -> used as name_color.

    Returns:
        (eye_line_ratio, name_color_hex).
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
    Choose a full-body outfit image to use for eye-line and scale selection:

    Preference:
        a/outfits/Base.webp, Formal.webp, Casual.webp
    Fallback:
        a/base.webp or any image found under char_dir.

    Args:
        char_dir: Character root directory.

    Returns:
        Path to an outfit image.
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


# =========================
# Tk UI: scale vs reference
# =========================

def prompt_for_scale(image_path: Path, user_eye_line_ratio: Optional[float] = None) -> float:
    """
    Side-by-side scaling UI using reference_sprites:

    - Left: reference in-game size (scale from its YAML).
    - Right: user's sprite at adjustable in-game scale (slider).
    - Optional eye-line guide on user sprite.

    Args:
        image_path: User sprite image path.
        user_eye_line_ratio: Optional eye-line ratio to draw as a guide.

    Returns:
        Chosen scale float.
    """
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


def finalize_character(
    char_dir: Path,
    display_name: str,
    voice: str,
    game_name: Optional[str],
) -> None:
    """
    Shared finalization step for a character:
      - Pick representative outfit.
      - Eye line + name_color.
      - Scale vs reference.
      - Flatten pose/outfit combos into ST pose letters.
      - Write character.yml.
    """
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


# =========================
# Gemini generation helpers (single-shot)
# =========================

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
        image_path: Source image for the character.
        out_stem: Output stem for the base pose.
        gender_style: "f" or "m".

    Returns:
        Path to the saved normalized base pose.
    """
    print("  [Gemini] Normalizing base pose...")
    image_b64 = load_image_as_base64(image_path)
    prompt = build_initial_pose_prompt(gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved base pose to: {final_path}")
    return final_path


def generate_genderbend_pose_once(
    api_key: str,
    base_image_path: Path,
    out_stem: Path,
    original_gender_style: str,
    target_gender_style: str,
    target_archetype: str,
) -> Path:
    """
    Generate a gender-bent pose A from a normalized base pose.

    Args:
        api_key: Gemini API key.
        base_image_path: Normalized pose A for the original character.
        out_stem: Output stem for the gender-bent pose A.
        original_gender_style: "f" or "m" for the source character.
        target_gender_style: "f" or "m" for the gender-bent character.
        target_archetype: Archetype label for the gender-bent character.

    Returns:
        Path to the gender-bent pose image.
    """
    image_b64 = load_image_as_base64(base_image_path)
    prompt = build_genderbend_pose_prompt(original_gender_style, target_gender_style, target_archetype)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)

    print(f"  Saved gender-bent pose to: {final_path}")
    return final_path


def build_fusion_prompt(target_archetype: str) -> str:
    """
    Build the text prompt used when fusing two existing characters
    into a new one of a chosen archetype.

    We rely on the two input images as visual references.
    """
    gender_style = archetype_to_gender_style(target_archetype)
    gender_word = "girl" if gender_style == "f" else "boy"

    return (
        "Can you make a fusion of these two visual novel characters, where the resulting character has facial, hair, "
        "and body features from both of these inputted characters? The resulting character should end up as a "
        f"{target_archetype}, but should still have clear character features from both of them."
        f"Give the character a new casual outfit that makes sense for a {target_archetype}"
        "Crop the character from the mid-thigh up, facing mostly toward the viewer in a friendly, neutral base pose that works as a sprite."
        "Use a pure, flat magenta background (#FF00FF) behind the character, and make sure the character, outfit, and hair have none of that magenta background color on them."
    )

def generate_fusion_base_once(
    api_key: str,
    parent1_base_path: Path,
    parent2_base_path: Path,
    out_stem: Path,
    target_archetype: str,
) -> Path:
    """
    Use Gemini + the two parent base sprites to generate a single fused
    base sprite for the new character.

    Args:
        api_key: Gemini API key.
        parent1_base_path: Normalized base pose image for parent 1.
        parent2_base_path: Normalized base pose image for parent 2.
        out_stem: Output stem for the fused base pose.
        target_archetype: Archetype label for the resulting character.

    Returns:
        Path to the saved fused base pose.
    """
    prompt = build_fusion_prompt(target_archetype)

    # Use the two parent bases as references
    refs = [parent1_base_path, parent2_base_path]

    print("  [Gemini] Generating fused base sprite...")
    img_bytes = call_gemini_text_or_refs(api_key, prompt, refs)
    final_path = save_image_bytes_as_png(img_bytes, out_stem)
    print(f"  Saved fused base pose to: {final_path}")
    return final_path


def prompt_fusion_target_metadata(
    parent1_base_path: Path,
    parent2_base_path: Path,
) -> Tuple[str, str, str]:
    """
    Tk dialog for fusion setup:

      - Shows previews of the two parent base sprites.
      - Asks for:
          * Fusion voice (Girl / Boy),
          * Fusion name (auto-filled, editable),
          * Fusion archetype (filtered by voice).

    Returns:
        (fusion_display_name, fusion_voice, fusion_archetype_label)
    """
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)

    root = tk.Tk()
    root.configure(bg=BG_COLOR)
    root.title("Fusion Setup")

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    wrap_len = _wraplength_for(int(sw * 0.9))

    tk.Label(
        root,
        text="Review the two parent characters and choose the settings\n"
             "for the resulting fusion character.",
        font=TITLE_FONT,
        bg=BG_COLOR,
        wraplength=wrap_len,
        justify="center",
    ).grid(row=0, column=0, padx=10, pady=(10, 6), sticky="we")

    # Parent previews
    preview_frame = tk.Frame(root, bg=BG_COLOR)
    preview_frame.grid(row=1, column=0, padx=10, pady=(4, 4))

    def _load_preview(path: Path, max_size: int = 280) -> ImageTk.PhotoImage:
        img = Image.open(path).convert("RGBA")
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    img1_tk = _load_preview(parent1_base_path)
    img2_tk = _load_preview(parent2_base_path)

    # Keep references so Tk doesn't garbage collect them
    root._fusion_img1 = img1_tk  # type: ignore[attr-defined]
    root._fusion_img2 = img2_tk  # type: ignore[attr-defined]

    lf1 = tk.LabelFrame(preview_frame, text="Parent 1", bg=BG_COLOR)
    lf1.pack(side=tk.LEFT, padx=10)
    tk.Label(lf1, image=img1_tk, bg=BG_COLOR).pack()

    lf2 = tk.LabelFrame(preview_frame, text="Parent 2", bg=BG_COLOR)
    lf2.pack(side=tk.LEFT, padx=10)
    tk.Label(lf2, image=img2_tk, bg=BG_COLOR).pack()

    # Fusion metadata (voice, name, archetype)
    meta_frame = tk.Frame(root, bg=BG_COLOR)
    meta_frame.grid(row=2, column=0, padx=10, pady=(4, 4), sticky="we")

    voice_var = tk.StringVar(value="")
    name_var = tk.StringVar(value="")
    gender_style_var = {"value": None}

    tk.Label(
        meta_frame,
        text="Fusion voice:",
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
        meta_frame, text="Girl", width=10, command=lambda: set_voice("girl")
    ).grid(row=0, column=1, padx=4, pady=2, sticky="w")
    tk.Button(
        meta_frame, text="Boy", width=10, command=lambda: set_voice("boy")
    ).grid(row=0, column=2, padx=4, pady=2, sticky="w")

    tk.Label(
        meta_frame,
        text="Fusion name:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).grid(row=1, column=0, padx=(0, 6), pady=2, sticky="w")

    name_entry = tk.Entry(meta_frame, textvariable=name_var, width=28)
    name_entry.grid(row=1, column=1, columnspan=2, padx=4, pady=2, sticky="w")

    arch_frame = tk.Frame(root, bg=BG_COLOR)
    arch_frame.grid(row=3, column=0, pady=(4, 4))

    tk.Label(
        arch_frame,
        text="Fusion archetype:",
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

        if labels:
            arch_var.set(labels[0])
        else:
            arch_var.set("")

        for lbl in labels:
            menu.add_command(label=lbl, command=lambda v=lbl: arch_var.set(v))

    decision = {
        "ok": False,
        "name": "",
        "voice": "",
        "arch": "",
    }

    def on_ok():
        v = voice_var.get()
        nm = name_var.get().strip()
        arch = arch_var.get()
        gs = gender_style_var["value"]

        if not v or not arch or not gs:
            messagebox.showerror(
                "Missing data",
                "Please choose a voice and archetype for the fusion.",
            )
            return
        if not nm:
            nm = _pick_random_name_for_voice(v)

        decision["ok"] = True
        decision["name"] = nm
        decision["voice"] = v
        decision["arch"] = arch
        root.destroy()

    def on_cancel():
        sys.exit(0)

    btns = tk.Frame(root, bg=BG_COLOR)
    btns.grid(row=4, column=0, pady=(6, 10))
    tk.Button(btns, text="OK", width=16, command=on_ok).pack(side=tk.LEFT, padx=10)
    tk.Button(btns, text="Cancel and Exit", width=16, command=on_cancel).pack(
        side=tk.LEFT, padx=10
    )

    _center_and_clamp(root)
    root.mainloop()

    if not decision["ok"]:
        sys.exit(0)

    return decision["name"], decision["voice"], decision["arch"]


def generate_outfits_once(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    outfit_descriptions: Dict[str, str],
    include_base_outfit: bool = True,
) -> List[Path]:
    """
    Generate outfits for a pose.

    Layout:
      - If include_base_outfit=True:
          * Copies base pose as Base.png.
      - For each outfit_descriptions[key], generate <Key>.png.

    Returns:
        List of created image paths (including Base if requested).
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    # Optionally keep the normalized base as a 'Base' outfit
    if include_base_outfit:
        base_bytes = base_pose_path.read_bytes()
        base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
        base_out_path = (outfits_dir / "Base").with_suffix(".png")
        base_img.save(base_out_path, format="PNG", lossless=True)
        paths.append(base_out_path)

    # The normalization pose is still used as the *reference* image
    # for Gemini outfit edits regardless.
    image_b64 = load_image_as_base64(base_pose_path)

    for key, desc in outfit_descriptions.items():
        out_stem = outfits_dir / key.capitalize()
        prompt = build_outfit_prompt(desc, gender_style)
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        final_path = save_image_bytes_as_png(img_bytes, out_stem)

        print(f"  Saved outfit '{key}' to: {final_path}")
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

    Layout produced (for pose 'a' and outfit 'Base'):
        a/outfits/Base.png
        a/faces/face/0.webp ... N.webp  (if outfit_name == 'Base')
    Or for non-base outfits (e.g. 'Formal'):
        a/faces/Formal/0.webp ... N.webp

    0.webp is always the neutral outfit image itself.
    1.webp, 2.webp, ... are generated expressions.

    Args:
        api_key: Gemini API key.
        pose_dir: Pose directory (e.g. char_dir / "a").
        outfit_path: Path to the outfit image (Base/Formal/...).
        faces_root: Root directory for faces under the pose.
        expressions_sequence: Ordered list of (key, description) to generate.
                              The first entry should be neutral, but only the
                              non-neutral entries are sent to Gemini.

    Returns:
        List of expression image paths, including 0.webp neutral.
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

    # 0.webp = neutral (the outfit image itself)
    neutral_stem = out_dir / "0"
    outfit_img = Image.open(outfit_path).convert("RGBA")
    neutral_path = save_img_webp_or_png(outfit_img, neutral_stem)
    generated_paths.append(neutral_path)
    print(f"  [Expr] Using outfit as neutral '0' -> {neutral_path}")

    # 1..N = generated expressions
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


def generate_and_review_expressions_for_pose(
    api_key: str,
    char_dir: Path,
    pose_dir: Path,
    pose_label: str,
    expressions_sequence: List[Tuple[str, str]],
) -> None:
    """
    For a given pose directory (e.g., 'a'), iterate each outfit and:

      - Generate its full expression set using expressions_sequence.
      - Show review window for just that outfit.
      - Allow Accept / Regenerate / Cancel at outfit level.

    This works for both primary and gender-bent characters.

    Args:
        api_key: Gemini API key.
        char_dir: Character root directory.
        pose_dir: Pose directory (e.g. char_dir/"a").
        pose_label: String label for UI ("A", "GB-A", etc.).
        expressions_sequence: Ordered list of expressions (including neutral).
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

        while True:
            expr_paths = generate_expressions_for_single_outfit_once(
                api_key,
                pose_dir,
                outfit_path,
                faces_root,
                expressions_sequence=expressions_sequence,
            )

            infos = [
                (
                    p,
                    f"Pose {pose_label} – {outfit_name} – {p.relative_to(char_dir)}",
                )
                for p in expr_paths
            ]

            choice = review_images_for_step(
                infos,
                f"Review Expressions for Pose {pose_label} – {outfit_name}",
                "These expressions are generated for this single pose/outfit.\n"
                "Accept them, regenerate, or cancel.",
            )

            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)


def get_reference_images_for_archetype(archetype_label: str, max_images: int = 7) -> List[Path]:
    """
    Choose a small set of reference sprites to show Gemini the art style.

    Preference:
      1) Images from reference_sprites/<archetype_label>/ if that folder exists.
      2) Otherwise, PNGs directly under reference_sprites/.

    Args:
        archetype_label: Archetype string.
        max_images: Maximum number of references to include.

    Returns:
        List of paths to reference images.
    """
    paths: List[Path] = []

    arch_dir = REF_SPRITES_DIR / archetype_label
    if arch_dir.is_dir():
        for p in sorted(arch_dir.iterdir()):
            if p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg"):
                paths.append(p)
                if len(paths) >= max_images:
                    break

    if not paths and REF_SPRITES_DIR.is_dir():
        for p in sorted(REF_SPRITES_DIR.iterdir()):
            if p.suffix.lower() == ".png":
                paths.append(p)
                if len(paths) >= max_images:
                    break

    return paths


def build_prompt_for_idea(concept: str, archetype_label: str, gender_style: str) -> str:
    """
    Build the text prompt used when generating a brand new character from a concept.

    Args:
        concept: Free-text description of the character.
        archetype_label: Archetype label.
        gender_style: "f" or "m".

    Returns:
        Full text prompt string.
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


def generate_initial_character_from_prompt(
    api_key: str,
    concept: str,
    archetype_label: str,
    output_root: Path,
) -> Path:
    """
    Use Gemini + reference sprites to generate a brand new base character image
    from a text concept.

    Output path:
        <output_root>/_prompt_sources/<slug>.png

    Args:
        api_key: Gemini API key.
        concept: Character idea text.
        archetype_label: Archetype string.
        output_root: Root output directory.

    Returns:
        Path to the saved source sprite image.
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


# =========================
# Tk UI: source mode + prompt entry
# =========================

def prompt_source_mode() -> str:
    """
    Tk dialog asking whether to generate from an image or from a text prompt.

    Returns:
        "image"  -> user wants to pick an image file.
        "prompt" -> user wants to describe a character concept.
    """
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
        sys.exit(0)

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
      - A free-text character concept.
      - Voice (Girl / Boy).
      - A name (auto-filled, editable).
      - An archetype label (filtered by voice).

    Returns:
        (concept_text, archetype_label, voice, display_name, gender_style).
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

        if labels:
            arch_var.set(labels[0])
        else:
            arch_var.set("")

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

        decision["ok"] = True
        decision["concept"] = concept
        decision["archetype"] = arch
        decision["voice"] = v
        decision["name"] = nm
        decision["gstyle"] = gs
        root.destroy()

    def on_cancel():
        sys.exit(0)

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


# =========================
# Tk UI: generation options (outfits + expressions)
# =========================

def prompt_outfits_and_expressions() -> Tuple[
    List[str],
    List[Tuple[str, str]],
    Dict[str, Dict[str, Optional[str]]],
]:
    """
    Tk dialog asking which outfits and expressions to generate, and for each
    outfit whether to use a random prompt from the CSV or a custom prompt.

    Per-outfit controls:
      - Checkbox: generate this outfit type at all.
      - If generated:
          * Radio: "Random" (from outfit_prompts.csv / fallback),
          * Radio: "Custom" (use text entered in the prompt field).

    Notes:
      - Base outfit is always included for each pose regardless of selections.
      - Neutral expression (0) is always included.
      - This function only controls the *non-base* outfits.

    Returns:
        (
          selected_outfit_keys,
          expressions_sequence,
          outfit_prompt_config
        )

        where:
          selected_outfit_keys: list of outfit_keys like ["formal", "casual"].
          expressions_sequence: ordered list of (key, description),
                                including "0" as the first entry.
          outfit_prompt_config: dict keyed by outfit_key, e.g.:
              {
                "formal": {
                    "use_random": True/False,
                    "custom_prompt": str or None,
                },
                ...
              }
    """
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

    # ======================================================
    # Outfits column: per-outfit toggle + random/custom + text
    # ======================================================
    outfit_frame = tk.LabelFrame(
        body_frame,
        text="Additional outfits (Base is always included):",
        bg=BG_COLOR,
    )
    outfit_frame.grid(row=0, column=0, padx=5, pady=4, sticky="nsew")
    outfit_frame.grid_columnconfigure(0, weight=0)
    outfit_frame.grid_columnconfigure(1, weight=0)
    outfit_frame.grid_columnconfigure(2, weight=1)

    # Per-outfit state
    outfit_selected_vars: Dict[str, tk.IntVar] = {}
    outfit_mode_vars: Dict[str, tk.StringVar] = {}
    outfit_prompt_entries: Dict[str, tk.Entry] = {}

    # Hint label for all outfits
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
    ).grid(row=0, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 6))

    # Start rows after the hint
    row_idx = 1

    for key in ALL_OUTFIT_KEYS:
        # Whether this outfit is generated at all
        sel_var = tk.IntVar(value=1 if key in OUTFIT_KEYS else 0)
        outfit_selected_vars[key] = sel_var

        # Random vs custom mode for this outfit
        mode_var = tk.StringVar(value="random")
        outfit_mode_vars[key] = mode_var

        # Row frame
        row_frame = tk.Frame(outfit_frame, bg=BG_COLOR)
        row_frame.grid(row=row_idx, column=0, columnspan=3, sticky="we", pady=2)
        row_idx += 1
        row_frame.grid_columnconfigure(0, weight=0)
        row_frame.grid_columnconfigure(1, weight=0)
        row_frame.grid_columnconfigure(2, weight=1)

        # Checkbox: include this outfit
        chk = tk.Checkbutton(
            row_frame,
            text=key.capitalize(),
            variable=sel_var,
            bg=BG_COLOR,
            anchor="w",
        )
        chk.grid(row=0, column=0, padx=(6, 4), sticky="w")

        # Radio buttons: random vs custom
        rb_random = tk.Radiobutton(
            row_frame,
            text="Random",
            variable=mode_var,
            value="random",
            bg=BG_COLOR,
            anchor="w",
        )
        rb_random.grid(row=0, column=1, padx=(0, 4), sticky="w")

        rb_custom = tk.Radiobutton(
            row_frame,
            text="Custom",
            variable=mode_var,
            value="custom",
            bg=BG_COLOR,
            anchor="w",
        )
        rb_custom.grid(row=0, column=2, padx=(0, 4), sticky="w")

        # Entry for custom prompt
        entry = tk.Entry(row_frame, width=50)
        entry.grid(row=1, column=0, columnspan=3, padx=(24, 6), pady=(1, 2), sticky="we")
        outfit_prompt_entries[key] = entry

        # Enable/disable widgets for this outfit based on checkbox + mode
        def make_update_fn(
            outfit_key: str,
            _sel_var=sel_var,
            _mode_var=mode_var,
            _entry=entry,
            _rb_r=rb_random,
            _rb_c=rb_custom,
        ):
            def _update(*_args):
                if _sel_var.get() == 0:
                    # Outfit not selected: disable radios and entry
                    _rb_r.config(state=tk.DISABLED)
                    _rb_c.config(state=tk.DISABLED)
                    _entry.config(state=tk.DISABLED)
                else:
                    # Outfit selected: radios enabled, entry only if custom
                    _rb_r.config(state=tk.NORMAL)
                    _rb_c.config(state=tk.NORMAL)
                    if _mode_var.get() == "custom":
                        _entry.config(state=tk.NORMAL)
                    else:
                        _entry.config(state=tk.DISABLED)
            return _update

        updater = make_update_fn(key)
        sel_var.trace_add("write", updater)
        mode_var.trace_add("write", updater)
        updater()

    # ======================================================
    # Expressions column (same as before)
    # ======================================================
    expr_frame = tk.LabelFrame(
        body_frame,
        text="Expressions (neutral is always included):",
        bg=BG_COLOR,
    )
    expr_frame.grid(row=0, column=1, padx=5, pady=4, sticky="nsew")

    expr_vars: Dict[str, tk.IntVar] = {}
    for key, desc in EXPRESSIONS_SEQUENCE:
        if key == "0":
            tk.Label(
                expr_frame,
                text=f"0 – {desc} (always generated)",
                bg=BG_COLOR,
                anchor="w",
                justify="left",
                wraplength=wrap_len // 2,
            ).pack(anchor="w", padx=6, pady=2)
            continue
        var = tk.IntVar(value=1)  # default: generate all non-neutral expressions
        chk = tk.Checkbutton(
            expr_frame,
            text=f"{key} – {desc}",
            variable=var,
            bg=BG_COLOR,
            anchor="w",
            justify="left",
            wraplength=wrap_len // 2,
        )
        chk.pack(anchor="w", padx=6, pady=2)
        expr_vars[key] = var

    # ======================================================
    # Decision + buttons
    # ======================================================
    decision = {
        "ok": False,
        "outfits": [],             # type: List[str]
        "expr_seq": EXPRESSIONS_SEQUENCE,
        "config": {},              # type: Dict[str, Dict[str, Optional[str]]]
    }

    def on_ok():
        # Collect selected outfits and per-outfit config
        selected_outfits: List[str] = []
        cfg: Dict[str, Dict[str, Optional[str]]] = {}

        for key in ALL_OUTFIT_KEYS:
            if outfit_selected_vars[key].get() == 1:
                selected_outfits.append(key)
                mode = outfit_mode_vars[key].get()
                use_random = (mode == "random")
                custom_prompt_val: Optional[str] = None

                if not use_random:
                    # Custom mode: require non-empty prompt
                    txt = outfit_prompt_entries[key].get().strip()
                    if not txt:
                        messagebox.showerror(
                            "Missing custom prompt",
                            f"Please enter a custom prompt for {key.capitalize()}, "
                            f"or switch it back to Random, or uncheck it."
                        )
                        return
                    custom_prompt_val = txt

                cfg[key] = {
                    "use_random": use_random,
                    "custom_prompt": custom_prompt_val,
                }

        # Collect expressions: always include neutral, then chosen non-neutral
        new_seq: List[Tuple[str, str]] = []
        # Neutral first
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
        sys.exit(0)

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


# =========================
# Character pipeline (per source image)
# =========================

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

    Flow:

      1) Voice + name + archetype (unless preselected from prompt mode).
      2) Ask if this run should create:
           - A normal character sprite folder, or
           - A gender-bent character sprite folder
         The run will only create the chosen sprite folder.
      3) Ask which outfits and expressions to generate this time.
      4) Depending on choice:
           - NORMAL PIPELINE:
               Pose A, outfits, expressions, finalize.
           - GENDERBENT PIPELINE:
               Normalize original, gender-bent pose A, outfits,
               expressions, finalize.

    Args:
        api_key: Gemini API key.
        image_path: Source image path (or prompt-generated sprite).
        output_root: Root folder for character sprite folders.
        outfit_db: Outfit prompt database from CSV.
        game_name: Optional game name.
        preselected: Optional dict for prompt mode:
            {
                "voice": ...,
                "display_name": ...,
                "archetype_label": ...,
                "gender_style": ...
            }
    """
    print(f"\n=== Processing source image: {image_path.name} ===")

    # --- Primary character metadata (original, not necessarily saved as a folder) ---
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
        voice, display_name, archetype_label, gender_style = prompt_voice_archetype_and_name(image_path)

    # --- Ask what sprite type we are creating: normal / genderbend / fusion ---
    sprite_mode = prompt_sprite_type()
    print(f"[INFO] Sprite mode selected: {sprite_mode}")

    # --- Ask which outfits and expressions to generate ---
    (
        selected_outfit_keys,
        expressions_sequence,
        outfit_prompt_config,
    ) = prompt_outfits_and_expressions()

    print(f"[INFO] Selected outfits (Base always included): {selected_outfit_keys}")
    print(
        "[INFO] Selected expressions (including neutral): "
        f"{[key for key, _ in expressions_sequence]}"
    )
    print("[INFO] Per-outfit prompt config:")
    for key in selected_outfit_keys:
        cfg = outfit_prompt_config.get(key, {})
        mode_str = "random" if cfg.get("use_random", True) else "custom"
        print(f"  - {key}: {mode_str}")

    # =========================
    # NORMAL CHARACTER PIPELINE (single pose)
    # =========================
    if sprite_mode == "normal":
        char_folder_name = get_unique_folder_name(output_root, display_name)
        char_dir = output_root / char_folder_name
        char_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output character folder: {char_dir}")

        # Pose A: normalized base (with review + "use as outfit" toggle)
        a_dir = char_dir / "a"
        a_dir.mkdir(parents=True, exist_ok=True)
        a_base_stem = a_dir / "base"

        use_base_as_outfit = True  # default

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

        # Outfits for Pose A
        print("[INFO] Generating outfits for pose A...")
        while True:
            outfit_prompts_orig = build_outfit_prompts_with_config(
                archetype_label,
                gender_style,
                selected_outfit_keys,
                outfit_db,
                outfit_prompt_config,
            )

            a_out_paths = generate_outfits_once(
                api_key,
                a_base_path,
                a_dir / "outfits",
                gender_style,
                outfit_prompts_orig,
                include_base_outfit=use_base_as_outfit,
            )

            a_infos = [(p, f"Pose A – {p.name}") for p in a_out_paths]
            choice = review_images_for_step(
                a_infos,
                "Review Outfits for Pose A",
                "Accept these outfits, regenerate them (random outfits will pick new "
                "CSV prompts next time; custom outfits will keep the same prompts), "
                "or cancel.",
            )
            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)


        # Expressions for Pose A
        print("[INFO] Generating expressions for pose A (per outfit)...")
        generate_and_review_expressions_for_pose(
            api_key,
            char_dir,
            a_dir,
            "A",
            expressions_sequence=expressions_sequence,
        )

        # Finalization
        finalize_character(char_dir, display_name, voice, game_name)
        return

    # =========================
    # GENDER-BENT-ONLY PIPELINE (single pose)
    # =========================
    if sprite_mode == "genderbend":
        gb_archetype_label, gb_gender_style = prompt_genderbend_archetype(gender_style)
        gb_voice = "girl" if gb_gender_style == "f" else "boy"
        girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)
        gb_display_name = pick_random_name(gb_voice, girl_names, boy_names)
        gb_folder_name = get_unique_folder_name(output_root, gb_display_name)
        gb_char_dir = output_root / gb_folder_name
        gb_char_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output gender-bent character folder: {gb_char_dir}")

        # Normalize original sprite once into a temporary location
        tmp_dir = gb_char_dir / "_tmp_source"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_base_stem = tmp_dir / "base"
        tmp_normalized_base = generate_initial_pose_once(
            api_key,
            image_path,
            tmp_base_stem,
            gender_style,
        )

        # GB Pose A: gender-bent version of normalized base
        gb_a_dir = gb_char_dir / "a"
        gb_a_dir.mkdir(parents=True, exist_ok=True)
        gb_a_base_stem = gb_a_dir / "base"

        while True:
            gb_a_base_path = generate_genderbend_pose_once(
                api_key,
                tmp_normalized_base,
                gb_a_base_stem,
                gender_style,
                gb_gender_style,
                gb_archetype_label,
            )
            choice = review_images_for_step(
                [(gb_a_base_path, f"GB Pose A base (gender-bent): {gb_a_base_path.name}")],
                "Review Gender-Bent Base Pose",
                "Accept this gender-bent base pose, regenerate it, or cancel.",
            )
            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)

        # GB Outfits (single pose)
        print("[INFO] Generating outfits for GB pose A...")
        while True:
            outfit_prompts_gb = build_outfit_prompts_with_config(
                gb_archetype_label,
                gb_gender_style,
                selected_outfit_keys,
                outfit_db,
                outfit_prompt_config,
            )

            gb_a_out_paths = generate_outfits_once(
                api_key,
                gb_a_base_path,
                gb_a_dir / "outfits",
                gb_gender_style,
                outfit_prompts_gb,
            )
            gb_a_infos = [(p, f"GB Pose A – {p.name}") for p in gb_a_out_paths]
            choice = review_images_for_step(
                gb_a_infos,
                "Review Outfits for GB Pose A",
                "Accept these GB outfits, regenerate them (random outfits will pick new "
                "CSV prompts next time; custom outfits will keep the same prompts), "
                "or cancel.",
            )
            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)

        # GB Expressions (single pose)
        print("[INFO] Generating expressions for GB pose A (per outfit)...")
        generate_and_review_expressions_for_pose(
            api_key,
            gb_char_dir,
            gb_a_dir,
            "GB-A",
            expressions_sequence=expressions_sequence,
        )

        # Clean up temporary normalized base
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        # Finalization for GB character
        finalize_character(gb_char_dir, gb_display_name, gb_voice, game_name)
        return

    # =========================
    # FUSION PIPELINE (single pose)
    # =========================
    if sprite_mode == "fusion":
        # --- TEMP FOLDERS FOR PARENT NORMALIZATION ---
        fusion_tmp_dir = output_root / "_fusion_tmp"
        fusion_tmp_dir.mkdir(parents=True, exist_ok=True)

        # Parent 1 is the *current* run's source image
        parent1_dir = fusion_tmp_dir / "parent1"
        parent1_dir.mkdir(parents=True, exist_ok=True)
        parent1_base_stem = parent1_dir / "base"

        # Normalize parent 1 into our standard base pose
        parent1_base_path = generate_initial_pose_once(
            api_key,
            image_path,
            parent1_base_stem,
            gender_style,  # this is the gender_style for the first image
        )

        # --- ASK HOW TO GET PARENT 2 (image vs prompt) ---
        second_mode = prompt_source_mode()  # same UI as the very start of the script

        if second_mode == "image":
            # Pick a second image via file dialog
            root = tk.Tk()
            root.withdraw()
            initialdir = image_path.parent if image_path.parent.is_dir() else str(Path.cwd())
            filename = filedialog.askopenfilename(
                title="Choose second parent character image",
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
                raise SystemExit("No second image selected. Exiting.")
            parent2_source_path = Path(filename)

            # Get metadata for the second parent (voice, archetype, etc.)
            p2_voice, p2_name, p2_arch_label, p2_gender_style = \
                prompt_voice_archetype_and_name(parent2_source_path)

        else:
            # Second parent via prompt (same flow as your existing prompt mode)
            concept2, arch_label2, voice2, display_name2, gender_style2 = \
                prompt_character_idea_and_archetype()

            while True:
                parent2_source_path = generate_initial_character_from_prompt(
                    api_key,
                    concept2,
                    arch_label2,
                    output_root,
                )
                choice = review_images_for_step(
                    [(parent2_source_path,
                      f"Prompt-generated second parent: {parent2_source_path.name}")],
                    "Review Prompt-Generated Second Parent",
                    "Accept this as the second parent sprite, regenerate it, or cancel.",
                )
                if choice == "accept":
                    break
                if choice == "regenerate":
                    continue
                if choice == "cancel":
                    sys.exit(0)

            p2_voice, p2_name, p2_arch_label, p2_gender_style = (
                voice2,
                display_name2,
                arch_label2,
                gender_style2,
            )

        print(
            f"[INFO] Second parent metadata: name={p2_name}, "
            f"voice={p2_voice}, archetype={p2_arch_label}, "
            f"gender_style={p2_gender_style}"
        )

        # Normalize parent 2 into our standard base pose
        parent2_dir = fusion_tmp_dir / "parent2"
        parent2_dir.mkdir(parents=True, exist_ok=True)
        parent2_base_stem = parent2_dir / "base"
        parent2_base_path = generate_initial_pose_once(
            api_key,
            parent2_source_path,
            parent2_base_stem,
            p2_gender_style,
        )

        # --- ASK USER HOW THE FUSION RESULT SHOULD LOOK ---
        fusion_display_name, fusion_voice, fusion_archetype_label = \
            prompt_fusion_target_metadata(parent1_base_path, parent2_base_path)
        fusion_gender_style = archetype_to_gender_style(fusion_archetype_label)

        print(
            f"[INFO] Fusion result: name={fusion_display_name}, voice={fusion_voice}, "
            f"archetype={fusion_archetype_label}, gender_style={fusion_gender_style}"
        )

        # --- CREATE FUSION CHARACTER FOLDER (pose 'a' only) ---
        fusion_folder_name = get_unique_folder_name(output_root, fusion_display_name)
        fusion_char_dir = output_root / fusion_folder_name
        fusion_char_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output fusion character folder: {fusion_char_dir}")

        fusion_a_dir = fusion_char_dir / "a"
        fusion_a_dir.mkdir(parents=True, exist_ok=True)
        fusion_a_base_stem = fusion_a_dir / "base"

        # --- FUSION BASE POSE (A) WITH REGENERATE SUPPORT ---
        while True:
            fusion_a_base_path = generate_fusion_base_once(
                api_key,
                parent1_base_path,
                parent2_base_path,
                fusion_a_base_stem,
                fusion_archetype_label,
            )
            choice = review_images_for_step(
                [(fusion_a_base_path, f"Fusion base pose: {fusion_a_base_path.name}")],
                "Review Fusion Base Pose",
                "Accept this fused base pose, regenerate it, or cancel.",
            )
            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)

        # --- OUTFITS FOR FUSION POSE A ---
        print("[INFO] Generating outfits for fusion pose...")
        while True:
            fusion_outfit_prompts = build_outfit_prompts_with_config(
                fusion_archetype_label,
                fusion_gender_style,
                selected_outfit_keys,
                outfit_db,
                outfit_prompt_config,
            )

            fusion_a_out_paths = generate_outfits_once(
                api_key,
                fusion_a_base_path,
                fusion_a_dir / "outfits",
                fusion_gender_style,
                fusion_outfit_prompts,
            )

            a_infos = [(p, f"Fusion pose – {p.name}") for p in fusion_a_out_paths]
            choice = review_images_for_step(
                a_infos,
                "Review Outfits – Fusion Pose",
                "Accept these outfits, regenerate them (random outfits will pick new "
                "CSV prompts next time; custom outfits will keep the same prompts), "
                "or cancel.",
            )
            if choice == "accept":
                break
            if choice == "regenerate":
                continue
            if choice == "cancel":
                sys.exit(0)

        # --- EXPRESSIONS FOR FUSION POSE A (ONLY) ---
        print("[INFO] Generating expressions for fusion pose (per outfit)...")
        generate_and_review_expressions_for_pose(
            api_key,
            fusion_char_dir,
            fusion_a_dir,
            "Fusion-1",
            expressions_sequence=expressions_sequence,
        )

        # --- CLEAN UP TMP + FINALIZE CHARACTER ---
        try:
            shutil.rmtree(fusion_tmp_dir)
        except Exception:
            pass

        finalize_character(fusion_char_dir, fusion_display_name, fusion_voice, game_name)
        return




# =========================
# CLI entry point
# =========================

def find_character_images(input_dir: Path) -> List[Path]:
    """Return all image files in input_dir that could be characters."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    images.sort()
    return images


def main() -> None:
    """
    Parse arguments, validate API key, load outfit CSV, and run the pipeline
    for a single character at a time, using either:

      - An existing image file (chosen via file dialog), or
      - A new character generated from a text prompt.

    The batch mode of scanning --input-dir is intentionally removed in favor of
    an interactive one-at-a-time workflow.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Student Transfer sprite builder using Google Gemini:\n"
            "  - base pose (+ optional gender-bent version)\n"
            "  - outfits (Base + selected extras like Formal/Casual/Uniform/...)\n"
            "  - expressions per outfit (0 + selected non-neutral ones)\n"
            "  - eye line / name color / scale\n"
            "  - character.yml\n"
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Previously used for batch mode; now only used as the initial directory "
             "when choosing an image file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root folder to write final character sprite folders.",
    )
    parser.add_argument(
        "--game-name",
        type=str,
        default=None,
        help="Optional game name to write into character.yml (game field).",
    )

    args = parser.parse_args()

    random.seed(int.from_bytes(os.urandom(16), "big"))

    api_key = get_api_key()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir
    game_name: Optional[str] = args.game_name

    if not input_dir.exists():
        print(f"[WARN] Input directory does not exist: {input_dir}")

    outfit_db = load_outfit_prompts(OUTFIT_CSV_PATH)
    output_root.mkdir(parents=True, exist_ok=True)

    # Ask how to start (image vs prompt)
    mode = prompt_source_mode()

    if mode == "image":
        root = tk.Tk()
        root.withdraw()
        initialdir = input_dir if input_dir.is_dir() else str(Path.cwd())
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
        process_single_character(api_key, image_path, output_root, outfit_db, game_name)

    else:
        concept, arch_label, voice, display_name, gender_style = prompt_character_idea_and_archetype()

        # Generate + review the prompt-based base sprite
        while True:
            src_path = generate_initial_character_from_prompt(
                api_key,
                concept,
                arch_label,
                output_root,
            )

            choice = review_images_for_step(
                [(src_path, f"Prompt-generated base: {src_path.name}")],
                "Review Prompt-Generated Base Sprite",
                "Accept this as the starting sprite, regenerate it, or cancel.",
            )

            if choice == "accept":
                break
            if choice == "regenerate":
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


if __name__ == "__main__":
    main()
