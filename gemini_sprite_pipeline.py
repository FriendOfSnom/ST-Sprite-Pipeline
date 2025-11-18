#!/usr/bin/env python3
"""
gemini_sprite_pipeline.py (refactored + prompt mode)

Changes in this version:

- Improved background remover to:
  * Better remove tiny green pockets inside hair/arms, and
  * Reduce green "halos" along the edges
  without eating legitimate character pixels.

- Added an initial Tk "source mode" dialog:
  * "Generate from an image" -> use file chooser to pick a single image.
  * "Generate from a text prompt" -> describe a character concept and archetype,
    then Gemini generates a new base sprite using reference sprites.

- The rest of the pipeline for a character (poses, outfits, expressions,
  eye-line, scale, character.yml, flattening) is unchanged.

NOTE: Gender-bent sprites are still generated as Pose C within the same
character folder. In a future step we can refactor that into a separate
character folder if desired.
"""

import argparse
import base64
import csv
import json
import os
import random
import sys
import shutil
from collections import Counter, deque
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from rembg import remove
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox, filedialog
import yaml
import webbrowser


# =========================
# Global paths/constants
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path.home() / ".st_gemini_config.json"
OUTFIT_CSV_PATH = SCRIPT_DIR / "outfit_prompts.csv"
NAMES_CSV_PATH = SCRIPT_DIR / "names.csv"
REF_SPRITES_DIR = SCRIPT_DIR / "reference_sprites"

GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_IMAGE_MODEL}:generateContent"
)

# Tk UI style constants
BG_COLOR = "lightgray"
TITLE_FONT = ("Arial", 16, "bold")
INSTRUCTION_FONT = ("Arial", 12)
LINE_COLOR = "#00E5FF"
WINDOW_MARGIN = 10
WRAP_PADDING = 40


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
    dest_stem: Path without extension, e.g., char_dir/'a'/'base'.
    """
    dest_stem = Path(dest_stem)
    dest_stem.parent.mkdir(parents=True, exist_ok=True)

    safe = img.convert("RGBA")

    try:
        out_path = dest_stem.with_suffix(".webp")
        # Use lossless WebP so we don't introduce new color noise at edges.
        safe.save(out_path, format="WEBP", lossless=True, quality=100, method=6)
        return out_path
    except Exception as e:
        print(f"[WARN] WEBP save failed for {dest_stem.name}: {e}. Falling back to PNG.")
        out_path = dest_stem.with_suffix(".png")
        safe.save(out_path, format="PNG")
        return out_path


def save_image_bytes_as_webp(image_bytes: bytes, dest_stem: Path) -> Path:
    """
    Convert arbitrary image bytes (PNG from Gemini + stripped background)
    into WEBP (preferred) or PNG (fallback).
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    return save_img_webp_or_png(img, dest_stem)


def strip_background(image_bytes: bytes) -> bytes:
    """
    Remove the background using rembg, then do two cleanup passes:

      1) Halo removal: delete pixels very close to the *original* flat green
         background color that sit right next to transparency (kills green fringe)
         but keep this threshold fairly tight so we do not eat green clothing.

      2) Interior island removal: delete small enclosed blobs of pixels that are
         close to the background color but surrounded by the character
         (kills tiny pockets inside hair/arms/etc).

    If we *cannot* confidently detect a flat green-ish background on the original
    image, we just return rembg's result without the extra cleanup to avoid
    chewing into dark features like eyes/hair.
    """
    try:
        orig = Image.open(BytesIO(image_bytes)).convert("RGBA")
        ow, oh = orig.size
        opix = orig.load()

        border = max(2, min(ow, oh) // 40)
        samples = []

        # Sample opaque border pixels from the original image
        for y in range(oh):
            for x in range(ow):
                if not (x < border or x >= ow - border or y < border or y >= oh - border):
                    continue
                r, g, b, a = opix[x, y]
                if a > 240:  # definitely visible background, not transparent
                    samples.append((r, g, b))

        bg_color = None
        if len(samples) >= 50:
            avg_r = sum(s[0] for s in samples) / len(samples)
            avg_g = sum(s[1] for s in samples) / len(samples)
            avg_b = sum(s[2] for s in samples) / len(samples)

            def channel_var(idx: int) -> float:
                vals = [s[idx] for s in samples]
                mean = sum(vals) / len(vals)
                return sum((v - mean) ** 2 for v in vals) / len(vals)

            var_r = channel_var(0)
            var_g = channel_var(1)
            var_b = channel_var(2)

            # "Flat" = low variance across the border.
            # "Green screen" = green channel clearly dominates.
            if max(var_r, var_g, var_b) < 15.0 ** 2 and (avg_g - max(avg_r, avg_b)) > 20.0:
                bg_color = (avg_r, avg_g, avg_b)

        # --- Step 1: let rembg do its segmentation ---
        seg_bytes = remove(image_bytes)
        img = Image.open(BytesIO(seg_bytes)).convert("RGBA")

        # If we couldn't confidently detect a flat green background, *don't* do
        # the aggressive cleanup. Just return rembg's result.
        if bg_color is None:
            out_buf = BytesIO()
            img.save(out_buf, format="PNG")
            return out_buf.getvalue()

        width, height = img.size
        pixels = img.load()

        def color_dist_sq(c1, c2) -> float:
            dr = c1[0] - c2[0]
            dg = c1[1] - c2[1]
            db = c1[2] - c2[2]
            return dr * dr + dg * dg + db * db

        # --- Step 2: halo cleanup along transparency edges (targets green fringe) ---
        # Slightly *tighter* than before to avoid eating near-green outfit colors.
        halo_threshold_sq = 32.0 ** 2  # only pixels very close to the bg color
        to_clear = []
        max_radius = 1  # just immediate neighbors

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                if color_dist_sq((r, g, b), bg_color) > halo_threshold_sq:
                    continue

                # If this bg-colored pixel touches transparency, treat it as halo.
                for ny in range(max(0, y - max_radius), min(height, y + max_radius + 1)):
                    cleared = False
                    for nx in range(max(0, x - max_radius), min(width, x + max_radius + 1)):
                        if nx == x and ny == y:
                            continue
                        _, _, _, na = pixels[nx, ny]
                        if na == 0:
                            to_clear.append((x, y))
                            cleared = True
                            break
                    if cleared:
                        break

        for x, y in to_clear:
            r, g, b, _ = pixels[x, y]
            pixels[x, y] = (r, g, b, 0)

        # --- Step 3: small enclosed islands of bg color inside the character ---
        # We allow a slightly *looser* color match here so we catch more
        # tiny pockets inside the hair, but still limit to small blobs.
        island_threshold_sq = 60.0 ** 2
        tiny_island_size = max(20, (width * height) // 4000)
        tiny_island_size = min(tiny_island_size, 800)

        bg_like = [[False] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                if color_dist_sq((r, g, b), bg_color) <= island_threshold_sq:
                    bg_like[y][x] = True

        visited = [[False] * width for _ in range(height)]
        neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(height):
            for x in range(width):
                if not bg_like[y][x] or visited[y][x]:
                    continue

                stack = [(x, y)]
                visited[y][x] = True
                comp_pixels = []
                touches_border = False

                while stack:
                    cx, cy = stack.pop()
                    comp_pixels.append((cx, cy))

                    if cx == 0 or cy == 0 or cx == width - 1 or cy == height - 1:
                        touches_border = True

                    for dx, dy in neighbors_4:
                        nx = cx + dx
                        ny = cy + dy
                        if nx < 0 or nx >= width or ny < 0 or ny >= height:
                            continue
                        if visited[ny][nx] or not bg_like[ny][nx]:
                            continue
                        visited[ny][nx] = True
                        stack.append((nx, ny))

                # If it's a small island that does *not* touch the border, nuke it.
                if (not touches_border) and (len(comp_pixels) <= tiny_island_size):
                    for cx, cy in comp_pixels:
                        r, g, b, _ = pixels[cx, cy]
                        pixels[cx, cy] = (r, g, b, 0)

        out_buf = BytesIO()
        img.save(out_buf, format="PNG")
        return out_buf.getvalue()

    except Exception as e:
        print(f"  [WARN] Background stripping failed, returning original bytes: {e}")
        return image_bytes


def load_image_as_base64(path: Path) -> str:
    """
    Load an image from disk, re-encode it as PNG in memory, and return base64-encoded bytes.

    This keeps the input format to Gemini consistent (actual PNG bytes that match
    the declared mime_type), even if we store sprites as WEBP on disk.
    """
    img = Image.open(path).convert("RGBA")
    buf = BytesIO()
    img.save(buf, format="PNG")
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
    Returns raw PNG bytes or None.
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

    Includes a small retry loop to handle occasional transient failures where
    the API responds successfully but no image bytes are present.
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

    This is used for "generate from prompt" mode:
      - prompt: full text prompt for the character concept.
      - ref_images: list of reference sprite paths to anchor the art style.

    Returns PNG bytes (with background stripped) from the first candidate.
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
            img.save(buf, format="PNG")
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

EXPRESSIONS: Dict[str, str] = {
    "0": "a neutral and relaxed expression",
    "2": "a happy smile, like they are chuckling, or perhaps winking",
    "3": "a sad expression, like they are hurt or about to cry",
    "4": "an annoyed or angry expression",
    "7": "a flushed, embarrassed and flustered expression, cheeks blushing, eyes a little unfocused",
}

# Ordered list of expressions we actually use per outfit.
# 0 is neutral (the outfit image itself).
EXPRESSIONS_SEQUENCE: List[Tuple[str, str]] = [
    ("0", EXPRESSIONS["0"]),  # neutral
    ("2", EXPRESSIONS["2"]),  # happy
    ("3", EXPRESSIONS["3"]),  # sad
    ("4", EXPRESSIONS["4"]),  # annoyed / angry
    ("7", EXPRESSIONS["7"]),  # flustered / crushy
]

GENDER_ARCHETYPES = [
    ("young woman", "f"),
    ("adult woman", "f"),
    ("motherly woman", "f"),
    ("young man", "m"),
    ("adult man", "m"),
    ("fatherly man", "m"),
]

OUTFIT_KEYS: List[str] = ["formal", "casual"]


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
    return f"a simple outfit that fits this {gender_word}'s personality"


def choose_outfit_prompts_for_archetype(
    archetype_label: str,
    gender_style: str,
    outfit_keys: List[str],
    outfit_db: Dict[str, Dict[str, List[str]]],
) -> Dict[str, str]:
    """
    For a given archetype and gender, choose one prompt per outfit_key.
    If CSV has prompts, pick one at random. Else use a generic description.
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
    Take poses like:

        a/outfits/Base.webp, Formal.webp, Casual.webp
        a/faces/face/*.webp           (Base expressions)
        a/faces/Formal/*.webp         (Formal expressions)
        a/faces/Casual/*.webp         (Casual expressions)
        b/...
        c/...

    And convert them into separate ST poses with single outfits:

        <char>/a/
            outfits/outfit.webp   (transparent)
            faces/face/*.webp     (pose A + Base)

        <char>/b/
            outfits/outfit.webp
            faces/face/*.webp     (pose A + Formal)

        ...

    Returns the list of final pose letters in order.
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

            outfit_name = outfit_path.stem  # "Base", "Formal", "Casual", etc.
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

            # Copy expressions images
            for src in sorted(src_expr_dir.iterdir()):
                if not src.is_file():
                    continue
                dest = new_faces_dir / src.name
                shutil.copy2(src, dest)

            # Transparent outfit, same size as outfit image
            try:
                outfit_img = Image.open(outfit_path).convert("RGBA")
                w, h = outfit_img.size
                transparent = Image.new("RGBA", (w, h), (0, 0, 0, 0))

                # Use the original outfit type as the file name:
                # Base.webp, Formal.webp, Casual.webp, etc.
                outfit_name = outfit_path.stem  # e.g. "Base", "Formal", "Casual"
                out_name = outfit_name + outfit_path.suffix.lower()

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

    # Remove original pose folders
    for pose_dir in original_pose_dirs:
        try:
            shutil.rmtree(pose_dir)
        except Exception as e:
            print(f"[WARN] Failed to remove original pose folder {pose_dir}: {e}")

    # Move flattened poses into the character folder
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
    """Prompt to normalize the original sprite (mid-thigh, green background)."""
    return (
        "Edit the input image of our visual novel character, and if it is not already, "
        "crop the character so we only see from the mid-thigh up. "
        "Use a pure, flat green background (#00FF00) behind the character, and make sure "
        "the character and outfit have none of the background color on them. "
        "Do not change any design details of the character; only reframe and place them "
        "onto that green background."
    )


def build_pose_prompt(gender_style: str) -> str:
    """Prompt to generate a second pose while keeping design, crop, and outfit fixed."""
    gender_clause = "feminine" if gender_style == "f" else "masculine"
    if gender_style == "f":
        pose_examples = (
            "a clearly different cute or cool feminine pose that is not too over the top, "
            "but would fit well in a visual novel"
        )
    else:
        pose_examples = (
            "a clearly different relaxed or friendly masculine pose that is not too over the top, "
            "but would fit well in a visual novel"
        )

    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Make sure the character is in a new pose that matches this description: {pose_examples}. "
        "Do not change anything about the character or image besides the pose of the character, "
        "including the crop from the mid-thigh up, image size, and outfit. "
        "Go ahead and change the hair style of the character to match the new pose, but do not change the overall "
        "hair length or color. The character should not be holding anything in their hand, and should "
        "be cropped from the mid-thigh on up. "
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character "
        "and outfit have none of the background color on them."
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
            f"a clearly different cool or relaxed pose that suits a {target_archetype} "
            "and would fit well in a visual novel. Make sure the character is not holding anything."
        )

    return (
        "Edit the input visual novel sprite, in the same art style. "
        f"Transform this {original_clause} character into a gender-bent {target_archetype} version of themselves, "
        f"clearly presenting in a {target_clause} way, while keeping their recognizable traits: similar face shape, "
        "hair color, eye color, and general vibe. "
        f"Put them in {pose_examples} "
        "Go ahead and change the hair style to better match the new pose and gender presentation, but keep the hair recognizably "
        "related to the original design. Change the body shape and clothing so that the character clearly presents as "
        "the new gender, but keep the overall crop from the mid-thigh up and do not change the image size. "
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit "
        "have none of the background color on them."
        "Crop it so that we only see from the mid-thigh on up."
    )


def build_expression_prompt(expression_desc: str) -> str:
    """Prompt to change only the facial expression, pixel-aligned with the input."""
    return (
        "Edit the input visual novel sprite in the same art style. "
        f"Only change the facial expression to match this description: {expression_desc}. "
        "Keep the hair volume, hair outlines, and the hair style, all the exact same. "
        "Do not change the hairstyle, crop from the mid-thigh up, image size, lighting, or background. "
        "You can change the arm and hand positions slightly, if it make sense for the this specific expression. But the body itself shouldn't move from its position."
        "Please just edit the expression on the existing head, but if the pose changes due to the expression, please make sure its as minimum as possible."
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit "
        "have none of the background color on them."
        )


def build_outfit_prompt(base_outfit_desc: str, gender_style: str) -> str:
    """Prompt to change only the clothing to base_outfit_desc on the given pose."""
    gender_clause = "girl" if gender_style == "f" else "boy"
    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Please change only the clothing, hair, and outfit to match this description: {base_outfit_desc}. "
        "Do not change the pose, head tilt, body proportions, crop from the mid-thigh up, or image size."
        "Keep everything else the same and just change the outfit."
        "Change the hair style to match the outfit, but dont change the hair length or general look."
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit "
        "have none of the background color on them."
        "Do not change the body, chest, and hip proportions to be different from the original."
        "Do not change hair length, even if its in a different style."
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
    Returns: (voice, display_name, archetype_label, gender_style).
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

    # --- name entry ---
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

    # archetype
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


def prompt_genderbend_archetype(original_gender_style: str) -> Tuple[str, str]:
    """
    Tk dialog to choose a gender-bent archetype from the opposite gender group.
    Returns: (gb_archetype_label, gb_gender_style).
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

    image_infos: list of (Path, caption_string)
    Returns: "accept", "regenerate", or "cancel".
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
    max_thumb_height = min(600, canvas_h - 40)  # allow fairly large thumbnails

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


# =========================
# Tk UI: eye line + name color (hair color)
# =========================

def prompt_for_eye_and_hair(image_path: Path) -> Tuple[float, str]:
    """
    Tk UI to choose:
      - Eye line (click once, as a height ratio).
      - Hair color (click once to sample RGB) -> used as name_color.
    Returns: (eye_line_ratio, name_color_hex).
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
    # Use more vertical space so the character appears larger in this UI.
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
    Choose a full-body outfit image to use for eye-line and scale:
    Prefer Pose A outfits (Base/Formal/Casual), fall back to a/base.webp.
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
      - Pick representative outfit
      - Eye line + name_color
      - Scale vs reference
      - Flatten pose/outfit combos into ST pose letters
      - Write character.yml
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
    """Normalize the original sprite into pose A."""
    print("  [Gemini] Normalizing base pose...")
    image_b64 = load_image_as_base64(image_path)
    prompt = build_initial_pose_prompt(gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_webp(img_bytes, out_stem)
    print(f"  Saved base pose to: {final_path}")
    return final_path


def generate_second_pose_once(
    api_key: str,
    base_image_path: Path,
    out_stem: Path,
    gender_style: str,
) -> Path:
    """Generate pose B (new pose)."""
    image_b64 = load_image_as_base64(base_image_path)
    prompt = build_pose_prompt(gender_style)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_webp(img_bytes, out_stem)
    print(f"  Saved second pose to: {final_path}")
    return final_path


def generate_genderbend_pose_once(
    api_key: str,
    base_image_path: Path,
    out_stem: Path,
    original_gender_style: str,
    target_gender_style: str,
    target_archetype: str,
) -> Path:
    """Generate pose C (gender-bent)."""
    image_b64 = load_image_as_base64(base_image_path)
    prompt = build_genderbend_pose_prompt(original_gender_style, target_gender_style, target_archetype)
    img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    final_path = save_image_bytes_as_webp(img_bytes, out_stem)
    print(f"  Saved gender-bent pose to: {final_path}")
    return final_path


def generate_outfits_once(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    outfit_descriptions: Dict[str, str],
) -> List[Path]:
    """
    Generate outfits for a pose:
      - Copies base pose as Base.webp.
      - Generates Formal.webp, Casual.webp, etc.
    Returns list of created image paths for preview.
    """
    outfits_dir.mkdir(parents=True, exist_ok=True)

    base_bytes = base_pose_path.read_bytes()
    base_img = Image.open(BytesIO(base_bytes)).convert("RGBA")
    base_out_path = save_img_webp_or_png(base_img, outfits_dir / "Base")

    paths: List[Path] = [base_out_path]
    image_b64 = load_image_as_base64(base_pose_path)

    for key, desc in outfit_descriptions.items():
        out_stem = outfits_dir / key.capitalize()
        prompt = build_outfit_prompt(desc, gender_style)
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        final_path = save_image_bytes_as_webp(img_bytes, out_stem)
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
        a/outfits/Base.webp
        a/faces/face/0.webp ... N.webp  (if outfit_name == 'Base')
    Or for non-base outfits (e.g. 'Formal'):
        a/faces/Formal/0.webp ... N.webp

    0.webp is the neutral outfit image itself.
    1.webp, 2.webp, ... are generated expressions.
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
        final_path = save_image_bytes_as_webp(img_bytes, out_stem)
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
) -> None:
    """
    For a given pose directory (a, b), iterate each outfit and:
      - Generate its full expression set.
      - Immediately show review window for just that outfit.
      - Allow Accept / Regenerate / Cancel at outfit level.

    This is shared by both the primary character and the gender-bent character.
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
      1) PNG/WEBP/JPG from reference_sprites/<archetype_label>/ if that folder exists.
      2) Otherwise, any PNGs directly under reference_sprites/.
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
    """
    gender_word = "girl" if gender_style == "f" else "boy"
    return (
        f"Create concept art for an original {archetype_label} {gender_word} character "
        f"for a Japanese-style visual novel. The character idea is:\n\n"
        f"{concept}\n\n"
        "Match the art style and rendering of the reference sprites exactly so they look "
        "like they come from the same game. The character should be cropped from the "
        "mid-thigh up, facing mostly toward the viewer in a friendly, neutral pose that "
        "would work as a base sprite. They should not be holding anything in their hands. "
        "Use a pure, flat green background (#00FF00) behind the character, with no "
        "green background color on the character or outfit itself. Use clean line art "
        "and vibrant but not overly saturated colors that match the reference style."
    )


def generate_initial_character_from_prompt(
    api_key: str,
    concept: str,
    archetype_label: str,
    output_root: Path,
) -> Path:
    """
    Use Gemini + reference sprites to generate a brand new base character image
    from a text concept. The result is saved under:

        <output_root>/_prompt_sources/<slug>.webp

    and that path is returned so it can be fed into the normal pipeline.
    """
    gender_style = archetype_to_gender_style(archetype_label)
    refs = get_reference_images_for_archetype(archetype_label)
    if refs:
        print(f"[INFO] Using {len(refs)} reference sprite(s) for archetype '{archetype_label}'.")
    else:
        print(f"[WARN] No reference sprites found for archetype '{archetype_label}'. "
              "Gemini will rely on the text prompt alone.")

    full_prompt = build_prompt_for_idea(concept, archetype_label, gender_style)
    print("[Gemini] Generating new character from text prompt...")
    img_bytes = call_gemini_text_or_refs(api_key, full_prompt, refs)

    # Simple slug from archetype + random token so multiple runs do not clash.
    rand_token = hex(random.getrandbits(32))[2:]
    slug = f"{archetype_label.replace(' ', '_')}_{rand_token}"
    prompt_src_dir = output_root / "_prompt_sources"
    prompt_src_dir.mkdir(parents=True, exist_ok=True)
    out_stem = prompt_src_dir / slug

    final_path = save_image_bytes_as_webp(img_bytes, out_stem)
    print(f"[INFO] Saved prompt-generated source sprite to: {final_path}")
    return final_path


# =========================
# Tk UI: source mode + prompt entry
# =========================

def prompt_source_mode() -> str:
    """
    Tk dialog asking whether to generate from an image or from a text prompt.

    Returns:
        "image"  -> user wants to pick an image file
        "prompt" -> user wants to describe a character concept
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
      - A free-text character concept
      - Voice (Girl / Boy)
      - A name (auto-filled from names.csv by voice, editable)
      - An archetype label (young woman, adult man, etc., filtered by voice)

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

    # Concept text
    text_frame = tk.Frame(root, bg=BG_COLOR)
    text_frame.grid(row=1, column=0, padx=10, pady=(4, 4), sticky="nsew")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    txt = tk.Text(text_frame, width=60, height=8, wrap="word")
    txt.pack(fill="both", expand=True)

    # Voice + name row
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
        # Fill name if empty
        if not name_var.get().strip():
            name_var.set(_pick_random_name_for_voice(v))
        # Update archetype menu based on gender
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

    # Archetype row
    arch_frame = tk.Frame(root, bg=BG_COLOR)
    arch_frame.grid(row=3, column=0, pady=(4, 4))

    tk.Label(
        arch_frame,
        text="Archetype:",
        bg=BG_COLOR,
        fg="black",
        font=INSTRUCTION_FONT,
    ).pack(side=tk.LEFT, padx=(0, 6))

    arche_labels = [lbl for (lbl, _) in GENDER_ARCHETYPES]
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
    Run the full pipeline for a single source image:

    PRIMARY CHARACTER:
      1) Voice + random name + archetype.
      2) Pose A (normalize).
      3) Pose B (new pose, review loop).
      4) For A & B:
         - Generate outfits (A can reroll prompts; B reuses A prompts).
         - For each outfit: generate expressions, review per outfit.
      5) Eye line + name color + scale.
      6) Flatten pose+outfit combos into ST poses.
      7) Write character.yml.

    OPTIONAL GENDER-BENT CHARACTER (separate folder):
      1) Ask if we should generate a GB version.
      2) If yes: choose GB archetype.
      3) Create a new character folder with a random opposite-gender name.
      4) Pose A' = gender-bent version of primary Pose A (with review loop).
      5) Pose B' = new pose for GB character (with review loop).
      6) Outfits & expressions as above, using GB archetype/prompts.
      7) Eye line + scale + flatten + character.yml for GB character.
    """
    print(f"\n=== Processing source image: {image_path.name} ===")

    # --- Primary character: basic setup (voice, name, archetype) ---
    # --- Basic setup: voice, name, archetype ---
    if preselected is not None:
        voice = preselected["voice"]
        display_name = preselected["display_name"]
        archetype_label = preselected["archetype_label"]
        gender_style = preselected["gender_style"]
        print(f"[INFO] Using preselected voice/name/archetype for {display_name}: "
              f"voice={voice}, archetype={archetype_label}, gender_style={gender_style}")
    else:
        voice, display_name, archetype_label, gender_style = prompt_voice_archetype_and_name(image_path)

    char_folder_name = get_unique_folder_name(output_root, display_name)
    char_dir = output_root / char_folder_name
    char_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output character folder: {char_dir}")

    # --- Ask once whether to create a separate gender-bent character ---
    root_toggle = tk.Tk()
    root_toggle.withdraw()
    do_genderbend = messagebox.askyesno(
        "Gender-Bent Character",
        "Also generate a gender-bent version of this character as a separate character folder?",
    )
    root_toggle.destroy()

    # =========================
    # PRIMARY CHARACTER: POSES A and B
    # =========================

    # Pose A: normalized base
    a_dir = char_dir / "a"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_base_stem = a_dir / "base"
    a_base_path = generate_initial_pose_once(api_key, image_path, a_base_stem, gender_style)

    # Pose B: new pose with review loop
    b_dir = char_dir / "b"
    b_dir.mkdir(parents=True, exist_ok=True)
    b_base_stem = b_dir / "base"

    while True:
        b_base_path = generate_second_pose_once(api_key, a_base_path, b_base_stem, gender_style)
        choice = review_images_for_step(
            [(b_base_path, f"Pose B base: {b_base_path.name}")],
            "Review Pose B (New Pose)",
            "Accept this new pose, regenerate it, or cancel.",
        )

        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # =========================
    # PRIMARY CHARACTER: OUTFITS
    # =========================

    # Pose A outfits (fresh random prompts each regenerate)
    print("[INFO] Generating outfits for pose A...")
    while True:
        outfit_prompts_orig = choose_outfit_prompts_for_archetype(
            archetype_label,
            gender_style,
            OUTFIT_KEYS,
            outfit_db,
        )

        a_out_paths = generate_outfits_once(
            api_key,
            a_base_path,
            a_dir / "outfits",
            gender_style,
            outfit_prompts_orig,
        )
        a_infos = [(p, f"Pose A – {p.name}") for p in a_out_paths]
        choice = review_images_for_step(
            a_infos,
            "Review Outfits for Pose A",
            "Accept these outfits, regenerate them (new random prompts), or cancel.",
        )
        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # Pose B outfits (reuse Pose A's accepted outfit prompts)
    print("[INFO] Generating outfits for pose B...")
    while True:
        b_out_paths = generate_outfits_once(
            api_key,
            b_base_path,
            b_dir / "outfits",
            gender_style,
            outfit_prompts_orig,
        )
        b_infos = [(p, f"Pose B – {p.name}") for p in b_out_paths]
        a_infos = [(p, f"Pose A (reference) – {p.name}") for p in a_out_paths]
        choice = review_images_for_step(
            b_infos + a_infos,
            "Review Outfits for Pose B",
            "Compare Pose B outfits against Pose A. Accept, regenerate B, or cancel.",
        )
        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # =========================
    # PRIMARY CHARACTER: EXPRESSIONS
    # =========================

    print("[INFO] Generating expressions for pose A (per outfit)...")
    generate_and_review_expressions_for_pose(api_key, char_dir, a_dir, "A")

    print("[INFO] Generating expressions for pose B (per outfit)...")
    generate_and_review_expressions_for_pose(api_key, char_dir, b_dir, "B")

    # =========================
    # PRIMARY CHARACTER: FINALIZE
    # =========================

    finalize_character(char_dir, display_name, voice, game_name)

    # =========================
    # OPTIONAL: GENDER-BENT CHARACTER
    # =========================

    if not do_genderbend:
        return

    # --- Choose GB archetype (opposite gender group) ---
    gb_archetype_label, gb_gender_style = prompt_genderbend_archetype(gender_style)
    gb_voice = "girl" if gb_gender_style == "f" else "boy"

    # Random opposite-gender name for GB character
    girl_names, boy_names = load_name_pool(NAMES_CSV_PATH)
    gb_display_name = pick_random_name(gb_voice, girl_names, boy_names)
    gb_folder_name = get_unique_folder_name(output_root, gb_display_name)
    gb_char_dir = output_root / gb_folder_name
    gb_char_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output gender-bent character folder: {gb_char_dir}")

    # ----- GB Pose A': gender-bent version of primary Pose A -----
    gb_a_dir = gb_char_dir / "a"
    gb_a_dir.mkdir(parents=True, exist_ok=True)
    gb_a_base_stem = gb_a_dir / "base"

    while True:
        gb_a_base_path = generate_genderbend_pose_once(
            api_key,
            a_base_path,
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

    # ----- GB Pose B': new pose for GB character -----
    gb_b_dir = gb_char_dir / "b"
    gb_b_dir.mkdir(parents=True, exist_ok=True)
    gb_b_base_stem = gb_b_dir / "base"

    while True:
        gb_b_base_path = generate_second_pose_once(
            api_key,
            gb_a_base_path,
            gb_b_base_stem,
            gb_gender_style,
        )
        choice = review_images_for_step(
            [(gb_b_base_path, f"GB Pose B base: {gb_b_base_path.name}")],
            "Review GB Pose B (New Pose)",
            "Accept this new GB pose, regenerate it, or cancel.",
        )

        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # ----- GB Outfits -----
    print("[INFO] Generating outfits for GB pose A...")
    while True:
        outfit_prompts_gb = choose_outfit_prompts_for_archetype(
            gb_archetype_label,
            gb_gender_style,
            OUTFIT_KEYS,
            outfit_db,
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
            "Accept these GB outfits, regenerate them (new random prompts), or cancel.",
        )
        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    print("[INFO] Generating outfits for GB pose B...")
    while True:
        gb_b_out_paths = generate_outfits_once(
            api_key,
            gb_b_base_path,
            gb_b_dir / "outfits",
            gb_gender_style,
            outfit_prompts_gb,
        )
        gb_b_infos = [(p, f"GB Pose B – {p.name}") for p in gb_b_out_paths]
        gb_a_infos = [(p, f"GB Pose A (reference) – {p.name}") for p in gb_a_out_paths]
        choice = review_images_for_step(
            gb_b_infos + gb_a_infos,
            "Review Outfits for GB Pose B",
            "Compare GB Pose B outfits against GB Pose A. Accept, regenerate B, or cancel.",
        )
        if choice == "accept":
            break
        if choice == "regenerate":
            continue
        if choice == "cancel":
            sys.exit(0)

    # ----- GB Expressions -----
    print("[INFO] Generating expressions for GB pose A (per outfit)...")
    generate_and_review_expressions_for_pose(api_key, gb_char_dir, gb_a_dir, "GB-A")

    print("[INFO] Generating expressions for GB pose B (per outfit)...")
    generate_and_review_expressions_for_pose(api_key, gb_char_dir, gb_b_dir, "GB-B")

    # ----- GB Finalization -----
    finalize_character(gb_char_dir, gb_display_name, gb_voice, game_name)


    # =========================
    # Expressions per pose/outfit
    # =========================

    print("[INFO] Generating expressions for pose A (per outfit)...")
    generate_and_review_expressions_for_pose(a_dir, "A")

    print("[INFO] Generating expressions for pose B (per outfit)...")
    generate_and_review_expressions_for_pose(b_dir, "B")

    print("[INFO] Generating expressions for pose C (per outfit)...")
    generate_and_review_expressions_for_pose(c_dir, "C")

    # =========================
    # Eye line, name_color, scale
    # =========================

    rep_outfit = pick_representative_outfit(char_dir)

    print("[INFO] Collecting eye line and name color...")
    eye_line, name_color = prompt_for_eye_and_hair(rep_outfit)

    print("[INFO] Collecting scale vs reference...")
    scale = prompt_for_scale(rep_outfit, user_eye_line_ratio=eye_line)

    # =========================
    # Flatten + character.yml
    # =========================

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

    print(f"=== Finished character: {display_name} ({char_folder_name}) ===")


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
    for a *single* character at a time, using either:

      - An existing image file (chosen via file dialog), or
      - A new character generated from a text prompt.

    The old batch mode by scanning --input-dir is intentionally removed in
    favor of a more interactive one-at-a-time workflow.
    """
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Student Transfer sprite builder using Google Gemini:\n"
            "  - base pose, new pose, gender-bent pose\n"
            "  - outfits (Base/Formal/Casual)\n"
            "  - expressions 0..4 per outfit\n"
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

    # Seed RNG with strong OS entropy so different runs do not repeat the same outfit/name picks.
    random.seed(int.from_bytes(os.urandom(16), "big"))

    api_key = get_api_key()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir
    game_name: Optional[str] = args.game_name

    if not input_dir.exists():
        print(f"[WARN] Input directory does not exist: {input_dir}")

    outfit_db = load_outfit_prompts(OUTFIT_CSV_PATH)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- New: ask how to start (image vs prompt) ---
    mode = prompt_source_mode()

    if mode == "image":
        # Use a file chooser to pick a single character image.
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

    else:  # mode == "prompt"
        # Get concept + voice + name + archetype up front
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

        # Now run the usual pipeline, reusing the preselected info
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