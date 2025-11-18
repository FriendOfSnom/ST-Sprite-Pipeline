#!/usr/bin/env python3
"""
gemini_character_builder.py

Simplified Gemini sprite builder:

- Input: folder with one image per character.
- For each image:
  * Ask for an archetype (young man, adult woman, etc.).
  * Generate poses:
      - a: normalized base pose (crop, green bg)
      - b: new pose
      - c: gender-bent pose
  * For each pose:
      - Generate outfits first, using single-image edits.
        - Outfit prompts are pulled from outfit_prompts.csv by archetype + outfit_key.
        - One random prompt per outfit_key is chosen for the character and reused
          across poses (a and b share prompts; c uses its own).
        - User can accept or regenerate outfits for each pose.
      - Then generate expressions (simple, 9 slots), with accept/regenerate.

- Output layout:
    <output_root>/<char_name>/<pose_label>/
        base.png
        outfits/base.png + <outfits>.png
        faces/0.png ... faces/8.png
"""

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import webbrowser
import requests
import random

from rembg import remove
from io import BytesIO
from collections import Counter
from PIL import Image


# ----------------------- Gemini API configuration ---------------------

CONFIG_PATH = Path.home() / ".st_gemini_config.json"

GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_IMAGE_MODEL}:generateContent"
)

# CSV with outfit prompts; same folder as this script.
OUTFIT_CSV_PATH = Path(__file__).with_name("outfit_prompts.csv")


# ---------------- expressions (simplified descriptions) --------------------


EXPRESSIONS: Dict[str, str] = {
    "0": "keeping the same style and head position, a neutral and relaxed expression",
    "1": "keeping the same style and head position, a neutral and relaxed expression, but like they are talking and their mouth is open.",
    "2": "keeping the same style and head position, a playfully winking expression",
    "3": "keeping the same style and head position, a big happy smile, like they are laughing or very excited",
    "4": "keeping the same style and head position, a sad expression, like they are hurt or about to cry",
    "5": "keeping the same style and head position, an annoyed or angry expression",
    "6": "keeping the same style and head position, a surprised expression, like they just saw something unexpected",
    "7": "keeping the same style and head position, a worried or anxious expression",
    "8": "keeping the same style and head position, a shy and embarassed expression, with a dreamy smile",
    "9": "keeping the same style and head position, a disgusted expression, like something is gross",
}


# ---------------------- gender archetype options ----------------------

# (label, gender_style)
# gender_style is 'f' or 'm' so the rest of the pipeline can stay simple.
GENDER_ARCHETYPES = [
    ("young woman",   "f"),
    ("adult woman",   "f"),
    ("motherly woman","f"),
    ("young man",     "m"),
    ("adult man",     "m"),
    ("fatherly man",  "m"),
]

# Outfit types we generate per pose (besides the base outfit).
OUTFIT_KEYS: List[str] = ["formal", "casual"]


# ------------------------- config helpers -----------------------------

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
    """Write config to disk with best-effort user-only permissions."""
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass


def interactive_api_key_setup() -> str:
    """Prompt the user to create/paste a Gemini API key and save it."""
    print("\nIt looks like you haven't configured a Gemini API key yet.")
    print("To use Google Gemini's image model, we need an API key.")
    print("I will open the Gemini API key page in your browser.")
    print("Please create a new API key (if you don't already have one), then copy it.")
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
    print("You will not be asked for it again unless you delete that file.")
    return api_key


def get_api_key() -> str:
    """
    Fetch the Gemini API key from environment or config,
    or run the interactive setup on first use.
    """
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    cfg = load_config()
    if cfg.get("api_key"):
        return cfg["api_key"]

    return interactive_api_key_setup()


# --------------------------- outfit prompts CSV -----------------------------

def load_outfit_prompts(csv_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Load outfit prompts from a CSV file.

    CSV format (with header):
        archetype,outfit_key,prompt

    Returns:
        {
          "young man": {
            "formal": [prompt1, prompt2, ...],
            "casual": [...],
          },
          "adult woman": {...},
          ...
        }
    """
    db: Dict[str, Dict[str, List[str]]] = {}

    if not csv_path.is_file():
        print(f"[WARN] Outfit CSV not found at {csv_path}. Falling back to generic prompts.")
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
    """
    Fallback generic outfit description if no CSV prompt is available.
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
    return f"a simple outfit that fits this {gender_word}'s personality"


def choose_outfit_prompts_for_archetype(
    archetype_label: str,
    gender_style: str,
    outfit_keys: List[str],
    outfit_db: Dict[str, Dict[str, List[str]]],
) -> Dict[str, str]:
    """
    For a given archetype and gender, choose one prompt per outfit_key.

    If the CSV has prompts for that archetype + outfit_key, pick one at random.
    Otherwise, fall back to build_simple_outfit_description.
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


# --------------------------- image helpers ----------------------------

def load_image_as_base64(path: Path) -> str:
    """Return the file at `path` as a base64-encoded string."""
    with path.open("rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode("utf-8")


def strip_background(image_bytes: bytes) -> bytes:
    """
    Remove a flat or near-flat green-screen background while trying hard
    not to eat legitimate green clothing/details.

    Strategy:
      1. Sample the border to estimate the background color.
      2. Build a background mask (looser on the border, stricter inside).
      3. Remove connected components that touch the border (true background)
         plus small/avg-color-close 'holes' inside the sprite.
      4. Do a conservative edge cleanup to remove green halos.
      5. If nothing useful was removed, fall back to rembg.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        width, height = image.size
        pixels = image.load()

        # -------- 1) Sample border colors to estimate background --------
        border_thickness = max(4, min(width, height) // 20)
        sample_colors: List[tuple[int, int, int]] = []

        def add_samples(x0: int, y0: int, x1: int, y1: int) -> None:
            for y in range(y0, y1):
                for x in range(x0, x1):
                    r, g, b, a = pixels[x, y]
                    if a > 0:
                        sample_colors.append((r, g, b))

        # Sample all borders.
        add_samples(0, 0, width, border_thickness)
        add_samples(0, height - border_thickness, width, height)
        add_samples(0, 0, border_thickness, height)
        add_samples(width - border_thickness, 0, width, height)

        if not sample_colors:
            # No opaque border pixels; let rembg handle it.
            return remove(image_bytes)

        bucket_size = 16

        def quantize_color(c: tuple[int, int, int]) -> tuple[int, int, int]:
            r, g, b = c
            return (r // bucket_size, g // bucket_size, b // bucket_size)

        bucket_counter: Counter[tuple[int, int, int]] = Counter(
            quantize_color(c) for c in sample_colors
        )
        most_common = bucket_counter.most_common(1)
        if not most_common:
            return remove(image_bytes)

        bg_bucket, _ = most_common[0]
        cluster_pixels = [c for c in sample_colors if quantize_color(c) == bg_bucket]
        if not cluster_pixels:
            return remove(image_bytes)

        avg_r = sum(c[0] for c in cluster_pixels) / len(cluster_pixels)
        avg_g = sum(c[1] for c in cluster_pixels) / len(cluster_pixels)
        avg_b = sum(c[2] for c in cluster_pixels) / len(cluster_pixels)
        bg_color = (avg_r, avg_g, avg_b)

        # Helper to recognize "true screen green" (close to #00FF00)
        def is_screen_green(r: int, g: int, b: int) -> bool:
            # Very bright green, very little red/blue
            return g >= 220 and r <= 25 and b <= 25

        # Slightly looser threshold on the border, stricter in the interior
        interior_conn_threshold = 26.0
        border_conn_threshold = 36.0
        interior_conn_threshold_sq = interior_conn_threshold ** 2
        border_conn_threshold_sq = border_conn_threshold ** 2

        def color_dist_sq(c1: tuple[float, float, float], c2: tuple[float, float, float]) -> float:
            dr = c1[0] - c2[0]
            dg = c1[1] - c2[1]
            db = c1[2] - c2[2]
            return dr * dr + dg * dg + db * db

        def is_background_like(r: int, g: int, b: int, *, on_border: bool) -> bool:
            # Absolute "screen green" rule first.
            if is_screen_green(r, g, b):
                return True
            dist_sq = color_dist_sq((r, g, b), bg_color)
            if on_border:
                return dist_sq <= border_conn_threshold_sq
            return dist_sq <= interior_conn_threshold_sq

        # -------- 2) Build initial background mask --------
        bg_mask = [[False] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                on_border = (x < 2) or (y < 2) or (x > width - 3) or (y > height - 3)
                if is_background_like(r, g, b, on_border=on_border):
                    bg_mask[y][x] = True

        from collections import deque

        visited = [[False] * width for _ in range(height)]
        neighbors_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def bfs_component(start_x: int, start_y: int) -> dict:
            """
            Flood-fill a connected background-like component and compute:
            - list of pixels
            - whether it touches the image border
            - average color
            """
            queue: "deque[tuple[int, int]]" = deque()
            queue.append((start_x, start_y))
            visited[start_y][start_x] = True

            comp_pixels: List[tuple[int, int]] = []
            sum_r = sum_g = sum_b = 0
            touches_border = False

            while queue:
                x, y = queue.popleft()
                comp_pixels.append((x, y))
                r, g, b, a = pixels[x, y]
                sum_r += r
                sum_g += g
                sum_b += b

                if x == 0 or x == width - 1 or y == 0 or y == height - 1:
                    touches_border = True

                for dx, dy in neighbors_4:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    if visited[ny][nx]:
                        continue
                    if not bg_mask[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    queue.append((nx, ny))

            n = len(comp_pixels)
            if n > 0:
                avg_cr = sum_r / n
                avg_cg = sum_g / n
                avg_cb = sum_b / n
            else:
                avg_cr = avg_cg = avg_cb = 0

            return {
                "pixels": comp_pixels,
                "touches_border": touches_border,
                "avg_color": (avg_cr, avg_cg, avg_cb),
            }

        components = []
        for y in range(height):
            for x in range(width):
                if not bg_mask[y][x] or visited[y][x]:
                    continue
                components.append(bfs_component(x, y))

        if not components:
            # No background-like components; let rembg handle it.
            return remove(image_bytes)

        # -------- 3) Remove border components + "hole" components --------
        hole_color_threshold = 18.0
        hole_color_threshold_sq = hole_color_threshold ** 2

        # Shrink "tiny hole" size a bit so we only auto-nuke genuinely tiny blobs
        max_hole_size = (width * height) // 25
        tiny_hole_size = max(5, (width * height) // 1200)

        removed_any = False

        for comp in components:
            pixels_list = comp["pixels"]
            touches_border = comp["touches_border"]
            avg_c = comp["avg_color"]
            comp_size = len(pixels_list)

            if touches_border:
                # Big outer background; always remove it.
                for x, y in pixels_list:
                    r, g, b, a = pixels[x, y]
                    if a != 0:
                        pixels[x, y] = (r, g, b, 0)
                        removed_any = True
            else:
                # Small "holes" are almost certainly background specks inside hair, etc.
                if comp_size <= tiny_hole_size:
                    for x, y in pixels_list:
                        r, g, b, a = pixels[x, y]
                        if a != 0:
                            pixels[x, y] = (r, g, b, 0)
                            removed_any = True
                    continue

                # Very large components are suspicious; skip them unless color matches.
                if comp_size > max_hole_size:
                    continue

                # Medium-sized components: only remove if their average color
                # is very close to the background color.
                dist_sq = color_dist_sq(avg_c, bg_color)
                if dist_sq <= hole_color_threshold_sq:
                    for x, y in pixels_list:
                        r, g, b, a = pixels[x, y]
                        if a != 0:
                            pixels[x, y] = (r, g, b, 0)
                            removed_any = True

        # -------- 4) Edge halo cleanup (much more conservative) --------
        if not removed_any:
            # If we didn't remove anything at all, this likely isn't a green-screen.
            # Don't run halo cleanup; just let rembg have a go.
            return remove(image_bytes)

        edge_threshold = 40.0
        edge_threshold_sq = edge_threshold * edge_threshold

        def is_bg_tinted_pixel(r: int, g: int, b: int) -> bool:
            """
            Returns True for pixels that are very close to the background color
            (or pure screen green). We no longer use a pure "g >> r,b" heuristic
            because it tends to eat real clothing/eye greens.
            """
            if is_screen_green(r, g, b):
                return True
            dist_sq = color_dist_sq((r, g, b), bg_color)
            return dist_sq <= edge_threshold_sq

        to_clear_edge: List[tuple[int, int]] = []
        max_radius = 2  # smaller radius: stay close to the actual matte edge

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]
                if a == 0:
                    continue
                if not is_bg_tinted_pixel(r, g, b):
                    continue

                # Only treat a pixel as halo if it has at least *two* transparent
                # neighbors in a small window — single transparent neighbor often
                # happens in fine details (eyes, folds) we want to preserve.
                trans_neighbors = 0
                for ny in range(max(0, y - max_radius), min(height, y + max_radius + 1)):
                    for nx in range(max(0, x - max_radius), min(width, x + max_radius + 1)):
                        if nx == x and ny == y:
                            continue
                        _, _, _, na = pixels[nx, ny]
                        if na == 0:
                            trans_neighbors += 1
                            if trans_neighbors >= 2:
                                to_clear_edge.append((x, y))
                                ny = height  # break both loops
                                break

        for x, y in to_clear_edge:
            r, g, b, a = pixels[x, y]
            pixels[x, y] = (r, g, b, 0)

        # -------- 5) Encode as PNG --------
        out_buf = BytesIO()
        image.save(out_buf, format="PNG")
        return out_buf.getvalue()

    except Exception as e:
        print(f"  [WARN] Background stripping failed, using original image bytes: {e}")
        return image_bytes


def save_bytes(path: Path, data: bytes) -> None:
    """Write raw bytes to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(data)


# ---------------------- Gemini HTTP helper --------------------


def call_gemini_image_edit(api_key: str, prompt: str, image_b64: str) -> bytes:
    """
    Call the Gemini image model once with a text prompt and a single input image.
    Returns PNG bytes with background stripped when possible.
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
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    resp = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
    if not resp.ok:
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

    data = resp.json()
    candidates = data.get("candidates", [])
    for cand in candidates:
        content = cand.get("content", {})
        for part in content.get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and "data" in blob:
                raw_bytes = base64.b64decode(blob["data"])
                return strip_background(raw_bytes)

    raise RuntimeError("No image data found in Gemini response.")


# ---------------------- prompt construction helpers ------------------


def build_initial_pose_prompt(gender_style: str, char_name: Optional[str] = None) -> str:
    """
    Prompt to normalize the original sprite:
    - Same character, outfit, and expression.
    - Crop to mid-thigh up.
    - Put on flat green background for keying.
    """
    return (
        "Edit the input image of our visual novel character, and if it is not already, crop the character so we only see from the mid-thigh up. "
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit have none of the background color on them. "
        "Do not change any design details of the character; only reframe and place them onto that green background."
    )


def build_pose_prompt(gender_style: str, char_name: Optional[str] = None) -> str:
    """
    Prompt to generate a second pose while keeping design, crop and outfit fixed.
    """
    gender_clause = "feminine" if gender_style == "f" else "masculine"
    if gender_style == "f":
        pose_examples = (
            "a clearly different cute or cool feminine pose that is not too over the top, but would fit well in a visual novel"
        )
    else:
        pose_examples = (
            "a clearly different cool or relaxed masculine pose that is not too over the top, but would fit well in a visual novel"
        )

    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Make sure the character is in a new pose that matches this description: {pose_examples}. "
        "Do not change anything about the character or image besides the pose of the character, including the crop from the mid-thigh up, image size, and outfit. "
        "Change the hair style of the character to match the new pose, but do not change the overall hair design or color. "
        "The character should not be holding anything in their hand, and should be cropped from the mid-thigh on up."
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit have none of the background color on them. "
        "In other words, keep everything the same about the input image, just change the pose the character is in, and make sure to only show from the mid-thigh on up."
    )


def build_genderbend_pose_prompt(
    original_gender_style: str,
    target_gender_style: str,
    target_archetype: str,
    char_name: Optional[str] = None,
) -> str:
    """
    Prompt to generate a gender-bent version of the character with a new archetype.
    """
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
        "Change the hairstyle to better match the new pose and gender presentation, but keep the hair recognizably "
        "related to the original design. "
        "Change the body shape and clothing so that the character clearly presents as the new gender, but keep the "
        "overall crop from the mid-thigh up and do not change the image size. "
        "Use a pure, flat green background (#00FF00) behind the character, and make sure the character and outfit "
        "have none of the background color on them."
    )


def build_expression_prompt(expression_desc: str, char_name: Optional[str] = None) -> str:
    """
    Prompt to change only the facial expression.
    """
    return (
        "Edit the inputed visual novel sprite, in the same art style. "
        f"Only change the facial expression to match this description: {expression_desc}. "
        "Do not change anything about the character or image besides the expression, including the crop from the mid-thigh up, image size, pose, outfit, hair style, or how the character's hair frames their face. "
        "Do not move the arms, legs, or change the head tilt. Please just edit the face."
        "Keep the background exactly the same as in the input."
    )


def build_outfit_prompt(
    base_outfit_desc: str,
    gender_style: str,
    char_name: Optional[str] = None,
) -> str:
    """
    Prompt to change only the clothing to `base_outfit_desc` on the given pose.
    """
    gender_clause = "girl" if gender_style == "f" else "boy"
    return (
        f"Edit the inputed {gender_clause} visual novel sprite, in the same art style. "
        f"Please change only the clothing and outfit to match this description: {base_outfit_desc}. "
        "Do not change the pose, head tilt, body proportions, crop from the mid-thigh up, image size, hairstyle, or background. "
        "In other words, keep everything the same about the input image, just change the outfit."
    )


# ------------------------- character pipeline helpers -------------------------


def generate_initial_pose(
    api_key: str,
    image_path: Path,
    out_pose_path: Path,
    gender_style: str,
    char_name: str,
) -> None:
    """
    Normalize the original sprite into pose 'a' base.png:
    - Crop to mid-thigh up.
    - Put on a flat green background.
    - Keep character design as-is.
    """
    print(f"  [Gemini] Normalizing base pose for {char_name}...")
    image_b64 = load_image_as_base64(image_path)
    prompt = build_initial_pose_prompt(gender_style=gender_style, char_name=char_name)

    try:
        img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to normalize base pose for {char_name}: {e}")

    save_bytes(out_pose_path, img_bytes)
    print(f"  Saved normalized base pose to: {out_pose_path}")


def generate_second_pose(
    api_key: str,
    base_image_path: Path,
    out_pose_path: Path,
    gender_style: str,
    char_name: str,
) -> None:
    """
    Interactively generate a second pose (pose 'b') from pose 'a'.
    """
    image_b64 = load_image_as_base64(base_image_path)

    while True:
        prompt = build_pose_prompt(gender_style=gender_style, char_name=char_name)
        print(f"  [Gemini] Requesting new pose for {char_name}...")
        try:
            img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        except RuntimeError as e:
            print(f"  [ERROR] Failed to generate pose: {e}")
            choice = input("    Retry pose generation? [y/N]: ").strip().lower()
            if choice != "y":
                raise
            continue

        save_bytes(out_pose_path, img_bytes)
        print(f"  Saved new pose to: {out_pose_path}")
        print("  Please open that image in your image viewer to inspect it.")
        choice = input("  Do you accept this pose? [y/N/r] (y = yes, r = regenerate): ").strip().lower()
        if choice == "y":
            print("  Pose accepted.")
            return
        if choice == "r":
            print("  Regenerating pose...")
            continue
        raise RuntimeError("User aborted pose generation.")


def generate_genderbend_pose(
    api_key: str,
    base_image_path: Path,
    out_pose_path: Path,
    original_gender_style: str,
    target_gender_style: str,
    target_archetype: str,
    char_name: str,
) -> None:
    """
    Interactively generate a gender-bent pose (pose 'c') from pose 'a'.
    """
    image_b64 = load_image_as_base64(base_image_path)

    while True:
        prompt = build_genderbend_pose_prompt(
            original_gender_style=original_gender_style,
            target_gender_style=target_gender_style,
            target_archetype=target_archetype,
            char_name=char_name,
        )
        print(f"  [Gemini] Requesting gender-bent pose for {char_name} as a {target_archetype}...")
        try:
            img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        except RuntimeError as e:
            print(f"  [ERROR] Failed to generate gender-bent pose: {e}")
            choice = input("    Retry gender-bent pose generation? [y/N]: ").strip().lower()
            if choice != "y":
                raise
            continue

        save_bytes(out_pose_path, img_bytes)
        print(f"  Saved gender-bent pose to: {out_pose_path}")
        print("  Please open that image in your image viewer to inspect it.")
        choice = input(
            "  Do you accept this gender-bent pose? [y/N/r] (y = yes, r = regenerate): "
        ).strip().lower()
        if choice == "y":
            print("  Gender-bent pose accepted.")
            return
        if choice == "r":
            print("  Regenerating gender-bent pose...")
            continue
        raise RuntimeError("User aborted gender-bent pose generation.")


def generate_expressions_for_pose(
    api_key: str,
    base_pose_path: Path,
    faces_dir: Path,
    char_name: str,
) -> None:
    """
    Generate the full set of expressions for a single pose image (single pass).
    """
    image_b64 = load_image_as_base64(base_pose_path)
    faces_dir.mkdir(parents=True, exist_ok=True)

    for key, desc in EXPRESSIONS.items():
        out_path = faces_dir / f"{key}.png"
        print(f"  [Gemini] Generating expression '{key}' → {out_path.name}")
        prompt = build_expression_prompt(expression_desc=desc, char_name=char_name)
        try:
            img_bytes = call_gemini_image_edit(api_key, prompt, image_b64)
        except RuntimeError as e:
            print(f"    [ERROR] Failed to generate expression '{key}': {e}")
            continue
        save_bytes(out_path, img_bytes)
        print(f"    Saved {out_path}")


def generate_expressions_with_retry(
    api_key: str,
    base_pose_path: Path,
    faces_dir: Path,
    char_name: str,
    pose_label: str,
) -> None:
    """
    Wrapper around `generate_expressions_for_pose` with accept/regenerate loop.
    """
    while True:
        if faces_dir.exists():
            for png in faces_dir.glob("*.png"):
                try:
                    png.unlink()
                except Exception:
                    pass

        print(f"  Generating expressions for pose {pose_label}...")
        generate_expressions_for_pose(api_key, base_pose_path, faces_dir, char_name)

        print(f"  Expressions for pose {pose_label} are in: {faces_dir}")
        choice = input(
            f"  Do you accept the expressions for pose {pose_label}? [y/N/r] (y = yes, r = regenerate): "
        ).strip().lower()

        if choice == "y":
            print(f"  Expressions for pose {pose_label} accepted.")
            return
        if choice == "r":
            print(f"  Regenerating expressions for pose {pose_label}...")
            continue
        raise RuntimeError("User aborted expression generation.")


def generate_outfits_for_pose_single_image(
    api_key: str,
    base_pose_path: Path,
    outfits_dir: Path,
    gender_style: str,
    char_name: str,
    outfit_descriptions: Dict[str, str],
    pose_label: str,
) -> None:
    """
    Generate a simple set of outfits for a single pose using single-image edits.

    - Copies the base pose as outfits/base.png.
    - For each outfit key in `outfit_descriptions`, runs one Gemini edit with the
      corresponding text prompt.
    - After generating the whole set, asks the user to accept or regenerate.
    """
    while True:
        outfits_dir.mkdir(parents=True, exist_ok=True)

        # Clear old outfiles so regen does not mix files.
        for png in outfits_dir.glob("*.png"):
            try:
                png.unlink()
            except Exception:
                pass

        base_bytes = base_pose_path.read_bytes()
        base_out_path = outfits_dir / "base.png"
        save_bytes(base_out_path, base_bytes)
        print(f"  Copied base pose as default outfit for pose {pose_label}: {base_out_path}")

        image_b64 = load_image_as_base64(base_pose_path)

        for key, desc in outfit_descriptions.items():
            out_path = outfits_dir / f"{key}.png"
            print(
                f"  [Gemini] Generating outfit '{key}' for {char_name} on pose {pose_label} "
                "using single-image outfit prompt..."
            )
            prompt = build_outfit_prompt(
                base_outfit_desc=desc,
                gender_style=gender_style,
                char_name=char_name,
            )
            try:
                img_bytes = call_gemini_image_edit(
                    api_key=api_key,
                    prompt=prompt,
                    image_b64=image_b64,
                )
            except RuntimeError as e:
                print(f"    [ERROR] Failed to generate outfit '{key}': {e}")
                continue

            save_bytes(out_path, img_bytes)
            print(f"    Saved outfit '{key}' to {out_path}")

        print(f"  Finished generating outfits for pose {pose_label} in: {outfits_dir}")
        print("  Please open those images in your viewer to inspect them.")
        choice = input(
            f"  Do you accept the outfits for pose {pose_label}? [y/N/r] (y = yes, r = regenerate): "
        ).strip().lower()

        if choice == "y":
            print(f"  Outfits for pose {pose_label} accepted.")
            return
        if choice == "r":
            print(f"  Regenerating outfits for pose {pose_label}...")
            continue
        raise RuntimeError("User aborted outfit generation.")


# ------------------------- main character pipeline -------------------------


def process_single_character(
    api_key: str,
    image_path: Path,
    output_root: Path,
    outfit_db: Dict[str, Dict[str, List[str]]],
) -> None:
    """
    Run the full pipeline for a single character image:
    - Ask for gender archetype.
    - Generate poses a (normalized), b (new pose), c (gender-bent).
    - For each pose:
        * Outfits first (single-image edits, using per-character prompts).
        * Then expressions (with accept/regenerate).
    """
    char_name = image_path.stem
    print(f"\n=== Processing character: {char_name} ===")
    print(f"  Source image: {image_path}")

    # Archetype selection for original gender.
    print("  How would you describe this character?")
    for idx, (label, _) in enumerate(GENDER_ARCHETYPES, start=1):
        print(f"    {idx}) {label}")
    while True:
        choice = input("  Enter a number (1-6) for the best fit: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(GENDER_ARCHETYPES):
                original_archetype_label, gender_style = GENDER_ARCHETYPES[idx - 1]
                break
        print("  Please enter a valid number from the list above.")

    print(
        f"  Selected archetype for {char_name}: "
        f"{original_archetype_label} ({'feminine' if gender_style == 'f' else 'masculine'})"
    )

    # Choose outfit prompts for this archetype (used for poses a and b).
    original_outfit_prompts = choose_outfit_prompts_for_archetype(
        archetype_label=original_archetype_label,
        gender_style=gender_style,
        outfit_keys=OUTFIT_KEYS,
        outfit_db=outfit_db,
    )

    char_dir = output_root / char_name

    # Pose a: normalized from original image.
    a_dir = char_dir / "a"
    a_dir.mkdir(parents=True, exist_ok=True)
    a_base = a_dir / "base.png"
    generate_initial_pose(
        api_key=api_key,
        image_path=image_path,
        out_pose_path=a_base,
        gender_style=gender_style,
        char_name=char_name,
    )

    # Pose b: new pose from pose a.
    b_dir = char_dir / "b"
    b_dir.mkdir(parents=True, exist_ok=True)
    b_base = b_dir / "base.png"
    generate_second_pose(
        api_key=api_key,
        base_image_path=a_base,
        out_pose_path=b_base,
        gender_style=gender_style,
        char_name=char_name,
    )

    # Pose c: gender-bent from pose a.
    gb_gender_style = "f" if gender_style == "m" else "m"
    opposite_options = [label for (label, g) in GENDER_ARCHETYPES if g == gb_gender_style]

    print("\n  For the gender-bent version of this character, how should they present?")
    for idx, label in enumerate(opposite_options, start=1):
        print(f"    {idx}) {label}")
    while True:
        gb_choice = input(f"  Enter a number (1-{len(opposite_options)}): ").strip()
        if gb_choice.isdigit():
            idx = int(gb_choice)
            if 1 <= idx <= len(opposite_options):
                gb_archetype_label = opposite_options[idx - 1]
                break
        print("  Please enter a valid number from the list above.")

    print(
        f"  Selected gender-bent archetype for {char_name}: "
        f"{gb_archetype_label} ({'feminine' if gb_gender_style == 'f' else 'masculine'})"
    )

    # Choose outfit prompts for the gender-bent archetype (pose c).
    gb_outfit_prompts = choose_outfit_prompts_for_archetype(
        archetype_label=gb_archetype_label,
        gender_style=gb_gender_style,
        outfit_keys=OUTFIT_KEYS,
        outfit_db=outfit_db,
    )

    c_dir = char_dir / "c"
    c_dir.mkdir(parents=True, exist_ok=True)
    c_base = c_dir / "base.png"
    generate_genderbend_pose(
        api_key=api_key,
        base_image_path=a_base,
        out_pose_path=c_base,
        original_gender_style=gender_style,
        target_gender_style=gb_gender_style,
        target_archetype=gb_archetype_label,
        char_name=char_name,
    )

    # ----------------- outfits (all poses first) -----------------

    print("  Generating outfits for poses a, b, and c...")

    a_outfits_dir = a_dir / "outfits"
    generate_outfits_for_pose_single_image(
        api_key=api_key,
        base_pose_path=a_base,
        outfits_dir=a_outfits_dir,
        gender_style=gender_style,
        char_name=char_name,
        outfit_descriptions=original_outfit_prompts,
        pose_label="a",
    )

    b_outfits_dir = b_dir / "outfits"
    generate_outfits_for_pose_single_image(
        api_key=api_key,
        base_pose_path=b_base,
        outfits_dir=b_outfits_dir,
        gender_style=gender_style,
        char_name=char_name,
        outfit_descriptions=original_outfit_prompts,
        pose_label="b",
    )

    c_outfits_dir = c_dir / "outfits"
    generate_outfits_for_pose_single_image(
        api_key=api_key,
        base_pose_path=c_base,
        outfits_dir=c_outfits_dir,
        gender_style=gb_gender_style,
        char_name=char_name,
        outfit_descriptions=gb_outfit_prompts,
        pose_label="c (gender-bent)",
    )

    # ----------------- expressions (after outfits) -----------------

    print("  Generating expressions for poses a, b, and c...")

    generate_expressions_with_retry(
        api_key=api_key,
        base_pose_path=a_base,
        faces_dir=a_dir / "faces",
        char_name=char_name,
        pose_label="a",
    )

    generate_expressions_with_retry(
        api_key=api_key,
        base_pose_path=b_base,
        faces_dir=b_dir / "faces",
        char_name=char_name,
        pose_label="b",
    )

    generate_expressions_with_retry(
        api_key=api_key,
        base_pose_path=c_base,
        faces_dir=c_dir / "faces",
        char_name=char_name,
        pose_label="c (gender-bent)",
    )

    print(f"=== Finished character: {char_name} ===")


# --------------------------- CLI entrypoint ---------------------------

def find_character_images(input_dir: Path) -> List[Path]:
    """Return all image files in `input_dir` that could be characters."""
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]
    images.sort()
    return images


def main() -> None:
    """Parse arguments, validate API key, load outfit CSV, and run the pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Build VN-ready character assets using Google Gemini image editing:\n"
            "  - normalized base pose\n"
            "  - second pose\n"
            "  - gender-bent pose\n"
            "  - outfit set per pose (single-image edits, CSV-driven prompts)\n"
            "  - expression set per pose\n"
            "Input: a folder with one image per character.\n"
            "Output: structured char/pose/faces/outfits folders."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing one image per character.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root folder to write generated character assets.",
    )
    args = parser.parse_args()

    random.seed(None)

    api_key = get_api_key()

    input_dir: Path = args.input_dir
    output_root: Path = args.output_dir

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    images = find_character_images(input_dir)
    if not images:
        raise SystemExit(f"No character images found in: {input_dir}")

    outfit_db = load_outfit_prompts(OUTFIT_CSV_PATH)

    print(f"Found {len(images)} character image(s) to process.")
    for image_path in images:
        process_single_character(api_key, image_path, output_root, outfit_db)

    print("All characters processed.")


if __name__ == "__main__":
    main()
