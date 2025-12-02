"""
Gemini API client for sprite generation.

Handles authentication, API calls, retries, and response parsing for Google Gemini.
"""

import base64
import json
import os
import webbrowser
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image

from gsp_constants import CONFIG_PATH, GEMINI_API_URL


# =============================================================================
# Configuration Management
# =============================================================================

def load_config() -> dict:
    """
    Load configuration from ~/.st_gemini_config.json if present.

    Returns:
        Dictionary containing configuration, or empty dict if not found.
    """
    if CONFIG_PATH.is_file():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    """
    Save configuration dictionary to CONFIG_PATH.

    Sets file permissions to 0o600 for security (API keys).

    Args:
        config: Configuration dictionary to save.
    """
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except Exception:
        pass  # Permissions may not be supported on all platforms


def interactive_api_key_setup() -> str:
    """
    Prompt user for Gemini API key and save it to config.

    Opens browser to API key page and prompts for input.

    Returns:
        The API key entered by the user.

    Raises:
        SystemExit: If no API key is entered.
    """
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

    config = load_config()
    config["api_key"] = api_key
    save_config(config)
    print(f"Saved API key to {CONFIG_PATH}.")
    return api_key


def get_api_key() -> str:
    """
    Return Gemini API key from environment variable or config file.

    Checks GEMINI_API_KEY environment variable first, then config file.
    If neither exists, prompts user interactively.

    Returns:
        Valid Gemini API key.
    """
    # Check environment variable first
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key

    # Check config file
    config = load_config()
    if config.get("api_key"):
        return config["api_key"]

    # Interactive setup if neither exists
    return interactive_api_key_setup()


# =============================================================================
# Image Utilities
# =============================================================================

def load_image_as_base64(path: Path) -> str:
    """
    Load image from disk, re-encode as PNG, return base64 string.

    Ensures consistent format for Gemini API regardless of source format.

    Args:
        path: Path to image file.

    Returns:
        Base64-encoded PNG image data.
    """
    img = Image.open(path).convert("RGBA")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    raw_bytes = buffer.getvalue()
    return base64.b64encode(raw_bytes).decode("utf-8")


def _extract_inline_image_from_response(data: dict) -> Optional[bytes]:
    """
    Extract the first inline image bytes from a Gemini JSON response.

    Handles both 'inlineData' and 'inline_data' field naming.

    Args:
        data: Parsed JSON response from Gemini API.

    Returns:
        Decoded image bytes, or None if no image found.
    """
    candidates = data.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            # Handle both naming conventions
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and "data" in blob:
                return base64.b64decode(blob["data"])
    return None


# =============================================================================
# Background Removal (ML-Based)
# =============================================================================

def strip_background(image_bytes: bytes) -> bytes:
    """
    Remove background from image using ML-based method (rembg) with cleanup.

    Two-stage process:
    1. ML-based removal (rembg) - identifies subject vs background
    2. Cleanup pass - removes any remaining colored background pixels

    This handles edge cases like:
    - Leftover background fragments (pink/magenta areas)
    - Shadow halos and artifacts
    - Background "pockets" inside character shapes
    - Fine hair details preserved while removing surrounding pixels

    Falls back to legacy threshold method if rembg is not available.

    Args:
        image_bytes: Input image as bytes

    Returns:
        Image bytes with background removed (transparent PNG)

    Note:
        First run will download ~200MB model (cached for future use).
        Processing takes ~2-3 seconds per image on CPU.
    """
    try:
        from rembg import remove
        import numpy as np

        print("[INFO] Using ML-based background removal (rembg)...")

        # Stage 1: ML-based removal
        rembg_output = remove(image_bytes)

        # Stage 2: Cleanup pass - remove any remaining colored background
        img = Image.open(BytesIO(rembg_output)).convert("RGBA")
        img_array = np.array(img)

        # Sample border pixels to identify background color
        height, width = img_array.shape[:2]
        border_pixels = []

        # Sample top and bottom edges
        for x in range(0, width, max(1, width // 50)):  # Sample ~50 points per edge
            # Top edge
            if img_array[0, x, 3] > 0:  # If not transparent
                border_pixels.append(img_array[0, x, :3])
            # Bottom edge
            if img_array[height-1, x, 3] > 0:
                border_pixels.append(img_array[height-1, x, :3])

        # Sample left and right edges
        for y in range(0, height, max(1, height // 50)):
            # Left edge
            if img_array[y, 0, 3] > 0:
                border_pixels.append(img_array[y, 0, :3])
            # Right edge
            if img_array[y, width-1, 3] > 0:
                border_pixels.append(img_array[y, width-1, :3])

        # If we found border pixels, clean up matching colors
        if len(border_pixels) > 0:
            border_pixels = np.array(border_pixels)
            bg_color = np.median(border_pixels, axis=0)

            # Calculate color distance from background
            rgb = img_array[:, :, :3].astype(float)
            alpha = img_array[:, :, 3]

            # Calculate color distance for ALL pixels (including semi-transparent)
            color_dist = np.sqrt(np.sum((rgb - bg_color)**2, axis=2))

            # More aggressive cleanup:
            # 1. Remove fully/mostly opaque pixels close to background color
            BG_CLEANUP_THRESH = 50  # Slightly more forgiving threshold
            opaque_cleanup = (color_dist < BG_CLEANUP_THRESH) & (alpha > 20)

            # 2. Remove semi-transparent pixels that are grayish or close to background
            #    These are the weird gray artifacts rembg leaves behind
            semi_transparent = (alpha > 0) & (alpha < 200)
            is_grayish = (color_dist < 80)  # More aggressive for semi-transparent
            semi_cleanup = semi_transparent & is_grayish

            # Combine both cleanup masks
            cleanup_mask = opaque_cleanup | semi_cleanup
            img_array[cleanup_mask, 3] = 0  # Make transparent

            print(f"[INFO] Cleaned up {np.sum(cleanup_mask)} background/artifact pixels")

        # Convert back to image
        cleaned_img = Image.fromarray(img_array, mode="RGBA")
        buffer = BytesIO()
        cleaned_img.save(buffer, format="PNG")
        return buffer.getvalue()

    except ImportError:
        print("[WARN] rembg not installed. Falling back to legacy threshold method.")
        print("[WARN] Install with: pip install 'rembg>=2.0.57'")
        from .background_removal_legacy import strip_background_legacy
        return strip_background_legacy(image_bytes)
    except Exception as e:
        print(f"[WARN] rembg failed ({e}), falling back to legacy method...")
        from .background_removal_legacy import strip_background_legacy
        return strip_background_legacy(image_bytes)


# =============================================================================
# Gemini API Calls
# =============================================================================

def _call_gemini_with_parts(
    api_key: str,
    parts: List[dict],
    context: str
) -> bytes:
    """
    Call Gemini API with custom parts array and retry logic.

    Handles retries for transient errors (429, 500, 502, 503, 504).
    Strips background from returned image.

    Args:
        api_key: Google Gemini API key.
        parts: List of content parts (text, images, etc.).
        context: Description of the operation for error messages.

    Returns:
        Image bytes with background stripped.

    Raises:
        RuntimeError: If API call fails after all retries.
    """
    payload = {"contents": [{"parts": parts}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                data=json.dumps(payload)
            )

            # Handle retryable errors
            if not response.ok:
                if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    print(
                        f"[WARN] Gemini API error {response.status_code} ({context}) "
                        f"attempt {attempt}; retrying..."
                    )
                    last_error = f"Gemini API error {response.status_code}: {response.text}"
                    continue
                raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

            data = response.json()
            raw_bytes = _extract_inline_image_from_response(data)

            if raw_bytes is not None:
                return strip_background(raw_bytes)

            last_error = f"No image data in Gemini response ({context})."
            if attempt < max_retries:
                print(
                    f"[WARN] Gemini response missing image ({context}) "
                    f"attempt {attempt}; retrying..."
                )
                continue
            raise RuntimeError(last_error)

        except Exception as e:
            last_error = str(e)
            if attempt < max_retries:
                print(
                    f"[WARN] Gemini call failed ({context}) "
                    f"attempt {attempt}; retrying: {e}"
                )
                continue
            raise RuntimeError(
                f"Gemini call failed after {max_retries} attempts ({context}): {last_error}"
            )


def call_gemini_image_edit(api_key: str, prompt: str, image_b64: str) -> bytes:
    """
    Call Gemini image model with an input image and text prompt for editing.

    Args:
        api_key: Google Gemini API key.
        prompt: Text prompt describing the desired edit.
        image_b64: Base64-encoded input image.

    Returns:
        Generated/edited image bytes.
    """
    parts: List[dict] = [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/png", "data": image_b64}},
    ]
    return _call_gemini_with_parts(api_key, parts, "image_edit")


def call_gemini_text_or_refs(
    api_key: str,
    prompt: str,
    ref_images: Optional[List[Path]] = None
) -> bytes:
    """
    Call Gemini with a text prompt and optional reference images.

    Used for generating new characters from text descriptions with
    style references.

    Args:
        api_key: Google Gemini API key.
        prompt: Text prompt describing what to generate.
        ref_images: Optional list of reference image paths for style guidance.

    Returns:
        Generated image bytes.
    """
    parts: List[dict] = [{"text": prompt}]

    # Add reference images if provided
    if ref_images:
        for ref_path in ref_images:
            try:
                ref_b64 = load_image_as_base64(ref_path)
                parts.append({
                    "inline_data": {"mime_type": "image/png", "data": ref_b64}
                })
            except Exception as e:
                print(f"[WARN] Could not load reference image {ref_path}: {e}")

    return _call_gemini_with_parts(api_key, parts, "text_or_refs")
