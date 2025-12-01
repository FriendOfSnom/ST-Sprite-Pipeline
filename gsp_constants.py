#!/usr/bin/env python3
"""
gsp_constants.py

All global paths, constants, and static tables for the Gemini sprite pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base directory for scripts
SCRIPT_DIR = Path(__file__).resolve().parent

# Paths for configuration and data files
CONFIG_PATH = Path.home() / ".st_gemini_config.json"
OUTFIT_CSV_PATH = SCRIPT_DIR / "outfit_prompts.csv"
NAMES_CSV_PATH = SCRIPT_DIR / "names.csv"
REF_SPRITES_DIR = SCRIPT_DIR / "reference_sprites"

# Gemini API constants
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_IMAGE_MODEL}:generateContent"
)

# Background color we ask Gemini to use.
GBG_COLOR = (255, 0, 255)  # pure magenta (#FF00FF)

# Tk UI style constants
BG_COLOR = "lightgray"
TITLE_FONT = ("Arial", 16, "bold")
INSTRUCTION_FONT = ("Arial", 12)
LINE_COLOR = "#00E5FF"
WINDOW_MARGIN = 10
WRAP_PADDING = 40

# Outfit keys:
# Base is always included by the pipeline, but you can choose which additional
# outfits to generate. These keys should match outfit_key values in the CSV.
ALL_OUTFIT_KEYS: List[str] = ["formal", "casual", "uniform", "athletic", "swimsuit"]

# Default subset used when the user does not change anything.
OUTFIT_KEYS: List[str] = ["formal", "casual"]

# Expression description mapping (not all are always used)
EXPRESSIONS: Dict[str, str] = {
    "0": "a neutral and relaxed expression",
    "1": "a happy smile with a wink, like they are chuckling.",
    "2": "a sad expression, like they are hurt or about to cry",
    "3": "an annoyed or angry expression",
    "4": "a flushed, embarrassed and flustered expression, cheeks blushing, eyes a little unfocused",
    "5": "a big happy smile, like they are laughing or very excited",
    "6": "a surprised expression, like they just saw something unexpected",
    "7": "a worried or anxious expression",
    "8": "a mildly disgusted expression.",
}

# Default ordered list of expressions we actually use per outfit.
# The first entry is always neutral.
EXPRESSIONS_SEQUENCE: List[Tuple[str, str]] = [

    # =========================================
    # CORE EXPRESSIONS (0–9)
    # Most used in VN-style writing.
    # =========================================
    ("0",  "soft smile, mouth closed"),
    ("1",  "neutral, relaxed face, mouth open as if talking"),
    ("2",  "big happy smile, mouth open"),
    ("3",  "worried, eyebrows slightly furrowed, small frown"),
    ("4",  "angry, eyebrows sharply furrowed, mouth open"),
    ("5",  "embarrassed, light blush, eyes slightly averted"),
    ("6",  "surprised, wide eyes, mouth slightly open"),
    ("7",  "sad, eyes downcast, slight frown"),
    ("8",  "thinking, one eyebrow raised, mouth neutral"),
    ("9",  "closed-eye happy smile"),

    # =========================================
    # CORE+ EXPRESSIONS (10–14)
    # Very commonly needed and distinct.
    # =========================================

    ("10", "unimpressed, half-lidded eyes, flat mouth"),
    ("11", "laughing, eyes closed, mouth open wide"),
    ("12", "deadpan, half-lidded eyes, blank expression"),
    ("13", "shy smile, light blush, small soft smile"),
    ("14", "smug, self-satisfied grin, eyebrows slightly raised"),

    # =========================================
    # OPTIONAL EXPRESSIONS (15–24)
    # Less essential but high-value.
    # =========================================

    ("15", "mischievous smirk, one eyebrow raised"),
    ("16", "playful wink while smiling"),
    ("17", "pouting, cheeks slightly puffed, lips pressed together"),
    ("18", "tired, half-lidded eyes, small frown"),
    ("19", "nervous, wide eyes, tense small mouth"),
    ("20", "determined, eyebrows lowered, firm confident smile"),
    ("21", "about to cry, tears welling"),
    ("22", "crying, visible tears on cheeks"),
    ("23", "concerned, soft eyes, small downturned mouth"),

    # =========================================
    # SPICY SAFETY TIER (25–27)
    # The 'horny-adjacent' expressions that stay SFW.
    # =========================================

    ("25", "flustered, heavy blush, wide eyes, small tense mouth"),
    ("26", "bashful, looking slightly away, soft blush, timid smile"),
    ("27", "romantic longing, soft blush, half-lidded eyes, lips gently parted"),
]


# Archetypes and their gender style codes
GENDER_ARCHETYPES: List[Tuple[str, str]] = [
    ("young woman", "f"),
    ("adult woman", "f"),
    ("motherly woman", "f"),
    ("young man", "m"),
    ("adult man", "m"),
    ("fatherly man", "m"),
]
