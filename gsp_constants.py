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
    ("0",  "neutral expression, relaxed face, mouth closed"),
    ("1",  "small gentle smile"),
    ("2",  "big smile with mouth open"),
    ("3",  "mischievous smirk"),
    ("4",  "closed-eye happy smile"),
    ("5",  "playful wink while smiling"),
    ("6",  "surprised, eyes wide, mouth slightly open"),
    ("7",  "shocked, eyes wide, mouth wide open"),
    ("8",  "confused, one eyebrow raised"),
    ("9",  "annoyed, eyebrows furrowed, mouth slightly open"),
    ("10", "angry, eyebrows sharply furrowed, mouth open as if shouting"),
    ("11", "sad, eyes downcast, mouth relaxed"),
    ("12", "about to cry, eyes glossy with tears"),
    ("13", "crying, tears visible on cheeks"),
    ("14", "embarrassed blush, small frown"),
    ("15", "embarrassed blush, small smile"),
    ("16", "determined, eyebrows lowered, small confident smile"),
    ("17", "tired, half-lidded eyes, small frown"),
    ("18", "nervous, eyes a bit wide, mouth tense"),
    ("19", "pouting, cheeks slightly puffed, lips pressed together"),
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
