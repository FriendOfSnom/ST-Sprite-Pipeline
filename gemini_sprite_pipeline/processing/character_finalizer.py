"""
Character finalization and expression sheet generation.

Handles the final steps of character creation including eye line/scale selection,
pose flattening, and expression sheet generation.
"""

import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from ..ui.dialogs import prompt_for_eye_and_hair, prompt_for_scale
from .image_utils import pick_representative_outfit
from .pose_processor import flatten_pose_outfits_to_letter_poses, write_character_yml


def finalize_character(
    char_dir: Path,
    display_name: str,
    voice: str,
    game_name: Optional[str],
) -> None:
    """
    Pick eye line, name color, scale, flatten poses, and write character.yml.

    This is the final step that converts the generated sprites into a
    ready-to-use visual novel character.

    Args:
        char_dir: Character directory.
        display_name: Character's display name.
        voice: Character voice.
        game_name: Optional game name for metadata.
    """
    rep_outfit = pick_representative_outfit(char_dir)

    print("[INFO] Collecting eye line and name color...")
    eye_line, name_color = prompt_for_eye_and_hair(rep_outfit)

    print("[INFO] Collecting scale vs reference...")
    scale = prompt_for_scale(rep_outfit, user_eye_line_ratio=eye_line)

    print("[INFO] Flattening pose/outfit combinations into letter poses...")
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

    poses_yaml: Dict[str, Dict[str, str]] = {
        letter: {"facing": "right"} for letter in final_pose_letters
    }

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


def generate_expression_sheets_for_root(root_folder: Path) -> None:
    """
    Run expression_sheet_maker.py on the given root folder so that expression
    sheets are generated for all characters under it.

    This is used at the end of the pipeline so that a newly created character
    immediately has expression sheets available for Ren'Py scripting.

    Args:
        root_folder: Root directory containing character folders.
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
