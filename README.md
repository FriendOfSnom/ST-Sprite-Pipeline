# Student Transfer Sprite Pipeline Tool

A cross-platform, interactive pipeline for **downloading**, **sorting**, and **organizing** sprites into a ready-to-use format for the Student Transfer visual novel.

This suite of tools helps automate the process of collecting raw image data, manually categorizing it, cropping it consistently, and preparing final sprite folders complete with metadata YAML files.

---

## Features

* Bulk image downloader with resumption and error logging.
* Manual sorting helper that guides users to build the correct folder hierarchy.
* Interactive crop-and-metadata collection with Tkinter UIs.
* Body & Expression cropping with remembered cuts.
* Eye line, name color, voice, and scale selection.
* Scale matching to reference sprites with real-time comparison.
* Generates game-ready Student Transfer character folders with consistent naming and metadata.
* **Expression Sheet Generator**: creates printable expression sheets per character pose, scaling all expressions using the character's metadata.
* Windows batch script for one-click setup and launching.

---

## Requirements

* **Python 3.8 or higher**
* Dependencies in `requirements.txt`

## Quick Start (Windows)

Windows users can just run:

```
start-windows.bat
```

This will:

1. Create a virtual environment (if it doesn't exist).
2. Install dependencies.
3. Launch the interactive pipeline controller.

---

## Typical Workflow

The pipeline consists of **four interactive steps**, which can be run together in one session or separately at any time.

### 1. Downloader

* Prompts for a gallery start page URL.
* Downloads images to a chosen folder in your OS Downloads directory.
* Supports resuming partial downloads and retrying failed links.
* Logs all progress and errors.

### 2. Manual Sorting Helper

* Prompts for your download folder path.
* Opens your system's file explorer to help you organize sprites into:

```
CharacterFolder/
    PoseOrOutfit/
        image1.png
        image2.png
```

* Gives clear instructions on the required structure.

### 3. Organizer / Finalizer

* Guides you through:

  * Auto-cropping transparent padding.
  * Cropping below legs and below chin for expressions.
  * Reusing previous crop lines with thumbnail previews.
  * Selecting eye line ratio by clicking on eyes.
  * Picking hair color from the image.
  * Choosing voice (girl/boy).
  * Adjusting scale relative to reference sprites.

* Saves finalized folders with:

  * Outfits and face crops.
  * A `character.yml` metadata file per character.

### 4. Expression Sheet Generator

* Reads each character's `character.yml` file to get the correct scale.
* Loads all expressions for each pose.
* Scales them consistently according to the character’s scale value.
* Arranges them in a printable grid with labels (e.g., `a_0`, `b_1`).
* Saves all expression sheets into:

```
CharacterFolder/
    expression_sheets/
        a_sheet.png
        b_sheet.png
```

---

## File Layout

```
project-root/
├── downloader.py
├── manual_sort_helper.py
├── organize_sprites.py
├── expression_sheet_maker.py
├── pipeline_runner.py
├── requirements.txt
├── start-windows.bat
├── names.csv
└── reference_sprites/   <-- (you supply these!)
```

* **reference\_sprites/** *(required but not included)*

  * Contains your standard-sized reference sprites and matching `.yml` files.
  * Each `.yml` describes eye line and scale for that reference.

---

## Notes & Best Practices

* You can run the entire pipeline at once or restart from any step using `pipeline_runner.py`.
* The Windows batch script is optional but recommended for easy setup and startup each time.
* The pipeline is designed to make consistent, game-ready assets from arbitrary online sources.
* All user inputs are saved in clearly named folders with consistent structure for easy integration into Student Transfer projects.

---
