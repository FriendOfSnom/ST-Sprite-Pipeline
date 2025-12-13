# Student Transfer Sprite Pipeline Tool (v1.1.1)

A cross-platform, interactive pipeline for **downloading**, **sorting**, **organizing**, **downscaling**, and **documenting** sprites into a ready-to-use format for the Student Transfer visual novel.

This suite of tools automates the process of collecting raw image data, categorizing it, cropping consistently, downscaling to in-game size, and preparing final sprite folders complete with metadata YAML files and expression sheets.

---

## New in v1.1.1

* **Sprite Bulk Downscaler (Step 4)**  
  Downscales finalized sprites to their correct in-game size using a **gamma-aware LANCZOS pipeline** with multi-step quality downscaling.  
  - Option to overwrite in-place or write to a separate destination.  
  - No backup folders are created by default (game-safe).  
  - Updates each `character.yml` to set `scale: 1.0` and preserve `original_scale`.

* **Expression Sheet Generator improvements (Step 5)**  
  - Sheets are now saved directly **inside each pose folder** (e.g., `<pose>/<pose>_sheet.png`), alongside `faces/` and `outfits/`, to avoid conflicts with the game.  
  - Expression images are scaled correctly based on `character.yml`.

* **Downloader upgrades**  
  - Supports **ExHentai cookies** via pasteable cookie headers, with local caching in `~/.st_sprite_tool/auth.json`.  
  - Adjustable per-image download delay.  
  - More robust resume and logging system.

* **Pipeline Runner expanded**  
  - Now includes **five steps** instead of four:
    1. Downloader  
    2. Manual Sorting Helper  
    3. Organizer and Finalizer  
    4. Downscale Sprites (optional)  
    5. Generate Expression Sheets  
  - Paths flow forward automatically between steps.  
  - Any step can still be run standalone.

* **macOS launcher script**  
  - A new `start-mac.command` ensures Homebrew Python + Tk compatibility and sets up a venv with the correct environment for Tkinter + Pillow.

---

## Features

* **Downloader**: Bulk image crawler for E-Hentai/ExHentai with resume, retries, and cookie support.  
* **Manual Sorting Helper**: Interactive guide that opens file explorer and explains the required folder structure.  
* **Organizer & Finalizer**: Tkinter UI for cropping, metadata collection, naming, and scaling.  
* **Bulk Downscaler**: High-quality downscaling of finalized sprites to their true in-game size.  
* **Expression Sheet Generator**: Creates printable per-pose sheets with labeled expressions.  
* Generates **game-ready Student Transfer folders** with consistent structure and YAML metadata.  
* Cross-platform launchers for **Windows** and **macOS**.

---

## Requirements

* **Python 3.8 or higher**
* Dependencies in `requirements.txt`:
  ```
  requests
  beautifulsoup4
  pillow>=10.3
  pyyaml
  pandas
  ```

---

## Quick Start

### Windows
Run:
```bash
start-windows.bat
```
This will:
1. Create a virtual environment (if it doesn't exist).  
2. Install dependencies.  
3. Launch the pipeline controller.  

### macOS
Run:
```bash
./start-mac.command
```
This will:
1. Ensure Homebrew Python + Tcl/Tk are installed.  
2. Create a virtual environment.  
3. Install dependencies.  
4. Launch the pipeline controller.  

---

## Workflow

The pipeline is interactive and can be run step-by-step or in full sequence.

### 1. Downloader
* Input gallery start page URL.  
* Downloads images into your OS **Downloads** folder.  
* Supports resume, retries, logging, and ExHentai cookies.

### 2. Manual Sorting Helper
* Guides you to organize sprites into:
  ```
  CharacterFolder/
      PoseOrOutfit/
          sprite1.png
          sprite2.png
  ```
* Opens file explorer automatically.

### 3. Organizer / Finalizer
* UI-driven process:
  - Auto-crop transparent padding.  
  - Crop legs and chin with remembered cuts.  
  - Select eye line, hair color, voice.  
  - Adjust scale against reference sprites (in-game accurate).  
* Saves:
  - Cropped outfits + face sprites.  
  - `character.yml` metadata file.

### 4. Downscale Sprites (optional)
* Reads `character.yml` and resizes all images to true in-game size.  
* In-place overwrite (recommended) or export to a new folder.  
* Prevents double-scaling by resetting YAML `scale` to `1.0`.

### 5. Generate Expression Sheets
* Loads each pose’s expressions.  
* Applies character scale automatically.  
* Saves sheets next to poses:
  ```
  CharacterFolder/
      a/
         faces/
         outfits/
         a_sheet.png
  ```

---

## File Layout

```
project-root/
├── bulk_downscale.py
├── downloader.py
├── manual_sort_helper.py
├── organize_sprites.py
├── expression_sheet_maker.py
├── pipeline_runner.py
├── requirements.txt
├── start-windows.bat
├── start-mac.command
├── names.csv
└── reference_sprites/
```

---

## Notes & Best Practices

* Run everything via `pipeline_runner.py` (or the launcher scripts).  
* Restart from any step with the same folder if needed.  
* Backups are **not** made by default—game directories remain clean.  
* ExHentai requires cookie login info—see downloader instructions.  
* `reference_sprites/` must include both `.png` and `.yml` with scale values for reference matching.  

---

## Changelog

### v1.1.1
* Added **Bulk Downscaler (Step 4)** for high-quality, in-game downscaling.
* Improved **Expression Sheet Generator**: sheets saved directly in pose folders, respecting `character.yml` scale.
* Expanded **Downloader** with ExHentai cookie support, local caching, and adjustable per-image delays.
* Enhanced **Pipeline Runner**: five-step workflow, better path passing, standalone step support.
* Added **macOS launcher script** (`start-mac.command`) with Homebrew-based Tk/Python setup.

### v1.0.0
* Initial release of the Student Transfer Sprite Pipeline Tool.
* Included: Downloader, Manual Sorting Helper, Organizer/Finalizer, Expression Sheet Generator.
* Windows batch script for quick setup and launch.
