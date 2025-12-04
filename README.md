# Visual Novel Development Toolkit (v2.0.0)

A comprehensive, AI-powered toolkit for creating visual novels with Ren'Py. This suite provides everything needed to create professional VN projects: from project setup and AI character generation to expression sheet creation.

Perfect for game developers, writers, and creators who want to build accessible visual novels with custom character systems.

---

## Three-Tool Workflow

### **Tool 1: Ren'Py Project Scaffolder**
Creates production-ready Ren'Py projects with custom character support.

**What it does:**
- Downloads Ren'Py SDK 8.5.0 automatically (if missing)
- Launches official Ren'Py launcher for project creation
- Injects custom character loading system
- Adds support for dynamic character positioning and expressions
- Optionally imports existing character folders
- Moves project to your chosen location

**Custom Features Added to Projects:**
- Auto-loads characters from `game/images/characters/` folder
- Custom position transforms: `centerleft`, `centerright`, `faceleft`, `faceright`
- Automatic character voice and color support
- Dynamic outfit and expression swapping
- Support for character metadata via `character.yml` files

---

### **Tool 2: Gemini Character Creator**
AI-powered character sprite generator using Google Gemini vision models.

**What it does:**
- Generate character sprites from text prompts or reference images
- Create multiple expressions (happy, sad, angry, neutral, etc.)
- Generate multiple outfits (casual, uniform, etc.)
- Automatically crop and scale sprites
- Create game-ready folder structures
- Generate `character.yml` metadata files

**Output Format:**
```
character_name/
├── character.yml          # Metadata (voice, scale, colors)
└── a/                     # Pose folder (letter poses)
    ├── faces/
    │   └── face/
    │       ├── happy.png
    │       ├── sad.png
    │       ├── angry.png
    │       └── neutral.png
    └── outfits/
        ├── casual.png
        └── uniform.png
```

**Integration:**
Characters created with Tool 2 are automatically compatible with projects created by Tool 1. Simply copy character folders to `game/images/characters/` and they'll load automatically!

---

### **Tool 3: Visual Scene Editor** (Planned)
A complete GUI-based scene editor for writing visual novels without code.

**What it will do:**
- Live preview window showing the current scene in real-time
- Drag-and-drop character positioning on screen
- Visual selectors for expressions, outfits, backgrounds, and sounds
- Dialogue editor with character selection dropdown
- Automatic expression sheet generation and display
- Preset position buttons (left, centerleft, center, centerright, right)
- Flip controls (faceleft, faceright)
- Transition selector for character entrances/exits
- Forward/back arrows to navigate scene timeline (undo/redo)
- "Next" button to generate and append Ren'Py code to script.rpy

**Vision:**
This tool will enable non-programmers to create complete visual novels using an intuitive GUI interface, automatically generating proper Ren'Py script code behind the scenes.

---

## Quick Start

### First-Time Setup

#### 1. Install Python Dependencies
```bash
# Windows
start-windows.bat

# macOS
./start-mac.command
```

This will create a virtual environment and install all required packages.

#### 2. Download Ren'Py SDK (Required for Tool 1)
The toolkit will prompt you to download the SDK when you first run Tool 1. Alternatively, you can download it manually:

```bash
python download_renpy_sdk.py
```

This downloads Ren'Py SDK 8.5.0 (~150 MB) to the toolkit folder.

---

## Requirements

* **Python 3.10 or higher**
* **Google Gemini API Key** (for Tool 2 - Character Creator)
  - Get your free API key at: https://aistudio.google.com/apikey
  - Set environment variable: `GEMINI_API_KEY=your_key_here`

### Python Dependencies
Installed automatically via launcher scripts:
```
google-generativeai
pillow>=10.3
pyyaml
requests
beautifulsoup4
pandas
rembg  # Optional: for background removal experiments
```

---

## Usage

### Running the Toolkit

**Windows:**
```bash
start-windows.bat
```

**macOS:**
```bash
./start-mac.command
```

This launches the main menu where you can select:
1. **Create new Ren'Py project** (Tool 1 - Project Scaffolder)
2. **Create new character sprites** (Tool 2 - Gemini Character Creator)
3. **Generate expression sheets** (Utility tool)
Q. Quit

**Note:** Tool 3 (Visual Scene Editor) is planned for future implementation.

---

## Workflow Example

### Creating a Complete Visual Novel Project

1. **Run Tool 1: Create a New Project**
   - Select option 1 from the menu
   - SDK downloads automatically (if not present)
   - Ren'Py launcher opens
   - Create your project with a name, resolution, and theme
   - Close Ren'Py when done
   - Browse to your project location (or let it stay in SDK/projects/)
   - Character system is automatically injected!

2. **Run Tool 2: Create Characters**
   - Select option 2 from the menu
   - Choose output folder (e.g., Desktop or project's `game/images/characters/`)
   - Enter character name and description
   - Generate expressions and outfits with AI
   - Review and crop results
   - Save to character folder

3. **Copy Characters to Project** (if needed)
   - If you didn't save directly to the project, copy character folders to:
     `YourProject/game/images/characters/`

4. **Generate Expression Sheets** (optional)
   - Select option 3 from the menu
   - Choose the folder containing your character folders
   - Expression sheets are generated for reference

5. **Write Your Story**
   - Option A: Use Tool 3 (Visual Scene Editor) when available - GUI-based scene creation
   - Option B: Edit script.rpy manually in your preferred text editor

6. **Launch Your Project in Ren'Py**
   - Open Ren'Py SDK
   - Select your project
   - Click "Launch Project"
   - Characters are automatically loaded and ready to use!

### Using Characters in Your Script

```renpy
label start:
    scene bg room

    # Show character with outfit and expression
    show alice casual happy at centerright
    alice "Hello! Nice to meet you!"

    # Change expression
    show alice casual sad
    alice "I'm feeling a bit down today..."

    # Flip character to face left
    show alice at faceleft
    alice "Looking this way!"

    # Change outfit
    show alice uniform neutral at centerleft
    alice "Ready for school!"
```

---

## Project Structure

```
Visual-Novel-Development-Toolkit/
├── gemini_sprite_pipeline/        # Tool 2: Character Creator modules
│   ├── core/                      # Core generation logic
│   ├── ui/                        # Tkinter UI components
│   ├── processing/                # Image processing
│   └── utils/                     # Utilities
├── renpy_template/                # Character system template files
│   └── game/
│       ├── character.py           # Character loading system
│       ├── body.py                # Body/Pose/Expression classes
│       ├── char_sprites.py        # Person/Ghost sprite classes
│       ├── pymage_size.py         # Image utilities
│       └── effects.rpy            # Custom transforms & animations
├── renpy-8.5.0-sdk/               # Ren'Py SDK (downloaded automatically)
├── reference_sprites/             # Reference characters for scaling
├── models/                        # Gemini model configurations
├── pipeline_runner.py             # Main launcher (menu system)
├── renpy_project_scaffolder.py    # Tool 1: Project creator
├── expression_sheet_maker.py      # Utility: Expression sheet generator
├── download_renpy_sdk.py          # SDK downloader
├── requirements.txt               # Python dependencies
├── start-windows.bat              # Windows launcher
├── start-mac.command              # macOS launcher
└── README.md                      # This file

Note: Tool 3 (Visual Scene Editor) is planned for future development.
```

---

## Academic Use

This toolkit was developed as a graduation project to make visual novel development more accessible. It demonstrates:
- **AI Integration**: Leveraging Google Gemini for creative asset generation
- **Automated Workflows**: Streamlining game development pipelines
- **Accessibility**: Enabling non-programmers to create professional VNs
- **Modular Architecture**: Clean separation of tools and responsibilities

**Perfect for:**
- Computer Science capstone projects
- Game development portfolios
- Independent VN creators
- Accessibility technology research

---

## Advanced Configuration

### Character System Features

Projects created with Tool 1 include a powerful character system:

**Position Transforms:**
- `left`, `centerleft`, `center`, `centerright`, `right` - Screen positions
- `faceleft`, `faceright` - Flip character sprites horizontally

**Character Methods (in script.rpy):**
```python
# Access character objects
alice.outfit = "uniform"           # Change outfit
alice.add_accessory("glasses")     # Add accessory
alice.remove_accessory("hat")      # Remove accessory
```

**Character YAML Format:**
```yaml
display_name: "Alice"
name_color: "#ff69b4"
scale: 0.85
voice: "girl"
eye_line: 0.42
default_outfit: "casual"
```

---

## Troubleshooting

### "Ren'Py SDK not found!"
- Run Tool 1 and select "Yes" when prompted to download the SDK
- Or manually run: `python download_renpy_sdk.py`

### "GEMINI_API_KEY not set"
- Get your API key from https://aistudio.google.com/apikey
- Set environment variable before running:
  ```bash
  # Windows (CMD)
  set GEMINI_API_KEY=your_key_here

  # Windows (PowerShell)
  $env:GEMINI_API_KEY="your_key_here"

  # macOS/Linux
  export GEMINI_API_KEY=your_key_here
  ```

### Characters not loading in game
- Verify folder structure matches Tool 2 output format
- Check that characters are in `game/images/characters/`
- Ensure `character.yml` exists in character folder
- Launch game and check console for error messages

### Tkinter errors on macOS
- Ensure you're using Homebrew Python (not system Python)
- The `start-mac.command` script handles this automatically

---

## Changelog

### v2.0.0 (Current)
* **Complete toolkit redesign** for academic use
* **Tool 1: Ren'Py Project Scaffolder** - Create projects with custom character system
* **Tool 2: Gemini Character Creator** - AI-powered sprite generation
* **Tool 3: Visual Scene Editor** - Planned GUI-based scene creation tool
* Expression sheet generator utility
* Removed all project-specific references
* Automatic Ren'Py SDK download and setup
* Integrated workflow across all tools
* Clean, production-ready character system injection
* Updated to Ren'Py 8.5.0
* Comprehensive documentation and examples

### v1.1.1 (Legacy)
* Original sprite pipeline with 5-step workflow
* Manual sorting and organization tools
* Bulk downscaler with gamma-aware processing
* Expression sheet improvements

### v1.0.0 (Legacy)
* Initial release of sprite pipeline tool

---

## License

This toolkit is provided as-is for educational and personal use.

## Contributing

This is a graduation project, but feedback and suggestions are welcome! Please open an issue on GitHub with any bugs, feature requests, or questions.

---

**Developed as a graduation project in Computer Science**
*Making visual novel development accessible to everyone*
