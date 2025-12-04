# How to Run Your Ren'Py Game

After creating a project with Tool 1, here's how to launch and test your visual novel:

## Method 1: Using Ren'Py SDK Launcher (Recommended)

1. **Open the Ren'Py SDK:**
   - Navigate to: `renpy-8.5.0-sdk/`
   - Windows: Double-click `renpy.exe`
   - Mac/Linux: Run `./renpy.sh`

2. **Select Your Project:**
   - Your project will appear in the left panel
   - Click on your project name to select it

3. **Launch the Game:**
   - Click the **"Launch Project"** button
   - Your game will start in a new window

## Method 2: Direct Command Line

From the SDK directory:
```bash
# Windows
renpy.exe "path\to\YourProject"

# Mac/Linux
./renpy.sh "/path/to/YourProject"
```

## Method 3: From Tool 1 Success Message

The success dialog at the end of Tool 1 shows:
- Project location
- Recommendation to launch in Ren'Py

Simply open the SDK launcher and follow Method 1.

---

## During Development

**Editing Scripts:**
- Use Tool 3 (Visual Scene Editor) for GUI-based editing
- Or edit `.rpy` files directly in any text editor
- Ren'Py auto-reloads when you save changes (press Shift+R in game)

**Testing Changes:**
- Save your script file
- Press **Shift+R** in the running game to reload
- Or restart the game from the SDK launcher

**Common Shortcuts in Game:**
- `Shift+R` - Reload game (after script changes)
- `Shift+D` - Developer menu
- `Shift+O` - Console
- `Esc` - Main menu

---

## Building for Distribution

Once your game is complete:

1. Open Ren'Py SDK
2. Select your project
3. Click **"Build Distributions"**
4. Select platforms (Windows, Mac, Linux, etc.)
5. Click "Build"

This creates standalone executables in `YourProject-dists/` that players can run without Ren'Py.

---

## Troubleshooting

**"Project not found in SDK"**
- Make sure your project has a `game/` subfolder
- Check that `game/` contains at least `script.rpy` and `options.rpy`

**"Character not loading"**
- Verify character folder is in `game/images/characters/`
- Check that `character.yml` exists
- Launch game and check console for error messages

**"Changes not appearing"**
- Save the script file first
- Press Shift+R in game to reload
- If still not working, restart the game from SDK

---

**Note:** The game folder itself does not contain a "run" script during development. You always need the Ren'Py SDK to test and run games. Only after building distributions do you get standalone executables.
