#!/usr/bin/env bash
# start-mac.command — Homebrew-first launcher (fixed: Tk preflight doesn't require Pillow)
set -euo pipefail

print_header() { echo; echo "================================="; echo "  $1"; echo "================================="; echo; }

# Tk-only sanity test (no Pillow): returns 0 if Tk 8.6+ can create a Canvas
check_tk_only() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
try:
    import tkinter as tk
except Exception:
    sys.exit(1)
if getattr(tk, "TkVersion", 0) < 8.6:
    sys.exit(1)
r = tk.Tk(); r.withdraw()
c = tk.Canvas(r, width=10, height=10); c.pack()
c.create_line(0,5,10,5,fill="#00E5FF",width=2)
r.update_idletasks(); r.destroy()
PY
}

# Full check with Pillow (only after deps installed)
check_tk_with_pillow() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
try:
    import tkinter as tk
    from PIL import Image, ImageTk
except Exception:
    sys.exit(1)
if getattr(tk, "TkVersion", 0) < 8.6:
    sys.exit(1)
r = tk.Tk(); r.withdraw()
im = Image.new("RGBA",(10,10),(0,0,0,255))
c = tk.Canvas(r,width=10,height=10); c.pack()
tki = ImageTk.PhotoImage(im)
c.create_image(0,0,anchor="nw",image=tki)
c.create_line(0,5,10,5,fill="#00E5FF",width=2)
r.update_idletasks(); r.destroy()
PY
}

diagnose_tk() {
  "$1" - <<'PY'
import os, sys, ctypes.util, traceback
print("---- Python & Tk Diagnostics ----")
print("sys.executable:", sys.executable)
print("sys.version:", sys.version.replace("\n"," "))
try:
    import _tkinter as _t
    print("_tkinter file:", getattr(_t, "__file__", "<unknown>"))
except Exception as e:
    print("Import _tkinter FAILED:", e)
try:
    import tkinter as tk
    print("tkinter imported OK; TkVersion:", getattr(tk, "TkVersion", "<n/a>"))
    r = tk.Tk(); r.withdraw(); r.update_idletasks(); r.destroy()
    print("Tk root created/destroyed OK.")
except Exception:
    print("Tk init FAILED:")
    traceback.print_exc()
print("ctypes.find_library('tk'):", ctypes.util.find_library('tk'))
print("ctypes.find_library('tcl'):", ctypes.util.find_library('tcl'))
for k in ("TCL_LIBRARY","TK_LIBRARY","DYLD_FRAMEWORK_PATH","DYLD_LIBRARY_PATH","PATH"):
    print(f"{k}=", os.environ.get(k, ""))
print("---- End Diagnostics ----")
PY
}

# --- Go to repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
print_header "ST Downloader and Organizer — macOS Launcher"
echo "Working directory: $SCRIPT_DIR"

# --- Homebrew ahead of pyenv
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
if ! command -v brew >/dev/null 2>&1; then
  echo "ERROR: Homebrew is required. Install from https://brew.sh and re-run."
  exit 1
fi

# --- Ensure brewed python & tcl-tk & python-tk
brew list --versions python >/dev/null 2>&1 || brew install python
PY_PREFIX="$(brew --prefix python)"
PYTHON_BIN="$PY_PREFIX/bin/python3"

# Install python-tk that matches MAJOR.MINOR (e.g., 3.13)
PY_MM="$("$PYTHON_BIN" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
brew list --versions "python-tk@${PY_MM}" >/dev/null 2>&1 || brew install "python-tk@${PY_MM}"

# Prefer brewed tcl-tk (9 is fine; Tk reports 9.0 on your machine)
brew list --versions tcl-tk >/dev/null 2>&1 || brew install tcl-tk
TK_PREFIX="$(brew --prefix tcl-tk)"

# --- Env wiring so Tk can find its libs
export PATH="$PY_PREFIX/bin:$TK_PREFIX/bin:${PATH}"
export LDFLAGS="-L$TK_PREFIX/lib ${LDFLAGS:-}"
export CPPFLAGS="-I$TK_PREFIX/include ${CPPFLAGS:-}"
export PKG_CONFIG_PATH="$TK_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export DYLD_FRAMEWORK_PATH="$TK_PREFIX/Frameworks:${DYLD_FRAMEWORK_PATH:-}"
export DYLD_LIBRARY_PATH="$TK_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"
export TK_SILENCE_DEPRECATION=1

# Best-effort hints (ok with Tk 8.6 or 9.0 present)
[[ -d "$TK_PREFIX/lib/tcl9.0" ]] && export TCL_LIBRARY="$TK_PREFIX/lib/tcl9.0"
[[ -d "$TK_PREFIX/lib/tk9.0"  ]] && export TK_LIBRARY="$TK_PREFIX/lib/tk9.0"
[[ -z "${TCL_LIBRARY:-}" && -d "$TK_PREFIX/lib/tcl8.6" ]] && export TCL_LIBRARY="$TK_PREFIX/lib/tcl8.6"
[[ -z "${TK_LIBRARY:-}"  && -d "$TK_PREFIX/lib/tk8.6"  ]] && export TK_LIBRARY="$TK_PREFIX/lib/tk8.6"

echo "Using Python: $PYTHON_BIN"

# --- Preflight: Tk-only check (no Pillow yet)
if ! check_tk_only "$PYTHON_BIN"; then
  echo "ERROR: Tk failed to initialize with brewed Python."
  diagnose_tk "$PYTHON_BIN"
  echo "Try: brew reinstall tcl-tk python \"python-tk@${PY_MM}\""
  exit 1
fi

# --- Create/recreate venv with brewed Python
VENV_PY='./venv/bin/python'
if [[ -d "venv" ]]; then
  if [[ ! -x "$VENV_PY" ]] || ! "$VENV_PY" -c 'import sys,os;print(os.path.realpath(sys.executable))' | grep -q "$(realpath "$PYTHON_BIN")"; then
    echo "Recreating virtual environment with brewed Python..."
    rm -rf venv
    "$PYTHON_BIN" -m venv venv
  else
    echo "Using existing virtual environment (./venv)"
  fi
else
  print_header "Creating virtual environment (./venv)"
  "$PYTHON_BIN" -m venv venv
fi

# --- Activate venv
# shellcheck disable=SC1091
source "venv/bin/activate"
deactivate_on_exit() { type deactivate >/dev/null 2>&1 && deactivate || true; }
trap deactivate_on_exit EXIT

# --- Log Tk version & stabilize scaling
python - <<'PY' || true
import tkinter as tk
print("Detected Tk version:", tk.TkVersion)
try:
    r=tk.Tk(); r.tk.call('tk','scaling',1.0); r.destroy()
except Exception:
    pass
PY

# --- Install deps
print_header "Upgrading pip"
python -m pip install --upgrade pip

print_header "Installing requirements"
if [[ -f "requirements.txt" ]]; then
  pip install -r requirements.txt
else
  echo "WARNING: requirements.txt not found; skipping."
fi

# --- Now (optionally) verify Tk+Pillow inside venv; don't fail if Pillow isn't in reqs
if ! check_tk_with_pillow "python"; then
  echo "NOTE: Tk OK, but Pillow+ImageTk precheck failed (maybe Pillow not installed yet)."
  echo "Continuing to launch; the app will install/use Pillow from requirements."
fi

# --- Optional: point the pipeline at a bundled upscaler binary
ESRGAN_DIR="$SCRIPT_DIR/Mac_Helper/realesrgan"
ESRGAN_BIN="$ESRGAN_DIR/realesrgan-ncnn-vulkan"

if [[ -x "$ESRGAN_BIN" ]]; then
  export REAL_ESRGAN_BIN="$ESRGAN_BIN"
  echo "Found bundled upscaler: $REAL_ESRGAN_BIN"
elif [[ -f "$ESRGAN_BIN" ]]; then
  # Ensure it's executable on first run
  chmod +x "$ESRGAN_BIN" || true
  if [[ -x "$ESRGAN_BIN" ]]; then
    export REAL_ESRGAN_BIN="$ESRGAN_BIN"
    echo "Prepared bundled upscaler: $REAL_ESRGAN_BIN"
  else
    echo "Bundled upscaler not executable yet (optional). Downscale-only runs will still work."
  fi
else
  echo "Bundled upscaler not found (optional). Downscale-only runs will still work."
fi
echo

# --- Launch
print_header "Launching the pipeline controller"
python pipeline_runner.py

echo; echo "Done."
