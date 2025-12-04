"""
gemini_sprite_pipeline package

A modular system for generating visual novel character sprites using
Google Gemini AI.

Main components:
- api: Gemini API client and prompt builders
- ui: Tkinter user interface components
- processing: Image processing and generation workflows
- models: Data structures and configuration

Main entry point:
- run_pipeline: Core pipeline function called by pipeline_runner.py
"""

from pathlib import Path

# Import the main orchestrator from the script file
import sys
import os

# Add parent directory to path to import the script module
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import run_pipeline from the script file (not the package)
# This allows backward compatibility with pipeline_runner.py
try:
    from gemini_sprite_pipeline_script import run_pipeline
except ImportError:
    # Fallback: if the script file still has the old name, try that
    import importlib.util
    _script_path = _parent_dir / "gemini_sprite_pipeline.py"
    if _script_path.exists():
        spec = importlib.util.spec_from_file_location("_gsp_script", _script_path)
        if spec and spec.loader:
            _module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_module)
            run_pipeline = _module.run_pipeline

__version__ = "2.0.0"
__all__ = ["api", "ui", "processing", "models", "run_pipeline"]
