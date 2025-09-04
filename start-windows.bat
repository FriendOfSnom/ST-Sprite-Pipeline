@echo off
echo.
echo =================================
echo  ST Downloader and Organizer
echo  Windows Startup Script
echo =================================
echo.

REM ------------------------------------------
REM Always run relative to this script's folder
cd /d %~dp0
echo Changed working directory to: %cd%
echo.

REM ------------------------------------------
REM Create venv if missing
if not exist "venv\" (
    echo No virtual environment found. Creating one now...
    python -m venv venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Make sure Python is installed and in PATH!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
) else (
    echo Existing virtual environment found.
    echo.
)

REM ------------------------------------------
REM Activate venv and ensure dependencies
echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing/updating required packages from requirements.txt...
pip install -r requirements.txt

echo.
echo Dependencies are up to date.
echo.

REM ------------------------------------------
REM Run the pipeline
echo Launching the pipeline controller...
python pipeline_runner.py

echo.
echo Deactivating virtual environment...
call venv\Scripts\deactivate

echo.
echo Done! Exiting.
pause
