@echo off
echo.
echo ================================
echo  ST Downloader and Organizer
echo  Windows Startup Script
echo ================================
echo.

REM ------------------------------------------
REM Always run relative to this script's folder
cd /d %~dp0
echo Changed working directory to: %cd%
REM ------------------------------------------

REM ------------------------
REM Check if venv exists
REM ------------------------
IF NOT EXIST "venv\" (
    echo No virtual environment found. Creating one now...
    python -m venv venv
    IF ERRORLEVEL 1 (
        echo Failed to create virtual environment. Make sure Python is installed and in PATH!
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.

    echo Installing required packages...
    call venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    call venv\Scripts\deactivate
    echo Dependencies installed.
) ELSE (
    echo Existing virtual environment found.
)

REM ------------------------
REM Activate and run the pipeline
REM ------------------------
echo.
echo Launching the pipeline controller...
call venv\Scripts\activate
python pipeline_runner.py
call venv\Scripts\deactivate

echo.
echo Done! Exiting.
pause
