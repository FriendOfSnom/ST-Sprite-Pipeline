@echo off
setlocal EnableExtensions

echo =================================
echo  ST Downloader and Organizer
echo  Windows Startup Script
echo =================================

rem --- always run from this script's folder
cd /d "%~dp0"
if errorlevel 1 goto err_cd
echo Changed working directory to: %cd%

rem --- create venv if missing
if exist "venv\Scripts\python.exe" goto venv_exists
echo No virtual environment found. Creating one now...
python -m venv "venv"
if errorlevel 1 goto err_makevenv
echo Virtual environment created successfully.

:venv_exists
echo Existing virtual environment found (or just created).

rem --- activate venv
echo Activating virtual environment...
call "venv\Scripts\activate"
if errorlevel 1 goto err_activate

rem --- ensure deps
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto err_pip_upgrade

echo Installing/updating required packages from requirements.txt...
pip install -r "requirements.txt"
if errorlevel 1 goto err_pip_install

echo Dependencies are up to date.

rem --- optional upscaler path (no parens, no blocks)
set "APP_ROOT=%~dp0"
set "ESRGAN_BIN=%APP_ROOT%Windows_Helper\realesrgan\realesrgan-ncnn-vulkan.exe"
if exist "%ESRGAN_BIN%" set "REAL_ESRGAN_BIN=%ESRGAN_BIN%"
if exist "%ESRGAN_BIN%" echo Found bundled upscaler: "%REAL_ESRGAN_BIN%"
if not exist "%ESRGAN_BIN%" echo Bundled upscaler not found (optional). Downscale-only runs will still work.

rem --- run the pipeline
echo Launching the pipeline controller...
python -X dev -u "pipeline_runner.py"
set "RC=%ERRORLEVEL%"
echo pipeline_runner.py exited with code %RC%

rem --- deactivate venv (best effort)
call venv\Scripts\deactivate >nul 2>&1

if "%RC%"=="0" goto ok
echo [ERROR] Pipeline returned a non-zero exit code.
pause
exit /b %RC%

:ok
echo Done!
pause
exit /b 0

:err_cd
echo [ERROR] Could not change directory to script folder.
pause
exit /b 1

:err_makevenv
echo [ERROR] Failed to create virtual environment. Make sure Python is installed and in PATH!
pause
exit /b 1

:err_activate
echo [ERROR] Failed to activate virtual environment.
pause
exit /b 1

:err_pip_upgrade
echo [ERROR] pip upgrade failed.
pause
exit /b 1

:err_pip_install
echo [ERROR] pip install -r requirements.txt failed.
pause
exit /b 1
