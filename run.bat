@echo off
setlocal

rem Change directory to the location of this script
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 or higher is required but was not found in PATH.
    pause
    exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Python 3.9 or higher is required to run Auto Object Focus.
    pause
    exit /b 1
)

set "MISSING="
for /f "delims=" %%i in ('python -c "import importlib.util; mapping={'ultralytics':'ultralytics','cv2':'opencv-python','numpy':'numpy','pyautogui':'pyautogui','mss':'mss','gradio':'gradio','pynput':'pynput'}; missing=[pkg for mod,pkg in mapping.items() if importlib.util.find_spec(mod) is None]; print(' '.join(missing))"') do set "MISSING=%%i"

if defined MISSING (
    echo Missing dependencies detected: %MISSING%
    echo Installing required packages. This may take a moment...
    python -m pip install %MISSING%
    if errorlevel 1 (
        echo Failed to install the required Python packages.
        pause
        exit /b 1
    )
) else (
    echo All Python dependencies are already installed.
)

set "PYTHONPATH=%~dp0src;%PYTHONPATH%"

echo Launching the Auto Object Focus web interface...
python -m auto_focus.webui
if errorlevel 1 (
    echo Failed to start the Auto Object Focus web interface.
    pause
    exit /b 1
)

pause
exit /b 0
