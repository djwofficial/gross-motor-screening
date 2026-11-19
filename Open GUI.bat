@echo off
rem Windows + NVIDIA: double-click to open the live window (webcam + stick figure + live score).
rem Same as: python -m app.ui --camera 0 --device cuda --show-score
cd /d "%~dp0"
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -m app.ui --camera 0 --device cuda --show-score %*
if errorlevel 1 pause
