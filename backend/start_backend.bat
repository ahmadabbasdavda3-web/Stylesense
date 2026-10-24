@echo off
cd /d "%~dp0"
echo Starting Smart Wardrobe Backend...
if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000
) else (
    python -m uvicorn main:app --reload --port 8000
)
pause
