@echo off
cd /d "%~dp0"
echo ========================================================
echo Starting Smart Wardrobe Application (Backend + Frontend)
echo ========================================================
if exist "%~dp0backend\venv\Scripts\python.exe" (
    start "Smart Wardrobe Backend" cmd /k "cd /d %~dp0backend && "%~dp0backend\venv\Scripts\python.exe" -m uvicorn main:app --reload --port 8000"
) else (
    start "Smart Wardrobe Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --reload --port 8000"
)
echo Waiting for backend to initialize...
timeout /t 3 >nul
start "" "%~dp0frontend\index.html"
echo Smart Wardrobe interface launched in browser!
echo Press any key to exit this launcher window.
pause >nul
