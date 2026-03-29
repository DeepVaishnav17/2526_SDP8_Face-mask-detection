@echo off
echo ============================================
echo   FaceGuard v2 - Face Mask Detection System
echo ============================================
echo.
echo [1/3] Installing Python dependencies...
cd backend
pip install -r requirements.txt
cd ..

echo.
echo [2/3] Installing Node dependencies...
cd frontend
call npm install
cd ..

echo.
echo [3/3] Starting servers...
echo.

start "FaceGuard Backend" cmd /k "cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"
timeout /t 4 /nobreak > NUL
start "FaceGuard Frontend" cmd /k "cd frontend && npm start"

echo.
echo ============================================
echo   Open http://localhost:3000 in Chrome
echo   NOTE: First run downloads AI models (~300MB)
echo   Backend will say "ready" when models loaded
echo ============================================
pause
