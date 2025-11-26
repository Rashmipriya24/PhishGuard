@echo off
echo ========================================
echo    PhishGuard - Quick Start (Windows)
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
python --version

echo [2/4] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    pause
    exit /b 1
)
node --version

echo.
echo [3/4] Starting Backend Server...
echo.
start cmd /k "cd backend && title PhishGuard Backend && echo Installing dependencies... && pip install -r requirements.txt && echo. && echo Starting backend server... && python start_backend.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo [4/4] Starting Frontend Application...
echo.
start cmd /k "cd frontend && title PhishGuard Frontend && echo Installing dependencies... && npm install && echo. && echo Starting frontend... && npm run dev"

echo.
echo ========================================
echo    PhishGuard is starting up!
echo ========================================
echo.
echo Backend API:  http://localhost:8000
echo API Docs:     http://localhost:8000/docs
echo Frontend:     http://localhost:5173
echo.
echo Two terminal windows have been opened:
echo   1. Backend Server (Python/FastAPI)
echo   2. Frontend Application (React/Vite)
echo.
echo Close those windows to stop the servers.
echo.
pause
