@echo off
setlocal

REM Change to the folder where this script lives
cd /d "%~dp0"

set NODE_OPTIONS=--max_old_space_size=8192

echo.
echo ==========================================
echo  MSL Project - Setup and Start Script
echo ==========================================
echo.

REM ============================
REM 1. Backend - Python packages
REM ============================
echo.
echo [Backend] Restoring Python packages...
echo.

cd backend
call python -m pip install -r requirements.txt
if "%errorlevel%" neq "0" (
    echo [Backend] Failed to restore Python packages
    exit /B %errorlevel%
)

echo.
echo [Backend] Starting backend server on port 8000 with Uvicorn...
echo.

REM Open backend in a new terminal window using uvicorn and the ASGI wrapper
start "MSL Backend" cmd /k "cd /d %~dp0backend && python -m uvicorn app:asgi_app --host 0.0.0.0 --port 8000 --reload"

REM ============================
REM 2. Frontend - npm packages
REM ============================
echo.
echo [Frontend] Restoring npm packages...
echo.

cd ..\frontend
call npm install
if "%errorlevel%" neq "0" (
    echo [Frontend] Failed to restore npm packages
    exit /B %errorlevel%
)

echo.
echo [Frontend] Starting Vite dev server on port 5173...
echo.

REM Open frontend dev server in a new terminal window
start "MSL Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

REM ============================
REM 3. Open browser
REM ============================
echo.
echo Opening browser at http://localhost:5173 ...
echo.
start http://localhost:5173

echo.
echo All done. Backend: 8000 (uvicorn), Frontend: 5173 (Vite)
echo.

endlocal
exit /B 0
