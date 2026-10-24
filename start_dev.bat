@echo off
setlocal
REM ============================================================
REM  SOP-AI demo - one-click dev launcher (ASCII only, no UTF-8)
REM  - Frees backend/frontend ports automatically
REM  - Bootstraps .venv and npm deps on first run
REM  - Backend : uvicorn src.api:app   (default port 8000)
REM  - Frontend: vite dev server       (default port 5173)
REM ============================================================

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    exit /b 1
)

echo [1/5] Freeing port %BACKEND_PORT% (backend) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort %BACKEND_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('  stopping PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }; exit 0"

echo [2/5] Freeing port %FRONTEND_PORT% (frontend) ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort %FRONTEND_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('  stopping PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }; exit 0"

echo [3/5] Checking python venv ...
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo   Creating .venv and installing requirements, please wait ...
    python -m venv "%ROOT%\.venv"
    "%ROOT%\.venv\Scripts\python.exe" -m pip install --upgrade pip
    "%ROOT%\.venv\Scripts\python.exe" -m pip install -r "%ROOT%\requirements.txt"
) else (
    echo   .venv OK
)

echo [4/5] Checking frontend deps ...
if not exist "%ROOT%\web\package.json" (
    echo   SKIP: web\ frontend not scaffolded yet (planned at M2, see docs/01).
) else if not exist "%ROOT%\web\node_modules" (
    echo   Installing npm dependencies, please wait ...
    pushd "%ROOT%\web"
    call npm install
    popd
) else (
    echo   node_modules OK
)

echo [5/5] Starting services ...

if exist "%ROOT%\src\api.py" (
    start "SOP-AI backend  :%BACKEND_PORT%" /D "%ROOT%" cmd /k ".venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port %BACKEND_PORT% --reload"
) else (
    echo   SKIP: src\api.py not found.
)

if exist "%ROOT%\web\package.json" (
    start "SOP-AI frontend :%FRONTEND_PORT%" /D "%ROOT%\web" cmd /k "npm run dev -- --port %FRONTEND_PORT% --strictPort"
) else (
    echo   SKIP: web\ frontend not scaffolded yet.
)

echo.
echo Backend API  : http://127.0.0.1:%BACKEND_PORT%/docs
echo Frontend dev : http://localhost:%FRONTEND_PORT%   (after web\ is scaffolded)
echo Close the two opened windows to stop the services.
endlocal
