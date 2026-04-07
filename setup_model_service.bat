@echo off
REM Quick Start: Model Service Setup & Launch (Windows)
REM 
REM Usage:
REM   setup_model_service.bat          - Install dependencies
REM   setup_model_service.bat demo     - Start in demo mode (simulation)
REM   setup_model_service.bat prod     - Start in production mode (real model)

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo Model Service Setup ^& Launch (Windows)
echo ==========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set MODEL_SERVICE_DIR=%SCRIPT_DIR%model-service
set VENV_DIR=%MODEL_SERVICE_DIR%\venv

REM Create virtual environment
:setup_venv
if not exist "%VENV_DIR%" (
    echo 📦 Creating Python virtual environment...
    cd /d "%MODEL_SERVICE_DIR%"
    python -m venv venv
    
    echo ⬇️  Installing dependencies...
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    echo ✓ Virtual environment ready
) else (
    echo ✓ Virtual environment already exists
)

REM Parse command
if "%1%"=="" goto setup
if /i "%1%"=="setup" goto setup
if /i "%1%"=="install" goto setup
if /i "%1%"=="demo" goto demo
if /i "%1%"=="dev" goto demo
if /i "%1%"=="test" goto demo
if /i "%1%"=="prod" goto prod
if /i "%1%"=="production" goto prod

REM Default: show usage
echo Usage: setup_model_service.bat {setup^|demo^|prod}
echo.
echo Commands:
echo   setup       Install dependencies (default)
echo   demo        Start in demo/test mode (simulation)
echo   prod        Start in production mode (requires checkpoint)
echo.
exit /b 1

:setup
echo Setting up model service...
goto :eof

:demo
echo.
echo 🚀 Starting Model Service in DEMO MODE (simulation)
echo.
echo    Model checkpoint: NOT REQUIRED (using simulation)
echo    GPU: Not required
echo    Service URL: http://localhost:8001
echo.

cd /d "%MODEL_SERVICE_DIR%"
call venv\Scripts\activate.bat

echo 📡 Launching service...
uvicorn src.serve.service:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --reload

goto :eof

:prod
echo.
echo 🚀 Starting Model Service in PRODUCTION MODE (real model)
echo.

REM Check if checkpoint exists
if not exist "%MODEL_SERVICE_DIR%\checkpoints\best_model.pth" (
    echo ❌ ERROR: Model checkpoint not found at:
    echo    %MODEL_SERVICE_DIR%\checkpoints\best_model.pth
    echo.
    echo To use production mode, you need a trained model checkpoint.
    echo.
    echo Options:
    echo   1. Train a model:
    echo      python src\train.py --data-dir data\faceforensics++ --epochs 50
    echo.
    echo   2. Use demo mode instead:
    echo      setup_model_service.bat demo
    echo.
    exit /b 1
)

echo    Device: GPU (CUDA) if available, otherwise CPU
echo    Service URL: http://localhost:8001
echo.

cd /d "%MODEL_SERVICE_DIR%"
call venv\Scripts\activate.bat

echo 📡 Launching service...
set DEVICE=cuda
set PORT=8001
uvicorn src.serve.service:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --workers 1

goto :eof

endlocal
