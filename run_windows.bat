@echo off
REM AI Content Generator - Windows Launcher
REM This script sets up and launches the AI Content Generator on Windows

echo.
echo ========================================
echo   AI Content Generator - Windows
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.8+ from python.org
    echo.
    pause
    exit /b 1
)

REM Display Python version
echo 🐍 Python version:
python --version
echo.

REM Check if we're in a virtual environment
python -c "import sys; print('✅ Virtual environment active' if sys.prefix != sys.base_prefix else '⚠️  Not in virtual environment')"
echo.

REM Check system specs
echo 🖥️  Checking system specifications...
python system_specs.py
echo.

REM Check for uv package manager
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing uv package manager...
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set PATH=%USERPROFILE%\.cargo\bin;%PATH%
)

echo ✅ uv package manager ready
echo.

REM Check if required packages are installed
echo 📦 Checking dependencies...
python -c "
import sys
missing = []
try:
    import torch
    print('✅ PyTorch available')
except ImportError:
    missing.append('torch')
    print('❌ PyTorch not installed')

try:
    import gradio
    print('✅ Gradio available')
except ImportError:
    missing.append('gradio')
    print('❌ Gradio not installed')

try:
    import diffusers
    print('✅ Diffusers available')
except ImportError:
    missing.append('diffusers')
    print('❌ Diffusers not installed')

if missing:
    print(f'\\n⚠️  Missing packages: {missing}')
    print('Install with: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\\n✅ All core dependencies available!')
"

if errorlevel 1 (
    echo.
    echo 📥 Would you like to install missing dependencies? [Y/N]
    set /p install_deps=
    if /i "%install_deps%"=="Y" (
        echo.
        echo 📦 Installing dependencies with uv...
        if exist "pyproject.toml" (
            REM Check for NVIDIA GPU
            nvidia-smi >nul 2>&1
            if %errorlevel% equ 0 (
                echo ✅ NVIDIA GPU detected, installing PyTorch with CUDA support...
                uv add torch torchvision torchaudio --index pytorch-cu121
            ) else (
                echo ⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch...
                uv add torch torchvision torchaudio --index pytorch-cpu
            )
            uv sync
        ) else (
            REM Check for NVIDIA GPU
            nvidia-smi >nul 2>&1
            if %errorlevel% equ 0 (
                echo ✅ NVIDIA GPU detected, installing PyTorch with CUDA support...
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ) else (
                echo ⚠️  No NVIDIA GPU detected, installing CPU-only PyTorch...
                uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            )
            uv pip install diffusers transformers accelerate gradio psutil
        )
        if errorlevel 1 (
            echo ❌ Installation failed!
            pause
            exit /b 1
        )
        echo ✅ Dependencies installed successfully!
    ) else (
        echo ⚠️  Some features may not work without dependencies
    )
)

echo.
echo 🚀 Starting AI Content Generator...
echo.
echo 💡 The interface will open in your web browser
echo 🌐 Default URL: http://localhost:7860
echo 🛑 Press Ctrl+C to stop the server
echo.

REM Launch the application
python app.py %*

REM If we get here, the app has stopped
echo.
echo 👋 AI Content Generator stopped
pause