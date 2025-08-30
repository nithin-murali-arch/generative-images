#!/bin/bash

# macOS AI Content Generator Launcher
# Optimized for macOS with Apple Silicon and Intel Macs

set -e

echo "🍎 macOS AI Content Generator Launcher"
echo "======================================"

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Detect architecture
ARCH=$(uname -m)
echo "🔍 Detected architecture: $ARCH"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

echo "✅ Homebrew found"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "📦 Installing Python 3 via Homebrew..."
    brew install python
fi

PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python $PYTHON_VERSION found"

# Check for uv package manager
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv package manager..."
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo "📦 Installing curl first..."
        brew install curl
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

echo "✅ uv package manager ready"

# Install dependencies using uv
echo "📦 Installing Python dependencies..."
if [ -f "pyproject.toml" ]; then
    # macOS: Always use CPU version unless Apple Silicon with MPS
    if [[ "$ARCH" == "arm64" ]]; then
        echo "🚀 Apple Silicon detected - installing PyTorch with MPS support..."
        uv add torch torchvision torchaudio
    else
        echo "💻 Intel Mac detected - installing CPU-only PyTorch..."
        uv add torch torchvision torchaudio --index pytorch-cpu
    fi
    uv sync
else
    echo "⚠️  pyproject.toml not found, installing basic dependencies..."
    if [[ "$ARCH" == "arm64" ]]; then
        echo "🚀 Apple Silicon detected - installing PyTorch with MPS support..."
        uv pip install torch torchvision torchaudio
    else
        echo "💻 Intel Mac detected - installing CPU-only PyTorch..."
        uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    uv pip install diffusers transformers accelerate
    uv pip install gradio psutil
fi

# Check for Apple Silicon optimizations
if [[ "$ARCH" == "arm64" ]]; then
    echo "🚀 Apple Silicon detected - enabling MPS acceleration"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
else
    echo "💻 Intel Mac detected - using CPU acceleration"
fi

# Run thermal safety check
echo "🌡️  Running thermal safety check..."
if python3 test_thermal_safety.py; then
    echo "✅ Thermal safety check passed"
else
    echo "⚠️  Thermal safety check failed - proceeding with caution"
fi

# Display system information
echo ""
echo "🖥️  System Information:"
python3 system_specs.py

echo ""
echo "🚀 Starting AI Content Generator..."
echo "   Access the web interface at: http://localhost:7860"
echo "   Press Ctrl+C to stop"
echo ""

# Launch the application
python3 app.py

echo ""
echo "👋 AI Content Generator stopped"