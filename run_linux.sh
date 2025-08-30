#!/bin/bash
# AI Content Generator - Linux/Ubuntu Launcher
# This script sets up and launches the AI Content Generator on Linux/Ubuntu

set -e  # Exit on any error

echo ""
echo "========================================"
echo "  AI Content Generator - Linux/Ubuntu"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found! Please install Python 3.8+ with:"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  Arch: sudo pacman -S python python-pip"
    exit 1
fi

# Display Python version
echo "🐍 Python version:"
python3 --version
echo ""

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment active: $VIRTUAL_ENV"
else
    print_warning "Not in virtual environment (recommended to use one)"
fi
echo ""

# Check system specs
echo "🖥️ Checking system specifications..."
python3 system_specs.py
echo ""

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "
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
    print('Install with: pip3 install -r requirements.txt')
    sys.exit(1)
else:
    print('\\n✅ All core dependencies available!')
"

# Check if dependencies installation is needed
if [ $? -ne 0 ]; then
    echo ""
    read -p "📥 Would you like to install missing dependencies? [y/N]: " install_deps
    if [[ $install_deps =~ ^[Yy]$ ]]; then
        echo ""
        print_info "Installing dependencies..."
        
        # Check if pip3 is available
        if ! command -v pip3 &> /dev/null; then
            print_error "pip3 not found! Install with:"
            echo "  Ubuntu/Debian: sudo apt install python3-pip"
            exit 1
        fi
        
        # Install PyTorch with CUDA support (if available)
        if command -v nvidia-smi &> /dev/null; then
            print_info "NVIDIA GPU detected, installing PyTorch with CUDA support..."
            pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        else
            print_warning "No NVIDIA GPU detected, installing CPU-only PyTorch..."
            pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        fi
        
        # Install other requirements
        pip3 install -r requirements.txt
        
        if [ $? -eq 0 ]; then
            print_status "Dependencies installed successfully!"
        else
            print_error "Installation failed!"
            exit 1
        fi
    else
        print_warning "Some features may not work without dependencies"
    fi
fi

echo ""
echo "🚀 Starting AI Content Generator..."
echo ""
print_info "The interface will open in your web browser"
print_info "Default URL: http://localhost:7860"
print_info "Press Ctrl+C to stop the server"
echo ""

# Check if we can open browser automatically
if command -v xdg-open &> /dev/null; then
    BROWSER_CMD="xdg-open"
elif command -v gnome-open &> /dev/null; then
    BROWSER_CMD="gnome-open"
else
    BROWSER_CMD=""
fi

# Launch the application
if [ "$BROWSER_CMD" != "" ]; then
    print_info "Will attempt to open browser automatically..."
    # Launch app in background and open browser after a delay
    python3 app.py "$@" &
    APP_PID=$!
    
    # Wait a moment for the server to start
    sleep 3
    
    # Try to open browser
    $BROWSER_CMD http://localhost:7860 2>/dev/null || true
    
    # Wait for the app to finish
    wait $APP_PID
else
    # Just launch the app normally
    python3 app.py "$@"
fi

# If we get here, the app has stopped
echo ""
print_info "AI Content Generator stopped"