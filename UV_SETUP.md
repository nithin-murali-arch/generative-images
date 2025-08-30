# 🚀 UV Setup Guide for AI Content Generator

## Why Use UV?

UV is a **fast Python package manager** that provides several advantages:

### ✅ **Resource Efficiency**
- **Shared cache**: Downloaded packages are cached globally and shared across projects
- **No duplication**: Same package versions stored only once on your system
- **Space savings**: Significant disk space savings compared to pip
- **Faster installs**: Cached packages install almost instantly

### ✅ **Speed Benefits**
- **10-100x faster** than pip for package resolution and installation
- **Parallel downloads**: Multiple packages downloaded simultaneously
- **Smart caching**: Reuses previously downloaded packages

### ✅ **Better Dependency Management**
- **Lock files**: Precise, reproducible dependency versions
- **Conflict resolution**: Better handling of dependency conflicts
- **Cross-platform**: Consistent behavior across different operating systems

## 📦 **What Gets Shared vs. Downloaded**

### 🔄 **Shared Resources (No Re-download)**
- **Python packages**: All pip packages are cached and shared
- **Dependencies**: Common libraries like NumPy, Pillow shared across projects
- **Virtual environments**: Efficient virtual environment management

### 📥 **Still Downloaded Per-Project**
- **AI Models**: Large model files (GB) still download to `~/.cache/huggingface/`
- **Model sharing**: But models are shared across ALL Python projects
- **First-time only**: Models download only when first used

## 🛠️ **Installation Steps**

### 1. Install UV
```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Using pip
pip install uv
```

### 2. Clone and Setup Project
```bash
# Clone the repository
git clone <your-repo-url>
cd ai-content-generator

# Create virtual environment and install dependencies
uv sync

# Install PyTorch with CUDA support (choose your version)
uv add torch torchvision --index pytorch-cu121

# Or for older GPUs:
uv add torch torchvision --index pytorch-cu118

# Or for CPU only:
uv add torch torchvision --index pytorch-cpu
```

### 3. Optional Dependencies
```bash
# Install hardware optimization (if supported)
uv add --optional optimization

# Install API server components
uv add --optional api

# Install development tools
uv add --optional dev

# Install everything
uv add --optional all
```

### 4. Run the Application
```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Run the application
python app.py

# Or use the installed script
ai-generator
```

## 📊 **Storage Comparison**

### Traditional pip (Multiple Projects)
```
Project A/
├── venv/ (2GB - PyTorch, etc.)
└── ~/.cache/huggingface/ (20GB models)

Project B/  
├── venv/ (2GB - Same packages!)
└── ~/.cache/huggingface/ (Same 20GB models)

Total: ~24GB (4GB wasted on duplicate packages)
```

### With UV
```
~/.cache/uv/ (2GB - Shared packages)
~/.cache/huggingface/ (20GB - Shared models)

Project A/ (.venv/ - Just links, ~50MB)
Project B/ (.venv/ - Just links, ~50MB)

Total: ~22GB (2GB saved, faster installs)
```

## 🔧 **UV Commands Cheat Sheet**

```bash
# Project setup
uv init                    # Initialize new project
uv sync                    # Install all dependencies
uv add <package>           # Add new dependency
uv remove <package>        # Remove dependency
uv lock                    # Update lock file

# Environment management
uv venv                    # Create virtual environment
uv pip install <package>  # Install in current env
uv pip list               # List installed packages

# Running commands
uv run python app.py      # Run with automatic env activation
uv run pytest            # Run tests
uv run black .            # Format code
```

## 🎯 **Model Download Behavior**

### First Time Setup
```bash
# 1. UV installs Python packages (fast, from cache if available)
uv sync  # ~30 seconds

# 2. Models download when first used (one-time per model)
python app.py
# -> Stable Diffusion 1.5 downloads (~4GB, 5-10 minutes)
# -> SDXL downloads (~6GB, 8-15 minutes)  
# -> FLUX downloads (~24GB, 20-40 minutes)
```

### Subsequent Projects
```bash
# 1. UV reuses cached packages (very fast)
uv sync  # ~5 seconds

# 2. Models already cached (instant)
python app.py  # Starts immediately, models already available
```

## 💡 **Pro Tips**

### 🚀 **Speed Optimization**
```bash
# Pre-download models for offline use
python -c "
from src.core.model_registry import get_model_registry
registry = get_model_registry()
# This will trigger model downloads
"
```

### 🧹 **Cache Management**
```bash
# Check cache size
uv cache info

# Clean old cached packages
uv cache clean

# Keep models, clean packages only
uv cache clean --package
```

### 🔄 **Multiple CUDA Versions**
```bash
# Switch between CUDA versions easily
uv remove torch torchvision
uv add torch torchvision --index pytorch-cu118  # Switch to CUDA 11.8
```

## 🆚 **UV vs Pip Comparison**

| Feature | UV | Pip |
|---------|----|----|
| **Speed** | 10-100x faster | Standard |
| **Cache Sharing** | ✅ Global cache | ❌ Per-project |
| **Disk Usage** | 📉 Minimal | 📈 High duplication |
| **Lock Files** | ✅ Automatic | ❌ Manual |
| **Parallel Downloads** | ✅ Yes | ❌ Sequential |
| **Dependency Resolution** | ✅ Advanced | ⚠️ Basic |

## 🎉 **Result**

With UV, you get:
- **Faster installations** (especially after first setup)
- **Less disk usage** (shared package cache)
- **Better reproducibility** (automatic lock files)
- **Same model sharing** (Hugging Face cache still shared)
- **Easier environment management**

The AI models themselves still download once and are shared across all projects, but the Python package management becomes much more efficient! 🚀