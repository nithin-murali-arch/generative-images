# 🎨 AI Content Generator

A cross-platform AI content generator with automatic hardware optimization, thermal safety, and support for the latest image and video generation models.

## ✨ Key Features

- **🖼️ Image Generation**: Latest models (FLUX.1, SDXL, Stable Diffusion)
- **🎬 Video Generation**: Text-to-video and image-to-video support
- **🌡️ Thermal Safety**: Real-time temperature monitoring with automatic cooling
- **🖥️ Hardware Optimization**: Automatic detection and optimization for 2GB-24GB+ VRAM
- **🌍 Cross-Platform**: Windows, Linux, and macOS support
- **⚡ Smart Model Selection**: Only shows compatible models for your hardware

## 🚀 Quick Start

### One-Click Launch (Recommended)

#### Windows
```cmd
run_windows.bat
```

#### Linux/macOS
```bash
chmod +x run_linux.sh && ./run_linux.sh
# or for macOS specifically:
chmod +x run_macos.sh && ./run_macos.sh
```

#### Universal (Any Platform)
```bash
python3 launch.py
```

### Manual Installation

#### Using UV (Fast Package Manager)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup project
uv sync
python3 app.py
```

#### Using pip (Traditional)
```bash
# Create virtual environment
python3 -m venv ai_generator
source ai_generator/bin/activate  # Linux/Mac
# or: ai_generator\\Scripts\\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Launch
python3 app.py
```

## 🖥️ Hardware Support

### Automatic Hardware Detection
The system automatically detects your GPU and recommends optimal models:

| Hardware Tier | VRAM | Example GPUs | Recommended Models | Performance |
|---------------|------|--------------|-------------------|-------------|
| **Lightweight** | 2-4GB | GTX 1650, RTX 3050 | SD 1.5, Tiny SD | 2-5s per image |
| **Mid-Tier** | 6-12GB | RTX 3070, RTX 4060 Ti | SDXL, SDXL Turbo | 3-8s per image |
| **High-End** | 16-24GB | RTX 4080, RTX 4090 | FLUX.1 Schnell | 2-6s per image |
| **Ultra** | 24GB+ | RTX 6000, A100 | FLUX.1 Dev | 1-4s per image |

### Thermal Safety
- **Real-time monitoring** of CPU and GPU temperatures
- **Automatic pause** at 70°C, resume at 45°C
- **Emergency shutdown** at 90°C to prevent hardware damage
- **Cross-platform** temperature sensor detection

## 🎯 Smart Features

### Hardware-Aware Model Selection
- Only shows models compatible with your VRAM
- Automatic model downloads for your hardware tier
- Performance estimates based on your GPU
- Dedicated vs integrated GPU detection

### Thermal Protection
- Continuous temperature monitoring during generation
- Automatic workload pausing when temperatures exceed safe limits
- No mocking or fallbacks - real sensor readings only
- Platform-specific thermal sensor integration

### Cross-Platform Optimization
- **Windows**: WMI hardware detection, CUDA optimization
- **Linux**: lm-sensors integration, package manager detection
- **macOS**: Apple Silicon MPS support, system_profiler integration

## 📁 Project Structure

```
ai-content-generator/
├── src/
│   ├── core/                 # Core system components
│   │   ├── model_registry.py      # Latest model definitions
│   │   ├── thermal_monitor.py     # Real-time thermal safety
│   │   ├── cross_platform_hardware.py  # Hardware detection
│   │   └── model_downloader.py    # Intelligent model downloads
│   ├── pipelines/           # Generation pipelines
│   └── ui/                  # Modern Gradio interface
├── .kiro/steering/          # Safety policies and guidelines
├── run_windows.bat          # Windows one-click launcher
├── run_linux.sh            # Linux one-click launcher
├── run_macos.sh            # macOS one-click launcher
├── launch.py               # Universal cross-platform launcher
├── app.py                  # Main application entry point
└── pyproject.toml          # Modern Python project configuration
```

## 🛡️ Safety & Compliance

### No-Mocking Policy
- **Real hardware detection only** - no fallbacks or estimates
- **Actual thermal readings** - no simulated temperatures
- **Fail-fast behavior** - stops if hardware detection fails
- **Comprehensive logging** - full audit trail of operations

### Thermal Safety Protocol
```python
# Automatic safety checks before AI workloads
from src.core.thermal_monitor import ensure_thermal_safety

if not ensure_thermal_safety():
    # System automatically waits for cooling or shuts down
    raise RuntimeError("System too hot - operations paused")
```

## 🔧 Advanced Usage

### System Information
```bash
# Check hardware specifications
python3 system_specs.py

# Test thermal safety system
python3 test_thermal_safety.py

# Check dependencies
python3 launch.py --check
```

### Model Management
- Models auto-download based on hardware compatibility
- Shared model cache across projects (`~/.cache/huggingface/`)
- Background downloads don't block interface
- One-click model switching in UI

### Performance Optimization
- Automatic VRAM optimization based on detected hardware
- CPU offloading for memory-constrained systems
- Attention slicing and gradient checkpointing
- Platform-specific acceleration (CUDA, MPS, CPU)

## 🌡️ Thermal Monitoring

The system includes comprehensive thermal protection:

- **Real-time monitoring** of all CPU and GPU sensors
- **Automatic operation pause** when temperatures reach 70°C
- **Resume operations** when temperatures drop below 45°C
- **Emergency shutdown** at 90°C to prevent hardware damage
- **Cross-platform sensor support** (Linux lm-sensors, Windows WMI, macOS system calls)

## 🎉 Getting Started

1. **Run launcher**: Use platform-specific script or `python3 launch.py`
2. **Hardware detection**: System automatically detects GPU and VRAM
3. **Model recommendations**: See only compatible models for your hardware
4. **Auto-download**: Recommended models download automatically
5. **Start generating**: Use the web interface at http://localhost:7860

## 📞 Support

If you encounter issues:

1. **Check system specs**: `python3 system_specs.py`
2. **Verify thermal safety**: `python3 test_thermal_safety.py`
3. **Test dependencies**: `python3 launch.py --check`
4. **Check logs**: Look for error messages in console output
5. **Update drivers**: Ensure GPU drivers are current

## 🏗️ Development

The system is built with:
- **Modern Python** (3.8+) with type hints
- **Cross-platform compatibility** (Windows/Linux/macOS)
- **Real hardware detection** (no mocking or simulation)
- **Thermal safety enforcement** (mandatory temperature monitoring)
- **Modular architecture** (easy to extend and maintain)

## 📄 License

MIT License - See LICENSE file for details.

---

**Ready to create amazing AI content safely and efficiently across any platform!** 🚀