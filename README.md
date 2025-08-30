# 🎨 AI Content Generator

A modern, user-friendly interface for generating images and videos using the latest AI models. Features automatic hardware optimization and both easy and advanced modes for users of all skill levels.

## ✨ Features

- **🖼️ Image Generation**: Create stunning images from text prompts using the latest models
- **🎬 Video Generation**: Generate videos from text or images with state-of-the-art models  
- **🔄 Easy/Advanced Modes**: Simple interface for beginners, full control for power users
- **⚡ Hardware Optimization**: Automatic optimization for 2GB to 24GB+ VRAM
- **🤖 Latest Models**: Updated model registry with FLUX.1, SDXL, Stable Video Diffusion
- **🎯 Smart Fallbacks**: Automatic model selection based on your hardware

## 🖥️ Supported Hardware

### Minimum Requirements
- **GPU**: GTX 1050 (2GB VRAM) or equivalent
- **RAM**: 8GB system RAM  
- **Storage**: 20GB free space for models
- **OS**: Windows 10/11, macOS, or Linux with CUDA support

### Recommended Configurations
- **Budget**: GTX 1650 (4GB) → Stable Diffusion 1.5, fast image generation
- **Mid-Range**: RTX 3070 (8GB) → SDXL, high-quality images
- **High-End**: RTX 4090 (24GB) → FLUX.1, video generation, best quality

## 🚀 Quick Start

### 🌍 **Cross-Platform Support**
Works on **Windows**, **Linux**, and **macOS** with automatic hardware detection!

### 1. **Platform-Specific Quick Launch**

#### 🪟 **Windows** (One-Click)
```cmd
# Double-click or run in Command Prompt
run_windows.bat
```

#### 🐧 **Linux/Ubuntu** (One-Click)
```bash
# Make executable and run
chmod +x run_linux.sh
./run_linux.sh
```

#### 🍎 **macOS** (One-Click)
```bash
# Use the Linux script (works on macOS)
chmod +x run_linux.sh
./run_linux.sh
```

#### 🌐 **Universal Launcher** (Any Platform)
```bash
# Cross-platform launcher with auto-detection
python3 launch.py

# Check system specs first
python3 system_specs.py

# Check dependencies only
python3 launch.py --check
```

### 2. **Manual Installation** (If Needed)

#### Using UV (Recommended - Faster)
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh  # Linux/Mac
# or
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Setup project
uv sync
uv add torch torchvision --index pytorch-cu121
```

#### Using pip (Traditional)
```bash
# Windows
python -m venv ai_generator
ai_generator\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Linux/macOS
python3 -m venv ai_generator
source ai_generator/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

### 3. **Launch & Generate**
1. **Hardware Detection**: System automatically detects your GPU and VRAM
2. **Model Recommendations**: Shows only compatible models for your hardware
3. **Auto-Download**: Recommended models download automatically
4. **Start Creating**: Use Easy mode for quick results!

### 📖 **Detailed Setup Guides**
- 🌍 **[Cross-Platform Setup](CROSS_PLATFORM_SETUP.md)** - Comprehensive platform-specific instructions
- 🚀 **[UV Setup Guide](UV_SETUP.md)** - Fast package management with UV
- 🖥️ **[Hardware Features](HARDWARE_AWARE_FEATURES.md)** - Hardware detection and optimization

## Project Structure

```
academic-multimodal-llm-system/
├── src/
│   ├── core/                 # Core interfaces and configuration
│   ├── pipelines/           # Image and video generation pipelines
│   ├── data/                # Copyright-aware data management
│   ├── hardware/            # Hardware detection and optimization
│   ├── ui/                  # Gradio research interface
│   └── api/                 # REST API endpoints
├── tests/                   # Unit and integration tests
├── config/                  # Configuration files
├── data/                    # Training datasets (organized by license)
├── models/                  # Downloaded and fine-tuned models
├── experiments/             # Experiment results and logs
└── cache/                   # Temporary files and model cache
```

## Copyright Compliance Modes

### Open Source Only
- **Content**: Public Domain + Creative Commons
- **Sources**: Wikimedia Commons, Unsplash, Pexels, Archive.org
- **Use Case**: Commercial-safe research and development

### Research Safe (Default)
- **Content**: Open Source + Fair Use Research
- **Sources**: Above + Flickr, DeviantArt, YouTube (research exemption)
- **Use Case**: Academic research with fair use justification

### Full Dataset
- **Content**: All content including copyrighted material
- **Sources**: All sources including commercial stock photos
- **Use Case**: Research comparison studies only (clearly labeled)

## Hardware Optimization

The system automatically detects your hardware and applies appropriate optimizations:

### GTX 1650 (4GB VRAM)
- Aggressive memory optimization
- CPU offloading for LLM tasks
- SD 1.5 with attention slicing
- 512x512 resolution, 30-60s generation time

### RTX 3070 (8GB VRAM)
- Balanced optimization
- SDXL-Turbo support
- 768x768 resolution, 10-20s generation time

### RTX 4090 (24GB VRAM)
- Minimal optimization needed
- FLUX.1-schnell support
- 1024x1024 resolution, 5-10s generation time
- Concurrent model loading

## Research Ethics

This system is designed with academic research ethics in mind:

- **Transparent Licensing**: All content is classified and attributed
- **Fair Use Compliance**: Research exemptions are clearly documented
- **Attribution Tracking**: Source attribution is maintained throughout
- **Selective Training**: Choose which license types to include in training
- **Audit Trail**: Complete provenance tracking for academic integrity

## Development Phases

The system supports incremental development:

1. **Phase 1**: Basic image generation setup
2. **Phase 2**: Video generation integration
3. **Phase 3**: Ethical data collection
4. **Phase 4**: Fine-tuning experiments
5. **Phase 5**: LLM integration and workflows
6. **Phase 6**: Analysis and documentation

## Contributing

This is an academic research project. Contributions should focus on:

- Ethical AI practices and copyright compliance
- Hardware optimization for consumer GPUs
- Research methodology and experiment tracking
- Academic integrity and transparency

## License

MIT License - See LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{academic_multimodal_llm_2024,
  title={Academic Multimodal LLM Experiment System},
  author={Academic Research Team},
  year={2024},
  url={https://github.com/example/academic-multimodal-llm-system}
}
```

## Disclaimer

This system is designed for academic research purposes. Users are responsible for ensuring compliance with copyright laws and institutional ethics guidelines in their jurisdiction.