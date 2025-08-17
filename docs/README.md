# Academic Multimodal LLM Experiment System

## Overview

The Academic Multimodal LLM Experiment System is a comprehensive research platform designed for experimenting with Large Language Models (LLMs) and multimodal AI generation, including image and video generation using Stable Diffusion models. The system provides a web-based interface, comprehensive API, and extensive logging and experiment tracking capabilities.

## Key Features

- **AI Image Generation**: Generate images using Stable Diffusion models (v1.5, v2.1, SDXL)
- **AI Video Generation**: Create videos using Stable Video Diffusion and other video models
- **Hardware Optimization**: Automatic GPU detection and memory management
- **Experiment Tracking**: Comprehensive logging and experiment database
- **Web Interface**: Gradio-based UI for easy interaction
- **REST API**: Full API access for programmatic control
- **Compliance Controls**: Copyright and licensing compliance management
- **Memory Management**: Advanced CUDA memory optimization and cleanup

## System Architecture

The system is built with a modular architecture consisting of several key components:

```
src/
├── core/           # Core system components and interfaces
├── ui/            # User interface components (Gradio)
├── api/           # REST API server and endpoints
├── pipelines/     # AI generation pipelines (image, video)
├── hardware/      # Hardware detection and optimization
└── data/          # Data management and experiment tracking
```

## Quick Start

### Prerequisites

- Python 3.13+
- CUDA-compatible GPU (recommended)
- Windows 10/11 (tested)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Projects
   ```

2. **Install dependencies**:
   ```bash
   py -3.13 -m pip install -r requirements.txt
   ```

3. **Launch the system**:
   ```bash
   py -3.13 launch_real.py
   ```

4. **Access the web interface**:
   Open http://127.0.0.1:7861 in your browser

## System Components

### Core System (`src/core/`)

- **`system_integration.py`**: Main orchestrator that connects all components
- **`interfaces.py`**: Data structures and enums for the system
- **`generation_workflow.py`**: Manages the complete generation process
- **`gpu_optimizer.py`**: GPU-specific optimizations and memory management

### User Interface (`src/ui/`)

- **`research_interface_real.py`**: Main Gradio interface with real AI generation
- **`research_interface_simple.py`**: Simplified interface for testing
- **`launcher.py`**: Interface launcher with proper path configuration

### AI Pipelines (`src/pipelines/`)

- **`image_generation.py`**: Stable Diffusion image generation pipeline
- **`video_generation.py`**: Video generation using various models
- **`lora_training.py`**: LoRA fine-tuning capabilities

### Hardware Management (`src/hardware/`)

- **`detector.py`**: Automatic hardware detection
- **`memory_manager.py`**: Memory optimization strategies
- **`profiles.py`**: Hardware-specific optimization profiles

### API Server (`src/api/`)

- **`server.py`**: FastAPI-based REST server
- **`model_management.py`**: Model loading and management
- **`dependencies.py`**: API dependencies and utilities

## Usage

### Web Interface

1. **Image Generation**:
   - Select a model from the dropdown
   - Enter your prompt
   - Adjust parameters (width, height, steps, guidance scale)
   - Click "Generate Image"

2. **Video Generation**:
   - Switch to the "Video Generation" tab
   - Enter video prompt and parameters
   - Set frames and FPS
   - Click "Generate Video"

### API Usage

The system provides a comprehensive REST API:

```python
import requests

# Generate an image
response = requests.post("http://localhost:8000/generate/image", json={
    "prompt": "A beautiful sunset over mountains",
    "width": 512,
    "height": 512,
    "steps": 20,
    "guidance_scale": 7.5
})

# Generate a video
response = requests.post("http://localhost:8000/generate/video", json={
    "prompt": "A cat walking in a garden",
    "width": 512,
    "height": 512,
    "frames": 32,
    "fps": 20
})
```

### Command Line

```bash
# Launch with specific configuration
py -3.13 main.py --mode ui --port 7861

# Run experiments
py -3.13 main.py --mode experiment --config experiments/config.json

# API server only
py -3.13 main.py --mode api --port 8000
```

## Configuration

### System Configuration (`config/system_config.json`)

```json
{
  "hardware": {
    "gpu_model": "GTX 1650",
    "vram_gb": 4,
    "auto_optimize": true
  },
  "models": {
    "default_image_model": "stable-diffusion-v1-5",
    "default_video_model": "stable-video-diffusion"
  },
  "generation": {
    "max_concurrent": 1,
    "default_steps": 20,
    "default_guidance_scale": 7.5
  }
}
```

### Hardware Profiles (`src/hardware/profiles.py`)

The system automatically detects your hardware and applies appropriate optimizations:

- **GTX 1650 (4GB)**: Conservative memory usage, CPU offloading disabled
- **RTX 3060 (12GB)**: Balanced optimization with XFormers support
- **RTX 4090 (24GB)**: Aggressive optimization with full features

## Memory Management

The system includes advanced memory management features:

- **Automatic cleanup**: Memory is freed after each generation
- **Model switching**: Previous models are unloaded when switching
- **VRAM optimization**: Configurable memory limits and cleanup intervals
- **CPU offloading**: Automatic fallback to CPU when GPU memory is insufficient

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce image resolution
   - Lower the number of steps
   - Enable memory cleanup
   - Check if other applications are using GPU memory

2. **Import Errors**:
   - Ensure you're using Python 3.13
   - Check that all dependencies are installed
   - Verify the `src/` directory is in your Python path

3. **Model Loading Failures**:
   - Check internet connection for model downloads
   - Verify sufficient disk space
   - Check GPU memory availability

### Debug Mode

Enable detailed logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

- **For 4GB GPUs**: Use 512x512 resolution, 20 steps max
- **For 8GB GPUs**: Can handle 768x768, 30 steps
- **For 12GB+ GPUs**: Full resolution support, 50+ steps

## Development

### Adding New Models

1. **Update model registry** in `src/pipelines/image_generation.py`
2. **Add hardware requirements** in `src/hardware/profiles.py`
3. **Update UI model list** in `src/ui/research_interface_real.py`

### Extending Pipelines

1. **Create new pipeline class** inheriting from `IGenerationPipeline`
2. **Implement required methods**: `generate()`, `initialize()`, `cleanup()`
3. **Register with system integration**

### Testing

```bash
# Run unit tests
py -3.13 -m pytest tests/unit/

# Run integration tests
py -3.13 -m pytest tests/integration/

# Run performance tests
py -3.13 -m pytest tests/performance/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs in the `logs/` directory
- Open an issue on the repository
- Check the system status in the web interface

## Changelog

### Version 1.0.0
- Initial release with image and video generation
- Hardware optimization and memory management
- Web interface and REST API
- Experiment tracking and logging 