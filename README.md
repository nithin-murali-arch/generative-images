# Academic Multimodal LLM Experiment System

A research-focused platform for ethical experimentation with multimodal AI generation on consumer gaming hardware. This system emphasizes copyright compliance, hardware optimization, and systematic research methodology while providing a seamless interface for generating images and videos from text prompts.

## Features

- **Copyright-Aware Data Management**: Automatic classification and ethical handling of training data
- **Hardware Optimization**: Adaptive optimization for 4GB to 24GB+ VRAM configurations
- **Multimodal Generation**: Support for both image and video generation with various models
- **Research Ethics**: Built-in compliance modes and attribution tracking
- **Local Development**: Complete local setup without cloud dependencies
- **Experiment Tracking**: Comprehensive logging and result comparison

## Supported Hardware

### Minimum Requirements
- **GPU**: GTX 1650 (4GB VRAM) or equivalent
- **RAM**: 16GB system RAM
- **Storage**: 50GB free space for models and data
- **OS**: Windows 10/11 with CUDA support

### Recommended Configurations
- **Entry Level**: GTX 1650 (4GB) - SD 1.5, basic image generation
- **Mid Range**: RTX 3070 (8GB) - SDXL-Turbo, faster generation
- **High End**: RTX 4090 (24GB) - FLUX.1, video generation, concurrent models

## Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv multimodal_research
multimodal_research\Scripts\activate  # Windows

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Install Ollama for LLM Serving
```bash
# Install Ollama
winget install ollama

# Download lightweight LLM
ollama pull phi3:mini  # 2.3GB model for 4GB VRAM systems
# OR for higher-end systems:
ollama pull llama3.1:8b  # 4.7GB model for 16GB+ VRAM
```

### 3. Initialize System
```python
from src.core.config import ConfigManager
from src.hardware.detector import HardwareDetector

# Initialize configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Detect hardware and optimize
detector = HardwareDetector()
hardware_config = detector.detect_hardware()
config_manager.update_hardware_config(hardware_config)
```

### 4. Run Basic Generation
```python
from src.main import AcademicMultimodalSystem

# Initialize system
system = AcademicMultimodalSystem()

# Generate image with copyright compliance
result = system.generate_image(
    prompt="A serene landscape with mountains",
    compliance_mode="research_safe"  # Uses only PD + CC + Fair Use content
)
```

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