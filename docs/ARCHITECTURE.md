# System Architecture

## Overview

The Academic Multimodal LLM Experiment System is designed with a modular, layered architecture that separates concerns and provides clear interfaces between components. The system follows the principle of loose coupling and high cohesion, making it easy to extend, test, and maintain.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
├─────────────────────────────────────────────────────────────┤
│  Gradio UI  │  REST API  │  CLI  │  Experiment Interface  │
├─────────────────────────────────────────────────────────────┤
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│  System Integration  │  Workflow Coordinator  │  Controllers │
├─────────────────────────────────────────────────────────────┤
│                     Domain Layer                            │
├─────────────────────────────────────────────────────────────┤
│  Image Pipeline  │  Video Pipeline  │  LoRA Training      │
├─────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Hardware Mgmt  │  Memory Mgmt  │  Logging  │  Storage     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. System Integration (`src/core/system_integration.py`)

The central orchestrator that coordinates all system components.

**Responsibilities:**
- Initialize and manage all pipelines
- Coordinate hardware detection and optimization
- Manage experiment tracking and logging
- Provide unified interface for generation requests

**Key Methods:**
```python
class SystemIntegration:
    def initialize(self, config: Dict[str, Any]) -> bool
    def execute_complete_generation_workflow(self, prompt: str, ...) -> GenerationResult
    def get_system_status(self) -> Dict[str, Any]
    def cleanup(self) -> None
```

**Dependencies:**
- Hardware configuration
- Image and video pipelines
- Experiment tracker
- LLM controller

### 2. Generation Pipelines

#### Image Generation Pipeline (`src/pipelines/image_generation.py`)

Handles all image generation using Stable Diffusion models.

**Features:**
- Multiple model support (v1.5, v2.1, SDXL)
- Hardware-adaptive optimization
- Memory management
- Quality control

**Key Methods:**
```python
class ImageGenerationPipeline:
    def initialize(self, hardware_config: HardwareConfig) -> bool
    def generate(self, request: GenerationRequest) -> GenerationResult
    def optimize_for_hardware(self, hardware_config: HardwareConfig) -> None
    def cleanup(self) -> None
```

#### Video Generation Pipeline (`src/pipelines/video_generation.py`)

Manages video generation using various video diffusion models.

**Features:**
- Stable Video Diffusion support
- AnimateDiff integration
- Temporal consistency
- Frame processing optimization

**Key Methods:**
```python
class VideoGenerationPipeline:
    def initialize(self, hardware_config: HardwareConfig) -> bool
    def generate(self, request: GenerationRequest) -> GenerationResult
    def _generate_video_hybrid(self, params: Dict[str, Any]) -> List[np.ndarray]
    def _save_video(self, frames: List[np.ndarray], ...) -> str
```

### 3. Hardware Management (`src/hardware/`)

#### Hardware Detector (`src/hardware/detector.py`)

Automatically detects and profiles system hardware.

**Capabilities:**
- GPU detection and profiling
- VRAM measurement
- CUDA capability detection
- CPU core counting

#### Memory Manager (`src/hardware/memory_manager.py`)

Optimizes memory usage based on hardware capabilities.

**Strategies:**
- **Aggressive**: Maximum performance, higher memory usage
- **Balanced**: Good performance, moderate memory usage
- **Minimal**: Conservative memory usage, basic performance

#### Hardware Profiles (`src/hardware/profiles.py`)

Pre-configured optimization profiles for different GPU types.

**Profiles:**
```python
HARDWARE_PROFILES = {
    "gtx_1650_4gb": {
        "torch_dtype": "float32",
        "enable_attention_slicing": True,
        "enable_vae_slicing": True,
        "cpu_offload": False,
        "xformers": False
    },
    "rtx_3060_12gb": {
        "torch_dtype": "float16",
        "enable_attention_slicing": False,
        "enable_vae_slicing": False,
        "cpu_offload": False,
        "xformers": True
    }
}
```

### 4. User Interface (`src/ui/`)

#### Research Interface (`src/ui/research_interface_real.py`)

Main Gradio-based web interface for user interaction.

**Features:**
- Image generation interface
- Video generation interface
- Model selection and configuration
- Real-time status monitoring
- Memory management controls

**Key Components:**
```python
class ResearchInterface:
    def __init__(self)
    def _create_interface(self) -> gr.Blocks
    def _create_image_generation_tab(self)
    def _create_video_generation_tab(self)
    def _generate_image(self, prompt: str, ...) -> Tuple[str, Dict, str]
    def _generate_video(self, prompt: str, ...) -> Tuple[str, Dict, str]
```

### 5. API Server (`src/api/`)

#### FastAPI Server (`src/api/server.py`)

RESTful API for programmatic access to the system.

**Endpoints:**
```python
# Image generation
POST /generate/image
POST /generate/image/async

# Video generation
POST /generate/video
POST /generate/video/async

# System management
GET /status
GET /models
POST /models/switch
```

## Data Flow

### 1. Image Generation Flow

```
User Input → UI → System Integration → Image Pipeline → Hardware Optimization → Model Loading → Generation → Post-processing → Output
```

**Detailed Flow:**
1. User enters prompt and parameters in UI
2. UI creates `GenerationRequest` object
3. Request sent to `SystemIntegration.execute_complete_generation_workflow()`
4. System selects appropriate image pipeline
5. Pipeline loads model if not already loaded
6. Hardware optimizations applied (memory management, attention slicing)
7. Image generation using Stable Diffusion
8. Post-processing (saving, metadata)
9. Result returned to UI

### 2. Video Generation Flow

```
User Input → UI → System Integration → Video Pipeline → Model Selection → Frame Generation → Temporal Consistency → Video Assembly → Output
```

**Detailed Flow:**
1. User enters video prompt and parameters
2. UI creates video generation request
3. System selects optimal video model based on hardware
4. Video pipeline initializes with hardware optimizations
5. Frames generated using video diffusion model
6. Temporal consistency applied between frames
7. Video assembled and saved
8. Result returned to UI

### 3. Hardware Optimization Flow

```
System Start → Hardware Detection → Profile Selection → Memory Manager → Pipeline Optimization → Runtime Monitoring
```

**Detailed Flow:**
1. System detects GPU and VRAM
2. Selects appropriate hardware profile
3. Memory manager applies optimization strategy
4. Pipelines configured with hardware-specific settings
5. Runtime monitoring tracks memory usage
6. Automatic cleanup when thresholds exceeded

## Interfaces and Contracts

### 1. Generation Request Interface

```python
@dataclass
class GenerationRequest:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    compliance_mode: ComplianceMode
    additional_params: Optional[Dict[str, Any]] = None
```

### 2. Generation Result Interface

```python
@dataclass
class GenerationResult:
    success: bool
    output_path: Optional[str]
    generation_time: float
    model_used: str
    quality_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    compliance_info: Optional[Dict[str, Any]] = None
```

### 3. Hardware Configuration Interface

```python
@dataclass
class HardwareConfig:
    gpu_model: str
    vram_gb: float
    cuda_capability: Tuple[int, int]
    cpu_cores: int
    memory_gb: float
```

## Error Handling and Resilience

### 1. Graceful Degradation

The system is designed to continue operating even when some components fail:

- **Model Loading Failures**: Fallback to alternative models
- **GPU Memory Issues**: Automatic CPU offloading
- **Pipeline Failures**: Error reporting with detailed diagnostics
- **Hardware Issues**: Mock mode for testing

### 2. Error Recovery

- **Automatic Retry**: Failed operations are retried with reduced parameters
- **Memory Cleanup**: Automatic cleanup when memory issues occur
- **Model Switching**: Automatic fallback to compatible models
- **Logging**: Comprehensive error logging for debugging

### 3. Monitoring and Alerting

- **Real-time Status**: Live system status in UI
- **Memory Monitoring**: Continuous VRAM usage tracking
- **Performance Metrics**: Generation time and quality tracking
- **Error Reporting**: Detailed error messages with context

## Performance Characteristics

### 1. Memory Usage

- **Base Memory**: ~2GB for system components
- **Model Loading**: 2-8GB depending on model size
- **Generation Memory**: 1-4GB depending on resolution and steps
- **Peak Memory**: 6-12GB for high-resolution generation

### 2. Generation Speed

- **Image Generation**: 10-60 seconds depending on parameters
- **Video Generation**: 30-300 seconds depending on length and quality
- **Model Loading**: 5-30 seconds depending on model size
- **Optimization**: 1-5 seconds for hardware-specific settings

### 3. Scalability

- **Concurrent Generations**: Configurable (default: 1)
- **Model Caching**: Automatic model reuse
- **Memory Sharing**: Efficient memory allocation between pipelines
- **Resource Pooling**: Shared resources for multiple operations

## Security and Compliance

### 1. Input Validation

- **Prompt Sanitization**: Removes potentially harmful content
- **Parameter Validation**: Ensures parameters are within safe ranges
- **Resource Limits**: Prevents resource exhaustion attacks

### 2. Compliance Controls

- **Copyright Checking**: Validates generated content against known copyrighted material
- **License Management**: Tracks model licenses and usage
- **Audit Logging**: Comprehensive logging of all operations

### 3. Access Control

- **API Authentication**: Optional API key authentication
- **Rate Limiting**: Configurable request rate limits
- **Resource Quotas**: Per-user resource allocation limits

## Testing Strategy

### 1. Unit Testing

- **Component Testing**: Individual component functionality
- **Interface Testing**: Contract compliance verification
- **Mock Testing**: Isolated testing with mocked dependencies

### 2. Integration Testing

- **Pipeline Testing**: End-to-end generation workflows
- **Hardware Testing**: Hardware detection and optimization
- **API Testing**: REST endpoint functionality

### 3. Performance Testing

- **Load Testing**: Concurrent generation performance
- **Memory Testing**: Memory usage and cleanup
- **Stress Testing**: Resource exhaustion scenarios

## Deployment and Operations

### 1. Environment Requirements

- **Python**: 3.13+
- **CUDA**: 11.8+ (for GPU acceleration)
- **Memory**: 8GB+ RAM, 4GB+ VRAM
- **Storage**: 10GB+ for models and outputs

### 2. Configuration Management

- **Environment Variables**: Runtime configuration
- **Configuration Files**: JSON-based system configuration
- **Hardware Profiles**: Automatic hardware detection and configuration
- **Model Registry**: Centralized model management

### 3. Monitoring and Maintenance

- **Health Checks**: System status monitoring
- **Log Rotation**: Automatic log management
- **Model Updates**: Automatic model version management
- **Performance Tuning**: Continuous optimization based on usage patterns 