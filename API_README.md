# Academic Multimodal LLM Experiment System - API Documentation

## Overview

This implementation provides a comprehensive REST API for the Academic Multimodal LLM Experiment System, enabling programmatic access to image/video generation, model management, and experiment tracking capabilities.

## Features Implemented

### ✅ Task 8.1: FastAPI Server Foundation

- **FastAPI Server**: Modern, high-performance API server with automatic documentation
- **Image Generation Endpoint**: `/generate/image` with comprehensive parameter validation
- **Video Generation Endpoint**: `/generate/video` with progress tracking support
- **Parameter Validation**: Pydantic models with proper bounds checking and validation
- **Background Tasks**: Asynchronous processing for long-running generation tasks
- **CORS Support**: Cross-origin resource sharing for web integration
- **Health Check**: `/health` endpoint for monitoring

### ✅ Task 8.2: Model Management Endpoints

- **Model Status**: `/models/status` - VRAM monitoring and system information
- **Model Switching**: `/models/switch` - Dynamic model loading with memory management
- **Model Information**: `/models/info/{pipeline_type}/{model_name}` - Detailed model specs
- **Model Listing**: `/models/list/{pipeline_type}` - Available models by pipeline
- **Experiment Tracking**: `/models/experiment/save` - Research data persistence
- **System Metrics**: CPU, RAM, and disk usage monitoring

## API Endpoints

### Core Generation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate/image` | POST | Generate images from text prompts |
| `/generate/video` | POST | Generate videos from text prompts |
| `/tasks/{task_id}` | GET | Check generation task status |
| `/download/{task_id}` | GET | Download generated content |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/status` | GET | System status and VRAM usage |
| `/models/switch` | POST | Switch between models |
| `/models/list/{pipeline_type}` | GET | List available models |
| `/models/info/{pipeline_type}/{model_name}` | GET | Model information |

### Experiment Tracking

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/experiment/save` | POST | Save experiment results |
| `/models/experiment/{experiment_id}` | GET | Retrieve experiment data |
| `/models/experiment/list` | GET | List saved experiments |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | API health check |
| `/docs` | GET | Interactive API documentation |
| `/redoc` | GET | Alternative API documentation |

## Request/Response Examples

### Image Generation

```bash
curl -X POST "http://localhost:8000/generate/image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape with a crystal clear lake",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "guidance_scale": 7.5,
    "compliance_mode": "research_safe"
  }'
```

Response:
```json
{
  "task_id": "uuid-string",
  "status": "queued",
  "message": "Image generation task started",
  "estimated_time": 30.0
}
```

### Model Status

```bash
curl "http://localhost:8000/models/status"
```

Response:
```json
{
  "vram_info": {
    "total_mb": 8000,
    "used_mb": 2000,
    "free_mb": 6000,
    "utilization_percent": 25.0
  },
  "current_models": {
    "image": "stable-diffusion-v1-5",
    "video": "stable-video-diffusion"
  },
  "available_models": {
    "image": ["stable-diffusion-v1-5", "sdxl-turbo"],
    "video": ["stable-video-diffusion", "animatediff"]
  },
  "hardware_info": {
    "gpu_model": "RTX 4090",
    "vram_size_mb": 24000,
    "cuda_available": true
  }
}
```

## Architecture

### Components

1. **FastAPI Server** (`src/api/server.py`)
   - Main application with endpoint definitions
   - Request/response models with validation
   - Background task management

2. **Model Management** (`src/api/model_management.py`)
   - Model switching and monitoring
   - VRAM usage tracking
   - Experiment persistence

3. **Dependencies** (`src/api/dependencies.py`)
   - Shared state management
   - Component initialization
   - Dependency injection

4. **Experiment Tracker** (`src/data/experiment_tracker.py`)
   - SQLite-based experiment storage
   - Research data organization
   - Compliance tracking

### Key Features

- **Hardware Optimization**: Automatic VRAM monitoring and model switching
- **Copyright Compliance**: Built-in compliance mode enforcement
- **Progress Tracking**: Real-time status updates for long-running tasks
- **Error Handling**: Comprehensive error responses with helpful messages
- **Validation**: Pydantic models ensure data integrity
- **Documentation**: Auto-generated OpenAPI/Swagger documentation

## Running the Server

### Development Mode

```bash
python -c "from src.api.server import run_server; run_server(reload=True)"
```

### Production Mode

```bash
python -c "from src.api.server import run_server; run_server(host='0.0.0.0', port=8000)"
```

### Using uvicorn directly

```bash
uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```

## Testing

### Unit Tests

```bash
python -m pytest tests/unit/test_api_endpoints.py -v
```

### Integration Tests

```bash
python -m pytest tests/integration/test_api_integration.py -v
```

### Demo Script

```bash
python demo_api.py
```

## Configuration

The API automatically detects hardware configuration and initializes appropriate models. Configuration can be customized through:

- Hardware detection settings
- Model selection preferences
- Compliance mode defaults
- Performance optimization levels

## Dependencies

- **FastAPI**: Modern web framework
- **Pydantic**: Data validation and serialization
- **uvicorn**: ASGI server
- **SQLite**: Experiment data storage
- **psutil**: System monitoring

## Compliance and Ethics

The API enforces copyright compliance through:

- **Compliance Modes**: Open Source Only, Research Safe, Full Dataset
- **Attribution Tracking**: Automatic source attribution
- **License Classification**: Content license categorization
- **Research Ethics**: Academic integrity validation

## Performance Considerations

- **Asynchronous Processing**: Non-blocking generation tasks
- **Memory Management**: Automatic VRAM optimization
- **Model Switching**: Efficient model loading/unloading
- **Caching**: Intelligent model and result caching
- **Hardware Adaptation**: Automatic optimization for available resources

## Future Enhancements

- WebSocket support for real-time progress updates
- Batch processing capabilities
- Advanced scheduling and queuing
- Multi-user authentication and authorization
- Cloud deployment configurations
- Advanced monitoring and analytics

## Error Handling

The API provides comprehensive error handling with:

- HTTP status codes following REST conventions
- Detailed error messages with context
- Validation error details
- Hardware limitation warnings
- Compliance violation notifications

## Security

- Input validation and sanitization
- Rate limiting capabilities
- CORS configuration
- Request size limits
- Safe file handling

This implementation provides a solid foundation for the Academic Multimodal LLM Experiment System API, enabling researchers to programmatically access advanced AI generation capabilities while maintaining ethical compliance and optimal performance.