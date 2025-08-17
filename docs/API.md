# API Documentation

## Overview

The Academic Multimodal LLM Experiment System provides a comprehensive REST API for programmatic access to all system capabilities. The API is built using FastAPI and provides both synchronous and asynchronous endpoints for image and video generation.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API operates without authentication. For production use, consider implementing API key authentication.

## API Endpoints

### 1. System Status

#### Get System Status
```http
GET /status
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-17T20:30:00Z",
  "hardware": {
    "gpu_model": "GTX 1650",
    "vram_total_gb": 4.0,
    "vram_used_gb": 1.2,
    "vram_free_gb": 2.8,
    "cuda_version": "11.8"
  },
  "pipelines": {
    "image": {
      "initialized": true,
      "current_model": "stable-diffusion-v1-5",
      "available_models": ["stable-diffusion-v1-5", "stable-diffusion-2-1"]
    },
    "video": {
      "initialized": true,
      "current_model": "stable-video-diffusion",
      "available_models": ["stable-video-diffusion", "animatediff"]
    }
  },
  "experiments": {
    "total": 15,
    "today": 3,
    "last_experiment": "2025-08-17T19:45:00Z"
  }
}
```

#### Get Hardware Information
```http
GET /hardware
```

**Response:**
```json
{
  "gpu": {
    "model": "GTX 1650",
    "vram_gb": 4.0,
    "cuda_capability": [7, 5],
    "driver_version": "471.11"
  },
  "cpu": {
    "cores": 8,
    "model": "Intel Core i7-10700K",
    "memory_gb": 32.0
  },
  "optimization_profile": "gtx_1650_4gb"
}
```

### 2. Model Management

#### Get Available Models
```http
GET /models
```

**Response:**
```json
{
  "image_models": [
    {
      "name": "stable-diffusion-v1-5",
      "model_id": "runwayml/stable-diffusion-v1-5",
      "min_vram_mb": 2000,
      "max_resolution": 768,
      "status": "available"
    },
    {
      "name": "stable-diffusion-2-1",
      "model_id": "stabilityai/stable-diffusion-2-1",
      "min_vram_mb": 3000,
      "max_resolution": 768,
      "status": "available"
    }
  ],
  "video_models": [
    {
      "name": "stable-video-diffusion",
      "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
      "min_vram_mb": 4000,
      "max_frames": 64,
      "status": "available"
    }
  ]
}
```

#### Switch Model
```http
POST /models/switch
```

**Request Body:**
```json
{
  "pipeline_type": "image",
  "model_name": "stable-diffusion-2-1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Switched to stable-diffusion-2-1",
  "previous_model": "stable-diffusion-v1-5",
  "current_model": "stable-diffusion-2-1"
}
```

### 3. Image Generation

#### Generate Image (Synchronous)
```http
POST /generate/image
```

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over mountains with golden light",
  "negative_prompt": "blurry, low quality, distorted",
  "width": 512,
  "height": 512,
  "steps": 20,
  "guidance_scale": 7.5,
  "seed": 42,
  "compliance_mode": "research_safe",
  "model_name": "stable-diffusion-v1-5"
}
```

**Response:**
```json
{
  "success": true,
  "output_path": "outputs/images/generated_1755432807.png",
  "generation_time": 45.2,
  "model_used": "stable-diffusion-v1-5",
  "parameters": {
    "prompt": "A beautiful sunset over mountains with golden light",
    "negative_prompt": "blurry, low quality, distorted",
    "width": 512,
    "height": 512,
    "steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  },
  "quality_metrics": {
    "clarity_score": 0.85,
    "aesthetic_score": 0.78
  },
  "compliance_info": {
    "copyright_checked": true,
    "license_valid": true,
    "compliance_mode": "research_safe"
  }
}
```

#### Generate Image (Asynchronous)
```http
POST /generate/image/async
```

**Request Body:** Same as synchronous endpoint

**Response:**
```json
{
  "task_id": "task_12345",
  "status": "queued",
  "estimated_time": 60,
  "message": "Image generation queued successfully"
}
```

#### Get Async Task Status
```http
GET /tasks/{task_id}
```

**Response:**
```json
{
  "task_id": "task_12345",
  "status": "completed",
  "progress": 100,
  "result": {
    "success": true,
    "output_path": "outputs/images/generated_1755432807.png",
    "generation_time": 45.2
  }
}
```

### 4. Video Generation

#### Generate Video (Synchronous)
```http
POST /generate/video
```

**Request Body:**
```json
{
  "prompt": "A cat walking through a magical garden with flowers blooming",
  "negative_prompt": "blurry, low quality, distorted, fast movement",
  "width": 512,
  "height": 512,
  "frames": 32,
  "fps": 20,
  "compliance_mode": "research_safe",
  "model_name": "stable-video-diffusion"
}
```

**Response:**
```json
{
  "success": true,
  "output_path": "outputs/videos/generated_1755432807.mp4",
  "generation_time": 180.5,
  "model_used": "stable-video-diffusion",
  "parameters": {
    "prompt": "A cat walking through a magical garden with flowers blooming",
    "negative_prompt": "blurry, low quality, distorted, fast movement",
    "width": 512,
    "height": 512,
    "frames": 32,
    "fps": 20
  },
  "quality_metrics": {
    "temporal_consistency": 0.82,
    "motion_smoothness": 0.75,
    "overall_quality": 0.78
  },
  "compliance_info": {
    "copyright_checked": true,
    "license_valid": true,
    "compliance_mode": "research_safe"
  }
}
```

#### Generate Video (Asynchronous)
```http
POST /generate/video/async
```

**Request Body:** Same as synchronous endpoint

**Response:**
```json
{
  "task_id": "task_67890",
  "status": "queued",
  "estimated_time": 300,
  "message": "Video generation queued successfully"
}
```

### 5. Experiment Management

#### Get Experiments
```http
GET /experiments
```

**Query Parameters:**
- `limit`: Number of experiments to return (default: 50)
- `offset`: Number of experiments to skip (default: 0)
- `type`: Filter by type (`image`, `video`, `all`)
- `date_from`: Filter from date (ISO format)
- `date_to`: Filter to date (ISO format)

**Response:**
```json
{
  "experiments": [
    {
      "id": "exp_123",
      "timestamp": "2025-08-17T19:45:00Z",
      "type": "image",
      "prompt": "A beautiful sunset over mountains",
      "model": "stable-diffusion-v1-5",
      "parameters": {
        "width": 512,
        "height": 512,
        "steps": 20
      },
      "output_path": "outputs/images/generated_1755432807.png",
      "generation_time": 45.2,
      "compliance_mode": "research_safe"
    }
  ],
  "total": 15,
  "limit": 50,
  "offset": 0
}
```

#### Get Experiment Details
```http
GET /experiments/{experiment_id}
```

**Response:**
```json
{
  "id": "exp_123",
  "timestamp": "2025-08-17T19:45:00Z",
  "type": "image",
  "prompt": "A beautiful sunset over mountains",
  "negative_prompt": "blurry, low quality",
  "model": "stable-diffusion-v1-5",
  "parameters": {
    "width": 512,
    "height": 512,
    "steps": 20,
    "guidance_scale": 7.5,
    "seed": 42
  },
  "output_path": "outputs/images/generated_1755432807.png",
  "generation_time": 45.2,
  "compliance_mode": "research_safe",
  "quality_metrics": {
    "clarity_score": 0.85,
    "aesthetic_score": 0.78
  },
  "hardware_used": {
    "gpu": "GTX 1650",
    "vram_used_gb": 2.1
  }
}
```

### 6. Memory Management

#### Get Memory Status
```http
GET /memory
```

**Response:**
```json
{
  "gpu": {
    "total_gb": 4.0,
    "allocated_gb": 2.1,
    "reserved_gb": 2.5,
    "free_gb": 1.5,
    "utilization_percent": 52.5
  },
  "system": {
    "total_gb": 32.0,
    "used_gb": 18.5,
    "free_gb": 13.5
  },
  "models": {
    "loaded_models": ["stable-diffusion-v1-5"],
    "estimated_vram_gb": 2.1
  }
}
```

#### Clear Memory Cache
```http
POST /memory/clear
```

**Response:**
```json
{
  "success": true,
  "message": "Memory cache cleared successfully",
  "freed_vram_gb": 0.8,
  "new_status": {
    "allocated_gb": 1.3,
    "reserved_gb": 1.7,
    "free_gb": 2.3
  }
}
```

### 7. System Control

#### Reload Models
```http
POST /system/reload
```

**Request Body:**
```json
{
  "pipeline_type": "image",
  "force": false
}
```

**Response:**
```json
{
  "success": true,
  "message": "Models reloaded successfully",
  "reloaded_models": ["stable-diffusion-v1-5", "stable-diffusion-2-1"]
}
```

#### Shutdown System
```http
POST /system/shutdown
```

**Response:**
```json
{
  "success": true,
  "message": "System shutdown initiated",
  "shutdown_time": "2025-08-17T20:30:00Z"
}
```

## Error Handling

### Error Response Format

All error responses follow this format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameters provided",
    "details": {
      "field": "width",
      "issue": "Value must be between 64 and 2048"
    }
  },
  "timestamp": "2025-08-17T20:30:00Z"
}
```

### Common Error Codes

- `VALIDATION_ERROR`: Invalid request parameters
- `MODEL_NOT_FOUND`: Requested model is not available
- `INSUFFICIENT_MEMORY`: Not enough GPU memory for operation
- `GENERATION_FAILED`: AI generation process failed
- `SYSTEM_ERROR`: Internal system error
- `SERVICE_UNAVAILABLE`: Service temporarily unavailable

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default Limit**: 100 requests per minute per IP
- **Generation Endpoints**: 10 requests per minute per IP
- **Status Endpoints**: No rate limiting

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## WebSocket Support

For real-time updates during long-running operations:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/tasks');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    console.log(`Progress: ${data.progress}%`);
  } else if (data.type === 'completed') {
    console.log('Task completed:', data.result);
  }
};
```

## SDK Examples

### Python Client

```python
import requests
import json

class AISystemClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def generate_image(self, prompt, **kwargs):
        """Generate an image using the AI system."""
        url = f"{self.base_url}/generate/image"
        data = {
            "prompt": prompt,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "steps": kwargs.get("steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 7.5)
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_video(self, prompt, **kwargs):
        """Generate a video using the AI system."""
        url = f"{self.base_url}/generate/video"
        data = {
            "prompt": prompt,
            "width": kwargs.get("width", 512),
            "height": kwargs.get("height", 512),
            "frames": kwargs.get("frames", 32),
            "fps": kwargs.get("fps", 20)
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def get_status(self):
        """Get system status."""
        url = f"{self.base_url}/status"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Usage example
client = AISystemClient()

# Generate an image
result = client.generate_image(
    "A majestic dragon flying over a medieval castle",
    width=768,
    height=768,
    steps=30
)
print(f"Image generated: {result['output_path']}")

# Generate a video
video_result = client.generate_video(
    "A butterfly emerging from a cocoon",
    frames=64,
    fps=24
)
print(f"Video generated: {video_result['output_path']}")
```

### JavaScript Client

```javascript
class AISystemClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async generateImage(prompt, options = {}) {
        const url = `${this.baseUrl}/generate/image`;
        const data = {
            prompt,
            width: options.width || 512,
            height: options.height || 512,
            steps: options.steps || 20,
            guidance_scale: options.guidance_scale || 7.5
        };
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async generateVideo(prompt, options = {}) {
        const url = `${this.baseUrl}/generate/video`;
        const data = {
            prompt,
            width: options.width || 512,
            height: options.height || 512,
            frames: options.frames || 32,
            fps: options.fps || 20
        };
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getStatus() {
        const url = `${this.baseUrl}/status`;
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
}

// Usage example
const client = new AISystemClient();

// Generate an image
client.generateImage('A serene lake at sunset')
    .then(result => {
        console.log('Image generated:', result.output_path);
    })
    .catch(error => {
        console.error('Error:', error);
    });

// Generate a video
client.generateVideo('A flower blooming in time-lapse', { frames: 64, fps: 30 })
    .then(result => {
        console.log('Video generated:', result.output_path);
    })
    .catch(error => {
        console.error('Error:', error);
    });
```

## Testing the API

### Using curl

```bash
# Get system status
curl -X GET "http://localhost:8000/status"

# Generate an image
curl -X POST "http://localhost:8000/generate/image" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "width": 512,
    "height": 512,
    "steps": 20
  }'

# Generate a video
curl -X POST "http://localhost:8000/generate/video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat walking in a garden",
    "width": 512,
    "height": 512,
    "frames": 32,
    "fps": 20
  }'
```

### Using Postman

1. Import the collection from `docs/postman_collection.json`
2. Set the base URL to `http://localhost:8000`
3. Test the endpoints with various parameters

## Performance Considerations

- **Image Generation**: Typically 10-60 seconds depending on parameters
- **Video Generation**: 30-300 seconds depending on length and quality
- **API Response Time**: <100ms for status endpoints, varies for generation
- **Concurrent Requests**: Limited by GPU memory and system resources
- **File Downloads**: Generated files are available for download via the output path

## Security Notes

- The API currently operates without authentication
- Input validation is performed on all endpoints
- Resource limits prevent abuse
- Generated content is logged for compliance purposes
- Consider implementing authentication for production use 