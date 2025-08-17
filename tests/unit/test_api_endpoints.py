"""
Unit tests for API endpoints.

Tests the basic functionality of the FastAPI server endpoints including
image/video generation, model management, and experiment tracking.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

from src.api.server import app, api_state
from src.core.interfaces import HardwareConfig, GenerationResult, ComplianceMode


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_hardware_config():
    """Mock hardware configuration."""
    return HardwareConfig(
        gpu_model="Test GPU",
        vram_size=8000,
        cuda_available=True,
        cpu_cores=8,
        ram_size=16000,
        optimization_level="balanced"
    )


@pytest.fixture
def mock_api_state(mock_hardware_config):
    """Mock API state with initialized components."""
    state = Mock()
    state.is_initialized = True
    state.hardware_config = mock_hardware_config
    
    # Mock LLM controller
    state.llm_controller = Mock()
    state.llm_controller.manage_context.return_value = Mock(
        conversation_id="test_conversation",
        current_mode=ComplianceMode.RESEARCH_SAFE,
        history=[],
        user_preferences={}
    )
    
    # Mock image pipeline
    state.image_pipeline = Mock()
    state.image_pipeline.current_model = "stable-diffusion-v1-5"
    state.image_pipeline.get_available_models.return_value = [
        "stable-diffusion-v1-5", "sdxl-turbo"
    ]
    state.image_pipeline.generate.return_value = GenerationResult(
        success=True,
        output_path="test_output.png",
        generation_time=30.0,
        model_used="stable-diffusion-v1-5",
        quality_metrics={"resolution": 512*512}
    )
    
    # Mock video pipeline
    state.video_pipeline = Mock()
    state.video_pipeline.current_model = "stable-video-diffusion"
    state.video_pipeline.get_available_models.return_value = [
        "stable-video-diffusion", "animatediff"
    ]
    state.video_pipeline.generate.return_value = GenerationResult(
        success=True,
        output_path="test_output.mp4",
        generation_time=300.0,
        model_used="stable-video-diffusion",
        quality_metrics={"frames": 14, "fps": 7}
    )
    
    # Mock experiment tracker
    state.experiment_tracker = Mock()
    state.experiment_tracker.save_experiment.return_value = "exp_123456"
    
    return state


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check returns correct status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"


class TestImageGenerationEndpoint:
    """Test image generation endpoint."""
    
    @patch('src.api.server.get_api_state')
    def test_generate_image_success(self, mock_get_state, client, mock_api_state):
        """Test successful image generation request."""
        mock_get_state.return_value = mock_api_state
        
        request_data = {
            "prompt": "A beautiful landscape",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["message"] == "Image generation task started"
        assert "estimated_time" in data
    
    def test_generate_image_invalid_prompt(self, client):
        """Test image generation with invalid prompt."""
        request_data = {
            "prompt": "",  # Empty prompt should fail validation
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_image_invalid_compliance_mode(self, client):
        """Test image generation with invalid compliance mode."""
        request_data = {
            "prompt": "A beautiful landscape",
            "compliance_mode": "invalid_mode"
        }
        
        response = client.post("/generate/image", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_image_invalid_dimensions(self, client):
        """Test image generation with invalid dimensions."""
        request_data = {
            "prompt": "A beautiful landscape",
            "width": 100,  # Too small
            "height": 3000,  # Too large
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestVideoGenerationEndpoint:
    """Test video generation endpoint."""
    
    @patch('src.api.server.get_api_state')
    def test_generate_video_success(self, mock_get_state, client, mock_api_state):
        """Test successful video generation request."""
        mock_get_state.return_value = mock_api_state
        
        request_data = {
            "prompt": "A flowing river",
            "width": 512,
            "height": 512,
            "num_frames": 14,
            "fps": 7,
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/video", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["message"] == "Video generation task started"
        assert "estimated_time" in data
    
    def test_generate_video_invalid_frames(self, client):
        """Test video generation with invalid frame count."""
        request_data = {
            "prompt": "A flowing river",
            "num_frames": 100,  # Too many frames
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/video", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestTaskStatusEndpoint:
    """Test task status endpoint."""
    
    def test_get_task_status_not_found(self, client):
        """Test getting status of non-existent task."""
        response = client.get("/tasks/nonexistent_task_id")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]
    
    @patch('src.api.server.api_state')
    def test_get_task_status_success(self, mock_state, client):
        """Test getting status of existing task."""
        task_id = "test_task_123"
        mock_state.active_tasks = {
            task_id: {
                'status': 'processing',
                'created_at': datetime.now(),
                'type': 'image',
                'progress': 50.0
            }
        }
        
        response = client.get(f"/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "processing"
        assert data["type"] == "image"


class TestModelManagementEndpoints:
    """Test model management endpoints."""
    
    @patch('src.api.server.get_api_state')
    def test_get_model_status(self, mock_get_state, client, mock_api_state):
        """Test getting model status."""
        mock_get_state.return_value = mock_api_state
        
        response = client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "vram_info" in data
        assert "current_models" in data
        assert "available_models" in data
        assert "hardware_info" in data
        assert "system_metrics" in data
    
    @patch('src.api.server.get_api_state')
    def test_switch_model_success(self, mock_get_state, client, mock_api_state):
        """Test successful model switching."""
        mock_get_state.return_value = mock_api_state
        mock_api_state.image_pipeline.switch_model.return_value = True
        mock_api_state.image_pipeline.get_model_info.return_value = {
            "min_vram_mb": 4000,
            "max_resolution": 512
        }
        
        request_data = {
            "model_name": "sdxl-turbo",
            "pipeline_type": "image"
        }
        
        response = client.post("/models/switch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "sdxl-turbo"
        assert data["pipeline_type"] == "image"
    
    @patch('src.api.server.get_api_state')
    def test_switch_model_invalid_pipeline(self, mock_get_state, client, mock_api_state):
        """Test model switching with invalid pipeline type."""
        mock_get_state.return_value = mock_api_state
        
        request_data = {
            "model_name": "test-model",
            "pipeline_type": "invalid"
        }
        
        response = client.post("/models/switch", json=request_data)
        
        assert response.status_code == 400
        assert "pipeline_type must be" in response.json()["detail"]
    
    @patch('src.api.server.get_api_state')
    def test_list_models(self, mock_get_state, client, mock_api_state):
        """Test listing available models."""
        mock_get_state.return_value = mock_api_state
        
        # Mock model info for each available model
        def mock_get_model_info(model_name):
            return {
                "min_vram_mb": 4000,
                "supports_negative_prompt": True,
                "supports_guidance_scale": True
            }
        
        mock_api_state.image_pipeline.get_model_info.side_effect = mock_get_model_info
        
        response = client.get("/models/list/image")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check first model structure
        model = data[0]
        assert "name" in model
        assert "pipeline_type" in model
        assert "is_loaded" in model
        assert "vram_requirement_mb" in model
        assert "supports_features" in model


class TestExperimentEndpoints:
    """Test experiment tracking endpoints."""
    
    @patch('src.api.server.get_api_state')
    def test_save_experiment_success(self, mock_get_state, client, mock_api_state):
        """Test successful experiment saving."""
        mock_get_state.return_value = mock_api_state
        
        request_data = {
            "experiment_name": "Test Experiment",
            "description": "A test experiment",
            "tags": ["test", "image"],
            "results": {
                "generation_time": 30.0,
                "quality_score": 0.85
            },
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/models/experiment/save", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["experiment_id"].startswith("exp_")
        assert data["status"] == "saved"
    
    def test_save_experiment_invalid_compliance(self, client):
        """Test experiment saving with invalid compliance mode."""
        request_data = {
            "experiment_name": "Test Experiment",
            "results": {"test": "data"},
            "compliance_mode": "invalid_mode"
        }
        
        response = client.post("/models/experiment/save", json=request_data)
        
        assert response.status_code == 400
        assert "Invalid compliance_mode" in response.json()["detail"]


class TestParameterValidation:
    """Test parameter validation across endpoints."""
    
    def test_image_generation_parameter_bounds(self, client):
        """Test image generation parameter boundary validation."""
        # Test minimum values
        request_data = {
            "prompt": "test",
            "width": 255,  # Below minimum
            "height": 255,  # Below minimum
            "num_inference_steps": 0,  # Below minimum
            "guidance_scale": 0.5,  # Below minimum
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        assert response.status_code == 422
        
        # Test maximum values
        request_data = {
            "prompt": "test",
            "width": 3000,  # Above maximum
            "height": 3000,  # Above maximum
            "num_inference_steps": 200,  # Above maximum
            "guidance_scale": 25.0,  # Above maximum
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        assert response.status_code == 422
    
    def test_video_generation_parameter_bounds(self, client):
        """Test video generation parameter boundary validation."""
        # Test invalid frame count
        request_data = {
            "prompt": "test",
            "num_frames": 50,  # Above maximum
            "fps": 100,  # Above maximum
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/video", json=request_data)
        assert response.status_code == 422


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations and background tasks."""
    
    @patch('src.api.server.get_api_state')
    async def test_background_task_execution(self, mock_get_state, mock_api_state):
        """Test that background tasks are properly executed."""
        mock_get_state.return_value = mock_api_state
        
        # This would require more complex mocking of the background task system
        # For now, we test that the endpoint accepts the request
        client = TestClient(app)
        
        request_data = {
            "prompt": "A test image",
            "compliance_mode": "research_safe"
        }
        
        response = client.post("/generate/image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"


if __name__ == "__main__":
    pytest.main([__file__])