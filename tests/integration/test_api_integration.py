"""
Integration tests for API endpoints.

Tests the full integration of API endpoints with actual pipeline components
and system initialization.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from pathlib import Path

from src.api.server import app, api_state
from src.core.interfaces import HardwareConfig, ComplianceMode


@pytest.fixture(scope="module")
def test_hardware_config():
    """Create test hardware configuration."""
    return HardwareConfig(
        gpu_model="Test GPU GTX 1650",
        vram_size=4000,  # 4GB VRAM for testing
        cuda_available=False,  # Use CPU for testing
        cpu_cores=4,
        ram_size=8000,
        optimization_level="balanced"
    )


@pytest.fixture(scope="module")
def integration_client():
    """Create test client with actual initialization."""
    return TestClient(app)


class TestAPIInitialization:
    """Test API server initialization."""
    
    @patch('src.hardware.detector.HardwareDetector')
    @patch('src.pipelines.image_generation.ImageGenerationPipeline')
    @patch('src.pipelines.video_generation.VideoGenerationPipeline')
    @patch('src.core.llm_controller.LLMController')
    @patch('src.data.experiment_tracker.ExperimentTracker')
    def test_api_initialization_success(
        self, 
        mock_tracker, 
        mock_controller, 
        mock_video_pipeline, 
        mock_image_pipeline, 
        mock_detector,
        test_hardware_config
    ):
        """Test successful API initialization with mocked components."""
        
        # Mock hardware detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect_hardware.return_value = test_hardware_config
        mock_detector.return_value = mock_detector_instance
        
        # Mock pipelines
        mock_image_instance = Mock()
        mock_image_instance.initialize.return_value = True
        mock_image_pipeline.return_value = mock_image_instance
        
        mock_video_instance = Mock()
        mock_video_instance.initialize.return_value = True
        mock_video_pipeline.return_value = mock_video_instance
        
        # Mock controller
        mock_controller_instance = Mock()
        mock_controller.return_value = mock_controller_instance
        
        # Mock experiment tracker
        mock_tracker_instance = Mock()
        mock_tracker.return_value = mock_tracker_instance
        
        # Reset API state to force re-initialization
        from src.api.dependencies import api_state
        api_state.is_initialized = False
        
        # Test initialization
        client = TestClient(app)
        
        # Make a request to trigger initialization (use an endpoint that requires state)
        response = client.get("/models/status")
        
        # The response might fail due to mocking, but initialization should be triggered
        # We're mainly testing that the initialization process works
        
        # Verify initialization was called
        mock_detector.assert_called_once()
        mock_image_pipeline.assert_called_once()
        mock_video_pipeline.assert_called_once()
        mock_controller.assert_called_once()


class TestEndToEndImageGeneration:
    """Test end-to-end image generation workflow."""
    
    @patch('src.api.server.get_api_state')
    def test_complete_image_generation_workflow(self, mock_get_state, integration_client):
        """Test complete image generation from request to result."""
        
        # Mock API state with realistic behavior
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="Test GPU",
            vram_size=8000,
            cuda_available=False,
            cpu_cores=4,
            ram_size=8000,
            optimization_level="balanced"
        )
        
        # Mock LLM controller
        mock_context = Mock()
        mock_context.conversation_id = "test_conv"
        mock_context.current_mode = ComplianceMode.RESEARCH_SAFE
        mock_context.history = []
        mock_context.user_preferences = {}
        
        mock_state.llm_controller = Mock()
        mock_state.llm_controller.manage_context.return_value = mock_context
        
        # Mock image pipeline with realistic generation result
        mock_state.image_pipeline = Mock()
        mock_state.image_pipeline.current_model = "stable-diffusion-v1-5"
        
        # Create a mock generation result
        from src.core.interfaces import GenerationResult
        mock_result = GenerationResult(
            success=True,
            output_path=Path("test_outputs/test_image.png"),
            generation_time=25.5,
            model_used="stable-diffusion-v1-5",
            quality_metrics={
                "resolution": 512 * 512,
                "aspect_ratio": 1.0,
                "inference_steps": 20
            },
            compliance_info={
                "compliance_mode": "research_safe",
                "model_used": "stable-diffusion-v1-5",
                "training_data_license": "CreativeML Open RAIL-M"
            }
        )
        
        mock_state.image_pipeline.generate.return_value = mock_result
        mock_get_state.return_value = mock_state
        
        # Step 1: Submit generation request
        request_data = {
            "prompt": "A serene mountain landscape with a lake",
            "width": 512,
            "height": 512,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "compliance_mode": "research_safe"
        }
        
        response = integration_client.post("/generate/image", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        task_id = data["task_id"]
        
        assert data["status"] == "queued"
        assert "estimated_time" in data
        
        # Step 2: Check task status (simulate processing time)
        time.sleep(0.1)  # Brief pause to simulate processing
        
        # Mock the task in active_tasks for status check
        api_state.active_tasks[task_id] = {
            'status': 'completed',
            'created_at': time.time(),
            'type': 'image',
            'result': {
                'success': True,
                'output_path': str(mock_result.output_path),
                'generation_time': mock_result.generation_time,
                'model_used': mock_result.model_used,
                'quality_metrics': mock_result.quality_metrics
            }
        }
        
        status_response = integration_client.get(f"/tasks/{task_id}")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        
        assert status_data["task_id"] == task_id
        assert status_data["status"] == "completed"
        assert status_data["type"] == "image"
        assert "result" in status_data
        
        # Verify generation was called with correct parameters
        mock_state.image_pipeline.generate.assert_called_once()
        call_args = mock_state.image_pipeline.generate.call_args[0][0]
        assert call_args.prompt == request_data["prompt"]
        assert call_args.compliance_mode == ComplianceMode.RESEARCH_SAFE


class TestEndToEndVideoGeneration:
    """Test end-to-end video generation workflow."""
    
    @patch('src.api.server.get_api_state')
    def test_complete_video_generation_workflow(self, mock_get_state, integration_client):
        """Test complete video generation from request to result."""
        
        # Mock API state
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="Test GPU RTX 3070",
            vram_size=8000,
            cuda_available=False,
            cpu_cores=8,
            ram_size=16000,
            optimization_level="balanced"
        )
        
        # Mock LLM controller
        mock_context = Mock()
        mock_context.conversation_id = "test_conv_video"
        mock_context.current_mode = ComplianceMode.RESEARCH_SAFE
        
        mock_state.llm_controller = Mock()
        mock_state.llm_controller.manage_context.return_value = mock_context
        
        # Mock video pipeline
        mock_state.video_pipeline = Mock()
        mock_state.video_pipeline.current_model = "stable-video-diffusion"
        
        from src.core.interfaces import GenerationResult
        mock_result = GenerationResult(
            success=True,
            output_path=Path("test_outputs/test_video.mp4"),
            generation_time=180.0,
            model_used="stable-video-diffusion",
            quality_metrics={
                "frames": 14,
                "fps": 7,
                "duration": 2.0,
                "resolution": 512 * 512
            }
        )
        
        mock_state.video_pipeline.generate.return_value = mock_result
        mock_get_state.return_value = mock_state
        
        # Submit video generation request
        request_data = {
            "prompt": "Ocean waves crashing on a beach",
            "width": 512,
            "height": 512,
            "num_frames": 14,
            "fps": 7,
            "num_inference_steps": 25,
            "compliance_mode": "research_safe"
        }
        
        response = integration_client.post("/generate/video", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "queued"
        assert "task_id" in data
        assert "estimated_time" in data
        
        # Verify video generation was set up correctly
        mock_state.llm_controller.manage_context.assert_called_once()


class TestModelManagementIntegration:
    """Test model management integration."""
    
    @patch('src.api.server.get_api_state')
    def test_model_status_integration(self, mock_get_state, integration_client):
        """Test model status endpoint integration."""
        
        # Mock comprehensive API state
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="RTX 4090",
            vram_size=24000,
            cuda_available=True,
            cpu_cores=16,
            ram_size=32000,
            optimization_level="aggressive"
        )
        
        # Mock image pipeline
        mock_state.image_pipeline = Mock()
        mock_state.image_pipeline.current_model = "flux.1-schnell"
        mock_state.image_pipeline.get_available_models.return_value = [
            "stable-diffusion-v1-5",
            "sdxl-turbo", 
            "flux.1-schnell"
        ]
        
        # Mock video pipeline
        mock_state.video_pipeline = Mock()
        mock_state.video_pipeline.current_model = "i2vgen-xl"
        mock_state.video_pipeline.get_available_models.return_value = [
            "stable-video-diffusion",
            "animatediff",
            "i2vgen-xl"
        ]
        
        mock_get_state.return_value = mock_state
        
        response = integration_client.get("/models/status")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "vram_info" in data
        assert "current_models" in data
        assert "available_models" in data
        assert "hardware_info" in data
        assert "system_metrics" in data
        
        # Verify current models
        assert data["current_models"]["image"] == "flux.1-schnell"
        assert data["current_models"]["video"] == "i2vgen-xl"
        
        # Verify available models
        assert "stable-diffusion-v1-5" in data["available_models"]["image"]
        assert "stable-video-diffusion" in data["available_models"]["video"]
        
        # Verify hardware info
        hardware_info = data["hardware_info"]
        assert hardware_info["gpu_model"] == "RTX 4090"
        assert hardware_info["vram_size_mb"] == 24000
    
    @patch('src.api.server.get_api_state')
    def test_model_switching_integration(self, mock_get_state, integration_client):
        """Test model switching integration."""
        
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="RTX 3070",
            vram_size=8000,
            cuda_available=True,
            cpu_cores=8,
            ram_size=16000,
            optimization_level="balanced"
        )
        
        # Mock image pipeline for switching
        mock_state.image_pipeline = Mock()
        mock_state.image_pipeline.get_available_models.return_value = [
            "stable-diffusion-v1-5", "sdxl-turbo"
        ]
        mock_state.image_pipeline.switch_model.return_value = True
        mock_state.image_pipeline.get_model_info.return_value = {
            "min_vram_mb": 7000,
            "recommended_vram_mb": 8000,
            "max_resolution": 1024,
            "supports_negative_prompt": False,
            "supports_guidance_scale": False
        }
        
        mock_get_state.return_value = mock_state
        
        # Test model switch
        switch_request = {
            "model_name": "sdxl-turbo",
            "pipeline_type": "image"
        }
        
        response = integration_client.post("/models/switch", json=switch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_name"] == "sdxl-turbo"
        assert data["pipeline_type"] == "image"
        assert "switch_time_seconds" in data
        assert "vram_usage" in data
        
        # Verify switch was called
        mock_state.image_pipeline.switch_model.assert_called_once_with("sdxl-turbo")


class TestExperimentTrackingIntegration:
    """Test experiment tracking integration."""
    
    @patch('src.api.server.get_api_state')
    def test_experiment_save_integration(self, mock_get_state, integration_client):
        """Test experiment saving integration."""
        
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="RTX 4090",
            vram_size=24000,
            cuda_available=True,
            cpu_cores=16,
            ram_size=32000,
            optimization_level="aggressive"
        )
        
        # Mock pipelines for experiment context
        mock_state.image_pipeline = Mock()
        mock_state.image_pipeline.current_model = "flux.1-schnell"
        mock_state.video_pipeline = Mock()
        mock_state.video_pipeline.current_model = None
        
        # Mock experiment tracker
        mock_state.experiment_tracker = Mock()
        mock_state.experiment_tracker.save_experiment.return_value = "exp_20240101_123456"
        
        mock_get_state.return_value = mock_state
        
        # Test experiment save
        experiment_data = {
            "experiment_name": "High-Quality Landscape Generation",
            "description": "Testing FLUX.1-schnell for landscape generation",
            "tags": ["landscape", "flux", "high-quality"],
            "results": {
                "generation_time": 15.2,
                "quality_score": 0.92,
                "user_satisfaction": 4.5,
                "model_used": "flux.1-schnell",
                "prompt": "A serene mountain landscape",
                "parameters": {
                    "width": 1024,
                    "height": 1024,
                    "steps": 4
                }
            },
            "compliance_mode": "research_safe"
        }
        
        response = integration_client.post("/models/experiment/save", json=experiment_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["experiment_id"] == "exp_20240101_123456"
        assert data["status"] == "saved"
        assert "successfully" in data["message"]
        
        # Verify experiment tracker was called
        mock_state.experiment_tracker.save_experiment.assert_called_once()
        
        # Verify the saved data structure
        saved_data = mock_state.experiment_tracker.save_experiment.call_args[0][0]
        assert saved_data["name"] == experiment_data["experiment_name"]
        assert saved_data["compliance_mode"] == "research_safe"
        assert "hardware_info" in saved_data
        assert "system_state" in saved_data


class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios."""
    
    @patch('src.api.server.get_api_state')
    def test_pipeline_failure_handling(self, mock_get_state, integration_client):
        """Test handling of pipeline failures."""
        
        mock_state = Mock()
        mock_state.is_initialized = True
        mock_state.hardware_config = HardwareConfig(
            gpu_model="Test GPU",
            vram_size=4000,
            cuda_available=False,
            cpu_cores=4,
            ram_size=8000,
            optimization_level="minimal"
        )
        
        # Mock LLM controller
        mock_context = Mock()
        mock_context.current_mode = ComplianceMode.RESEARCH_SAFE
        mock_state.llm_controller = Mock()
        mock_state.llm_controller.manage_context.return_value = mock_context
        
        # Mock image pipeline to fail
        mock_state.image_pipeline = Mock()
        from src.core.interfaces import GenerationResult
        mock_state.image_pipeline.generate.return_value = GenerationResult(
            success=False,
            output_path=None,
            generation_time=5.0,
            model_used="stable-diffusion-v1-5",
            error_message="CUDA out of memory"
        )
        
        mock_get_state.return_value = mock_state
        
        # Submit request that will fail
        request_data = {
            "prompt": "A complex scene",
            "compliance_mode": "research_safe"
        }
        
        response = integration_client.post("/generate/image", json=request_data)
        
        # Should still accept the request
        assert response.status_code == 200
        
        # The failure would be reflected in task status
        # (This would require more complex background task testing)
    
    def test_uninitialized_api_handling(self, integration_client):
        """Test handling when API is not properly initialized."""
        
        # Reset API state
        original_state = api_state.is_initialized
        api_state.is_initialized = False
        
        try:
            # This should trigger initialization
            response = integration_client.get("/health")
            assert response.status_code == 200
        finally:
            # Restore state
            api_state.is_initialized = original_state


if __name__ == "__main__":
    pytest.main([__file__])