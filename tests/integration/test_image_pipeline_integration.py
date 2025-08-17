"""
Integration tests for ImageGenerationPipeline with hardware detection and memory management.

Tests the integration between the image generation pipeline and other system components
like hardware detection, memory management, and profile management.
"""

import pytest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.core.interfaces import (
    HardwareConfig, GenerationRequest, OutputType, 
    StyleConfig, ComplianceMode, ConversationContext
)
from src.pipelines.image_generation import ImageGenerationPipeline, ImageModel
from src.hardware.detector import HardwareDetector
from src.hardware.memory_manager import MemoryManager
from src.hardware.profiles import HardwareProfileManager


class TestImagePipelineIntegration:
    """Integration tests for ImageGenerationPipeline."""
    
    @pytest.fixture
    def hardware_detector(self):
        """Hardware detector instance."""
        return HardwareDetector()
    
    @pytest.fixture
    def profile_manager(self):
        """Hardware profile manager instance."""
        return HardwareProfileManager()
    
    @pytest.fixture
    def pipeline(self):
        """ImageGenerationPipeline instance."""
        return ImageGenerationPipeline()
    
    @pytest.fixture
    def mock_hardware_config(self):
        """Mock hardware configuration."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    def test_pipeline_with_hardware_detection(self, pipeline, hardware_detector):
        """Test pipeline initialization with real hardware detection."""
        # Mock the hardware detection to return known values
        with patch.object(hardware_detector, 'detect_hardware') as mock_detect:
            mock_detect.return_value = HardwareConfig(
                vram_size=8192,
                gpu_model="RTX 3070",
                cpu_cores=8,
                ram_size=32768,
                cuda_available=True,
                optimization_level="balanced"
            )
            
            # Detect hardware
            hardware_config = hardware_detector.detect_hardware()
            
            # Mock dependencies for pipeline initialization
            with patch.object(pipeline, '_check_dependencies', return_value=True):
                with patch.object(pipeline, '_load_model', return_value=True):
                    
                    # Initialize pipeline with detected hardware
                    result = pipeline.initialize(hardware_config)
                    
                    assert result is True
                    assert pipeline.hardware_config == hardware_config
                    assert pipeline.memory_manager is not None
    
    def test_pipeline_with_profile_manager(self, pipeline, profile_manager, mock_hardware_config):
        """Test pipeline integration with hardware profile manager."""
        # Initialize pipeline
        pipeline.hardware_config = mock_hardware_config
        pipeline.profile_manager = profile_manager
        
        # Test model recommendations
        recommended_models = profile_manager.get_model_recommendations(mock_hardware_config)
        available_models = pipeline.get_available_models()
        
        # Verify recommended models are available
        for model in recommended_models:
            if model in pipeline.model_configs:
                assert model in available_models
        
        # Test optimization settings
        optimization_settings = profile_manager.get_optimization_settings(mock_hardware_config)
        assert isinstance(optimization_settings, dict)
        assert 'max_resolution' in optimization_settings
        assert 'xformers' in optimization_settings
    
    def test_pipeline_with_memory_manager(self, pipeline, mock_hardware_config):
        """Test pipeline integration with memory manager."""
        # Initialize memory manager
        memory_manager = MemoryManager(mock_hardware_config)
        pipeline.memory_manager = memory_manager
        pipeline.hardware_config = mock_hardware_config
        
        # Test model loading optimization
        model_name = ImageModel.STABLE_DIFFUSION_V1_5.value
        optimization_params = memory_manager.optimize_model_loading(model_name)
        
        assert isinstance(optimization_params, dict)
        assert 'torch_dtype' in optimization_params
        
        # Test memory status
        memory_status = memory_manager.get_memory_status()
        assert isinstance(memory_status, dict)
        assert 'strategy' in memory_status
        assert 'hardware' in memory_status
    
    def test_end_to_end_pipeline_flow(self, pipeline, mock_hardware_config):
        """Test complete pipeline flow from initialization to generation."""
        # Create generation request
        request = GenerationRequest(
            prompt="A beautiful landscape with mountains",
            output_type=OutputType.IMAGE,
            style_config=StyleConfig(
                generation_params={
                    'width': 512,
                    'height': 512,
                    'num_inference_steps': 20
                }
            ),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=mock_hardware_config,
            context=ConversationContext(
                conversation_id="test",
                history=[],
                current_mode=ComplianceMode.RESEARCH_SAFE,
                user_preferences={}
            )
        )
        
        # Mock model loading with side effect to set current_model
        def mock_load_model(model_name):
            pipeline.current_model = model_name
            pipeline.current_pipeline = Mock()
            return True
        
        # Mock all external dependencies
        with patch.object(pipeline, '_check_dependencies', return_value=True):
            with patch.object(pipeline, '_load_model', side_effect=mock_load_model):
                with patch.object(pipeline, '_generate_image') as mock_generate:
                    with patch.object(pipeline, '_save_image') as mock_save:
                        
                        # Mock image generation
                        mock_image = Mock()
                        mock_image.size = (512, 512)
                        mock_generate.return_value = mock_image
                        
                        # Mock image saving
                        output_path = Path("outputs/images/test_image.png")
                        mock_save.return_value = output_path
                        
                        # Initialize pipeline
                        init_result = pipeline.initialize(mock_hardware_config)
                        assert init_result is True
                        
                        # Generate image
                        result = pipeline.generate(request)
                        
                        # Verify result
                        assert result.success is True
                        assert result.output_path == output_path
                        assert result.generation_time > 0
                        assert result.model_used is not None
                        assert result.quality_metrics is not None
                        assert result.compliance_info is not None
    
    def test_model_switching_with_memory_management(self, pipeline, mock_hardware_config):
        """Test model switching with memory management integration."""
        # Initialize pipeline
        pipeline.hardware_config = mock_hardware_config
        pipeline.memory_manager = MemoryManager(mock_hardware_config)
        
        # Mock model loading with side effect to set current_model
        def mock_load_model(model_name):
            pipeline.current_model = model_name
            pipeline.current_pipeline = Mock()
            return True
        
        with patch.object(pipeline, '_load_model', side_effect=mock_load_model):
            
            # Load first model
            result1 = pipeline.switch_model(ImageModel.STABLE_DIFFUSION_V1_5.value)
            assert result1 is True
            assert pipeline.current_model == ImageModel.STABLE_DIFFUSION_V1_5.value
            
            # Switch to second model
            result2 = pipeline.switch_model(ImageModel.SDXL_TURBO.value)
            assert result2 is True
            assert pipeline.current_model == ImageModel.SDXL_TURBO.value
    
    def test_hardware_optimization_integration(self, pipeline, profile_manager):
        """Test hardware optimization integration across different configurations."""
        # Test different hardware configurations
        configs = [
            HardwareConfig(4096, "GTX 1650", 4, 16384, True, "aggressive"),
            HardwareConfig(8192, "RTX 3070", 8, 32768, True, "balanced"),
            HardwareConfig(24576, "RTX 4090", 16, 65536, True, "minimal")
        ]
        
        for config in configs:
            # Get profile for configuration
            profile = profile_manager.get_profile(config)
            
            # Verify profile matches hardware
            assert profile.min_vram_mb <= config.vram_size <= profile.max_vram_mb
            assert profile.optimization_level == config.optimization_level
            
            # Test pipeline optimization
            pipeline.hardware_config = config
            pipeline.optimize_for_hardware(config)
            
            # Verify available models match hardware capabilities
            available_models = pipeline.get_available_models()
            for model_name in available_models:
                model_config = pipeline.model_configs[model_name]
                assert model_config.min_vram_mb <= config.vram_size
    
    def test_compliance_mode_integration(self, pipeline, mock_hardware_config):
        """Test compliance mode integration with generation."""
        compliance_modes = [
            ComplianceMode.OPEN_SOURCE_ONLY,
            ComplianceMode.RESEARCH_SAFE,
            ComplianceMode.FULL_DATASET
        ]
        
        for mode in compliance_modes:
            request = GenerationRequest(
                prompt="Test prompt",
                output_type=OutputType.IMAGE,
                style_config=StyleConfig(),
                compliance_mode=mode,
                hardware_constraints=mock_hardware_config,
                context=ConversationContext(
                    conversation_id="test",
                    history=[],
                    current_mode=mode,
                    user_preferences={}
                )
            )
            
            # Test compliance info generation
            pipeline.current_model = ImageModel.STABLE_DIFFUSION_V1_5.value
            compliance_info = pipeline._get_compliance_info(request)
            
            assert compliance_info['compliance_mode'] == mode.value
            assert 'model_used' in compliance_info
            assert 'training_data_license' in compliance_info
    
    def test_error_handling_integration(self, pipeline, mock_hardware_config):
        """Test error handling across integrated components."""
        # Test initialization failure
        with patch.object(pipeline, '_check_dependencies', return_value=False):
            result = pipeline.initialize(mock_hardware_config)
            assert result is False
        
        # Test generation with uninitialized pipeline
        request = GenerationRequest(
            prompt="Test",
            output_type=OutputType.IMAGE,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=mock_hardware_config,
            context=ConversationContext("test", [], ComplianceMode.RESEARCH_SAFE, {})
        )
        
        result = pipeline.generate(request)
        assert result.success is False
        assert "not initialized" in result.error_message
    
    def test_performance_targets_integration(self, pipeline, profile_manager):
        """Test performance targets integration with hardware profiles."""
        configs = [
            HardwareConfig(4096, "GTX 1650", 4, 16384, True, "aggressive"),
            HardwareConfig(8192, "RTX 3070", 8, 32768, True, "balanced"),
            HardwareConfig(24576, "RTX 4090", 16, 65536, True, "minimal")
        ]
        
        for config in configs:
            # Get performance targets
            targets = profile_manager.get_performance_targets(config)
            
            # Verify targets are reasonable
            assert targets['image_generation_time_s'] > 0
            assert targets['model_switch_time_s'] > 0
            assert targets['max_vram_usage_percent'] > 0
            assert targets['max_vram_usage_percent'] <= 100
            
            # Higher VRAM should have better performance targets
            if config.vram_size >= 20000:  # High-end
                assert targets['image_generation_time_s'] <= 15
            elif config.vram_size >= 8000:  # Mid-range
                assert targets['image_generation_time_s'] <= 30
            else:  # Low-end
                assert targets['image_generation_time_s'] <= 90


if __name__ == '__main__':
    pytest.main([__file__])