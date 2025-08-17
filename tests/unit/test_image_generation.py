"""
Unit tests for the ImageGenerationPipeline class.

Tests model loading, hardware optimization, and basic generation functionality
without requiring actual model downloads or GPU hardware.
"""

import pytest
import unittest.mock as mock
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from src.core.interfaces import (
    HardwareConfig, GenerationRequest, OutputType, 
    StyleConfig, ComplianceMode, ConversationContext
)
from src.pipelines.image_generation import (
    ImageGenerationPipeline, ImageModel, ModelConfig, GenerationParams
)


class TestImageGenerationPipeline:
    """Test cases for ImageGenerationPipeline."""
    
    @pytest.fixture
    def hardware_config_4gb(self):
        """Hardware configuration for 4GB VRAM GPU."""
        return HardwareConfig(
            vram_size=4096,
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=16384,
            cuda_available=True,
            optimization_level="aggressive"
        )
    
    @pytest.fixture
    def hardware_config_8gb(self):
        """Hardware configuration for 8GB VRAM GPU."""
        return HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
    
    @pytest.fixture
    def hardware_config_24gb(self):
        """Hardware configuration for 24GB VRAM GPU."""
        return HardwareConfig(
            vram_size=24576,
            gpu_model="RTX 4090",
            cpu_cores=16,
            ram_size=65536,
            cuda_available=True,
            optimization_level="minimal"
        )
    
    @pytest.fixture
    def basic_request(self, hardware_config_8gb):
        """Basic generation request."""
        return GenerationRequest(
            prompt="A beautiful landscape",
            output_type=OutputType.IMAGE,
            style_config=StyleConfig(),
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config_8gb,
            context=ConversationContext(
                conversation_id="test",
                history=[],
                current_mode=ComplianceMode.RESEARCH_SAFE,
                user_preferences={}
            )
        )
    
    @pytest.fixture
    def pipeline(self):
        """ImageGenerationPipeline instance."""
        return ImageGenerationPipeline()
    
    def test_pipeline_creation(self, pipeline):
        """Test pipeline creation and initial state."""
        assert pipeline is not None
        assert not pipeline.is_initialized
        assert pipeline.current_model is None
        assert pipeline.current_pipeline is None
        assert len(pipeline.model_configs) == 3  # SD 1.5, SDXL-Turbo, FLUX
    
    def test_model_configs_initialization(self, pipeline):
        """Test model configurations are properly initialized."""
        configs = pipeline.model_configs
        
        # Check all expected models are present
        expected_models = [
            ImageModel.STABLE_DIFFUSION_V1_5.value,
            ImageModel.SDXL_TURBO.value,
            ImageModel.FLUX_SCHNELL.value
        ]
        
        for model in expected_models:
            assert model in configs
            config = configs[model]
            assert isinstance(config, ModelConfig)
            assert config.model_id
            assert config.pipeline_class
            assert config.min_vram_mb > 0
            assert config.recommended_vram_mb >= config.min_vram_mb
            assert config.max_resolution > 0
            assert config.default_steps > 0
    
    def test_get_available_models_no_hardware(self, pipeline):
        """Test getting available models without hardware config."""
        models = pipeline.get_available_models()
        assert len(models) == 3
        assert ImageModel.STABLE_DIFFUSION_V1_5.value in models
        assert ImageModel.SDXL_TURBO.value in models
        assert ImageModel.FLUX_SCHNELL.value in models
    
    def test_get_available_models_4gb_vram(self, pipeline, hardware_config_4gb):
        """Test getting available models for 4GB VRAM."""
        pipeline.hardware_config = hardware_config_4gb
        models = pipeline.get_available_models()
        
        # Only SD 1.5 should be available for 4GB VRAM
        assert ImageModel.STABLE_DIFFUSION_V1_5.value in models
        assert ImageModel.SDXL_TURBO.value not in models
        assert ImageModel.FLUX_SCHNELL.value not in models
    
    def test_get_available_models_8gb_vram(self, pipeline, hardware_config_8gb):
        """Test getting available models for 8GB VRAM."""
        pipeline.hardware_config = hardware_config_8gb
        models = pipeline.get_available_models()
        
        # SD 1.5 and SDXL-Turbo should be available
        assert ImageModel.STABLE_DIFFUSION_V1_5.value in models
        assert ImageModel.SDXL_TURBO.value in models
        assert ImageModel.FLUX_SCHNELL.value not in models
    
    def test_get_available_models_24gb_vram(self, pipeline, hardware_config_24gb):
        """Test getting available models for 24GB VRAM."""
        pipeline.hardware_config = hardware_config_24gb
        models = pipeline.get_available_models()
        
        # All models should be available
        assert ImageModel.STABLE_DIFFUSION_V1_5.value in models
        assert ImageModel.SDXL_TURBO.value in models
        assert ImageModel.FLUX_SCHNELL.value in models
    
    def test_get_model_info(self, pipeline):
        """Test getting model information."""
        model_name = ImageModel.STABLE_DIFFUSION_V1_5.value
        info = pipeline.get_model_info(model_name)
        
        assert info is not None
        assert info['model_id'] == "runwayml/stable-diffusion-v1-5"
        assert info['min_vram_mb'] == 3500
        assert info['max_resolution'] == 768
        assert info['supports_negative_prompt'] is True
        assert info['supports_guidance_scale'] is True
    
    def test_get_model_info_unknown_model(self, pipeline):
        """Test getting info for unknown model."""
        info = pipeline.get_model_info("unknown-model")
        assert info is None
    
    @patch('src.pipelines.image_generation.TORCH_AVAILABLE', True)
    @patch('src.pipelines.image_generation.DIFFUSERS_AVAILABLE', True)
    @patch('src.pipelines.image_generation.PIL_AVAILABLE', True)
    def test_check_dependencies_available(self, pipeline):
        """Test dependency checking when all dependencies are available."""
        pipeline.hardware_config = HardwareConfig(
            vram_size=8192, gpu_model="RTX 3070", cpu_cores=8,
            ram_size=32768, cuda_available=True, optimization_level="balanced"
        )
        
        with patch('src.pipelines.image_generation.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            assert pipeline._check_dependencies() is True
    
    @patch('src.pipelines.image_generation.TORCH_AVAILABLE', False)
    def test_check_dependencies_no_torch(self, pipeline):
        """Test dependency checking when PyTorch is not available."""
        assert pipeline._check_dependencies() is False
    
    @patch('src.pipelines.image_generation.DIFFUSERS_AVAILABLE', False)
    def test_check_dependencies_no_diffusers(self, pipeline):
        """Test dependency checking when Diffusers is not available."""
        with patch('src.pipelines.image_generation.TORCH_AVAILABLE', True):
            assert pipeline._check_dependencies() is False
    
    def test_select_optimal_model_4gb(self, pipeline, hardware_config_4gb):
        """Test optimal model selection for 4GB VRAM."""
        with patch.object(pipeline.profile_manager, 'get_model_recommendations') as mock_rec:
            mock_rec.return_value = [ImageModel.STABLE_DIFFUSION_V1_5.value]
            
            optimal = pipeline._select_optimal_model(hardware_config_4gb)
            assert optimal == ImageModel.STABLE_DIFFUSION_V1_5.value
    
    def test_select_optimal_model_8gb(self, pipeline, hardware_config_8gb):
        """Test optimal model selection for 8GB VRAM."""
        with patch.object(pipeline.profile_manager, 'get_model_recommendations') as mock_rec:
            mock_rec.return_value = [ImageModel.SDXL_TURBO.value, ImageModel.STABLE_DIFFUSION_V1_5.value]
            
            optimal = pipeline._select_optimal_model(hardware_config_8gb)
            assert optimal == ImageModel.SDXL_TURBO.value
    
    def test_select_optimal_model_24gb(self, pipeline, hardware_config_24gb):
        """Test optimal model selection for 24GB VRAM."""
        with patch.object(pipeline.profile_manager, 'get_model_recommendations') as mock_rec:
            mock_rec.return_value = [ImageModel.FLUX_SCHNELL.value, ImageModel.SDXL_TURBO.value]
            
            optimal = pipeline._select_optimal_model(hardware_config_24gb)
            assert optimal == ImageModel.FLUX_SCHNELL.value
    
    def test_select_model_for_request_with_style_config(self, pipeline, basic_request):
        """Test model selection when style config specifies a model."""
        # Mock style config with model specification
        style_config = Mock()
        style_config.model_name = ImageModel.SDXL_TURBO.value
        basic_request.style_config = style_config
        
        selected = pipeline._select_model_for_request(basic_request)
        assert selected == ImageModel.SDXL_TURBO.value
    
    def test_select_model_for_request_current_model(self, pipeline, basic_request):
        """Test model selection when current model is loaded."""
        pipeline.current_model = ImageModel.STABLE_DIFFUSION_V1_5.value
        
        selected = pipeline._select_model_for_request(basic_request)
        assert selected == ImageModel.STABLE_DIFFUSION_V1_5.value
    
    def test_parse_generation_params_basic(self, pipeline, basic_request):
        """Test parsing basic generation parameters."""
        pipeline.hardware_config = basic_request.hardware_constraints
        
        with patch.object(pipeline, '_select_model_for_request') as mock_select:
            mock_select.return_value = ImageModel.STABLE_DIFFUSION_V1_5.value
            
            with patch.object(pipeline.profile_manager, 'get_optimization_settings') as mock_opt:
                mock_opt.return_value = {'max_resolution': 768}
                
                params = pipeline._parse_generation_params(basic_request)
                
                assert isinstance(params, GenerationParams)
                assert params.prompt == "A beautiful landscape"
                assert params.width == 768
                assert params.height == 768
                assert params.num_inference_steps == 20  # SD 1.5 default
                assert params.guidance_scale == 7.5
    
    def test_parse_generation_params_with_style(self, pipeline, basic_request):
        """Test parsing generation parameters with style config."""
        pipeline.hardware_config = basic_request.hardware_constraints
        
        # Add style parameters
        basic_request.style_config.generation_params = {
            'negative_prompt': 'blurry, low quality',
            'width': 512,
            'height': 512,
            'num_inference_steps': 30,
            'guidance_scale': 8.0,
            'seed': 42
        }
        
        with patch.object(pipeline, '_select_model_for_request') as mock_select:
            mock_select.return_value = ImageModel.STABLE_DIFFUSION_V1_5.value
            
            with patch.object(pipeline.profile_manager, 'get_optimization_settings') as mock_opt:
                mock_opt.return_value = {'max_resolution': 768}
                
                params = pipeline._parse_generation_params(basic_request)
                
                assert params.negative_prompt == 'blurry, low quality'
                assert params.width == 512
                assert params.height == 512
                assert params.num_inference_steps == 30
                assert params.guidance_scale == 8.0
                assert params.seed == 42
    
    def test_parse_generation_params_turbo_model(self, pipeline, basic_request):
        """Test parsing parameters for SDXL-Turbo model."""
        pipeline.hardware_config = basic_request.hardware_constraints
        
        with patch.object(pipeline, '_select_model_for_request') as mock_select:
            mock_select.return_value = ImageModel.SDXL_TURBO.value
            
            with patch.object(pipeline.profile_manager, 'get_optimization_settings') as mock_opt:
                mock_opt.return_value = {'max_resolution': 1024}
                
                params = pipeline._parse_generation_params(basic_request)
                
                assert params.negative_prompt is None  # Turbo doesn't support negative prompts
                assert params.guidance_scale == 0.0    # Turbo uses guidance_scale=0.0
                assert params.num_inference_steps == 1  # Turbo default
    
    def test_calculate_quality_metrics(self, pipeline):
        """Test quality metrics calculation."""
        params = GenerationParams(
            prompt="test",
            width=512,
            height=768,
            num_inference_steps=20
        )
        
        # Mock image
        mock_image = Mock()
        mock_image.size = (512, 768)
        
        with patch('src.pipelines.image_generation.PIL_AVAILABLE', True):
            metrics = pipeline._calculate_quality_metrics(mock_image, params)
            
            assert metrics['resolution'] == 512 * 768
            assert metrics['aspect_ratio'] == 512 / 768
            assert metrics['inference_steps'] == 20
            assert metrics['actual_width'] == 512
            assert metrics['actual_height'] == 768
    
    def test_get_compliance_info(self, pipeline, basic_request):
        """Test compliance information generation."""
        pipeline.current_model = ImageModel.STABLE_DIFFUSION_V1_5.value
        
        compliance_info = pipeline._get_compliance_info(basic_request)
        
        assert compliance_info['compliance_mode'] == ComplianceMode.RESEARCH_SAFE.value
        assert compliance_info['model_used'] == ImageModel.STABLE_DIFFUSION_V1_5.value
        assert compliance_info['training_data_license'] == "CreativeML Open RAIL-M"
    
    def test_get_model_license_info(self, pipeline):
        """Test model license information retrieval."""
        sd15_license = pipeline._get_model_license_info(ImageModel.STABLE_DIFFUSION_V1_5.value)
        assert sd15_license == "CreativeML Open RAIL-M"
        
        sdxl_license = pipeline._get_model_license_info(ImageModel.SDXL_TURBO.value)
        assert sdxl_license == "CreativeML Open RAIL++-M"
        
        flux_license = pipeline._get_model_license_info(ImageModel.FLUX_SCHNELL.value)
        assert flux_license == "Apache 2.0"
        
        unknown_license = pipeline._get_model_license_info("unknown-model")
        assert unknown_license == "Unknown"
    
    @patch('src.pipelines.image_generation.TORCH_AVAILABLE', True)
    @patch('src.pipelines.image_generation.DIFFUSERS_AVAILABLE', True)
    @patch('src.pipelines.image_generation.PIL_AVAILABLE', True)
    def test_initialize_success(self, pipeline, hardware_config_8gb):
        """Test successful pipeline initialization."""
        with patch.object(pipeline, '_check_dependencies', return_value=True):
            with patch.object(pipeline, '_select_optimal_model', return_value=ImageModel.STABLE_DIFFUSION_V1_5.value):
                with patch.object(pipeline, '_load_model', return_value=True):
                    
                    result = pipeline.initialize(hardware_config_8gb)
                    
                    assert result is True
                    assert pipeline.is_initialized is True
                    assert pipeline.hardware_config == hardware_config_8gb
                    assert pipeline.memory_manager is not None
    
    def test_initialize_dependency_failure(self, pipeline, hardware_config_8gb):
        """Test pipeline initialization with dependency failure."""
        with patch.object(pipeline, '_check_dependencies', return_value=False):
            
            result = pipeline.initialize(hardware_config_8gb)
            
            assert result is False
            assert pipeline.is_initialized is False
    
    def test_generate_not_initialized(self, pipeline, basic_request):
        """Test generation when pipeline is not initialized."""
        result = pipeline.generate(basic_request)
        
        assert result.success is False
        assert result.error_message == "Pipeline not initialized"
        assert result.model_used == "none"
    
    @patch('src.pipelines.image_generation.time.time')
    def test_generate_model_load_failure(self, mock_time, pipeline, basic_request):
        """Test generation when model loading fails."""
        mock_time.return_value = 1000.0
        pipeline.is_initialized = True
        
        with patch.object(pipeline, '_parse_generation_params') as mock_parse:
            with patch.object(pipeline, '_select_model_for_request', return_value="test-model"):
                with patch.object(pipeline, '_ensure_model_loaded', return_value=False):
                    
                    result = pipeline.generate(basic_request)
                    
                    assert result.success is False
                    assert "Failed to load model" in result.error_message
                    assert result.model_used == "test-model"
    
    def test_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        # Set up pipeline with mock objects
        pipeline.current_pipeline = Mock()
        pipeline.current_model = "test-model"
        pipeline.memory_manager = Mock()
        
        pipeline.cleanup()
        
        assert pipeline.current_pipeline is None
        assert pipeline.current_model is None
        pipeline.memory_manager.clear_vram_cache.assert_called_once()
    
    def test_switch_model_unknown(self, pipeline):
        """Test switching to unknown model."""
        result = pipeline.switch_model("unknown-model")
        assert result is False
    
    def test_switch_model_same(self, pipeline):
        """Test switching to same model."""
        pipeline.current_model = ImageModel.STABLE_DIFFUSION_V1_5.value
        
        result = pipeline.switch_model(ImageModel.STABLE_DIFFUSION_V1_5.value)
        assert result is True
    
    def test_switch_model_success(self, pipeline):
        """Test successful model switching."""
        pipeline.current_model = ImageModel.STABLE_DIFFUSION_V1_5.value
        pipeline.memory_manager = Mock()
        
        with patch.object(pipeline, '_load_model', return_value=True):
            result = pipeline.switch_model(ImageModel.SDXL_TURBO.value)
            
            assert result is True
            pipeline.memory_manager.manage_model_switching.assert_called_once_with(
                ImageModel.STABLE_DIFFUSION_V1_5.value,
                ImageModel.SDXL_TURBO.value
            )


if __name__ == '__main__':
    pytest.main([__file__])