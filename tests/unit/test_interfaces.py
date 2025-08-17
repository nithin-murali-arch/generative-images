"""
Unit tests for core interfaces and data structures.

Tests the fundamental interfaces and data models to ensure they work correctly
and maintain consistency across the system.
"""

import pytest
from datetime import datetime
from pathlib import Path
from src.core.interfaces import (
    OutputType, LicenseType, ComplianceMode, HardwareConfig,
    StyleConfig, ConversationContext, GenerationRequest, ContentItem,
    GenerationResult, ExperimentResult, SystemError, MemoryError,
    ModelLoadError, ComplianceError, GenerationError, HardwareError
)


class TestEnums:
    """Test enum definitions and values."""
    
    def test_output_type_values(self):
        """Test OutputType enum has correct values."""
        assert OutputType.IMAGE.value == "image"
        assert OutputType.VIDEO.value == "video"
        assert OutputType.TEXT.value == "text"
        assert OutputType.MULTIMODAL.value == "multimodal"
    
    def test_license_type_values(self):
        """Test LicenseType enum has correct values."""
        assert LicenseType.PUBLIC_DOMAIN.value == "public_domain"
        assert LicenseType.CREATIVE_COMMONS.value == "creative_commons"
        assert LicenseType.FAIR_USE_RESEARCH.value == "fair_use_research"
        assert LicenseType.COPYRIGHTED.value == "copyrighted"
        assert LicenseType.UNKNOWN.value == "unknown"
    
    def test_compliance_mode_values(self):
        """Test ComplianceMode enum has correct values."""
        assert ComplianceMode.OPEN_SOURCE_ONLY.value == "open_only"
        assert ComplianceMode.RESEARCH_SAFE.value == "research_safe"
        assert ComplianceMode.FULL_DATASET.value == "full_dataset"


class TestDataStructures:
    """Test data structure creation and validation."""
    
    def test_hardware_config_creation(self):
        """Test HardwareConfig dataclass creation."""
        config = HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
        
        assert config.vram_size == 8192
        assert config.gpu_model == "RTX 3070"
        assert config.cuda_available is True
        assert config.optimization_level == "balanced"
    
    def test_style_config_creation(self):
        """Test StyleConfig dataclass creation."""
        config = StyleConfig(
            style_name="anime",
            lora_path=Path("models/anime_lora.safetensors"),
            generation_params={"steps": 20, "guidance_scale": 7.5}
        )
        
        assert config.style_name == "anime"
        assert isinstance(config.lora_path, Path)
        assert config.generation_params["steps"] == 20
    
    def test_conversation_context_creation(self):
        """Test ConversationContext dataclass creation."""
        context = ConversationContext(
            conversation_id="conv_123",
            history=[{"role": "user", "content": "Generate an image"}],
            current_mode=ComplianceMode.RESEARCH_SAFE,
            user_preferences={"quality": "high"}
        )
        
        assert context.conversation_id == "conv_123"
        assert len(context.history) == 1
        assert context.current_mode == ComplianceMode.RESEARCH_SAFE
    
    def test_generation_request_creation(self):
        """Test GenerationRequest dataclass creation."""
        hardware_config = HardwareConfig(
            vram_size=4096, gpu_model="GTX 1650", cpu_cores=4,
            ram_size=16384, cuda_available=True, optimization_level="aggressive"
        )
        
        style_config = StyleConfig()
        context = ConversationContext(
            conversation_id="test", history=[], 
            current_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            user_preferences={}
        )
        
        request = GenerationRequest(
            prompt="A beautiful sunset",
            output_type=OutputType.IMAGE,
            style_config=style_config,
            compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            hardware_constraints=hardware_config,
            context=context
        )
        
        assert request.prompt == "A beautiful sunset"
        assert request.output_type == OutputType.IMAGE
        assert request.compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY
    
    def test_content_item_creation(self):
        """Test ContentItem dataclass creation."""
        item = ContentItem(
            url="https://example.com/image.jpg",
            local_path=Path("data/images/image.jpg"),
            license_type=LicenseType.CREATIVE_COMMONS,
            attribution="Photo by Artist Name",
            metadata={"width": 1024, "height": 768},
            copyright_status="cc_by_4.0",
            research_safe=True
        )
        
        assert item.url == "https://example.com/image.jpg"
        assert item.license_type == LicenseType.CREATIVE_COMMONS
        assert item.research_safe is True
        assert item.metadata["width"] == 1024
    
    def test_generation_result_creation(self):
        """Test GenerationResult dataclass creation."""
        result = GenerationResult(
            success=True,
            output_path=Path("experiments/output.png"),
            generation_time=45.2,
            model_used="stable-diffusion-v1-5",
            quality_metrics={"aesthetic_score": 7.8},
            compliance_info={"license_types_used": ["public_domain"]}
        )
        
        assert result.success is True
        assert result.generation_time == 45.2
        assert result.model_used == "stable-diffusion-v1-5"
        assert result.quality_metrics["aesthetic_score"] == 7.8
    
    def test_experiment_result_creation(self):
        """Test ExperimentResult dataclass creation."""
        # Create required components
        hardware_config = HardwareConfig(
            vram_size=8192, gpu_model="RTX 3070", cpu_cores=8,
            ram_size=32768, cuda_available=True, optimization_level="balanced"
        )
        
        style_config = StyleConfig()
        context = ConversationContext(
            conversation_id="exp_test", history=[],
            current_mode=ComplianceMode.RESEARCH_SAFE, user_preferences={}
        )
        
        request = GenerationRequest(
            prompt="Test experiment",
            output_type=OutputType.IMAGE,
            style_config=style_config,
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=context
        )
        
        result = GenerationResult(
            success=True,
            output_path=Path("test_output.png"),
            generation_time=30.0,
            model_used="test_model"
        )
        
        experiment = ExperimentResult(
            experiment_id="exp_001",
            timestamp=datetime.now(),
            request=request,
            result=result,
            notes="Test experiment for validation"
        )
        
        assert experiment.experiment_id == "exp_001"
        assert isinstance(experiment.timestamp, datetime)
        assert experiment.request.prompt == "Test experiment"
        assert experiment.result.success is True
        assert experiment.notes == "Test experiment for validation"


class TestExceptions:
    """Test custom exception classes."""
    
    def test_system_error_inheritance(self):
        """Test that custom errors inherit from SystemError."""
        assert issubclass(MemoryError, SystemError)
        assert issubclass(ModelLoadError, SystemError)
        assert issubclass(ComplianceError, SystemError)
        assert issubclass(GenerationError, SystemError)
        assert issubclass(HardwareError, SystemError)
    
    def test_exception_creation(self):
        """Test exception creation with messages."""
        memory_error = MemoryError("VRAM exhausted")
        model_error = ModelLoadError("Failed to load model")
        compliance_error = ComplianceError("Copyright violation detected")
        generation_error = GenerationError("Generation failed")
        hardware_error = HardwareError("GPU not detected")
        
        assert str(memory_error) == "VRAM exhausted"
        assert str(model_error) == "Failed to load model"
        assert str(compliance_error) == "Copyright violation detected"
        assert str(generation_error) == "Generation failed"
        assert str(hardware_error) == "GPU not detected"


class TestDataValidation:
    """Test data validation and edge cases."""
    
    def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        hardware_config = HardwareConfig(
            vram_size=4096, gpu_model="Test", cpu_cores=4,
            ram_size=16384, cuda_available=True, optimization_level="balanced"
        )
        
        style_config = StyleConfig()
        context = ConversationContext(
            conversation_id="test", history=[],
            current_mode=ComplianceMode.OPEN_SOURCE_ONLY, user_preferences={}
        )
        
        # Empty prompt should still create valid request
        request = GenerationRequest(
            prompt="",
            output_type=OutputType.IMAGE,
            style_config=style_config,
            compliance_mode=ComplianceMode.OPEN_SOURCE_ONLY,
            hardware_constraints=hardware_config,
            context=context
        )
        
        assert request.prompt == ""
        assert isinstance(request, GenerationRequest)
    
    def test_none_values_handling(self):
        """Test handling of None values in optional fields."""
        result = GenerationResult(
            success=False,
            output_path=None,
            generation_time=0.0,
            model_used="test_model",
            error_message="Test error"
        )
        
        assert result.success is False
        assert result.output_path is None
        assert result.error_message == "Test error"
        assert result.quality_metrics is None
    
    def test_path_handling(self):
        """Test Path object handling in data structures."""
        item = ContentItem(
            url="test_url",
            local_path=Path("test/path/file.jpg"),
            license_type=LicenseType.UNKNOWN,
            attribution="test",
            metadata={},
            copyright_status="unknown",
            research_safe=False
        )
        
        assert isinstance(item.local_path, Path)
        assert str(item.local_path) == "test/path/file.jpg"


if __name__ == "__main__":
    pytest.main([__file__])