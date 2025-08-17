"""
Unit tests for configuration management.

Tests the ConfigManager class and hardware profile management to ensure
proper configuration loading, saving, and hardware profile matching.
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.core.config import ConfigManager, ModelConfig, SystemConfig
from src.core.interfaces import HardwareConfig, ComplianceMode


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary config file."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        self.config_manager = ConfigManager(self.config_path)
    
    def test_default_config_creation(self):
        """Test creation of default configuration."""
        config = self.config_manager.load_config()
        
        assert isinstance(config, SystemConfig)
        assert config.default_compliance_mode == ComplianceMode.RESEARCH_SAFE
        assert config.max_concurrent_generations == 1
        assert config.memory_cleanup_threshold == 0.85
        assert config.log_level == "INFO"
    
    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        # Load default config
        original_config = self.config_manager.load_config()
        
        # Modify config
        original_config.max_concurrent_generations = 2
        original_config.default_compliance_mode = ComplianceMode.OPEN_SOURCE_ONLY
        self.config_manager.config = original_config
        
        # Save config
        self.config_manager.save_config()
        
        # Create new manager and load
        new_manager = ConfigManager(self.config_path)
        loaded_config = new_manager.load_config()
        
        assert loaded_config.max_concurrent_generations == 2
        assert loaded_config.default_compliance_mode == ComplianceMode.OPEN_SOURCE_ONLY
    
    def test_hardware_config_update(self):
        """Test updating hardware configuration."""
        hardware_config = HardwareConfig(
            vram_size=8192,
            gpu_model="RTX 3070",
            cpu_cores=8,
            ram_size=32768,
            cuda_available=True,
            optimization_level="balanced"
        )
        
        self.config_manager.load_config()
        self.config_manager.update_hardware_config(hardware_config)
        
        # Reload and verify
        new_manager = ConfigManager(self.config_path)
        config = new_manager.load_config()
        
        assert config.hardware_config.vram_size == 8192
        assert config.hardware_config.gpu_model == "RTX 3070"
        assert config.hardware_config.optimization_level == "balanced"


class TestHardwareProfiles:
    """Test hardware profile matching and creation."""
    
    def setup_method(self):
        """Set up ConfigManager for testing."""
        self.config_manager = ConfigManager()
    
    def test_gtx_1650_profile_matching(self):
        """Test GTX 1650 profile matching."""
        profile = self.config_manager.get_hardware_profile("GTX 1650", 4096)
        
        assert profile["name"] == "GTX 1650 (4GB VRAM)"
        assert profile["optimization_level"] == "aggressive"
        assert profile["optimizations"]["attention_slicing"] is True
        assert profile["optimizations"]["cpu_offload"] is True
        assert "stable-diffusion-v1-5" in profile["recommended_models"]
    
    def test_rtx_3070_profile_matching(self):
        """Test RTX 3070 profile matching."""
        profile = self.config_manager.get_hardware_profile("RTX 3070", 8192)
        
        assert profile["name"] == "RTX 3070 (8GB VRAM)"
        assert profile["optimization_level"] == "balanced"
        assert profile["optimizations"]["attention_slicing"] is False
        assert profile["optimizations"]["cpu_offload"] is False
        assert "sdxl-turbo" in profile["recommended_models"]
    
    def test_rtx_4090_profile_matching(self):
        """Test RTX 4090 profile matching."""
        profile = self.config_manager.get_hardware_profile("RTX 4090", 24576)
        
        assert profile["name"] == "RTX 4090 (24GB VRAM)"
        assert profile["optimization_level"] == "minimal"
        assert profile["optimizations"]["batch_size"] == 2
        assert profile["optimizations"]["resolution"] == 1024
        assert "flux.1-schnell" in profile["recommended_models"]
    
    def test_unknown_hardware_adaptive_profile(self):
        """Test adaptive profile creation for unknown hardware."""
        profile = self.config_manager.get_hardware_profile("Unknown GPU", 6144)
        
        assert "Unknown GPU" in profile["name"]
        assert "6144" in profile["name"]
        assert profile["optimization_level"] == "balanced"  # 6GB should be balanced
        assert profile["optimizations"]["attention_slicing"] is True  # <8GB
        assert profile["optimizations"]["cpu_offload"] is False  # >=6GB
    
    def test_low_vram_adaptive_profile(self):
        """Test adaptive profile for very low VRAM."""
        profile = self.config_manager.get_hardware_profile("Low End GPU", 2048)
        
        assert profile["optimization_level"] == "aggressive"
        assert profile["optimizations"]["attention_slicing"] is True
        assert profile["optimizations"]["cpu_offload"] is True
        assert profile["optimizations"]["batch_size"] == 1
        assert profile["optimizations"]["resolution"] == 512
    
    def test_high_vram_adaptive_profile(self):
        """Test adaptive profile for high VRAM."""
        profile = self.config_manager.get_hardware_profile("High End GPU", 16384)
        
        assert profile["optimization_level"] == "minimal"
        assert profile["optimizations"]["attention_slicing"] is False
        assert profile["optimizations"]["cpu_offload"] is False
        assert profile["optimizations"]["batch_size"] == 2
        assert "flux.1-schnell" in profile["recommended_models"]


class TestModelConfigs:
    """Test model configuration management."""
    
    def setup_method(self):
        """Set up ConfigManager for testing."""
        self.config_manager = ConfigManager()
    
    def test_stable_diffusion_config(self):
        """Test Stable Diffusion v1.5 configuration."""
        config = self.config_manager.get_model_config("stable-diffusion-v1-5")
        
        assert config is not None
        assert config.name == "Stable Diffusion v1.5"
        assert config.path == "runwayml/stable-diffusion-v1-5"
        assert config.vram_requirement == 3500
        assert "text2img" in config.supported_features
        assert "lora" in config.supported_features
    
    def test_sdxl_turbo_config(self):
        """Test SDXL Turbo configuration."""
        config = self.config_manager.get_model_config("sdxl-turbo")
        
        assert config is not None
        assert config.name == "SDXL Turbo"
        assert config.vram_requirement == 7000
        assert "fast_inference" in config.supported_features
    
    def test_flux_config(self):
        """Test FLUX.1 Schnell configuration."""
        config = self.config_manager.get_model_config("flux.1-schnell")
        
        assert config is not None
        assert config.name == "FLUX.1 Schnell"
        assert config.vram_requirement == 20000
        assert "high_quality" in config.supported_features
    
    def test_llm_configs(self):
        """Test LLM model configurations."""
        llama_config = self.config_manager.get_model_config("llama-3.1-8b")
        phi_config = self.config_manager.get_model_config("phi-3-mini")
        
        assert llama_config.vram_requirement == 16000
        assert phi_config.vram_requirement == 2500
        assert "text_generation" in llama_config.supported_features
        assert "lightweight" in phi_config.supported_features
    
    def test_nonexistent_model_config(self):
        """Test handling of nonexistent model configuration."""
        config = self.config_manager.get_model_config("nonexistent-model")
        assert config is None


class TestGPUNameNormalization:
    """Test GPU name normalization for profile matching."""
    
    def setup_method(self):
        """Set up ConfigManager for testing."""
        self.config_manager = ConfigManager()
    
    def test_gtx_1650_variations(self):
        """Test various GTX 1650 name formats."""
        variations = [
            "GTX 1650",
            "GeForce GTX 1650",
            "NVIDIA GeForce GTX 1650",
            "gtx 1650",
            "GTX1650"
        ]
        
        for variation in variations:
            normalized = self.config_manager._normalize_gpu_name(variation)
            # Should match gtx_1650 pattern
            assert "gtx" in normalized and "1650" in normalized
    
    def test_rtx_3070_variations(self):
        """Test various RTX 3070 name formats."""
        variations = [
            "RTX 3070",
            "GeForce RTX 3070",
            "NVIDIA GeForce RTX 3070",
            "rtx 3070"
        ]
        
        for variation in variations:
            normalized = self.config_manager._normalize_gpu_name(variation)
            assert "rtx" in normalized and "3070" in normalized
    
    def test_unknown_gpu_normalization(self):
        """Test normalization of unknown GPU names."""
        unknown_gpu = "Some Unknown GPU Model"
        normalized = self.config_manager._normalize_gpu_name(unknown_gpu)
        
        assert normalized == "some_unknown_gpu_model"


class TestConfigValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_config_file(self):
        """Test handling of invalid configuration file."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "invalid_config.json"
        
        # Create invalid JSON file
        with open(config_path, 'w') as f:
            f.write("invalid json content")
        
        config_manager = ConfigManager(config_path)
        
        # Should handle invalid JSON gracefully and create default config
        with pytest.raises(json.JSONDecodeError):
            config_manager.load_config()
    
    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "nonexistent_config.json"
        
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Should create default config
        assert isinstance(config, SystemConfig)
        assert config_path.exists()  # Should have been created
    
    def test_config_directory_creation(self):
        """Test automatic creation of config directory."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "nested" / "config" / "system_config.json"
        
        config_manager = ConfigManager(config_path)
        config_manager.load_config()
        config_manager.save_config()
        
        assert config_path.exists()
        assert config_path.parent.exists()


if __name__ == "__main__":
    pytest.main([__file__])