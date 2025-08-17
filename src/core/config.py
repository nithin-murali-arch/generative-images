"""
Configuration management for the Academic Multimodal LLM Experiment System.

This module handles system-wide configuration including hardware profiles,
model configurations, and compliance settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from .interfaces import HardwareConfig, ComplianceMode, LicenseType


@dataclass
class ModelConfig:
    """Configuration for AI models."""
    name: str
    path: str
    vram_requirement: int  # MB
    optimization_flags: Dict[str, Any]
    supported_features: List[str]


@dataclass
class SystemConfig:
    """System-wide configuration."""
    # Paths
    data_dir: Path
    models_dir: Path
    experiments_dir: Path
    cache_dir: Path
    
    # Hardware
    hardware_config: Optional[HardwareConfig] = None
    
    # Compliance
    default_compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE
    
    # Performance
    max_concurrent_generations: int = 1
    memory_cleanup_threshold: float = 0.85  # 85% VRAM usage
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None


class ConfigManager:
    """Manages system configuration and hardware profiles."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/system_config.json")
        self.config: Optional[SystemConfig] = None
        self.hardware_profiles = self._load_hardware_profiles()
        self.model_configs = self._load_model_configs()
    
    def load_config(self) -> SystemConfig:
        """Load system configuration from file or create default."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            self.config = self._dict_to_system_config(config_data)
        else:
            self.config = self._create_default_config()
            self.save_config()
        
        return self.config
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self.config is None:
            raise ValueError("No configuration to save")
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self._system_config_to_dict(self.config), f, indent=2)
    
    def get_hardware_profile(self, gpu_model: str, vram_size: int) -> Dict[str, Any]:
        """Get hardware profile for specific GPU configuration."""
        # Normalize GPU model name
        gpu_key = self._normalize_gpu_name(gpu_model)
        
        # Find matching profile or create adaptive one
        for profile_name, profile in self.hardware_profiles.items():
            if (gpu_key in profile.get("compatible_gpus", []) and 
                vram_size >= profile.get("min_vram", 0)):
                return profile
        
        # Create adaptive profile for unknown hardware
        return self._create_adaptive_profile(gpu_model, vram_size)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model."""
        return self.model_configs.get(model_name)
    
    def update_hardware_config(self, hardware_config: HardwareConfig) -> None:
        """Update hardware configuration."""
        if self.config is None:
            self.load_config()
        self.config.hardware_config = hardware_config
        self.save_config()
    
    def _load_hardware_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load hardware optimization profiles."""
        return {
            "gtx_1650_4gb": {
                "name": "GTX 1650 (4GB VRAM)",
                "compatible_gpus": ["gtx_1650", "gtx_1660", "rtx_2060"],
                "min_vram": 3500,  # MB
                "max_vram": 6000,
                "optimization_level": "aggressive",
                "recommended_models": ["stable-diffusion-v1-5", "sd-tiny"],
                "optimizations": {
                    "attention_slicing": True,
                    "cpu_offload": True,
                    "mixed_precision": True,
                    "xformers": True,
                    "batch_size": 1,
                    "resolution": 512
                }
            },
            "rtx_3070_8gb": {
                "name": "RTX 3070 (8GB VRAM)",
                "compatible_gpus": ["rtx_3070", "rtx_3060_ti", "rtx_2080"],
                "min_vram": 7000,
                "max_vram": 10000,
                "optimization_level": "balanced",
                "recommended_models": ["sdxl-turbo", "stable-diffusion-v1-5"],
                "optimizations": {
                    "attention_slicing": False,
                    "cpu_offload": False,
                    "mixed_precision": True,
                    "xformers": True,
                    "batch_size": 1,
                    "resolution": 768
                }
            },
            "rtx_4090_24gb": {
                "name": "RTX 4090 (24GB VRAM)",
                "compatible_gpus": ["rtx_4090", "rtx_4080", "rtx_3090"],
                "min_vram": 20000,
                "max_vram": 30000,
                "optimization_level": "minimal",
                "recommended_models": ["flux.1-schnell", "sdxl-turbo", "stable-video-diffusion"],
                "optimizations": {
                    "attention_slicing": False,
                    "cpu_offload": False,
                    "mixed_precision": True,
                    "xformers": True,
                    "batch_size": 2,
                    "resolution": 1024
                }
            }
        }
    
    def _load_model_configs(self) -> Dict[str, ModelConfig]:
        """Load model configurations."""
        configs = {}
        
        # Image generation models
        configs["stable-diffusion-v1-5"] = ModelConfig(
            name="Stable Diffusion v1.5",
            path="runwayml/stable-diffusion-v1-5",
            vram_requirement=3500,
            optimization_flags={
                "torch_dtype": "float16",
                "use_safetensors": True
            },
            supported_features=["text2img", "img2img", "lora", "controlnet"]
        )
        
        configs["sdxl-turbo"] = ModelConfig(
            name="SDXL Turbo",
            path="stabilityai/sdxl-turbo",
            vram_requirement=7000,
            optimization_flags={
                "torch_dtype": "float16",
                "use_safetensors": True
            },
            supported_features=["text2img", "fast_inference"]
        )
        
        configs["flux.1-schnell"] = ModelConfig(
            name="FLUX.1 Schnell",
            path="black-forest-labs/FLUX.1-schnell",
            vram_requirement=20000,
            optimization_flags={
                "torch_dtype": "bfloat16",
                "use_safetensors": True
            },
            supported_features=["text2img", "high_quality"]
        )
        
        # Video generation models
        configs["stable-video-diffusion"] = ModelConfig(
            name="Stable Video Diffusion",
            path="stabilityai/stable-video-diffusion-img2vid-xt",
            vram_requirement=12000,
            optimization_flags={
                "torch_dtype": "float16",
                "use_safetensors": True
            },
            supported_features=["img2vid", "temporal_consistency"]
        )
        
        # LLM models
        configs["llama-3.1-8b"] = ModelConfig(
            name="Llama 3.1 8B",
            path="meta-llama/Meta-Llama-3.1-8B-Instruct",
            vram_requirement=16000,
            optimization_flags={
                "load_in_4bit": True,
                "device_map": "auto"
            },
            supported_features=["text_generation", "instruction_following"]
        )
        
        configs["phi-3-mini"] = ModelConfig(
            name="Phi-3 Mini",
            path="microsoft/Phi-3-mini-4k-instruct",
            vram_requirement=2500,
            optimization_flags={
                "torch_dtype": "float16",
                "device_map": "auto"
            },
            supported_features=["text_generation", "lightweight"]
        )
        
        return configs
    
    def _create_default_config(self) -> SystemConfig:
        """Create default system configuration."""
        base_dir = Path.cwd()
        return SystemConfig(
            data_dir=base_dir / "data",
            models_dir=base_dir / "models",
            experiments_dir=base_dir / "experiments",
            cache_dir=base_dir / "cache",
            default_compliance_mode=ComplianceMode.RESEARCH_SAFE,
            max_concurrent_generations=1,
            memory_cleanup_threshold=0.85,
            log_level="INFO",
            log_file=base_dir / "logs" / "system.log"
        )
    
    def _normalize_gpu_name(self, gpu_model: str) -> str:
        """Normalize GPU model name for profile matching."""
        gpu_lower = gpu_model.lower().replace(" ", "_")
        
        # Common normalizations
        if "gtx" in gpu_lower and "1650" in gpu_lower:
            return "gtx_1650"
        elif "rtx" in gpu_lower and "3070" in gpu_lower:
            return "rtx_3070"
        elif "rtx" in gpu_lower and "4090" in gpu_lower:
            return "rtx_4090"
        
        return gpu_lower
    
    def _create_adaptive_profile(self, gpu_model: str, vram_size: int) -> Dict[str, Any]:
        """Create adaptive profile for unknown hardware."""
        if vram_size < 6000:
            optimization_level = "aggressive"
            recommended_models = ["stable-diffusion-v1-5"]
        elif vram_size < 12000:
            optimization_level = "balanced"
            recommended_models = ["sdxl-turbo", "stable-diffusion-v1-5"]
        else:
            optimization_level = "minimal"
            recommended_models = ["flux.1-schnell", "sdxl-turbo"]
        
        return {
            "name": f"{gpu_model} ({vram_size}MB VRAM)",
            "compatible_gpus": [self._normalize_gpu_name(gpu_model)],
            "min_vram": vram_size - 500,
            "max_vram": vram_size + 500,
            "optimization_level": optimization_level,
            "recommended_models": recommended_models,
            "optimizations": {
                "attention_slicing": vram_size < 8000,
                "cpu_offload": vram_size < 6000,
                "mixed_precision": True,
                "xformers": True,
                "batch_size": 1 if vram_size < 12000 else 2,
                "resolution": 512 if vram_size < 8000 else 768
            }
        }
    
    def _dict_to_system_config(self, config_data: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to SystemConfig object."""
        # Convert paths
        for path_key in ["data_dir", "models_dir", "experiments_dir", "cache_dir", "log_file"]:
            if path_key in config_data and config_data[path_key]:
                config_data[path_key] = Path(config_data[path_key])
        
        # Convert enums
        if "default_compliance_mode" in config_data:
            config_data["default_compliance_mode"] = ComplianceMode(config_data["default_compliance_mode"])
        
        # Convert hardware config if present
        if "hardware_config" in config_data and config_data["hardware_config"]:
            hw_data = config_data["hardware_config"]
            config_data["hardware_config"] = HardwareConfig(**hw_data)
        
        return SystemConfig(**config_data)
    
    def _system_config_to_dict(self, config: SystemConfig) -> Dict[str, Any]:
        """Convert SystemConfig object to dictionary."""
        config_dict = asdict(config)
        
        # Convert paths to strings
        for path_key in ["data_dir", "models_dir", "experiments_dir", "cache_dir", "log_file"]:
            if config_dict[path_key]:
                config_dict[path_key] = str(config_dict[path_key])
        
        # Convert enums to values
        if config_dict["default_compliance_mode"]:
            config_dict["default_compliance_mode"] = config_dict["default_compliance_mode"].value
        
        return config_dict