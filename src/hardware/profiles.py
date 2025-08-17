"""
Hardware profile management for different GPU configurations.

This module manages hardware-specific optimization profiles and provides
configuration recommendations based on detected hardware capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..core.interfaces import HardwareConfig

logger = logging.getLogger(__name__)


@dataclass
class OptimizationProfile:
    """Hardware optimization profile with specific settings."""
    name: str
    compatible_gpus: List[str]
    min_vram_mb: int
    max_vram_mb: int
    optimization_level: str
    recommended_models: List[str]
    optimizations: Dict[str, Any]
    performance_targets: Dict[str, Any]


class HardwareProfileManager:
    """
    Manages hardware profiles and optimization strategies for different GPU configurations.
    
    Provides predefined profiles for common gaming GPUs and creates adaptive profiles
    for unknown hardware configurations.
    """
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
    
    def get_profile(self, hardware_config: HardwareConfig) -> OptimizationProfile:
        """
        Get optimization profile for given hardware configuration.
        
        Args:
            hardware_config: Hardware configuration to match
            
        Returns:
            OptimizationProfile: Best matching profile or adaptive profile
        """
        gpu_model = self._normalize_gpu_name(hardware_config.gpu_model)
        vram_mb = hardware_config.vram_size
        
        # Find exact match first
        for profile in self.profiles.values():
            if (gpu_model in profile.compatible_gpus and 
                profile.min_vram_mb <= vram_mb <= profile.max_vram_mb):
                logger.info(f"Using profile: {profile.name}")
                return profile
        
        # Find compatible profile by VRAM range
        for profile in self.profiles.values():
            if profile.min_vram_mb <= vram_mb <= profile.max_vram_mb:
                logger.info(f"Using compatible profile: {profile.name}")
                return profile
        
        # Create adaptive profile
        logger.info(f"Creating adaptive profile for {hardware_config.gpu_model}")
        return self._create_adaptive_profile(hardware_config)
    
    def get_model_recommendations(self, hardware_config: HardwareConfig) -> List[str]:
        """Get recommended models for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.recommended_models
    
    def get_optimization_settings(self, hardware_config: HardwareConfig) -> Dict[str, Any]:
        """Get optimization settings for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.optimizations.copy()
    
    def get_performance_targets(self, hardware_config: HardwareConfig) -> Dict[str, Any]:
        """Get performance targets for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.performance_targets.copy()
    
    def validate_model_compatibility(self, hardware_config: HardwareConfig, 
                                   model_name: str) -> bool:
        """Check if model is compatible with hardware configuration."""
        profile = self.get_profile(hardware_config)
        
        # Check if model is in recommended list
        if model_name in profile.recommended_models:
            return True
        
        # Check VRAM requirements for known models
        model_vram_requirements = {
            'stable-diffusion-v1-5': 3500,
            'sdxl-turbo': 7000,
            'flux.1-schnell': 20000,
            'stable-video-diffusion': 12000,
            'llama-3.1-8b': 16000,
            'phi-3-mini': 2500
        }
        
        required_vram = model_vram_requirements.get(model_name, 0)
        return hardware_config.vram_size >= required_vram
    
    def _initialize_profiles(self) -> Dict[str, OptimizationProfile]:
        """Initialize predefined hardware profiles."""
        profiles = {}
        
        # GTX 1650 / 4GB VRAM Profile
        profiles['gtx_1650_4gb'] = OptimizationProfile(
            name="GTX 1650 (4GB VRAM)",
            compatible_gpus=['gtx_1650', 'gtx_1660', 'rtx_2060_6gb'],
            min_vram_mb=3500,
            max_vram_mb=6000,
            optimization_level="aggressive",
            recommended_models=[
                'stable-diffusion-v1-5',
                'phi-3-mini'
            ],
            optimizations={
                'attention_slicing': True,
                'cpu_offload': True,
                'sequential_cpu_offload': True,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 512,
                'enable_vae_slicing': True,
                'enable_vae_tiling': True,
                'low_vram_mode': True,
                'gradient_checkpointing': True
            },
            performance_targets={
                'image_generation_time_s': 60,  # 512x512 image
                'video_generation_time_s': 900,  # 4-second clip
                'model_switch_time_s': 60,
                'max_vram_usage_percent': 90
            }
        )
        
        # RTX 3070 / 8GB VRAM Profile
        profiles['rtx_3070_8gb'] = OptimizationProfile(
            name="RTX 3070 (8GB VRAM)",
            compatible_gpus=['rtx_3070', 'rtx_3060_ti', 'rtx_2080', 'rtx_2080_super'],
            min_vram_mb=7000,
            max_vram_mb=10000,
            optimization_level="balanced",
            recommended_models=[
                'sdxl-turbo',
                'stable-diffusion-v1-5',
                'llama-3.1-8b'
            ],
            optimizations={
                'attention_slicing': False,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 768,
                'enable_vae_slicing': False,
                'enable_vae_tiling': True,
                'low_vram_mode': False,
                'gradient_checkpointing': False
            },
            performance_targets={
                'image_generation_time_s': 20,  # 768x768 image
                'video_generation_time_s': 300,  # 4-second clip
                'model_switch_time_s': 45,
                'max_vram_usage_percent': 85
            }
        )
        
        # RTX 4090 / 24GB VRAM Profile
        profiles['rtx_4090_24gb'] = OptimizationProfile(
            name="RTX 4090 (24GB VRAM)",
            compatible_gpus=['rtx_4090', 'rtx_4080', 'rtx_3090', 'rtx_3090_ti'],
            min_vram_mb=20000,
            max_vram_mb=30000,
            optimization_level="minimal",
            recommended_models=[
                'flux.1-schnell',
                'sdxl-turbo',
                'stable-video-diffusion',
                'llama-3.1-8b'
            ],
            optimizations={
                'attention_slicing': False,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 2,
                'max_resolution': 1024,
                'enable_vae_slicing': False,
                'enable_vae_tiling': False,
                'low_vram_mode': False,
                'gradient_checkpointing': False
            },
            performance_targets={
                'image_generation_time_s': 10,  # 1024x1024 image
                'video_generation_time_s': 120,  # 4-second clip
                'model_switch_time_s': 30,
                'max_vram_usage_percent': 80
            }
        )
        
        # Mid-range Profile (RTX 3060, RTX 2070, etc.)
        profiles['mid_range_6gb'] = OptimizationProfile(
            name="Mid-range GPU (6GB VRAM)",
            compatible_gpus=['rtx_3060', 'rtx_2070', 'rtx_2070_super', 'gtx_1080_ti'],
            min_vram_mb=5500,
            max_vram_mb=7000,
            optimization_level="balanced",
            recommended_models=[
                'stable-diffusion-v1-5',
                'phi-3-mini'
            ],
            optimizations={
                'attention_slicing': True,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 640,
                'enable_vae_slicing': True,
                'enable_vae_tiling': True,
                'low_vram_mode': False,
                'gradient_checkpointing': True
            },
            performance_targets={
                'image_generation_time_s': 30,  # 640x640 image
                'video_generation_time_s': 600,  # 4-second clip
                'model_switch_time_s': 50,
                'max_vram_usage_percent': 88
            }
        )
        
        return profiles
    
    def _normalize_gpu_name(self, gpu_model: str) -> str:
        """Normalize GPU model name for profile matching."""
        gpu_lower = gpu_model.lower().replace(" ", "_").replace("-", "_")
        
        # NVIDIA RTX 40 series
        if "rtx" in gpu_lower and "4090" in gpu_lower:
            return "rtx_4090"
        elif "rtx" in gpu_lower and "4080" in gpu_lower:
            return "rtx_4080"
        
        # NVIDIA RTX 30 series
        elif "rtx" in gpu_lower and "3090" in gpu_lower:
            return "rtx_3090"
        elif "rtx" in gpu_lower and "3080" in gpu_lower:
            return "rtx_3080"
        elif "rtx" in gpu_lower and "3070" in gpu_lower:
            return "rtx_3070"
        elif "rtx" in gpu_lower and "3060" in gpu_lower:
            return "rtx_3060"
        
        # NVIDIA RTX 20 series
        elif "rtx" in gpu_lower and "2080" in gpu_lower:
            return "rtx_2080"
        elif "rtx" in gpu_lower and "2070" in gpu_lower:
            return "rtx_2070"
        elif "rtx" in gpu_lower and "2060" in gpu_lower:
            return "rtx_2060"
        
        # NVIDIA GTX series
        elif "gtx" in gpu_lower and "1660" in gpu_lower:
            return "gtx_1660"
        elif "gtx" in gpu_lower and "1650" in gpu_lower:
            return "gtx_1650"
        elif "gtx" in gpu_lower and "1080" in gpu_lower:
            return "gtx_1080"
        elif "gtx" in gpu_lower and "1070" in gpu_lower:
            return "gtx_1070"
        
        # Return normalized name for unknown GPUs
        return gpu_lower
    
    def _create_adaptive_profile(self, hardware_config: HardwareConfig) -> OptimizationProfile:
        """Create adaptive profile for unknown hardware."""
        vram_mb = hardware_config.vram_size
        gpu_model = hardware_config.gpu_model
        
        # Determine optimization level and settings based on VRAM
        if vram_mb < 4000:
            optimization_level = "aggressive"
            recommended_models = ['phi-3-mini']
            optimizations = {
                'attention_slicing': True,
                'cpu_offload': True,
                'sequential_cpu_offload': True,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 512,
                'enable_vae_slicing': True,
                'enable_vae_tiling': True,
                'low_vram_mode': True,
                'gradient_checkpointing': True
            }
            performance_targets = {
                'image_generation_time_s': 90,
                'video_generation_time_s': 1200,
                'model_switch_time_s': 90,
                'max_vram_usage_percent': 95
            }
        elif vram_mb < 8000:
            optimization_level = "balanced"
            recommended_models = ['stable-diffusion-v1-5', 'phi-3-mini']
            optimizations = {
                'attention_slicing': True,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 640,
                'enable_vae_slicing': True,
                'enable_vae_tiling': True,
                'low_vram_mode': False,
                'gradient_checkpointing': True
            }
            performance_targets = {
                'image_generation_time_s': 40,
                'video_generation_time_s': 600,
                'model_switch_time_s': 60,
                'max_vram_usage_percent': 90
            }
        elif vram_mb < 16000:
            optimization_level = "balanced"
            recommended_models = ['sdxl-turbo', 'stable-diffusion-v1-5', 'llama-3.1-8b']
            optimizations = {
                'attention_slicing': False,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 1,
                'max_resolution': 768,
                'enable_vae_slicing': False,
                'enable_vae_tiling': True,
                'low_vram_mode': False,
                'gradient_checkpointing': False
            }
            performance_targets = {
                'image_generation_time_s': 25,
                'video_generation_time_s': 400,
                'model_switch_time_s': 45,
                'max_vram_usage_percent': 85
            }
        else:
            optimization_level = "minimal"
            recommended_models = ['flux.1-schnell', 'sdxl-turbo', 'stable-video-diffusion', 'llama-3.1-8b']
            optimizations = {
                'attention_slicing': False,
                'cpu_offload': False,
                'sequential_cpu_offload': False,
                'mixed_precision': True,
                'xformers': True,
                'batch_size': 2,
                'max_resolution': 1024,
                'enable_vae_slicing': False,
                'enable_vae_tiling': False,
                'low_vram_mode': False,
                'gradient_checkpointing': False
            }
            performance_targets = {
                'image_generation_time_s': 15,
                'video_generation_time_s': 180,
                'model_switch_time_s': 30,
                'max_vram_usage_percent': 80
            }
        
        return OptimizationProfile(
            name=f"Adaptive Profile - {gpu_model} ({vram_mb}MB)",
            compatible_gpus=[self._normalize_gpu_name(gpu_model)],
            min_vram_mb=max(0, vram_mb - 500),
            max_vram_mb=vram_mb + 500,
            optimization_level=optimization_level,
            recommended_models=recommended_models,
            optimizations=optimizations,
            performance_targets=performance_targets
        )