"""
Hardware profile manager for optimization settings.

This module provides hardware-specific optimization profiles for different
GPU configurations and system capabilities.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .interfaces import HardwareConfig

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware optimization profile."""
    name: str
    vram_min: int
    vram_max: int
    optimizations: Dict[str, Any]
    recommended_models: List[str]
    performance_targets: Dict[str, Any]


class HardwareProfileManager:
    """
    Manages hardware-specific optimization profiles.
    
    Provides model recommendations and optimization settings based on
    detected hardware capabilities.
    """
    
    def __init__(self):
        """Initialize hardware profile manager."""
        self.profiles = self._create_default_profiles()
        logger.info(f"HardwareProfileManager initialized with {len(self.profiles)} profiles")
    
    def _create_default_profiles(self) -> List[HardwareProfile]:
        """Create default hardware profiles."""
        return [
            # Low-end hardware (2-4GB VRAM)
            HardwareProfile(
                name="low_end",
                vram_min=0,
                vram_max=4000,
                optimizations={
                    "max_resolution": 512,
                    "attention_slicing": True,
                    "cpu_offload": True,
                    "sequential_cpu_offload": True,
                    "enable_xformers": False,
                    "fp16": True
                },
                recommended_models=["stable-diffusion-v1-5"],
                performance_targets={
                    "generation_time": 30.0,
                    "memory_usage": 0.8
                }
            ),
            
            # Mid-range hardware (4-8GB VRAM)
            HardwareProfile(
                name="mid_range",
                vram_min=4000,
                vram_max=8000,
                optimizations={
                    "max_resolution": 768,
                    "attention_slicing": True,
                    "cpu_offload": False,
                    "sequential_cpu_offload": False,
                    "enable_xformers": True,
                    "fp16": True
                },
                recommended_models=["sdxl-turbo", "stable-diffusion-v1-5"],
                performance_targets={
                    "generation_time": 15.0,
                    "memory_usage": 0.7
                }
            ),
            
            # High-end hardware (8GB+ VRAM)
            HardwareProfile(
                name="high_end",
                vram_min=8000,
                vram_max=999999,
                optimizations={
                    "max_resolution": 1024,
                    "attention_slicing": False,
                    "cpu_offload": False,
                    "sequential_cpu_offload": False,
                    "enable_xformers": True,
                    "fp16": True
                },
                recommended_models=["flux-schnell", "sdxl-turbo", "stable-diffusion-v1-5"],
                performance_targets={
                    "generation_time": 8.0,
                    "memory_usage": 0.6
                }
            )
        ]
    
    def get_profile(self, hardware_config: HardwareConfig) -> HardwareProfile:
        """Get hardware profile for given configuration."""
        vram_mb = hardware_config.vram_size
        
        for profile in self.profiles:
            if profile.vram_min <= vram_mb <= profile.vram_max:
                logger.debug(f"Selected profile '{profile.name}' for {vram_mb}MB VRAM")
                return profile
        
        # Fallback to low-end profile
        logger.warning(f"No profile found for {vram_mb}MB VRAM, using low_end profile")
        return self.profiles[0]
    
    def get_model_recommendations(self, hardware_config: HardwareConfig) -> List[str]:
        """Get recommended models for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.recommended_models.copy()
    
    def get_optimization_settings(self, hardware_config: HardwareConfig) -> Dict[str, Any]:
        """Get optimization settings for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.optimizations.copy()
    
    def get_performance_targets(self, hardware_config: HardwareConfig) -> Dict[str, Any]:
        """Get performance targets for hardware configuration."""
        profile = self.get_profile(hardware_config)
        return profile.performance_targets.copy()


# Global instance
_hardware_profile_manager: Optional[HardwareProfileManager] = None

def get_hardware_profile_manager() -> HardwareProfileManager:
    """Get the global hardware profile manager instance."""
    global _hardware_profile_manager
    if _hardware_profile_manager is None:
        _hardware_profile_manager = HardwareProfileManager()
    return _hardware_profile_manager