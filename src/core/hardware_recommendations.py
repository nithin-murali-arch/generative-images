"""
Hardware-based model recommendations and automatic setup.

This module provides intelligent model recommendations based on detected hardware
and automatically downloads appropriate models for the user's system.
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HardwareTier(Enum):
    """Hardware performance tiers."""
    BUDGET = "budget"        # 2-4GB VRAM
    MID_RANGE = "mid_range"  # 6-12GB VRAM  
    HIGH_END = "high_end"    # 16-24GB VRAM
    ENTHUSIAST = "enthusiast" # 24GB+ VRAM


@dataclass
class HardwareProfile:
    """Hardware profile with recommendations."""
    tier: HardwareTier
    vram_mb: int
    description: str
    recommended_models: List[str]
    auto_download_models: List[str]
    max_resolution: int
    expected_speed: str


class HardwareRecommendations:
    """
    Provides hardware-based model recommendations and setup.
    
    Features:
    - Automatic hardware tier detection
    - Model recommendations based on VRAM
    - Auto-download of appropriate models
    - Performance expectations
    """
    
    def __init__(self):
        self.profiles = self._initialize_profiles()
        logger.info("HardwareRecommendations initialized")
    
    def _initialize_profiles(self) -> Dict[HardwareTier, HardwareProfile]:
        """Initialize hardware profiles with model recommendations."""
        return {
            HardwareTier.BUDGET: HardwareProfile(
                tier=HardwareTier.BUDGET,
                vram_mb=4000,
                description="Budget Gaming (GTX 1650, RTX 3050)",
                recommended_models=["sd15", "tiny_sd"],
                auto_download_models=["sd15"],  # Auto-download SD 1.5
                max_resolution=512,
                expected_speed="Fast (2-5 seconds per image)"
            ),
            
            HardwareTier.MID_RANGE: HardwareProfile(
                tier=HardwareTier.MID_RANGE,
                vram_mb=10000,
                description="Mid-Range Gaming (RTX 3070, RTX 4060 Ti)",
                recommended_models=["sd15", "sdxl", "sdxl_turbo"],
                auto_download_models=["sd15", "sdxl_turbo"],  # Auto-download both
                max_resolution=1024,
                expected_speed="Balanced (3-8 seconds per image)"
            ),
            
            HardwareTier.HIGH_END: HardwareProfile(
                tier=HardwareTier.HIGH_END,
                vram_mb=20000,
                description="High-End Gaming (RTX 4080, RTX 4090)",
                recommended_models=["sd15", "sdxl", "sdxl_turbo", "flux_schnell"],
                auto_download_models=["sdxl_turbo", "flux_schnell"],  # Auto-download best models
                max_resolution=1024,
                expected_speed="Fast High-Quality (2-6 seconds per image)"
            ),
            
            HardwareTier.ENTHUSIAST: HardwareProfile(
                tier=HardwareTier.ENTHUSIAST,
                vram_mb=32000,
                description="Enthusiast/Workstation (RTX 6000, A100)",
                recommended_models=["sd15", "sdxl", "flux_schnell", "flux_dev"],
                auto_download_models=["flux_schnell"],  # Auto-download flagship model
                max_resolution=1536,
                expected_speed="Ultra Fast (1-4 seconds per image)"
            )
        }
    
    def detect_hardware_tier(self, vram_mb: int, gpu_model: str = "") -> HardwareTier:
        """
        Detect hardware tier based on VRAM and GPU model.
        
        Args:
            vram_mb: Available VRAM in MB
            gpu_model: GPU model name (optional)
            
        Returns:
            HardwareTier: Detected hardware tier
        """
        # VRAM-based detection with some overlap for flexibility
        if vram_mb >= 24000:
            return HardwareTier.ENTHUSIAST
        elif vram_mb >= 14000:
            return HardwareTier.HIGH_END
        elif vram_mb >= 6000:
            return HardwareTier.MID_RANGE
        else:
            return HardwareTier.BUDGET
    
    def get_profile(self, tier: HardwareTier) -> HardwareProfile:
        """Get hardware profile for a tier."""
        return self.profiles[tier]
    
    def get_recommendations(self, vram_mb: int, gpu_model: str = "") -> Dict[str, any]:
        """
        Get comprehensive recommendations for hardware.
        
        Args:
            vram_mb: Available VRAM in MB
            gpu_model: GPU model name
            
        Returns:
            Dict with recommendations
        """
        tier = self.detect_hardware_tier(vram_mb, gpu_model)
        profile = self.get_profile(tier)
        
        return {
            "tier": tier.value,
            "profile": profile,
            "recommended_models": profile.recommended_models,
            "auto_download_models": profile.auto_download_models,
            "settings": {
                "max_resolution": profile.max_resolution,
                "batch_size": 2 if tier in [HardwareTier.HIGH_END, HardwareTier.ENTHUSIAST] else 1,
                "precision": "float16",
                "memory_optimization": self._get_memory_optimization(tier),
                "enable_xformers": tier in [HardwareTier.HIGH_END, HardwareTier.ENTHUSIAST]
            },
            "performance": {
                "expected_speed": profile.expected_speed,
                "can_do_video": tier in [HardwareTier.MID_RANGE, HardwareTier.HIGH_END, HardwareTier.ENTHUSIAST],
                "recommended_video_models": self._get_video_recommendations(tier)
            }
        }
    
    def _get_memory_optimization(self, tier: HardwareTier) -> str:
        """Get recommended memory optimization level."""
        optimization_map = {
            HardwareTier.BUDGET: "aggressive",
            HardwareTier.MID_RANGE: "balanced", 
            HardwareTier.HIGH_END: "minimal",
            HardwareTier.ENTHUSIAST: "none"
        }
        return optimization_map[tier]
    
    def _get_video_recommendations(self, tier: HardwareTier) -> List[str]:
        """Get video model recommendations for hardware tier."""
        if tier == HardwareTier.BUDGET:
            return []  # No video generation for budget hardware
        elif tier == HardwareTier.MID_RANGE:
            return ["animatediff_v3"]
        elif tier == HardwareTier.HIGH_END:
            return ["svd_xt", "animatediff_v3"]
        else:  # ENTHUSIAST
            return ["svd_xt", "animatediff_v3", "cogvideox"]
    
    def get_setup_instructions(self, vram_mb: int, gpu_model: str = "") -> Dict[str, any]:
        """
        Get setup instructions for the user's hardware.
        
        Returns:
            Dict with setup steps and recommendations
        """
        recommendations = self.get_recommendations(vram_mb, gpu_model)
        tier = recommendations["tier"]
        profile = recommendations["profile"]
        
        # Generate setup steps
        setup_steps = []
        
        # Step 1: Hardware summary
        setup_steps.append({
            "step": 1,
            "title": "Hardware Detected",
            "description": f"{profile.description} - {vram_mb}MB VRAM",
            "status": "✅ Compatible"
        })
        
        # Step 2: Model downloads
        auto_models = profile.auto_download_models
        if auto_models:
            model_names = []
            total_size = 0
            
            # Get model info from registry
            try:
                from .model_registry import get_model_registry
                registry = get_model_registry()
                
                for model_key in auto_models:
                    model = registry.get_model(model_key)
                    if model:
                        model_names.append(model.model_name)
                        total_size += model.download_size_gb
            except:
                model_names = auto_models
                total_size = len(auto_models) * 5  # Rough estimate
            
            setup_steps.append({
                "step": 2,
                "title": "Recommended Models",
                "description": f"Auto-downloading: {', '.join(model_names)} (~{total_size:.1f}GB)",
                "status": "📥 Downloading"
            })
        
        # Step 3: Performance expectations
        setup_steps.append({
            "step": 3,
            "title": "Performance",
            "description": profile.expected_speed,
            "status": "🎯 Optimized"
        })
        
        # Step 4: Video capabilities
        if recommendations["performance"]["can_do_video"]:
            video_models = recommendations["performance"]["recommended_video_models"]
            setup_steps.append({
                "step": 4,
                "title": "Video Generation",
                "description": f"Supported with models: {', '.join(video_models)}",
                "status": "🎬 Available"
            })
        else:
            setup_steps.append({
                "step": 4,
                "title": "Video Generation", 
                "description": "Not recommended for this hardware tier",
                "status": "❌ Limited"
            })
        
        return {
            "tier": tier,
            "setup_steps": setup_steps,
            "recommendations": recommendations,
            "next_actions": [
                "Models will download automatically when first used",
                "Start with Easy mode for best experience",
                f"Recommended resolution: {profile.max_resolution}x{profile.max_resolution} or lower"
            ]
        }


# Global recommendations instance
_hardware_recommendations = None

def get_hardware_recommendations() -> HardwareRecommendations:
    """Get the global hardware recommendations instance."""
    global _hardware_recommendations
    if _hardware_recommendations is None:
        _hardware_recommendations = HardwareRecommendations()
    return _hardware_recommendations


# Convenience functions
def get_recommended_models(vram_mb: int) -> List[str]:
    """Get recommended models for VRAM amount."""
    recommendations = get_hardware_recommendations()
    return recommendations.get_recommendations(vram_mb)["recommended_models"]


def get_auto_download_models(vram_mb: int) -> List[str]:
    """Get models that should be auto-downloaded for VRAM amount."""
    recommendations = get_hardware_recommendations()
    return recommendations.get_recommendations(vram_mb)["auto_download_models"]


if __name__ == "__main__":
    # Test the recommendations
    recommendations = HardwareRecommendations()
    
    test_configs = [
        (4000, "GTX 1650"),
        (8000, "RTX 3070"),
        (16000, "RTX 4080"),
        (24000, "RTX 4090")
    ]
    
    for vram, gpu in test_configs:
        print(f"\n=== {gpu} ({vram}MB VRAM) ===")
        setup = recommendations.get_setup_instructions(vram, gpu)
        
        for step in setup["setup_steps"]:
            print(f"{step['step']}. {step['title']}: {step['description']} {step['status']}")
        
        print(f"Auto-download: {setup['recommendations']['auto_download_models']}")