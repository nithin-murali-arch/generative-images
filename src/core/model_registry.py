"""
Model registry with latest model versions and configurations.

This module maintains an up-to-date registry of the best available models
for image and video generation, with automatic fallbacks and hardware optimization.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of generative models."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_VIDEO = "image_to_video"


class ModelTier(Enum):
    """Model performance tiers based on hardware requirements."""
    LIGHTWEIGHT = "lightweight"  # 2-4GB VRAM, fast generation
    MID_TIER = "mid_tier"        # 6-12GB VRAM, balanced quality/speed
    HIGH_END = "high_end"        # 16-24GB VRAM, best quality
    ULTRA = "ultra"              # 24GB+ VRAM, cutting-edge models


@dataclass
class ModelConfig:
    """Configuration for a generative model."""
    model_id: str
    model_name: str
    model_type: ModelType
    tier: ModelTier
    
    # Hardware requirements
    min_vram_mb: int
    recommended_vram_mb: int
    min_ram_mb: int
    
    # Capabilities
    max_resolution: int
    default_resolution: Tuple[int, int]
    max_batch_size: int
    
    # Generation parameters
    default_steps: int
    min_steps: int
    max_steps: int
    supports_negative_prompt: bool
    supports_guidance_scale: bool
    
    # Video-specific (if applicable)
    max_frames: Optional[int] = None
    default_frames: Optional[int] = None
    supports_image_conditioning: bool = False
    
    # Model metadata
    pipeline_class: str = ""
    license: str = "Unknown"
    description: str = ""
    release_date: str = ""
    
    # Performance estimates (on RTX 4090)
    estimated_speed_512: Optional[float] = None  # seconds per image/video
    estimated_speed_1024: Optional[float] = None
    
    # Download information
    download_size_gb: float = 0.0
    is_downloaded: bool = False
    download_priority: int = 0  # Lower = higher priority for auto-download


class ModelRegistry:
    """Registry of available generative models with latest versions."""
    
    def __init__(self):
        self.models = self._initialize_models()
        logger.info(f"ModelRegistry initialized with {len(self.models)} models")
    
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize the model registry with latest versions."""
        models = {}
        
        # === IMAGE GENERATION MODELS ===
        
        # Stable Diffusion 1.5 (Still widely used, very compatible)
        models["sd15"] = ModelConfig(
            model_id="runwayml/stable-diffusion-v1-5",
            model_name="Stable Diffusion 1.5",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.LIGHTWEIGHT,
            min_vram_mb=2000,
            recommended_vram_mb=4000,
            min_ram_mb=8000,
            max_resolution=768,
            default_resolution=(512, 512),
            max_batch_size=4,
            default_steps=20,
            min_steps=1,
            max_steps=100,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            pipeline_class="StableDiffusionPipeline",
            license="CreML Open RAIL-M",
            description="Fast, reliable image generation with broad compatibility",
            release_date="2022-08",
            estimated_speed_512=2.5,
            estimated_speed_1024=8.0,
            download_size_gb=3.4,
            download_priority=1
        )
        
        # SDXL 1.0 (Updated from Turbo to full version)
        models["sdxl"] = ModelConfig(
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            model_name="Stable Diffusion XL 1.0",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.MID_TIER,
            min_vram_mb=6000,
            recommended_vram_mb=10000,
            min_ram_mb=16000,
            max_resolution=1024,
            default_resolution=(1024, 1024),
            max_batch_size=2,
            default_steps=30,
            min_steps=10,
            max_steps=100,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            pipeline_class="StableDiffusionXLPipeline",
            license="CreML Open RAIL++-M",
            description="High-quality 1024x1024 image generation",
            release_date="2023-07",
            estimated_speed_512=4.0,
            estimated_speed_1024=12.0,
            download_size_gb=6.9,
            download_priority=2
        )
        
        # SDXL Turbo (Fast version)
        models["sdxl_turbo"] = ModelConfig(
            model_id="stabilityai/sdxl-turbo",
            model_name="SDXL Turbo",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.MID_TIER,
            min_vram_mb=6000,
            recommended_vram_mb=8000,
            min_ram_mb=12000,
            max_resolution=1024,
            default_resolution=(512, 512),
            max_batch_size=2,
            default_steps=1,
            min_steps=1,
            max_steps=4,
            supports_negative_prompt=False,
            supports_guidance_scale=False,
            pipeline_class="StableDiffusionXLPipeline",
            license="CreML Open RAIL++-M",
            description="Ultra-fast single-step generation",
            release_date="2023-11",
            estimated_speed_512=0.8,
            estimated_speed_1024=2.5,
            download_size_gb=6.9,
            download_priority=3
        )
        
        # FLUX.1 Schnell (Latest from Black Forest Labs)
        models["flux_schnell"] = ModelConfig(
            model_id="black-forest-labs/FLUX.1-schnell",
            model_name="FLUX.1 Schnell",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.HIGH_END,
            min_vram_mb=16000,
            recommended_vram_mb=24000,
            min_ram_mb=32000,
            max_resolution=1024,
            default_resolution=(1024, 1024),
            max_batch_size=1,
            default_steps=4,
            min_steps=1,
            max_steps=8,
            supports_negative_prompt=False,
            supports_guidance_scale=False,
            pipeline_class="FluxPipeline",
            license="Apache 2.0",
            description="State-of-the-art image quality with fast generation",
            release_date="2024-08",
            estimated_speed_512=3.0,
            estimated_speed_1024=8.0,
            download_size_gb=23.8,
            download_priority=4
        )
        
        # FLUX.1 Dev (Higher quality version)
        models["flux_dev"] = ModelConfig(
            model_id="black-forest-labs/FLUX.1-dev",
            model_name="FLUX.1 Dev",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.ULTRA,
            min_vram_mb=20000,
            recommended_vram_mb=24000,
            min_ram_mb=32000,
            max_resolution=1024,
            default_resolution=(1024, 1024),
            max_batch_size=1,
            default_steps=20,
            min_steps=10,
            max_steps=50,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            pipeline_class="FluxPipeline",
            license="FLUX.1 [dev] Non-Commercial License",
            description="Highest quality image generation (non-commercial)",
            release_date="2024-08",
            estimated_speed_512=8.0,
            estimated_speed_1024=25.0
        )
        
        # === VIDEO GENERATION MODELS ===
        
        # Stable Video Diffusion XT (Latest version)
        models["svd_xt"] = ModelConfig(
            model_id="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
            model_name="Stable Video Diffusion XT 1.1",
            model_type=ModelType.IMAGE_TO_VIDEO,
            tier=ModelTier.HIGH_END,
            min_vram_mb=12000,
            recommended_vram_mb=16000,
            min_ram_mb=24000,
            max_resolution=1024,
            default_resolution=(576, 1024),
            max_batch_size=1,
            default_steps=25,
            min_steps=10,
            max_steps=50,
            supports_negative_prompt=False,
            supports_guidance_scale=True,
            max_frames=25,
            default_frames=14,
            supports_image_conditioning=True,
            pipeline_class="StableVideoDiffusionPipeline",
            license="Stability AI Community License",
            description="High-quality image-to-video generation",
            release_date="2024-01",
            estimated_speed_512=45.0,
            estimated_speed_1024=120.0
        )
        
        # AnimateDiff (Latest version with better motion)
        models["animatediff_v3"] = ModelConfig(
            model_id="guoyww/animatediff-motion-adapter-v1-5-3",
            model_name="AnimateDiff v3",
            model_type=ModelType.TEXT_TO_IMAGE,  # Text to video via SD base
            tier=ModelTier.MID_TIER,
            min_vram_mb=8000,
            recommended_vram_mb=12000,
            min_ram_mb=16000,
            max_resolution=512,
            default_resolution=(512, 512),
            max_batch_size=1,
            default_steps=25,
            min_steps=10,
            max_steps=50,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            max_frames=16,
            default_frames=16,
            supports_image_conditioning=False,
            pipeline_class="AnimateDiffPipeline",
            license="CreML Open RAIL-M",
            description="Text-to-video with motion control",
            release_date="2024-03",
            estimated_speed_512=60.0
        )
        
        # CogVideoX (New Chinese model, very good quality)
        models["cogvideox"] = ModelConfig(
            model_id="THUDM/CogVideoX-2b",
            model_name="CogVideoX 2B",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.HIGH_END,
            min_vram_mb=10000,
            recommended_vram_mb=16000,
            min_ram_mb=20000,
            max_resolution=720,
            default_resolution=(480, 720),
            max_batch_size=1,
            default_steps=50,
            min_steps=20,
            max_steps=100,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            max_frames=49,
            default_frames=49,
            supports_image_conditioning=False,
            pipeline_class="CogVideoXPipeline",
            license="CogVideoX License",
            description="High-quality text-to-video generation",
            release_date="2024-08",
            estimated_speed_512=180.0
        )
        
        # === LIGHTWEIGHT/FALLBACK MODELS ===
        
        # Tiny SD (For very low VRAM)
        models["tiny_sd"] = ModelConfig(
            model_id="segmind/tiny-sd",
            model_name="Tiny SD",
            model_type=ModelType.TEXT_TO_IMAGE,
            tier=ModelTier.LIGHTWEIGHT,
            min_vram_mb=1000,
            recommended_vram_mb=2000,
            min_ram_mb=4000,
            max_resolution=512,
            default_resolution=(512, 512),
            max_batch_size=4,
            default_steps=10,
            min_steps=1,
            max_steps=20,
            supports_negative_prompt=True,
            supports_guidance_scale=True,
            pipeline_class="StableDiffusionPipeline",
            license="Apache 2.0",
            description="Ultra-lightweight model for low-end hardware",
            release_date="2024-01",
            estimated_speed_512=1.0
        )
        
        return models
    
    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model configuration by ID."""
        return self.models.get(model_id)
    
    def get_models_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        """Get all models of a specific type."""
        return [model for model in self.models.values() if model.model_type == model_type]
    
    def get_models_by_tier(self, tier: ModelTier) -> List[ModelConfig]:
        """Get all models of a specific tier."""
        return [model for model in self.models.values() if model.tier == tier]
    
    def get_compatible_models(self, vram_mb: int, model_type: Optional[ModelType] = None) -> List[ModelConfig]:
        """Get models compatible with available VRAM."""
        compatible = []
        for model in self.models.values():
            if model.min_vram_mb <= vram_mb:
                if model_type is None or model.model_type == model_type:
                    compatible.append(model)
        
        # Sort by tier (quality) and VRAM requirements
        compatible.sort(key=lambda m: (m.tier.value, -m.min_vram_mb))
        return compatible
    
    def get_recommended_model(self, vram_mb: int, model_type: ModelType, 
                            prefer_speed: bool = False) -> Optional[ModelConfig]:
        """Get the recommended model for given constraints."""
        compatible = self.get_compatible_models(vram_mb, model_type)
        
        if not compatible:
            return None
        
        if prefer_speed:
            # Prefer LIGHTWEIGHT tier models
            fast_models = [m for m in compatible if m.tier == ModelTier.LIGHTWEIGHT]
            if fast_models:
                return fast_models[0]
        
        # Default: prefer highest quality that fits comfortably
        for model in reversed(compatible):  # Start with highest quality
            if model.recommended_vram_mb <= vram_mb:
                return model
        
        # Fallback to any compatible model
        return compatible[0]
    
    def get_fallback_model(self, original_model_id: str) -> Optional[ModelConfig]:
        """Get a fallback model if the original fails to load."""
        original = self.get_model(original_model_id)
        if not original:
            return None
        
        # Find a model of the same type with lower requirements
        same_type_models = self.get_models_by_type(original.model_type)
        fallbacks = [m for m in same_type_models 
                    if m.min_vram_mb < original.min_vram_mb]
        
        if fallbacks:
            # Return the one with highest quality among fallbacks
            fallbacks.sort(key=lambda m: -m.min_vram_mb)
            return fallbacks[0]
        
        return None
    
    def list_all_models(self) -> Dict[str, str]:
        """Get a simple mapping of model IDs to names."""
        return {model_id: config.model_name for model_id, config in self.models.items()}
    
    def get_model_info_summary(self, model_id: str) -> Optional[Dict]:
        """Get a summary of model information for UI display."""
        model = self.get_model(model_id)
        if not model:
            return None
        
        return {
            "name": model.model_name,
            "type": model.model_type.value,
            "tier": model.tier.value,
            "vram_min": f"{model.min_vram_mb}MB",
            "vram_recommended": f"{model.recommended_vram_mb}MB",
            "max_resolution": f"{model.max_resolution}px",
            "default_resolution": f"{model.default_resolution[0]}x{model.default_resolution[1]}",
            "description": model.description,
            "license": model.license,
            "estimated_speed_512": f"{model.estimated_speed_512}s" if model.estimated_speed_512 else "Unknown"
        }


# Global registry instance
_model_registry = None

def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


# Convenience functions
def get_latest_image_models() -> List[ModelConfig]:
    """Get the latest image generation models."""
    registry = get_model_registry()
    return registry.get_models_by_type(ModelType.TEXT_TO_IMAGE)


def get_latest_video_models() -> List[ModelConfig]:
    """Get the latest video generation models."""
    registry = get_model_registry()
    video_models = []
    video_models.extend(registry.get_models_by_type(ModelType.IMAGE_TO_VIDEO))
    # Add text-to-video models (some are classified as TEXT_TO_IMAGE but support video)
    text_models = registry.get_models_by_type(ModelType.TEXT_TO_IMAGE)
    video_models.extend([m for m in text_models if m.max_frames is not None])
    return video_models


def recommend_models_for_hardware(vram_mb: int) -> Dict[str, ModelConfig]:
    """Recommend the best models for given hardware."""
    registry = get_model_registry()
    
    recommendations = {}
    
    # Image generation
    image_model = registry.get_recommended_model(vram_mb, ModelType.TEXT_TO_IMAGE)
    if image_model:
        recommendations["image"] = image_model
    
    # Video generation
    video_models = get_latest_video_models()
    compatible_video = [m for m in video_models if m.min_vram_mb <= vram_mb]
    if compatible_video:
        # Prefer image-to-video models
        i2v_models = [m for m in compatible_video if m.model_type == ModelType.IMAGE_TO_VIDEO]
        if i2v_models:
            recommendations["video"] = i2v_models[0]
        else:
            recommendations["video"] = compatible_video[0]
    
    return recommendations


if __name__ == "__main__":
    # Test the registry
    registry = get_model_registry()
    
    print("Available models:")
    for model_id, model_name in registry.list_all_models().items():
        print(f"  {model_id}: {model_name}")
    
    print(f"\nRecommendations for 8GB VRAM:")
    recommendations = recommend_models_for_hardware(8000)
    for category, model in recommendations.items():
        print(f"  {category}: {model.model_name} ({model.model_id})")