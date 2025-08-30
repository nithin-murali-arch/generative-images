"""
Integration layer between the modern UI and the existing system components.

This module provides the bridge between the clean modern interface and the
underlying generation pipelines and system controllers.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import time

logger = logging.getLogger(__name__)

# Try to import system components
try:
    from ..core.interfaces import (
        GenerationRequest, GenerationResult, OutputType, 
        ComplianceMode, StyleConfig, HardwareConfig
    )
    from ..core.model_registry import get_model_registry, ModelType
    from ..pipelines.image_generation import ImageGenerationPipeline
    from ..pipelines.video_generation import VideoGenerationPipeline
    from ..hardware.detector import HardwareDetector
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"System components not available: {e}")
    SYSTEM_AVAILABLE = False
    
    # Mock classes for development
    class GenerationRequest:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class GenerationResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class HardwareConfig:
        def __init__(self):
            self.gpu_model = 'Unknown GPU'
            self.vram_size = 8000  # Default 8GB
            self.cpu_cores = 4
            self.ram_size = 16000  # Default 16GB
            self.cuda_available = False
            self.optimization_level = 'balanced'
    
    class HardwareDetector:
        def detect_hardware(self):
            return HardwareConfig()
    
    OutputType = type('OutputType', (), {'IMAGE': 'image', 'VIDEO': 'video'})
    ComplianceMode = type('ComplianceMode', (), {'RESEARCH_SAFE': 'research_safe'})


class SystemIntegration:
    """
    Integration layer that connects the modern UI to the generation system.
    
    Handles:
    - Model loading and management
    - Hardware detection and optimization
    - Generation request routing
    - Result processing and storage
    """
    
    def __init__(self):
        self.hardware_config: Optional[HardwareConfig] = None
        self.image_pipeline: Optional[ImageGenerationPipeline] = None
        self.video_pipeline: Optional[VideoGenerationPipeline] = None
        self.model_registry = None
        self.is_initialized = False
        
        logger.info("SystemIntegration created")
    
    def initialize(self) -> bool:
        """Initialize the system integration."""
        try:
            logger.info("Initializing system integration...")
            
            if not SYSTEM_AVAILABLE:
                logger.warning("System components not available - running in mock mode")
                self.is_initialized = True
                return True
            
            # Initialize model registry
            self.model_registry = get_model_registry()
            
            # Detect hardware using cross-platform detector
            try:
                from ..core.cross_platform_hardware import detect_cross_platform_hardware
                cross_platform_config = detect_cross_platform_hardware()
                
                # Convert to expected format
                self.hardware_config = type('HardwareConfig', (), {
                    'gpu_model': cross_platform_config.gpu_model,
                    'vram_size': cross_platform_config.vram_size,
                    'cpu_cores': cross_platform_config.cpu_cores,
                    'ram_size': cross_platform_config.ram_total_mb,
                    'cuda_available': cross_platform_config.cuda_available,
                    'optimization_level': cross_platform_config.optimization_level
                })()
                
                logger.info(f"Detected hardware: {self.hardware_config.gpu_model} with {self.hardware_config.vram_size}MB VRAM")
                
            except Exception as e:
                logger.warning(f"Cross-platform hardware detection failed: {e}")
                # Fallback to mock detector
                hardware_detector = HardwareDetector()
                self.hardware_config = hardware_detector.detect_hardware()
            
            # Initialize pipelines
            self.image_pipeline = ImageGenerationPipeline()
            if not self.image_pipeline.initialize(self.hardware_config):
                logger.warning("Image pipeline initialization failed")
            
            self.video_pipeline = VideoGenerationPipeline()
            if not self.video_pipeline.initialize(self.hardware_config):
                logger.warning("Video pipeline initialization failed")
            
            self.is_initialized = True
            logger.info("System integration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System integration initialization failed: {e}")
            return False
    
    def get_available_models(self, model_type: str = "image") -> Dict[str, Dict[str, Any]]:
        """Get available models for the specified type with detailed info."""
        if not self.model_registry:
            # Return mock models for development
            if model_type == "image":
                return {
                    "sd15": {
                        "name": "Stable Diffusion 1.5",
                        "description": "Fast, 4GB VRAM",
                        "tier": "lightweight",
                        "vram_mb": 4000,
                        "download_size_gb": 3.4,
                        "is_downloaded": False,
                        "can_run": True
                    }
                }
            else:
                return {
                    "svd_xt": {
                        "name": "Stable Video Diffusion XT",
                        "description": "High Quality, 12GB VRAM",
                        "tier": "mid_tier",
                        "vram_mb": 12000,
                        "download_size_gb": 9.7,
                        "is_downloaded": False,
                        "can_run": False
                    }
                }
        
        try:
            from ..core.model_downloader import get_model_downloader
            downloader = get_model_downloader()
            
            if model_type == "image":
                models = self.model_registry.get_models_by_type(ModelType.TEXT_TO_IMAGE)
            else:
                models = self.model_registry.get_models_by_type(ModelType.IMAGE_TO_VIDEO)
                # Add text-to-video models
                text_models = self.model_registry.get_models_by_type(ModelType.TEXT_TO_IMAGE)
                models.extend([m for m in text_models if m.max_frames is not None])
            
            # Build detailed model info
            available_models = {}
            user_vram = self.hardware_config.vram_size if self.hardware_config else 8000
            
            for model in models:
                model_key = model.model_name.lower().replace(' ', '_').replace('.', '_')
                can_run = model.min_vram_mb <= user_vram
                is_downloaded = downloader.is_model_downloaded(model.model_id)
                
                # Create tier description
                tier_descriptions = {
                    "lightweight": "💚 Lightweight",
                    "mid_tier": "🟡 Mid-Tier", 
                    "high_end": "🔴 High-End",
                    "ultra": "⚫ Ultra"
                }
                
                tier_desc = tier_descriptions.get(model.tier.value, model.tier.value)
                
                # Build description
                status_parts = []
                if is_downloaded:
                    status_parts.append("✅ Downloaded")
                else:
                    status_parts.append(f"📥 {model.download_size_gb:.1f}GB")
                
                if can_run:
                    status_parts.append(f"🎯 {model.min_vram_mb}MB VRAM")
                else:
                    status_parts.append(f"❌ Needs {model.min_vram_mb}MB VRAM")
                
                description = f"{tier_desc} • {' • '.join(status_parts)}"
                
                available_models[model_key] = {
                    "name": model.model_name,
                    "description": description,
                    "tier": model.tier.value,
                    "vram_mb": model.min_vram_mb,
                    "download_size_gb": model.download_size_gb,
                    "is_downloaded": is_downloaded,
                    "can_run": can_run,
                    "model_id": model.model_id,
                    "estimated_speed": model.estimated_speed_512
                }
            
            # Sort by: can_run (True first), then by tier, then by download priority
            def sort_key(item):
                model_info = item[1]
                tier_order = {"lightweight": 0, "mid_tier": 1, "high_end": 2, "ultra": 3}
                return (
                    not model_info["can_run"],  # False (can run) comes first
                    tier_order.get(model_info["tier"], 99),
                    model_info["vram_mb"]
                )
            
            sorted_models = dict(sorted(available_models.items(), key=sort_key))
            return sorted_models
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return {}
    
    def get_recommended_model(self, model_type: str = "image") -> str:
        """Get the recommended model for current hardware."""
        if not self.model_registry or not self.hardware_config:
            return "sd15" if model_type == "image" else "svd_xt"
        
        try:
            mt = ModelType.TEXT_TO_IMAGE if model_type == "image" else ModelType.IMAGE_TO_VIDEO
            recommended = self.model_registry.get_recommended_model(
                self.hardware_config.vram_size, mt
            )
            
            if recommended:
                return recommended.model_name.lower().replace(' ', '_')
            
        except Exception as e:
            logger.error(f"Failed to get recommended model: {e}")
        
        return "sd15" if model_type == "image" else "svd_xt"
    
    def generate_image(self, prompt: str, **kwargs) -> Tuple[Optional[Any], Dict[str, Any], str]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (image, generation_info, status_message)
        """
        try:
            if not prompt.strip():
                return None, {}, "❌ Please enter a prompt"
            
            logger.info(f"Generating image: {prompt[:50]}...")
            start_time = time.time()
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                output_type=OutputType.IMAGE,
                style_config=StyleConfig(),
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=self.hardware_config or {},
                additional_params=kwargs
            )
            
            # Generate using pipeline
            if self.image_pipeline and SYSTEM_AVAILABLE:
                result = self.image_pipeline.generate(request)
                
                if result.success:
                    generation_info = {
                        "prompt": prompt,
                        "model": result.model_used,
                        "generation_time": f"{result.generation_time:.2f}s",
                        "output_path": str(result.output_path) if result.output_path else None,
                        **kwargs
                    }
                    
                    # Load and return the image
                    if result.output_path and result.output_path.exists():
                        try:
                            from PIL import Image
                            image = Image.open(result.output_path)
                            return image, generation_info, "✅ Image generated successfully!"
                        except Exception as e:
                            logger.error(f"Failed to load generated image: {e}")
                    
                    return None, generation_info, "✅ Image generated (file saved)"
                else:
                    return None, {}, f"❌ Generation failed: {result.error_message}"
            
            else:
                # Mock generation for development
                generation_time = time.time() - start_time
                generation_info = {
                    "prompt": prompt,
                    "model": kwargs.get("model", "stable-diffusion-v1-5"),
                    "generation_time": f"{generation_time:.2f}s",
                    "mode": "mock",
                    **kwargs
                }
                
                return None, generation_info, "✅ Mock generation completed (install dependencies for real generation)"
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None, {}, f"❌ Generation error: {str(e)}"
    
    def generate_video(self, prompt: str, conditioning_image=None, **kwargs) -> Tuple[Optional[Any], Dict[str, Any], str]:
        """
        Generate a video from a text prompt and optional conditioning image.
        
        Args:
            prompt: Text description of the video
            conditioning_image: Optional starting image
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (video_path, generation_info, status_message)
        """
        try:
            if not prompt.strip():
                return None, {}, "❌ Please enter a prompt"
            
            logger.info(f"Generating video: {prompt[:50]}...")
            start_time = time.time()
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                output_type=OutputType.VIDEO,
                style_config=StyleConfig(),
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=self.hardware_config or {},
                additional_params={
                    "conditioning_image": conditioning_image,
                    **kwargs
                }
            )
            
            # Generate using pipeline
            if self.video_pipeline and SYSTEM_AVAILABLE:
                result = self.video_pipeline.generate(request)
                
                if result.success:
                    generation_info = {
                        "prompt": prompt,
                        "model": result.model_used,
                        "generation_time": f"{result.generation_time:.2f}s",
                        "output_path": str(result.output_path) if result.output_path else None,
                        "has_conditioning_image": conditioning_image is not None,
                        **kwargs
                    }
                    
                    return str(result.output_path) if result.output_path else None, generation_info, "✅ Video generated successfully!"
                else:
                    return None, {}, f"❌ Generation failed: {result.error_message}"
            
            else:
                # Mock generation for development
                generation_time = time.time() - start_time
                generation_info = {
                    "prompt": prompt,
                    "model": kwargs.get("model", "stable-video-diffusion"),
                    "generation_time": f"{generation_time:.2f}s",
                    "mode": "mock",
                    "has_conditioning_image": conditioning_image is not None,
                    **kwargs
                }
                
                return None, generation_info, "✅ Mock generation completed (install dependencies for real generation)"
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, {}, f"❌ Generation error: {str(e)}"
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information."""
        if not self.hardware_config:
            return {
                "gpu_model": "Unknown",
                "vram_total": 0,
                "vram_free": 0,
                "cuda_available": False
            }
        
        return {
            "gpu_model": self.hardware_config.gpu_model,
            "vram_total": self.hardware_config.vram_size,
            "vram_free": self.hardware_config.vram_size,  # Simplified
            "cuda_available": self.hardware_config.cuda_available,
            "optimization_level": self.hardware_config.optimization_level
        }
    
    def download_model(self, model_id: str) -> bool:
        """Download a model asynchronously."""
        try:
            from ..core.model_downloader import get_model_downloader
            downloader = get_model_downloader()
            return downloader.download_model(model_id, background=True)
        except Exception as e:
            logger.error(f"Failed to start model download: {e}")
            return False
    
    def get_download_progress(self, model_id: str) -> Dict[str, Any]:
        """Get download progress for a model."""
        try:
            from ..core.model_downloader import get_model_downloader
            downloader = get_model_downloader()
            progress = downloader.get_download_status(model_id)
            
            return {
                "status": progress.status.value,
                "progress_percent": progress.progress_percent,
                "downloaded_mb": progress.downloaded_mb,
                "total_mb": progress.total_mb,
                "speed_mbps": progress.speed_mbps,
                "eta_seconds": progress.eta_seconds,
                "error_message": progress.error_message
            }
        except Exception as e:
            logger.error(f"Failed to get download progress: {e}")
            return {"status": "unknown", "progress_percent": 0}
    
    def auto_download_recommended_models(self) -> List[str]:
        """Automatically download recommended models for current hardware."""
        if not self.hardware_config:
            return []
        
        try:
            from ..core.model_downloader import get_model_downloader
            downloader = get_model_downloader()
            return downloader.download_recommended_models(self.hardware_config.vram_size)
        except Exception as e:
            logger.error(f"Failed to auto-download models: {e}")
            return []
    
    def cleanup(self):
        """Clean up system resources."""
        logger.info("Cleaning up system integration...")
        
        if self.image_pipeline:
            self.image_pipeline.cleanup()
        
        if self.video_pipeline:
            self.video_pipeline.cleanup()
        
        logger.info("System integration cleanup completed")


# Global system integration instance
_system_integration = None

def get_system_integration() -> SystemIntegration:
    """Get the global system integration instance."""
    global _system_integration
    if _system_integration is None:
        _system_integration = SystemIntegration()
        _system_integration.initialize()
    return _system_integration