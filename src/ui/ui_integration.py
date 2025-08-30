"""
Integration layer between the modern UI and the existing system components.

This module provides the bridge between the clean modern interface and the
underlying generation pipelines and system controllers.
"""

import logging
import asyncio
import platform
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
    # Hardware detection is now in cross_platform_hardware
    SYSTEM_AVAILABLE = True
except ImportError as e:
    logger.warning(f"System components not available: {e}")
    SYSTEM_AVAILABLE = False
    
    # CRITICAL: NO MOCKS ALLOWED - System must fail if components unavailable
    logger.error("CRITICAL: Core system components not available")
    logger.error("Cannot proceed without real hardware detection and thermal monitoring")
    raise RuntimeError("System components unavailable - cannot operate safely without real hardware detection")


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
        """Initialize the system integration with mandatory safety checks."""
        try:
            logger.info("Initializing system integration with safety checks...")
            
            if not SYSTEM_AVAILABLE:
                logger.error("CRITICAL: System components not available")
                raise RuntimeError("Cannot operate without core system components")
            
            # MANDATORY: Initialize thermal monitoring FIRST
            from ..core.thermal_monitor import get_thermal_monitor, ensure_startup_thermal_safety
            
            logger.info("Starting thermal monitoring...")
            if not ensure_startup_thermal_safety():
                raise RuntimeError("System too hot - cannot proceed safely")
            
            # MANDATORY: Real hardware detection with validation
            from ..core.cross_platform_hardware import detect_cross_platform_hardware
            
            logger.info("Detecting hardware configuration...")
            cross_platform_config = detect_cross_platform_hardware()
            
            # CRITICAL: Validate hardware detection results
            if cross_platform_config.vram_size <= 0:
                raise RuntimeError(f"Invalid VRAM detection: {cross_platform_config.vram_size}MB")
            
            if cross_platform_config.gpu_model == "Unknown GPU":
                raise RuntimeError("GPU model detection failed - cannot proceed safely")
            
            if cross_platform_config.ram_total_mb <= 0:
                raise RuntimeError(f"Invalid RAM detection: {cross_platform_config.ram_total_mb}MB")
            
            # Convert to expected format (NO defaults, only real values)
            self.hardware_config = type('HardwareConfig', (), {
                'gpu_model': cross_platform_config.gpu_model,
                'vram_size': cross_platform_config.vram_size,
                'cpu_cores': cross_platform_config.cpu_cores,
                'ram_size': cross_platform_config.ram_total_mb,
                'cuda_available': cross_platform_config.cuda_available,
                'optimization_level': cross_platform_config.optimization_level
            })()
            
            logger.info(f"Hardware validated: {self.hardware_config.gpu_model} with {self.hardware_config.vram_size}MB VRAM")
            
            # Initialize model registry
            self.model_registry = get_model_registry()
            
            # Initialize pipelines with thermal monitoring
            logger.info("Initializing generation pipelines...")
            
            self.image_pipeline = ImageGenerationPipeline()
            if not self.image_pipeline.initialize(self.hardware_config):
                raise RuntimeError("Image pipeline initialization failed")
            
            self.video_pipeline = VideoGenerationPipeline()
            if not self.video_pipeline.initialize(self.hardware_config):
                raise RuntimeError("Video pipeline initialization failed")
            
            # Final thermal check before marking as initialized
            thermal_monitor = get_thermal_monitor()
            if not thermal_monitor.is_safe_for_startup():
                raise RuntimeError("System thermal state unsafe after initialization")
            
            # Log thermal status for initialization
            summary = thermal_monitor.get_thermal_summary()
            if summary.get("hot_components"):
                logger.warning(f"Hot components detected during initialization: {', '.join(summary['hot_components'])}")
                logger.info("System initialized but AI workloads will be monitored for thermal safety")
            
            self.is_initialized = True
            logger.info("System integration initialized successfully with thermal safety")
            return True
            
        except Exception as e:
            logger.error(f"CRITICAL: System integration initialization failed: {e}")
            logger.error(f"System: {platform.system()}")
            logger.error(f"Architecture: {platform.machine()}")
            raise RuntimeError(f"System initialization failed: {e}")
    
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
        Generate an image from a text prompt with thermal safety checks.
        
        Args:
            prompt: Text description of the image
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (image, generation_info, status_message)
        """
        try:
            if not prompt.strip():
                return None, {}, "❌ Please enter a prompt"
            
            # MANDATORY: Thermal safety check before generation
            from ..core.thermal_monitor import get_thermal_monitor
            
            thermal_monitor = get_thermal_monitor()
            if not thermal_monitor.is_safe_for_ai_workload():
                logger.warning("System too hot for AI generation - waiting for cooling")
                if not thermal_monitor.wait_for_cooling():
                    return None, {}, "❌ System too hot - generation cancelled for safety"
            
            logger.info(f"Generating image: {prompt[:50]}...")
            start_time = time.time()
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                output_type=OutputType.IMAGE,
                style_config=StyleConfig(),
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=self.hardware_config,
                additional_params=kwargs
            )
            
            # Generate using pipeline (NO FALLBACKS)
            if not self.image_pipeline:
                raise RuntimeError("Image pipeline not available")
            
            result = self.image_pipeline.generate(request)
            
            if result.success:
                generation_info = {
                    "prompt": prompt,
                    "model": result.model_used,
                    "generation_time": f"{result.generation_time:.2f}s",
                    "output_path": str(result.output_path) if result.output_path else None,
                    "thermal_safe": thermal_monitor.is_safe_for_ai_workload(),
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
                        raise RuntimeError(f"Failed to load generated image: {e}")
                
                return None, generation_info, "✅ Image generated (file saved)"
            else:
                logger.error(f"Image generation failed: {result.error_message}")
                return None, {}, f"❌ Generation failed: {result.error_message}"
            
        except Exception as e:
            logger.error(f"CRITICAL: Image generation failed: {e}")
            return None, {}, f"❌ Generation error: {str(e)}"
    
    def generate_video(self, prompt: str, conditioning_image=None, **kwargs) -> Tuple[Optional[Any], Dict[str, Any], str]:
        """
        Generate a video from a text prompt with thermal safety checks.
        
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
            
            # MANDATORY: Thermal safety check before generation
            from ..core.thermal_monitor import get_thermal_monitor
            
            thermal_monitor = get_thermal_monitor()
            if not thermal_monitor.is_safe_for_ai_workload():
                logger.warning("System too hot for video generation - waiting for cooling")
                if not thermal_monitor.wait_for_cooling():
                    return None, {}, "❌ System too hot - video generation cancelled for safety"
            
            logger.info(f"Generating video: {prompt[:50]}...")
            start_time = time.time()
            
            # Create generation request
            request = GenerationRequest(
                prompt=prompt,
                output_type=OutputType.VIDEO,
                style_config=StyleConfig(),
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=self.hardware_config,
                additional_params={
                    "conditioning_image": conditioning_image,
                    **kwargs
                }
            )
            
            # Generate using pipeline (NO FALLBACKS)
            if not self.video_pipeline:
                raise RuntimeError("Video pipeline not available")
            
            result = self.video_pipeline.generate(request)
            
            if result.success:
                generation_info = {
                    "prompt": prompt,
                    "model": result.model_used,
                    "generation_time": f"{result.generation_time:.2f}s",
                    "output_path": str(result.output_path) if result.output_path else None,
                    "has_conditioning_image": conditioning_image is not None,
                    "thermal_safe": thermal_monitor.is_safe_for_ai_workload(),
                    **kwargs
                }
                
                return str(result.output_path) if result.output_path else None, generation_info, "✅ Video generated successfully!"
            else:
                logger.error(f"Video generation failed: {result.error_message}")
                return None, {}, f"❌ Generation failed: {result.error_message}"
            
        except Exception as e:
            logger.error(f"CRITICAL: Video generation failed: {e}")
            return None, {}, f"❌ Generation error: {str(e)}"
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get current hardware information with thermal data."""
        if not self.hardware_config:
            raise RuntimeError("Hardware configuration not available")
        
        # Get real-time thermal information
        try:
            from ..core.thermal_monitor import get_thermal_monitor
            thermal_monitor = get_thermal_monitor()
            thermal_summary = thermal_monitor.get_thermal_summary()
        except Exception as e:
            logger.error(f"Failed to get thermal information: {e}")
            thermal_summary = {"status": "error", "safe": False}
        
        return {
            "gpu_model": self.hardware_config.gpu_model,
            "vram_total": self.hardware_config.vram_size,
            "vram_free": self.hardware_config.vram_size,  # TODO: Get real VRAM usage
            "cuda_available": self.hardware_config.cuda_available,
            "optimization_level": self.hardware_config.optimization_level,
            "thermal_status": thermal_summary["status"],
            "thermal_safe": thermal_summary["safe"],
            "max_temperature": thermal_summary.get("max_temp", 0),
            "avg_temperature": thermal_summary.get("avg_temp", 0)
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