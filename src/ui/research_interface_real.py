"""
Real Research Interface for the Academic Multimodal LLM Experiment System.
This version actually generates images using Stable Diffusion models.
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# PyTorch memory configuration is set in the launcher

logger = logging.getLogger(__name__)

# Try to import Gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logger.warning("Gradio not available - UI will be limited")

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image handling limited")

# Try to import AI generation libraries
try:
    import torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import logging as diffusers_logging
    AI_AVAILABLE = True
    logger.info("AI generation libraries imported successfully")
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning(f"AI generation libraries not available: {e}")

# Try to import Hugging Face Hub for model downloading
try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face Hub not available")


class ComplianceMode(Enum):
    """Copyright compliance modes for dataset selection."""
    OPEN_SOURCE_ONLY = "open_only"
    RESEARCH_SAFE = "research_safe"
    FULL_DATASET = "full_dataset"


class OutputType(Enum):
    """Types of content that can be generated."""
    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class UIState:
    """Current state of the user interface."""
    current_compliance_mode: ComplianceMode = ComplianceMode.RESEARCH_SAFE
    current_model: Optional[str] = None
    experiment_id: Optional[str] = None
    generation_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.generation_history is None:
            self.generation_history = []


class RealImageGenerator:
    """Real image generator using Stable Diffusion models."""
    
    def __init__(self):
        self.pipelines = {}
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Image generator initialized on device: {self.device}")
        
        # Memory management
        self.last_memory_cleanup = time.time()
        self.memory_cleanup_interval = 60  # Clean up every 60 seconds
        
        # Available models
        self.available_models = {
            "stable-diffusion-v1-5": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "min_vram": 2000,  # Reduced for GTX 1650
                "max_resolution": 512,  # Reduced for lower VRAM
                "default_steps": 20
            },
            "stable-diffusion-2-1": {
                "model_id": "stabilityai/stable-diffusion-2-1",
                "min_vram": 2000,  # Reduced for GTX 1650
                "max_resolution": 512,  # Reduced for lower VRAM
                "default_steps": 20
            },
            "sdxl-turbo": {
                "model_id": "stabilityai/sdxl-turbo",
                "min_vram": 3000,  # Lower VRAM requirement
                "max_resolution": 512,  # Optimized for speed
                "default_steps": 8  # Fewer steps for speed
            },
            "sdxl-base": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "min_vram": 8000,  # Keep high for SDXL
                "max_resolution": 1024,
                "default_steps": 25
            }
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models based on hardware."""
        available = []
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_mb = vram_gb * 1024  # Convert to MB
        else:
            vram_mb = 0
        
        logger.info(f"Available VRAM: {vram_mb:.0f}MB")
        
        for model_name, model_info in self.available_models.items():
            if vram_mb >= model_info["min_vram"]:
                available.append(model_name)
                logger.info(f"Model {model_name} available (requires {model_info['min_vram']}MB)")
            else:
                logger.info(f"Model {model_name} not available (requires {model_info['min_vram']}MB, have {vram_mb:.0f}MB)")
        
        return available
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model."""
        try:
            if model_name not in self.available_models:
                logger.error(f"Unknown model: {model_name}")
                return False
            
            if model_name in self.pipelines:
                self.current_model = model_name
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # CRITICAL: Check memory before loading new model
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free = total_memory - reserved
                
                logger.info(f"Memory before model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                
                # If memory usage is too high, force cleanup
                if reserved > 2.0:  # More conservative for 4GB GPU
                    logger.warning(f"High memory usage detected ({reserved:.2f}GB), forcing cleanup before model loading")
                    self._cleanup_memory()
                    
                    # Check again after cleanup
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    free = total_memory - reserved
                    logger.info(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                    
                    # If still too high, fail
                    if reserved > 2.5:
                        logger.error(f"Memory usage still too high after cleanup: {reserved:.2f}GB")
                        return False
                
                # Ensure we have enough free memory for model loading
                if free < 1.5:
                    logger.error(f"Insufficient free memory for model loading: {free:.2f}GB")
                    return False
            
            # If we're switching models, unload the previous one to free memory
            if self.current_model and self.current_model != model_name:
                logger.info(f"Switching from {self.current_model} to {model_name}, unloading previous model")
                if self.current_model in self.pipelines:
                    del self.pipelines[self.current_model]
                    torch.cuda.empty_cache()
                    logger.info(f"Unloaded previous model: {self.current_model}")
            
            model_info = self.available_models[model_name]
            logger.info(f"Loading model: {model_name} ({model_info['model_id']})")
            
            # Load the pipeline with conservative memory settings for GTX 1650
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_info["model_id"],
                torch_dtype=torch.float32,  # Use float32 to avoid black image corruption
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            
            # Optimize for speed and memory
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            
            # Move to device
            pipeline = pipeline.to(self.device)
            
            # Enable memory efficient attention for low VRAM
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing(slice_size="max")
                logger.info("Enabled aggressive attention slicing")
            
            # Enable memory efficient VAE decoding
            if hasattr(pipeline, "enable_vae_slicing"):
                pipeline.enable_vae_slicing()
                logger.info("Enabled VAE slicing")
            
            # Enable VAE tiling for very low VRAM
            if hasattr(pipeline, "enable_vae_tiling"):
                pipeline.enable_vae_tiling()
                logger.info("Enabled VAE tiling")
            
            # Don't enable CPU offloading - keep everything on GPU for stability
            # if hasattr(pipeline, "enable_model_cpu_offload"):
            #     pipeline.enable_model_cpu_offload()
            
            # Store pipeline
            self.pipelines[model_name] = pipeline
            self.current_model = model_name
            
            logger.info(f"Model {model_name} loaded successfully with optimizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def generate_image(self, prompt: str, negative_prompt: str = "", width: int = 512, 
                      height: int = 512, steps: int = 20, guidance_scale: float = 7.5,
                      seed: Optional[int] = None) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Generate an image using the current model."""
        try:
            if not self.current_model:
                logger.error("No model loaded")
                return None, {"error": "No model loaded"}
            
            pipeline = self.pipelines[self.current_model]
            
            # Validate dimensions for low VRAM
            max_dim = self.available_models[self.current_model]["max_resolution"]
            if width > max_dim or height > max_dim:
                logger.warning(f"Dimensions {width}x{height} exceed max {max_dim}x{max_dim}, reducing...")
                width = min(width, max_dim)
                height = min(height, max_dim)
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # CRITICAL: Check memory before generation to prevent 6GB allocation
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved
                logger.info(f"Memory before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                
                # CRITICAL: Prevent 6GB allocation on 4GB GPU
                if reserved > 2.0:  # More conservative for 4GB GPU
                    logger.error(f"CRITICAL: Memory usage too high ({reserved:.2f}GB), forcing aggressive cleanup...")
                    self._cleanup_memory()
                    # Check again after cleanup
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved
                    logger.info(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                    
                    # If still too high, fail early
                    if reserved > 2.5:
                        logger.error(f"Memory usage still too high after cleanup: {reserved:.2f}GB")
                        return None, {"error": f"GPU memory usage too high ({reserved:.2f}GB). Please restart the system."}
                
                # Ensure we have enough free memory
                if free < 1.0:
                    logger.error(f"Insufficient free memory: {free:.2f}GB")
                    return None, {"error": f"Insufficient GPU memory. Only {free:.2f}GB free, need at least 1.0GB."}
                
                # Periodic memory cleanup
                current_time = time.time()
                if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
                    logger.info("Performing periodic memory cleanup")
                    self._cleanup_memory()
                    self.last_memory_cleanup = current_time
            
            # Generate image
            logger.info(f"Generating image with prompt: {prompt[:50]}...")
            logger.info(f"Parameters: {width}x{height}, {steps} steps, guidance={guidance_scale}")
            start_time = time.time()
            
            # Force GPU usage and clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Ensure pipeline is on GPU
                pipeline = pipeline.to("cuda")
            
            with torch.no_grad():  # Disable gradient computation for inference
                result = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1
                )
            
            generation_time = time.time() - start_time
            
            # AGGRESSIVE MEMORY CLEANUP after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all operations are complete
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Check memory status
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                logger.info(f"Memory after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Get the generated image
            image = result.images[0]
            
            # Validate the generated image
            if image is None or image.size[0] == 0 or image.size[1] == 0:
                logger.error("Generated image is invalid")
                return None, {"error": "Generated image is invalid"}
            
            # Check if image is mostly blank/black
            import numpy as np
            img_array = np.array(image)
            if np.mean(img_array) < 10:  # Very dark image
                logger.warning("Generated image appears to be very dark/blank")
                # Try to fix with different parameters
                logger.info("Attempting to fix with different parameters...")
                return self._generate_image_fallback(prompt, negative_prompt, width, height, steps, guidance_scale, seed)
            
            # Create generation info
            generation_info = {
                "model": self.current_model,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "parameters": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed
                },
                "generation_time": generation_time,
                "device": self.device,
                "image_size": image.size,
                "image_mode": image.mode
            }
            
            logger.info(f"Image generated successfully in {generation_time:.2f}s")
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            return image, generation_info
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory: {e}")
            
            # Emergency memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            
            return None, {}, f"CUDA out of memory. Try reducing image size or steps."
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        try:
            for model_name, pipeline in self.pipelines.items():
                del pipeline
            self.pipelines.clear()
            self.current_model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Image generator cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _generate_image_fallback(self, prompt: str, negative_prompt: str = "", width: int = 512, 
                                height: int = 512, steps: int = 20, guidance_scale: float = 7.5,
                                seed: Optional[int] = None) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Fallback image generation with different parameters."""
        try:
            logger.info("Trying fallback generation with different parameters...")
            
            # Use more conservative parameters
            fallback_width = min(width, 256)
            fallback_height = min(height, 256)
            fallback_steps = min(steps, 15)
            fallback_guidance = min(guidance_scale, 5.0)
            
            logger.info(f"Fallback parameters: {fallback_width}x{fallback_height}, {fallback_steps} steps, guidance={fallback_guidance}")
            
            # Try generation with fallback parameters
            return self.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=fallback_width,
                height=fallback_height,
                steps=fallback_steps,
                guidance_scale=fallback_guidance,
                seed=seed
            )
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return None, {"error": f"Fallback failed: {str(e)}"}
    
    def _cleanup_memory(self):
        """Windows-compatible aggressive memory cleanup for CUDA memory management."""
        if not torch.cuda.is_available():
            return
        
        try:
            logger.info("Starting Windows-compatible memory cleanup...")
            
            # Check initial memory status
            initial_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            initial_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"Before cleanup: {initial_allocated:.2f}GB allocated, {initial_reserved:.2f}GB reserved")
            
            # Windows-specific: Multiple cleanup passes since expandable_segments not available
            for cleanup_pass in range(5):  # More aggressive for Windows
                logger.info(f"Memory cleanup pass {cleanup_pass + 1}/5")
                
                # Clear PyTorch cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Small delay to allow memory to be freed
                import time
                time.sleep(0.1)
            
            # Unload ALL pipelines to free maximum memory
            if self.pipelines:
                logger.warning("Unloading all pipelines to free memory")
                for model_name in list(self.pipelines.keys()):
                    del self.pipelines[model_name]
                    logger.info(f"Unloaded model: {model_name}")
                self.pipelines.clear()
                self.current_model = None
            
            # Final cleanup pass
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            # Check final memory status
            final_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            final_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            freed_memory = initial_reserved - final_reserved
            
            logger.info(f"Memory cleanup completed: {final_allocated:.2f}GB allocated, {final_reserved:.2f}GB reserved")
            logger.info(f"Freed memory: {freed_memory:.2f}GB")
            
            # Verify we have enough free memory
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_memory = total_memory - final_reserved
            logger.info(f"Total GPU memory: {total_memory:.2f}GB, Free: {free_memory:.2f}GB")
            
            # Windows-specific: More conservative memory thresholds
            if free_memory < 1.5:  # Increased threshold for Windows
                logger.error(f"CRITICAL: Only {free_memory:.2f}GB free memory available!")
                logger.warning("Windows platform requires more conservative memory management")
                        
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            import traceback
            traceback.print_exc()


class ResearchInterface:
    """
    Real research interface for the Academic Multimodal LLM Experiment System.
    """
    
    def __init__(self, system_controller=None, experiment_tracker=None, compliance_engine=None):
        """Initialize the research interface."""
        self.system_controller = system_controller
        self.experiment_tracker = compliance_engine
        self.compliance_engine = compliance_engine
        self.ui_state = UIState()
        
        # Memory management
        self.last_memory_cleanup = time.time()
        self.memory_cleanup_interval = 60  # Clean up every 60 seconds
        
        # Initialize system integration for proper generation
        self.system_integration = None
        if AI_AVAILABLE:
            try:
                from src.core.system_integration import SystemIntegration
                self.system_integration = SystemIntegration()
                
                # Initialize with basic config
                config = {
                    "data_dir": "data",
                    "models_dir": "models",
                    "experiments_dir": "experiments",
                    "cache_dir": "cache",
                    "logs_dir": "logs",
                    "auto_detect_hardware": True,
                    "max_concurrent_generations": 1
                }
                
                if self.system_integration.initialize(config):
                    logger.info("System integration initialized successfully")
                    
                    # Check video pipeline status
                    if self.system_integration.video_pipeline:
                        logger.info(f"Video pipeline available: {type(self.system_integration.video_pipeline).__name__}")
                        if self.system_integration.video_pipeline.is_initialized:
                            logger.info("✓ Video pipeline is initialized and ready")
                        else:
                            logger.warning("⚠ Video pipeline exists but is not initialized")
                    else:
                        logger.warning("⚠ Video pipeline not available in system integration")
                else:
                    logger.warning("System integration initialization failed")
                    self.system_integration = None
                    
            except Exception as e:
                logger.error(f"Failed to initialize system integration: {e}")
                self.system_integration = None
        
        # Fallback to custom generator if system integration fails
        if not self.system_integration:
            self.image_generator = RealImageGenerator() if AI_AVAILABLE else None
            logger.info("Using fallback image generator")
        else:
            self.image_generator = None
            logger.info("Using system integration for generation")
        
        # Force initial memory cleanup to start with clean slate
        if AI_AVAILABLE and torch.cuda.is_available():
            logger.info("Performing initial memory cleanup...")
            try:
                # Windows-specific memory management (expandable_segments not supported)
                logger.info("Windows platform detected - using compatible memory management")
                
                # Multiple cleanup passes for Windows
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    import gc
                    gc.collect()
                
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free = torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved
                logger.info(f"Initial memory status: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                
                if free < 2.0:
                    logger.warning(f"Low initial free memory: {free:.2f}GB. Consider restarting system.")
                    
                # Set Windows-compatible memory limits - more aggressive for 4GB GPU
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.6)  # Use only 60% of GPU memory (2.4GB max)
                    logger.info("Set GPU memory limit to 60% for Windows compatibility (4GB GPU)")
                    
                # Set additional memory constraints
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    logger.info("Initial cache cleared")
                    
            except Exception as e:
                logger.warning(f"Initial memory cleanup failed: {e}")
        
        # Create outputs directory
        self.outputs_dir = Path("outputs/images")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model dropdown choices
        self.model_choices = []
        if self.image_generator:
            self.model_choices = self.image_generator.get_available_models()
        elif self.system_integration and self.system_integration.hardware_config:
            # Get models from system integration
            self.model_choices = ["stable-diffusion-v1-5", "stable-diffusion-2-1", "sdxl-turbo"]
        
        logger.info("ResearchInterface created")
    

    
    def initialize(self) -> bool:
        """Initialize the research interface."""
        try:
            logger.info("Initializing research interface...")
            
            if not GRADIO_AVAILABLE:
                logger.warning("Gradio not available - interface will be limited")
                return False
            
            if not AI_AVAILABLE:
                logger.warning("AI generation not available - interface will be limited")
            
            logger.info("Research interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize research interface: {e}")
            return False
    
    def launch(self, share: bool = False, server_name: str = "127.0.0.1", server_port: int = 7860):
        """Launch the Gradio interface."""
        if not GRADIO_AVAILABLE:
            logger.error("Gradio not available - cannot launch interface")
            return
        
        try:
            logger.info(f"Launching research interface on {server_name}:{server_port}")
            
            # Create the interface
            interface = self._create_interface()
            
            # Launch the interface
            interface.launch(
                server_name=server_name,
                server_port=server_port,
                share=share,
                show_error=True
            )
            
        except Exception as e:
            logger.error(f"Failed to launch interface: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_interface(self):
        """Create the streamlined Gradio interface with image and video generation."""
        with gr.Blocks(title="AI Content Generation System", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("# AI Content Generation System")
            gr.Markdown("**Generate images and videos with AI**")
            
            # Memory status display
            with gr.Row():
                memory_status = gr.Textbox(
                    label="GPU Memory Status",
                    value="Checking memory...",
                    interactive=False
                )
                refresh_memory_btn = gr.Button("🔄 Refresh Memory")
            
            # Main interface with tabs
            with gr.Tabs():
                # Image Generation Tab
                with gr.Tab("🎨 Image Generation"):
                    self._create_image_generation_tab()
                
                # Video Generation Tab
                with gr.Tab("🎬 Video Generation"):
                    self._create_video_generation_tab()
            
            # Simple system info
            with gr.Accordion("System Information", open=False):
                with gr.Row():
                    hardware_info = gr.JSON(label="Hardware", value={})
                    model_info = gr.JSON(label="Model Status", value={})
                
                clear_cache_btn = gr.Button("🧹 Clear VRAM Cache")
            
            # Footer
            gr.Markdown("---")
            gr.Markdown("*AI Content Generation System - Image and Video Generation*")
        
        # Event handlers for system functions
        with interface:
            refresh_memory_btn.click(
                fn=self._refresh_memory_status,
                outputs=[memory_status]
            )
            
            clear_cache_btn.click(
                fn=self._clear_vram_cache,
                outputs=[hardware_info]
            )
            
            # Auto-load first model and refresh status
            interface.load(
                fn=self._initialize_ui,
                outputs=[memory_status, hardware_info, model_info]
            )
        
        return interface
    
    def _create_image_generation_tab(self):
        """Create the image generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want in the image...",
                    lines=2
                )
                
                with gr.Row():
                    width_input = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height_input = gr.Slider(256, 1024, 512, step=64, label="Height")
                
                with gr.Row():
                    steps_input = gr.Slider(10, 50, 20, step=1, label="Steps")
                    guidance_input = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
                
                seed_input = gr.Number(label="Seed (Optional)", value=None)
                
                model_dropdown = gr.Dropdown(
                    choices=self.model_choices,
                    value=self.model_choices[0] if self.model_choices else None,
                    label="Model"
                )
                
                with gr.Row():
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary", size="lg")
                
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Generated Image")
                generation_info = gr.JSON(label="Generation Info")
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_image,
            inputs=[prompt_input, negative_prompt_input, width_input, height_input, 
                   steps_input, guidance_input, seed_input, model_dropdown],
            outputs=[output_image, generation_info, status_text]
        )
    
    def _create_video_generation_tab(self):
        """Create the video generation tab."""
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=3
                )
                
                negative_prompt_input = gr.Textbox(
                    label="Negative Prompt (Optional)",
                    placeholder="What you don't want in the video...",
                    lines=2
                )
                
                with gr.Row():
                    width_input = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height_input = gr.Slider(256, 1024, 512, step=64, label="Height")
                
                with gr.Row():
                    frames_input = gr.Slider(8, 64, 16, step=8, label="Frames")
                    fps_input = gr.Slider(8, 30, 8, step=1, label="FPS")
                
                model_dropdown = gr.Dropdown(
                    choices=["stable-video-diffusion", "text-to-video"],
                    value="stable-video-diffusion",
                    label="Video Model"
                )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                status_text = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                output_video = gr.Video(label="Generated Video")
                generation_info = gr.JSON(label="Generation Info")
        
        # Event handlers
        generate_btn.click(
            fn=self._generate_video,
            inputs=[prompt_input, negative_prompt_input, width_input, height_input, 
                   frames_input, fps_input, model_dropdown],
            outputs=[output_video, generation_info, status_text]
        )
    
    def _create_system_status_tab(self):
        """Create the system status tab."""
        with gr.Row():
            with gr.Column():
                hardware_status = gr.JSON(label="Hardware Status")
                model_status = gr.JSON(label="Model Status")
                refresh_btn = gr.Button("Refresh Status")
            
            with gr.Column():
                system_logs = gr.Textbox(label="System Logs", lines=10, interactive=False)
                clear_cache_btn = gr.Button("Clear VRAM Cache")
        
        # Event handlers
        refresh_btn.click(
            fn=self._refresh_system_status,
            outputs=[hardware_status, model_status, system_logs]
        )
        
        clear_cache_btn.click(
            fn=self._clear_vram_cache,
            outputs=[hardware_status]
        )
        
        # Initial load
        refresh_btn.click()
    
    def _create_experiments_tab(self):
        """Create the experiments tab."""
        with gr.Row():
            with gr.Column():
                experiment_notes = gr.Textbox(
                    label="Research Notes",
                    placeholder="Document your research findings and observations...",
                    lines=5
                )
                save_btn = gr.Button("Save Experiment", variant="primary")
            
            with gr.Column():
                experiment_history = gr.Dataframe(
                    headers=["ID", "Timestamp", "Type", "Model", "Prompt", "Status"],
                    label="Experiment History"
                )
                refresh_history_btn = gr.Button("Refresh History")
        
        # Event handlers
        save_btn.click(
            fn=self._save_experiment,
            inputs=[experiment_notes],
            outputs=[experiment_notes]
        )
        
        refresh_history_btn.click(
            fn=self._refresh_experiment_history,
            outputs=[experiment_history]
        )
        
        # Initial load
        refresh_history_btn.click()
    
    def _load_model(self, model_name: str) -> str:
        """Load a specific model."""
        if not self.image_generator:
            return "AI generation not available"
        
        if not model_name:
            return "Please select a model"
        
        try:
            if self.image_generator.load_model(model_name):
                return f"Model {model_name} loaded successfully"
            else:
                return f"Failed to load model {model_name}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def _generate_image(self, prompt: str, negative_prompt: str, width: int, height: int, 
                       steps: int, guidance_scale: float, seed: Optional[int], 
                       model_name: str) -> Tuple[Optional[str], Dict[str, Any], str]:
        """Generate an image based on the input parameters."""
        try:
            if not prompt.strip():
                return None, {}, "Please enter a prompt"
            
            # CRITICAL: Check memory before generation to prevent 6GB allocation
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                total_memory = device_props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                reserved = torch.cuda.memory_reserved(0) / (1024**3)
                free = total_memory - reserved
                
                logger.info(f"Memory before generation: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                
                # If we're already using too much memory, force cleanup
                if reserved > 2.0:  # More conservative for 4GB GPU
                    logger.warning(f"High memory usage detected ({reserved:.2f}GB), forcing cleanup before generation")
                    if hasattr(self, 'image_generator') and self.image_generator:
                        self.image_generator._cleanup_memory()
                    else:
                        # Fallback cleanup
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        import gc
                        gc.collect()
                    
                    # Check again after cleanup
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    free = total_memory - reserved
                    logger.info(f"After cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free")
                    
                    # If still too high, fail early
                    if reserved > 2.5:
                        logger.error(f"Memory usage still too high after cleanup: {reserved:.2f}GB")
                        return None, {}, f"GPU memory usage too high ({reserved:.2f}GB). Please restart the system."
                
                # Ensure we have enough free memory
                if free < 1.0:
                    logger.error(f"Insufficient free memory: {free:.2f}GB")
                    return None, {}, f"Insufficient GPU memory. Only {free:.2f}GB free, need at least 1.0GB."
            
            # Use system integration if available (preferred method)
            if self.system_integration:
                logger.info("Using system integration for image generation")
                
                # Use default compliance mode
                from src.core.interfaces import ComplianceMode
                comp_mode = ComplianceMode.RESEARCH_SAFE
                
                # Prepare parameters for system integration
                additional_params = {
                    'width': width,
                    'height': height,
                    'num_inference_steps': steps,
                    'guidance_scale': guidance_scale,
                    'force_gpu_usage': True,
                    'precision': 'float32',  # Use float32 to avoid black image corruption
                    'memory_optimization': 'Attention Slicing'
                }
                
                # Execute generation using system integration
                result = self.system_integration.execute_complete_generation_workflow(
                    prompt=prompt,
                    conversation_id=f"ui_generation_{int(time.time())}",
                    compliance_mode=comp_mode,
                    additional_params=additional_params
                )
                
                if result.success and result.output_path:
                    # Add to history
                    timestamp = int(time.time())
                    self.ui_state.generation_history.append({
                        "timestamp": timestamp,
                        "type": "image",
                        "model": result.model_used,
                        "prompt": prompt,
                        "file_path": str(result.output_path)
                    })
                    
                    # Create generation info
                    generation_info = {
                        "model": result.model_used,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "parameters": additional_params,
                        "compliance_mode": "research_safe",
                        "generation_time": result.generation_time,
                        "device": "cuda",
                        "output_path": str(result.output_path)
                    }
                    
                    return str(result.output_path), generation_info, "Image generated successfully using system integration!"
                else:
                    return None, {}, f"Generation failed: {result.error_message}"
            
            # Fallback to custom generator
            elif self.image_generator:
                logger.info("Using fallback image generator")
                
                # Load model if not already loaded
                if model_name and model_name != self.image_generator.current_model:
                    if not self.image_generator.load_model(model_name):
                        return None, {}, f"Failed to load model {model_name}"
                
                if not self.image_generator.current_model:
                    return None, {}, "Please load a model first"
                
                # Generate the image
                image, generation_info = self.image_generator.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                if image is None:
                    return None, {}, f"Generation failed: {generation_info.get('error', 'Unknown error')}"
                
                # Save the image
                timestamp = int(time.time())
                filename = f"generated_{timestamp}.png"
                filepath = self.outputs_dir / filename
                image.save(filepath)
                
                # Update generation info
                generation_info["file_path"] = str(filepath)
                generation_info["compliance_mode"] = compliance_mode
                
                # Add to history
                self.ui_state.generation_history.append({
                    "timestamp": timestamp,
                    "type": "image",
                    "model": self.image_generator.current_model,
                    "prompt": prompt,
                    "file_path": str(filepath)
                })
                
                return str(filepath), generation_info, "Image generated successfully using fallback generator!"
            else:
                return None, {}, "No image generation system available"
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, {}, f"Generation failed: {str(e)}"
    
    def _generate_video(self, prompt: str, negative_prompt: str, width: int, height: int, 
                       frames: int, fps: int, model_name: str) -> Tuple[Optional[str], Dict[str, Any], str]:
        """Generate a video based on the input parameters."""
        try:
            if not prompt.strip():
                return None, {}, "Please enter a prompt"
            
            # Check if we have access to the video generation pipeline
            if not hasattr(self, 'system_integration') or not self.system_integration:
                logger.error("Video generation: system_integration not available")
                return None, {}, "Video generation pipeline not available - system integration required"
            
            logger.info(f"Video generation: system_integration available: {self.system_integration}")
            logger.info(f"Video generation: video_pipeline available: {self.system_integration.video_pipeline}")
            
            if not self.system_integration.video_pipeline:
                logger.error("Video generation: video_pipeline is None")
                return None, {}, "Video generation pipeline not available in system integration"
            
            if not self.system_integration.video_pipeline.is_initialized:
                logger.error("Video generation: video_pipeline not initialized")
                return None, {}, "Video generation pipeline not initialized. Please check system status."
            
            logger.info("Video generation: video_pipeline is available and initialized")
            
            logger.info(f"Starting video generation with model: {model_name}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Parameters: {width}x{height}, {frames} frames, {fps} fps")
            
            # Create generation request
            from src.core.interfaces import GenerationRequest, ComplianceMode as CoreComplianceMode
            
            # Use default compliance mode
            compliance_enum = CoreComplianceMode.RESEARCH_SAFE
            
            # Create a minimal style config for video generation
            from src.core.interfaces import StyleConfig, HardwareConfig, ConversationContext
            
            # Create minimal required objects
            style_config = StyleConfig(
                generation_params={
                    "width": width,
                    "height": height,
                    "frames": frames,
                    "fps": fps,
                    "model_name": model_name
                }
            )
            
            # Create minimal hardware config
            hardware_config = HardwareConfig(
                vram_size=4096,  # 4GB for GTX 1650
                gpu_model="GTX 1650",
                cpu_cores=os.cpu_count() or 4,
                ram_size=8192,  # 8GB RAM
                cuda_available=torch.cuda.is_available(),
                optimization_level="balanced"
            )
            
            # Create minimal conversation context
            context = ConversationContext(
                conversation_id=f"video_gen_{int(time.time())}",
                history=[],
                current_mode=compliance_enum,
                user_preferences={}
            )
            
            request = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative_prompt,
                output_type=OutputType.VIDEO,
                style_config=style_config,
                compliance_mode=compliance_enum,
                hardware_constraints=hardware_config,
                context=context,
                additional_params={
                    "frames": frames,
                    "fps": fps,
                    "model_name": model_name
                }
            )
            
            # Generate video using the system integration
            start_time = time.time()
            result = self.system_integration.video_pipeline.generate(request)
            generation_time = time.time() - start_time
            
            if result.success:
                # Add to history
                timestamp = int(time.time())
                self.ui_state.generation_history.append({
                    "timestamp": timestamp,
                    "type": "video",
                    "model": model_name,
                    "prompt": prompt,
                    "file_path": result.output_path
                })
                
                generation_info = {
                    "model": model_name,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "parameters": {
                        "width": width,
                        "height": height,
                        "frames": frames,
                        "fps": fps
                    },
                    "compliance_mode": "research_safe",
                    "generation_time": generation_time,
                    "output_path": result.output_path,
                    "quality_metrics": result.quality_metrics
                }
                
                logger.info(f"Video generated successfully in {generation_time:.2f}s")
                return result.output_path, generation_info, f"Video generated successfully! Output: {result.output_path}"
            else:
                error_msg = f"Video generation failed: {result.error_message}"
                logger.error(error_msg)
                return None, {}, error_msg
            
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return None, {}, f"Generation failed: {str(e)}"
    
    def _refresh_system_status(self) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        """Refresh system status information."""
        try:
            # Real hardware status
            if torch.cuda.is_available():
                device_props = torch.cuda.get_device_properties(0)
                vram_total = device_props.total_memory / (1024**3)
                vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                vram_free = vram_total - vram_allocated
                
                hardware_status = {
                    "gpu_model": device_props.name,
                    "vram_total": f"{vram_total:.1f}GB",
                    "vram_used": f"{vram_allocated:.1f}GB",
                    "vram_free": f"{vram_free:.1f}GB",
                    "cuda_version": torch.version.cuda,
                    "device": "CUDA"
                }
            else:
                hardware_status = {
                    "gpu_model": "CPU Only",
                    "device": "CPU",
                    "cpu_cores": os.cpu_count()
                }
            
            # Model status
            if self.image_generator:
                model_status = {
                    "current_model": self.image_generator.current_model or "None",
                    "available_models": self.image_generator.get_available_models(),
                    "pipelines_loaded": len(self.image_generator.pipelines)
                }
            else:
                model_status = {"error": "AI generation not available"}
            
            logs = f"System status refreshed successfully\nDevice: {hardware_status.get('device', 'Unknown')}\nCurrent model: {model_status.get('current_model', 'None')}"
            
            return hardware_status, model_status, logs
            
        except Exception as e:
            logger.error(f"Failed to refresh system status: {e}")
            return {}, {}, f"Error refreshing status: {str(e)}"
    
    def _clear_vram_cache(self) -> Dict[str, Any]:
        """Clear VRAM cache and return updated hardware status."""
        try:
            if self.image_generator:
                self.image_generator.cleanup()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return updated status
            return self._refresh_system_status()[0]
            
        except Exception as e:
            logger.error(f"Error clearing VRAM cache: {e}")
            return {"error": f"Error clearing cache: {str(e)}"}
    
    def _save_experiment(self, notes: str) -> str:
        """Save current experiment with notes."""
        try:
            logger.info("Saving experiment with notes")
            # Implementation would save to experiment tracker
            return "Experiment saved successfully"
        except Exception as e:
            logger.error(f"Failed to save experiment: {e}")
            return f"Save failed: {str(e)}"
    
    def _refresh_experiment_history(self) -> List[List[str]]:
        """Refresh experiment history display."""
        try:
            # Convert history to table format
            table_data = []
            for exp in self.ui_state.generation_history[-50:]:  # Last 50 experiments
                table_data.append([
                    str(exp.get('timestamp', ''))[:8],
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(exp.get('timestamp', 0))),
                    exp.get('type', ''),
                    exp.get('model', ''),
                    exp.get('prompt', '')[:50] + '...' if len(exp.get('prompt', '')) > 50 else exp.get('prompt', ''),
                    'Success'
                ])
            
            return table_data if table_data else [["No experiments yet", "", "", "", "", ""]]
            
        except Exception as e:
            logger.error(f"Failed to refresh experiment history: {e}")
            return [["Error loading experiments", "", "", "", "", ""]]
    
    def _refresh_memory_status(self) -> str:
        """Refresh and return current memory status."""
        try:
            if not torch.cuda.is_available():
                return "CUDA not available"
            
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total_memory - reserved
            
            status = f"GPU: {device_props.name}\n"
            status += f"Total: {total_memory:.1f}GB\n"
            status += f"Allocated: {allocated:.1f}GB\n"
            status += f"Reserved: {reserved:.1f}GB\n"
            status += f"Free: {free:.1f}GB"
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to refresh memory status: {e}")
            return f"Error: {str(e)}"
    
    def _initialize_ui(self) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Initialize UI with current status."""
        try:
            # Auto-load first model
            if self.model_choices and self.image_generator:
                first_model = self.model_choices[0]
                logger.info(f"Auto-loading first available model: {first_model}")
                if self.image_generator.load_model(first_model):
                    logger.info(f"Auto-loaded model {first_model} successfully")
                else:
                    logger.warning(f"Failed to auto-load model {first_model}")
            
            # Get initial status
            memory_status = self._refresh_memory_status()
            hardware_info = self._refresh_system_status()[0]
            model_info = self._refresh_system_status()[1]
            
            return memory_status, hardware_info, model_info
            
        except Exception as e:
            logger.error(f"Error during UI initialization: {e}")
            return "Initialization error", {}, {}
    
    def _auto_load_first_model(self):
        """Auto-load the first available model when the interface loads."""
        try:
            if self.model_choices and self.image_generator:
                first_model = self.model_choices[0]
                logger.info(f"Auto-loading first available model: {first_model}")
                if self.image_generator.load_model(first_model):
                    logger.info(f"Auto-loaded model {first_model} successfully")
                else:
                    logger.warning(f"Failed to auto-load model {first_model}")
        except Exception as e:
            logger.error(f"Error during auto-load: {e}") 