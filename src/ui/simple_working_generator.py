"""
Simple Working Image Generator - Based on the working test configuration
"""

import logging
import time
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

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

# Try to import PIL for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image handling limited")


class SimpleWorkingGenerator:
    """Simple working image generator using the proven configuration."""
    
    def __init__(self):
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "runwayml/stable-diffusion-v1-5"  # Use the working model
        logger.info(f"SimpleWorkingGenerator initialized on device: {self.device}")
    
    def initialize(self) -> bool:
        """Initialize the generator with the working configuration."""
        try:
            if not AI_AVAILABLE:
                logger.error("AI generation not available")
                return False
            
            logger.info(f"Loading model: {self.model_id}")
            
            # Use the exact working configuration from test_complete_system.py
            pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,  # Use float16 for memory efficiency
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device first
            pipeline = pipeline.to(self.device)
            
            # Apply the working optimizations
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
            
            # Enable attention slicing (the working memory optimization)
            if hasattr(pipeline, "enable_attention_slicing"):
                pipeline.enable_attention_slicing()
            
            # Store the pipeline
            self.pipeline = pipeline
            
            logger.info("Model loaded successfully with working configuration")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            return False
    
    def generate_image(self, prompt: str, width: int = 512, height: int = 512, 
                      steps: int = 20, guidance_scale: float = 7.5) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Generate an image using the working configuration."""
        try:
            if not self.pipeline:
                logger.error("Pipeline not initialized")
                return None, {"error": "Pipeline not initialized"}
            
            # Validate dimensions for GTX 1650 (4GB VRAM)
            max_dim = 512  # Conservative for 4GB VRAM
            if width > max_dim or height > max_dim:
                logger.warning(f"Dimensions {width}x{height} exceed max {max_dim}x{max_dim}, reducing...")
                width = min(width, max_dim)
                height = min(height, max_dim)
            
            # Clear CUDA cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Generating image: {width}x{height}, {steps} steps, guidance={guidance_scale}")
            start_time = time.time()
            
            # Generate using the working configuration
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1
                )
            
            generation_time = time.time() - start_time
            
            # Get the generated image
            image = result.images[0]
            
            # Validate the image
            if image is None or image.size[0] == 0 or image.size[1] == 0:
                logger.error("Generated image is invalid")
                return None, {"error": "Generated image is invalid"}
            
            # Create generation info
            generation_info = {
                "model": self.model_id,
                "prompt": prompt,
                "parameters": {
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance_scale": guidance_scale
                },
                "generation_time": generation_time,
                "device": self.device,
                "image_size": image.size,
                "image_mode": image.mode
            }
            
            logger.info(f"Image generated successfully in {generation_time:.2f}s")
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            
            return image, generation_info
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.pipeline:
                del self.pipeline
                self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SimpleWorkingGenerator cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def test_generator():
    """Test the simple working generator."""
    print("Testing SimpleWorkingGenerator...")
    
    generator = SimpleWorkingGenerator()
    
    if generator.initialize():
        print("✅ Generator initialized successfully")
        
        # Test generation
        prompt = "a beautiful red rose in a garden"
        print(f"Testing with prompt: '{prompt}'")
        
        image, info = generator.generate_image(
            prompt=prompt,
            width=256,  # Small size for testing
            height=256,
            steps=15,
            guidance_scale=7.5
        )
        
        if image:
            print("✅ Image generated successfully!")
            print(f"Size: {image.size}, Mode: {image.mode}")
            print(f"Info: {info}")
            
            # Save test image
            test_path = "test_working_generator.png"
            image.save(test_path)
            print(f"Test image saved to: {test_path}")
            
            # Check file size
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                print(f"File size: {file_size} bytes")
                if file_size > 1000:
                    print("✅ File size looks good!")
                else:
                    print("⚠️  File size too small - may be corrupted")
        else:
            print(f"❌ Generation failed: {info}")
        
        generator.cleanup()
    else:
        print("❌ Failed to initialize generator")


if __name__ == "__main__":
    test_generator() 