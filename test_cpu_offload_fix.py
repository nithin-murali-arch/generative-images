"""
Test to verify that the CPU offloading fix resolves the black image issue
"""

import logging
import time
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_generation():
    """Test simple image generation without CPU offloading."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        
        logger.info("Testing simple image generation...")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            return False
        
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load model with the working configuration (no CPU offloading)
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Loading model: {model_id}")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Apply Euler scheduler fix
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        logger.info("Applied Euler scheduler fix")
        
        # Move to GPU (no CPU offloading)
        pipeline = pipeline.to("cuda")
        logger.info("Pipeline moved to GPU")
        
        # Enable attention slicing for memory efficiency
        pipeline.enable_attention_slicing()
        logger.info("Enabled attention slicing")
        
        # Test generation
        prompt = "a beautiful red rose in a garden"
        logger.info(f"Generating image with prompt: '{prompt}'")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                width=256,  # Small size for testing
                height=256,
                num_inference_steps=20,
                guidance_scale=7.5,
                num_images_per_prompt=1
            )
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f}s")
        
        # Get the generated image
        image = result.images[0]
        logger.info(f"Image generated: {image.size}, mode: {image.mode}")
        
        # Save test image
        test_path = "test_cpu_offload_fix.png"
        image.save(test_path)
        logger.info(f"Test image saved to: {test_path}")
        
        # Check file size
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size > 10000:  # Should be much larger than 270 bytes
                logger.info("✅ SUCCESS: File size looks good - CPU offloading fix worked!")
                return True
            else:
                logger.error(f"❌ FAILED: File size too small ({file_size} bytes) - still corrupted")
                return False
        else:
            logger.error("❌ FAILED: Test image file not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_profile_configuration():
    """Test that the profile configuration is correct."""
    try:
        from src.hardware.profiles import HardwareProfileManager
        
        logger.info("Testing profile configuration...")
        
        profile_manager = HardwareProfileManager()
        
        # Test GTX 1650 profile
        profile = profile_manager.get_profile("GTX 1650", 4096)
        if profile:
            logger.info(f"Profile: {profile.name}")
            logger.info(f"CPU offload: {profile.optimizations.get('cpu_offload')}")
            logger.info(f"Sequential CPU offload: {profile.optimizations.get('sequential_cpu_offload')}")
            
            if (profile.optimizations.get('cpu_offload') == False and 
                profile.optimizations.get('sequential_cpu_offload') == False):
                logger.info("✅ SUCCESS: Profile correctly configured - CPU offloading disabled")
                return True
            else:
                logger.error("❌ FAILED: Profile still has CPU offloading enabled")
                return False
        else:
            logger.error("❌ FAILED: Could not get GTX 1650 profile")
            return False
            
    except Exception as e:
        logger.error(f"❌ Profile test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Testing CPU Offloading Fix")
    logger.info("=" * 50)
    
    # Test 1: Profile configuration
    logger.info("\n📋 Test 1: Profile Configuration")
    profile_ok = test_profile_configuration()
    
    # Test 2: Simple generation
    logger.info("\n🎨 Test 2: Simple Image Generation")
    generation_ok = test_simple_generation()
    
    # Summary
    logger.info("\n" + "=" * 50)
    if profile_ok and generation_ok:
        logger.info("🎉 ALL TESTS PASSED: CPU offloading fix is working!")
    else:
        logger.error("💥 SOME TESTS FAILED: CPU offloading fix needs more work")
    
    logger.info("=" * 50) 