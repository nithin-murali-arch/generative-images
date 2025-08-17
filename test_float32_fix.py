"""
Test if the issue is with float16 precision by trying float32
"""

import logging
import time
import os
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_float32_generation():
    """Test image generation with float32 instead of float16."""
    try:
        logger.info("Testing float32 image generation...")
        
        # Check CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            return False
        
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Import diffusers
        from diffusers import StableDiffusionPipeline
        
        # Load model with float32 (more memory but potentially more stable)
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Loading model: {model_id} with float32")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 instead of float16
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        logger.info("✅ Model loaded successfully with float32")
        
        # Move to GPU
        pipeline = pipeline.to("cuda")
        logger.info("✅ Pipeline moved to GPU")
        
        # Test generation
        prompt = "a red rose"
        logger.info(f"Generating: '{prompt}'")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                width=256,
                height=256,
                num_inference_steps=10,
                guidance_scale=7.5,
                num_images_per_prompt=1
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s")
        
        # Get image
        image = result.images[0]
        logger.info(f"✅ Image generated: {image.size}, mode: {image.mode}")
        
        # Save image
        test_path = "test_float32_fix.png"
        image.save(test_path)
        logger.info(f"✅ Image saved to: {test_path}")
        
        # Check file size
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size > 10000:
                logger.info("🎉 SUCCESS: Float32 fixed the issue!")
                return True
            else:
                logger.error(f"❌ FAILED: Float32 didn't help - still {file_size} bytes")
                return False
        else:
            logger.error("❌ FAILED: File not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_scheduler():
    """Test with a different scheduler."""
    try:
        logger.info("Testing with different scheduler...")
        
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        
        # Load model
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Loading model: {model_id}")
        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        # Use DPM++ 2M scheduler instead of default
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        logger.info("✅ Applied DPM++ 2M scheduler")
        
        # Move to GPU
        pipeline = pipeline.to("cuda")
        logger.info("✅ Pipeline moved to GPU")
        
        # Test generation
        prompt = "a red rose"
        logger.info(f"Generating: '{prompt}'")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                width=256,
                height=256,
                num_inference_steps=10,
                guidance_scale=7.5,
                num_images_per_prompt=1
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s")
        
        # Get image
        image = result.images[0]
        logger.info(f"✅ Image generated: {image.size}, mode: {image.mode}")
        
        # Save image
        test_path = "test_different_scheduler.png"
        image.save(test_path)
        logger.info(f"✅ Image saved to: {test_path}")
        
        # Check file size
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size > 10000:
                logger.info("🎉 SUCCESS: Different scheduler fixed the issue!")
                return True
            else:
                logger.error(f"❌ FAILED: Different scheduler didn't help - still {file_size} bytes")
                return False
        else:
            logger.error("❌ FAILED: File not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔧 Testing Float32 and Scheduler Fixes")
    logger.info("=" * 50)
    
    # Test 1: Float32
    logger.info("\n🔢 Test 1: Float32 Precision")
    float32_ok = test_float32_generation()
    
    # Test 2: Different scheduler
    logger.info("\n⏰ Test 2: Different Scheduler")
    scheduler_ok = test_different_scheduler()
    
    # Summary
    logger.info("\n" + "=" * 50)
    if float32_ok or scheduler_ok:
        logger.info("🎉 AT LEAST ONE TEST PASSED: Found a working configuration!")
    else:
        logger.error("💥 ALL TESTS FAILED: Issue is deeper than precision/scheduler")
    
    logger.info("=" * 50) 