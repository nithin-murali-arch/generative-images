"""
Minimal working test - bypass all complex pipeline logic
"""

import logging
import time
import os
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_generation():
    """Test minimal image generation with absolute basics."""
    try:
        logger.info("Testing minimal image generation...")
        
        # Check CUDA
        if not torch.cuda.is_available():
            logger.error("CUDA not available")
            return False
        
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Try to import diffusers
        try:
            from diffusers import StableDiffusionPipeline
            logger.info("✅ Diffusers imported successfully")
        except ImportError as e:
            logger.error(f"❌ Diffusers import failed: {e}")
            return False
        
        # Load model with absolute minimal parameters
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Loading model: {model_id}")
        
        # Use only the most basic parameters
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Move to GPU
        pipeline = pipeline.to("cuda")
        logger.info("✅ Pipeline moved to GPU")
        
        # Test generation with minimal parameters
        prompt = "a red rose"
        logger.info(f"Generating: '{prompt}'")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                width=256,
                height=256,
                num_inference_steps=10,  # Very few steps for testing
                guidance_scale=7.5,
                num_images_per_prompt=1
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Generation completed in {generation_time:.2f}s")
        
        # Get image
        image = result.images[0]
        logger.info(f"✅ Image generated: {image.size}, mode: {image.mode}")
        
        # Save image
        test_path = "test_minimal_working.png"
        image.save(test_path)
        logger.info(f"✅ Image saved to: {test_path}")
        
        # Check file size
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size > 10000:
                logger.info("🎉 SUCCESS: Large file size - image is working!")
                return True
            else:
                logger.error(f"❌ FAILED: File too small ({file_size} bytes)")
                return False
        else:
            logger.error("❌ FAILED: File not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_download():
    """Test if the model is properly downloaded."""
    try:
        from huggingface_hub import snapshot_download
        
        logger.info("Testing model download...")
        
        # Check if model exists locally
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_path = os.path.join(cache_dir, "models--runwayml--stable-diffusion-v1-5")
        
        if os.path.exists(model_path):
            logger.info(f"✅ Model found in cache: {model_path}")
            
            # Check file sizes
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith('.safetensors') or file.endswith('.bin'):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        logger.info(f"  {file}: {file_size / 1024**2:.1f} MB")
            
            return True
        else:
            logger.warning(f"⚠️  Model not found in cache: {model_path}")
            
            # Try to download
            logger.info("Attempting to download model...")
            snapshot_download(
                repo_id="runwayml/stable-diffusion-v1-5",
                local_dir="./models/stable-diffusion-v1-5",
                local_dir_use_symlinks=False
            )
            logger.info("✅ Model download completed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Model download test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔧 Minimal Working Test")
    logger.info("=" * 50)
    
    # Test 1: Model download
    logger.info("\n📥 Test 1: Model Download")
    download_ok = test_model_download()
    
    # Test 2: Minimal generation
    logger.info("\n🎨 Test 2: Minimal Generation")
    generation_ok = test_minimal_generation()
    
    # Summary
    logger.info("\n" + "=" * 50)
    if download_ok and generation_ok:
        logger.info("🎉 ALL TESTS PASSED: Minimal approach is working!")
    else:
        logger.error("💥 SOME TESTS FAILED: Need to investigate further")
    
    logger.info("=" * 50) 