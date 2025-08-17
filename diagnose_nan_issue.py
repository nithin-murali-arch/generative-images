#!/usr/bin/env python3
"""
Diagnose NaN Issue in Image Generation

This script diagnoses the NaN/black image issue that's causing blank outputs.
"""

import sys
import logging
import time
import torch
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def diagnose_model_output():
    """Diagnose what's happening in the model output."""
    logger.info("🔍 Diagnosing model output for NaN/black image issue...")
    
    try:
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model with minimal settings
        model_id = "runwayml/stable-diffusion-v1-5"
        
        logger.info("Loading model...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 to avoid precision issues
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to(device)
        
        # Test generation with debugging
        prompt = "a red apple"
        
        logger.info(f"Generating with prompt: '{prompt}'")
        
        # Hook into the pipeline to capture intermediate outputs
        original_decode = pipe.vae.decode
        
        def debug_decode(latents, return_dict=True):
            logger.info(f"VAE decode input - shape: {latents.shape}")
            logger.info(f"VAE decode input - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}")
            logger.info(f"VAE decode input - mean: {latents.mean().item():.6f}, std: {latents.std().item():.6f}")
            logger.info(f"VAE decode input - has NaN: {torch.isnan(latents).any().item()}")
            logger.info(f"VAE decode input - has Inf: {torch.isinf(latents).any().item()}")
            
            result = original_decode(latents, return_dict=return_dict)
            
            if hasattr(result, 'sample'):
                sample = result.sample
            else:
                sample = result
            
            logger.info(f"VAE decode output - shape: {sample.shape}")
            logger.info(f"VAE decode output - min: {sample.min().item():.6f}, max: {sample.max().item():.6f}")
            logger.info(f"VAE decode output - mean: {sample.mean().item():.6f}, std: {sample.std().item():.6f}")
            logger.info(f"VAE decode output - has NaN: {torch.isnan(sample).any().item()}")
            logger.info(f"VAE decode output - has Inf: {torch.isinf(sample).any().item()}")
            
            return result
        
        pipe.vae.decode = debug_decode
        
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512,
                output_type='pil'
            )
        
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Analyze the final image
            img_array = np.array(image)
            logger.info(f"Final image - shape: {img_array.shape}")
            logger.info(f"Final image - dtype: {img_array.dtype}")
            logger.info(f"Final image - min: {img_array.min()}, max: {img_array.max()}")
            logger.info(f"Final image - mean: {img_array.mean():.6f}, std: {img_array.std():.6f}")
            logger.info(f"Final image - unique values: {len(np.unique(img_array))}")
            
            # Save for inspection
            output_path = Path("debug_output.png")
            image.save(output_path)
            logger.info(f"Saved debug image to: {output_path}")
            
            # Check if all pixels are the same
            if img_array.std() < 0.1:
                logger.error("❌ Image is uniform/blank - this confirms the issue")
                
                # Check what the uniform value is
                uniform_value = img_array.flat[0]
                logger.info(f"Uniform pixel value: {uniform_value}")
                
                if uniform_value == 0:
                    logger.error("All pixels are BLACK (0) - VAE output issue")
                elif uniform_value == 255:
                    logger.error("All pixels are WHITE (255) - clipping issue")
                else:
                    logger.error(f"All pixels are GRAY ({uniform_value}) - unknown issue")
            else:
                logger.info("✅ Image has variation - generation working")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        return False
    finally:
        try:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def test_vae_directly():
    """Test the VAE component directly to isolate the issue."""
    logger.info("🔧 Testing VAE component directly...")
    
    try:
        from diffusers import AutoencoderKL
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load just the VAE
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float32
        )
        vae = vae.to(device)
        
        # Create a simple test latent
        batch_size = 1
        channels = 4
        height = 64  # 512/8
        width = 64   # 512/8
        
        # Create a test latent with known values
        test_latent = torch.randn(batch_size, channels, height, width, device=device, dtype=torch.float32)
        
        logger.info(f"Test latent - shape: {test_latent.shape}")
        logger.info(f"Test latent - min: {test_latent.min().item():.6f}, max: {test_latent.max().item():.6f}")
        logger.info(f"Test latent - mean: {test_latent.mean().item():.6f}, std: {test_latent.std().item():.6f}")
        
        # Decode the latent
        with torch.no_grad():
            decoded = vae.decode(test_latent).sample
        
        logger.info(f"Decoded - shape: {decoded.shape}")
        logger.info(f"Decoded - min: {decoded.min().item():.6f}, max: {decoded.max().item():.6f}")
        logger.info(f"Decoded - mean: {decoded.mean().item():.6f}, std: {decoded.std().item():.6f}")
        logger.info(f"Decoded - has NaN: {torch.isnan(decoded).any().item()}")
        logger.info(f"Decoded - has Inf: {torch.isinf(decoded).any().item()}")
        
        # Convert to image format
        decoded = (decoded / 2 + 0.5).clamp(0, 1)  # Normalize from [-1,1] to [0,1]
        decoded = (decoded * 255).round().to(torch.uint8)
        
        logger.info(f"After normalization - min: {decoded.min().item()}, max: {decoded.max().item()}")
        logger.info(f"After normalization - mean: {decoded.mean().item():.2f}, std: {decoded.std().item():.2f}")
        
        # Convert to numpy and save
        img_array = decoded.cpu().numpy().transpose(0, 2, 3, 1)[0]  # BCHW -> HWC
        
        from PIL import Image
        image = Image.fromarray(img_array)
        image.save("vae_test_output.png")
        
        logger.info("✅ VAE test completed - check vae_test_output.png")
        
        # Check if output is reasonable
        if img_array.std() > 10:
            logger.info("✅ VAE produces varied output")
            return True
        else:
            logger.error("❌ VAE produces uniform output")
            return False
        
    except Exception as e:
        logger.error(f"❌ VAE test failed: {e}")
        return False

def test_with_different_scheduler():
    """Test with a different scheduler to see if that fixes the issue."""
    logger.info("🔄 Testing with different scheduler...")
    
    try:
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Use a different scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        prompt = "a red apple on a white background"
        
        logger.info("Generating with Euler scheduler...")
        
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512
            )
        
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            img_array = np.array(image)
            
            logger.info(f"Euler result - mean: {img_array.mean():.2f}, std: {img_array.std():.2f}")
            
            image.save("euler_test_output.png")
            
            if img_array.std() > 10:
                logger.info("✅ Euler scheduler produces good output")
                return True
            else:
                logger.error("❌ Euler scheduler still produces uniform output")
                return False
        
    except Exception as e:
        logger.error(f"❌ Scheduler test failed: {e}")
        return False

def main():
    """Run comprehensive diagnosis."""
    logger.info("🚀 Diagnosing NaN/Black Image Issue")
    logger.info("="*50)
    
    tests = [
        ("Model Output Diagnosis", diagnose_model_output),
        ("VAE Direct Test", test_vae_directly),
        ("Different Scheduler Test", test_with_different_scheduler)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("DIAGNOSIS SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == 0:
        logger.error("\n💥 All diagnosis tests failed")
        logger.info("💡 Possible causes:")
        logger.info("   - Corrupted model weights")
        logger.info("   - GPU driver issues")
        logger.info("   - CUDA/PyTorch compatibility issues")
        logger.info("   - Insufficient VRAM causing silent failures")
        logger.info("\n🔧 Suggested fixes:")
        logger.info("   - Update GPU drivers")
        logger.info("   - Reinstall PyTorch with correct CUDA version")
        logger.info("   - Try CPU-only generation")
        logger.info("   - Clear Hugging Face cache and re-download models")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)