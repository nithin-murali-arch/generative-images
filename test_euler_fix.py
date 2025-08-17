#!/usr/bin/env python3
"""
Direct test of Euler scheduler fix for black images.
"""

import sys
import logging
import time
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_euler_fix_directly():
    """Test the Euler scheduler fix directly."""
    logger.info("🔧 Testing Euler scheduler fix directly...")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        from PIL import Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model with our fix
        model_id = "runwayml/stable-diffusion-v1-5"
        
        logger.info("Loading model with Euler scheduler fix...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 to avoid precision issues
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Apply the Euler scheduler fix (same as in our pipeline)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        logger.info("✅ Applied Euler scheduler fix")
        
        pipe = pipe.to(device)
        
        # Test generation
        prompt = "a red apple on a white background"
        
        logger.info(f"Generating with prompt: '{prompt}'")
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512,
                output_type='pil'
            )
        
        generation_time = time.time() - start_time
        
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Analyze image
            img_array = np.array(image)
            mean_val = img_array.mean()
            std_val = img_array.std()
            
            logger.info(f"✅ Generated in {generation_time:.2f}s")
            logger.info(f"📊 Image stats: mean={mean_val:.2f}, std={std_val:.2f}")
            
            # Save image
            output_path = Path("euler_fix_test.png")
            image.save(output_path)
            logger.info(f"💾 Saved: {output_path}")
            
            # Check if image has content
            if std_val > 10.0:
                logger.info("🎉 SUCCESS: Euler fix works! Image has good content variation")
                return True
            else:
                logger.error("❌ FAILED: Image still appears blank/uniform")
                logger.info(f"   Mean: {mean_val}, Std: {std_val}")
                
                # Check unique colors
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                logger.info(f"   Unique colors: {unique_colors}")
                
                return False
        else:
            logger.error("❌ No images returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Euler fix test failed: {e}")
        return False
    finally:
        try:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def test_different_parameters():
    """Test with different parameters to find working settings."""
    logger.info("🔍 Testing different parameters...")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Apply Euler scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        
        # Test different parameter combinations
        test_configs = [
            {"steps": 15, "guidance": 5.0, "name": "Low guidance"},
            {"steps": 20, "guidance": 7.5, "name": "Standard"},
            {"steps": 25, "guidance": 10.0, "name": "High guidance"},
            {"steps": 30, "guidance": 12.0, "name": "Very high guidance"}
        ]
        
        successful_configs = []
        
        for i, config in enumerate(test_configs):
            logger.info(f"\n🎨 Test {i+1}: {config['name']}")
            
            try:
                with torch.no_grad():
                    result = pipe(
                        "a red apple on a white background",
                        num_inference_steps=config["steps"],
                        guidance_scale=config["guidance"],
                        width=512,
                        height=512
                    )
                
                if hasattr(result, 'images') and len(result.images) > 0:
                    image = result.images[0]
                    img_array = np.array(image)
                    std_val = img_array.std()
                    
                    logger.info(f"   Steps: {config['steps']}, Guidance: {config['guidance']}")
                    logger.info(f"   Image std: {std_val:.2f}")
                    
                    # Save image
                    image.save(f"test_config_{i+1}_{config['name'].lower().replace(' ', '_')}.png")
                    
                    if std_val > 10.0:
                        logger.info("   ✅ Good variation")
                        successful_configs.append(config)
                    else:
                        logger.warning("   ⚠️ Low variation")
                
            except Exception as e:
                logger.error(f"   ❌ Failed: {e}")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)
        
        logger.info(f"\n📊 Results: {len(successful_configs)}/{len(test_configs)} configs successful")
        
        if successful_configs:
            logger.info("✅ Working configurations found:")
            for config in successful_configs:
                logger.info(f"   - {config['name']}: steps={config['steps']}, guidance={config['guidance']}")
            return True
        else:
            logger.error("❌ No working configurations found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Parameter test failed: {e}")
        return False

def main():
    """Run Euler fix tests."""
    logger.info("🚀 Testing Euler Scheduler Fix for Black Images")
    logger.info("="*50)
    
    tests = [
        ("Direct Euler Fix Test", test_euler_fix_directly),
        ("Parameter Variation Test", test_different_parameters)
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
    logger.info("EULER FIX TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed > 0:
        logger.info("\n🎉 Euler scheduler fix is working!")
        logger.info("💡 The system should now generate real images instead of black ones")
    else:
        logger.error("\n💥 Euler scheduler fix is not working")
        logger.info("💡 The issue may be deeper - possibly model corruption or GPU driver issues")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)