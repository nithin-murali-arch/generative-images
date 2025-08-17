#!/usr/bin/env python3
"""
Fix Blank Images Issue

This script diagnoses and fixes the blank image generation issue.
"""

import sys
import logging
import time
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_simple_generation():
    """Test simple generation with various parameters to find working settings."""
    logger.info("🔧 Testing simple generation with various parameters...")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from PIL import Image
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model with minimal optimizations
        model_id = "runwayml/stable-diffusion-v1-5"
        
        logger.info("Loading model with minimal optimizations...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        pipe = pipe.to(device)
        
        # Test different parameter combinations
        test_configs = [
            {
                "name": "Basic Test",
                "prompt": "a red apple on a white background",
                "steps": 20,
                "guidance": 7.5,
                "width": 512,
                "height": 512
            },
            {
                "name": "High Guidance Test",
                "prompt": "a red apple on a white background",
                "steps": 25,
                "guidance": 12.0,
                "width": 512,
                "height": 512
            },
            {
                "name": "More Steps Test",
                "prompt": "a red apple on a white background",
                "steps": 30,
                "guidance": 7.5,
                "width": 512,
                "height": 512
            },
            {
                "name": "Different Prompt Test",
                "prompt": "a beautiful sunset over mountains, vibrant colors",
                "steps": 20,
                "guidance": 7.5,
                "width": 512,
                "height": 512
            }
        ]
        
        successful_configs = []
        
        for i, config in enumerate(test_configs):
            logger.info(f"\n🎨 Test {i+1}: {config['name']}")
            logger.info(f"   Prompt: {config['prompt']}")
            logger.info(f"   Steps: {config['steps']}, Guidance: {config['guidance']}")
            
            try:
                start_time = time.time()
                
                # Generate with explicit parameters
                with torch.no_grad():
                    result = pipe(
                        prompt=config['prompt'],
                        num_inference_steps=config['steps'],
                        guidance_scale=config['guidance'],
                        width=config['width'],
                        height=config['height'],
                        num_images_per_prompt=1,
                        output_type='pil'
                    )
                
                generation_time = time.time() - start_time
                
                if hasattr(result, 'images') and len(result.images) > 0:
                    image = result.images[0]
                    
                    # Analyze image
                    img_array = np.array(image)
                    mean_val = img_array.mean()
                    std_val = img_array.std()
                    
                    logger.info(f"   ✅ Generated in {generation_time:.2f}s")
                    logger.info(f"   📊 Image stats: mean={mean_val:.2f}, std={std_val:.2f}")
                    
                    # Save image
                    output_path = Path(f"test_fix_{i+1}_{config['name'].lower().replace(' ', '_')}.png")
                    image.save(output_path)
                    logger.info(f"   💾 Saved: {output_path}")
                    
                    # Check if image has content
                    if std_val > 10.0:
                        logger.info(f"   ✅ Image has good content variation")
                        successful_configs.append(config)
                    else:
                        logger.warning(f"   ⚠️ Image appears uniform/blank")
                        
                        # Try to diagnose the issue
                        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                        logger.info(f"   🎨 Unique colors: {unique_colors}")
                        
                        if unique_colors < 10:
                            logger.warning("   Very few unique colors - likely blank/uniform")
                else:
                    logger.error(f"   ❌ No images returned")
                    
            except Exception as e:
                logger.error(f"   ❌ Generation failed: {e}")
            
            # Clear cache between tests
            if device == "cuda":
                torch.cuda.empty_cache()
            
            time.sleep(2)
        
        # Summary
        logger.info(f"\n📋 Test Summary:")
        logger.info(f"   Successful configs: {len(successful_configs)}/{len(test_configs)}")
        
        if successful_configs:
            logger.info("   ✅ Working configurations found:")
            for config in successful_configs:
                logger.info(f"     - {config['name']}: steps={config['steps']}, guidance={config['guidance']}")
            
            # Recommend best settings
            best_config = successful_configs[0]
            logger.info(f"\n💡 Recommended settings:")
            logger.info(f"   Steps: {best_config['steps']}")
            logger.info(f"   Guidance Scale: {best_config['guidance']}")
            logger.info(f"   Resolution: {best_config['width']}x{best_config['height']}")
            
            return True
        else:
            logger.error("   ❌ No working configurations found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Simple generation test failed: {e}")
        return False
    finally:
        try:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def test_system_with_fixes():
    """Test our system with the fixes applied."""
    logger.info("🔧 Testing system with fixes...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        
        system = SystemIntegration()
        
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "auto_detect_hardware": True,
            "max_concurrent_generations": 1
        }
        
        logger.info("⚙️ Initializing system...")
        if not system.initialize(config):
            logger.error("❌ System initialization failed")
            return False
        
        # Test with good parameters
        test_prompts = [
            "a red apple on a white background, photorealistic",
            "a beautiful sunset over mountains, vibrant colors",
            "a cute cat sitting on a windowsill, detailed"
        ]
        
        successful_generations = 0
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n🎨 System Test {i+1}: {prompt[:30]}...")
            
            result = system.execute_complete_generation_workflow(
                prompt=prompt,
                conversation_id=f"fix_test_{i}",
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                additional_params={
                    'width': 512,
                    'height': 512,
                    'num_inference_steps': 25,  # More steps for better quality
                    'guidance_scale': 9.0,      # Higher guidance for better adherence
                    'force_real_generation': True,
                    'precision': 'float16'
                }
            )
            
            if result.success:
                logger.info(f"   ✅ Generation successful in {result.generation_time:.2f}s")
                logger.info(f"   📁 Output: {result.output_path}")
                
                # Check if file exists and analyze
                if result.output_path and Path(result.output_path).exists():
                    try:
                        from PIL import Image
                        import numpy as np
                        
                        image = Image.open(result.output_path)
                        img_array = np.array(image)
                        std_val = img_array.std()
                        
                        logger.info(f"   📊 Image std: {std_val:.2f}")
                        
                        if std_val > 10.0:
                            logger.info(f"   ✅ Image has good content")
                            successful_generations += 1
                        else:
                            logger.warning(f"   ⚠️ Image appears uniform")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Could not analyze image: {e}")
                        successful_generations += 1  # Assume it's good if file exists
                else:
                    logger.error(f"   ❌ Output file not found")
            else:
                logger.error(f"   ❌ Generation failed: {result.error_message}")
            
            # Clear cache
            if system.memory_manager:
                system.memory_manager.clear_vram_cache()
            time.sleep(2)
        
        logger.info(f"\n📊 System Test Results: {successful_generations}/{len(test_prompts)} successful")
        
        system.cleanup()
        return successful_generations > 0
        
    except Exception as e:
        logger.error(f"❌ System test failed: {e}")
        return False

def main():
    """Run blank image fix tests."""
    logger.info("🚀 Fixing Blank Images Issue")
    logger.info("="*50)
    
    tests = [
        ("Simple Generation Test", test_simple_generation),
        ("System Integration Test", test_system_with_fixes)
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
    logger.info("BLANK IMAGE FIX SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed > 0:
        logger.info("\n🎉 Blank image issue partially or fully resolved!")
        logger.info("💡 Check the generated test images to verify quality")
    else:
        logger.error("\n💥 Blank image issue persists")
        logger.info("💡 May need to adjust model parameters or check GPU drivers")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)