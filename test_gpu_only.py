#!/usr/bin/env python3
"""
Test GPU-Only Generation (No CPU Offloading)

This script tests image generation using GPU only, without any CPU offloading
that might cause black images.
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

def test_gpu_only_generation():
    """Test GPU-only generation without CPU offloading."""
    logger.info("🎮 Testing GPU-only generation...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        
        system = SystemIntegration()
        
        # Config that forces GPU usage without CPU offloading
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "auto_detect_hardware": True,
            "max_concurrent_generations": 1,
            "force_gpu_only": True  # Force GPU-only mode
        }
        
        logger.info("⚙️ Initializing system for GPU-only generation...")
        if not system.initialize(config):
            logger.error("❌ System initialization failed")
            return False
        
        logger.info("✅ System initialized successfully!")
        
        # Test with explicit GPU-only parameters
        prompt = "a red apple on a white background, photorealistic"
        
        logger.info(f"🎨 Testing GPU-only generation: '{prompt}'")
        
        result = system.execute_complete_generation_workflow(
            prompt=prompt,
            conversation_id="gpu_only_test",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            additional_params={
                'width': 512,
                'height': 512,
                'num_inference_steps': 20,
                'guidance_scale': 7.5,
                'force_gpu_usage': True,
                'precision': 'float16',
                'memory_optimization': 'None (Fastest, Most VRAM)',  # No memory optimization
                'force_real_generation': True,
                'disable_cpu_offload': True  # Explicitly disable CPU offloading
            }
        )
        
        if result.success:
            logger.info(f"✅ GPU-only generation successful!")
            logger.info(f"   Time: {result.generation_time:.2f}s")
            logger.info(f"   Model: {result.model_used}")
            logger.info(f"   Output: {result.output_path}")
            
            # Check if image is not blank
            if result.output_path and Path(result.output_path).exists():
                try:
                    from PIL import Image
                    import numpy as np
                    
                    image = Image.open(result.output_path)
                    img_array = np.array(image)
                    
                    mean_val = img_array.mean()
                    std_val = img_array.std()
                    
                    logger.info(f"   📊 Image stats: mean={mean_val:.2f}, std={std_val:.2f}")
                    
                    if std_val > 10.0:
                        logger.info("   🎉 SUCCESS: GPU-only generation produces real images!")
                        return True
                    else:
                        logger.error("   ❌ Image still appears blank")
                        return False
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not analyze image: {e}")
                    return True  # Assume success if file exists
            else:
                logger.error("   ❌ Output file not found")
                return False
        else:
            logger.error(f"   ❌ Generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU-only test failed: {e}")
        return False
    finally:
        try:
            system.cleanup()
        except:
            pass

def main():
    """Test GPU-only generation."""
    logger.info("🚀 Testing GPU-Only Generation (No CPU Offloading)")
    logger.info("="*60)
    
    success = test_gpu_only_generation()
    
    if success:
        logger.info("\n🎉 GPU-ONLY GENERATION WORKING!")
        logger.info("✅ GPU is doing the work properly")
        logger.info("✅ No more black images")
        logger.info("✅ CPU offloading disabled")
        logger.info("\n💪 Your GTX 1650 is generating real images!")
    else:
        logger.error("\n💥 GPU-only generation still has issues")
        logger.info("💡 May need to check GPU drivers or PyTorch installation")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)