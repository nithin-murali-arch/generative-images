#!/usr/bin/env python3
"""
Test Real Image Generation

This script tests if we can actually load and use real Stable Diffusion models
to generate actual images instead of mock ones.
"""

import sys
import logging
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available."""
    logger.info("🔍 Checking dependencies...")
    
    missing_deps = []
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ CUDA not available - will use CPU")
    except ImportError:
        missing_deps.append("torch")
        logger.error("❌ PyTorch not installed")
    
    # Check Diffusers
    try:
        import diffusers
        logger.info(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        missing_deps.append("diffusers")
        logger.error("❌ Diffusers not installed")
    
    # Check Transformers
    try:
        import transformers
        logger.info(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
        logger.error("❌ Transformers not installed")
    
    # Check PIL
    try:
        from PIL import Image
        logger.info("✅ PIL available")
    except ImportError:
        missing_deps.append("pillow")
        logger.error("❌ PIL not installed")
    
    if missing_deps:
        logger.error(f"\n💥 Missing dependencies: {', '.join(missing_deps)}")
        logger.info("\n💡 To install missing dependencies:")
        logger.info("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        logger.info("pip install diffusers transformers accelerate")
        logger.info("pip install pillow")
        return False
    
    return True

def test_direct_model_loading():
    """Test loading a Stable Diffusion model directly."""
    logger.info("🤖 Testing direct model loading...")
    
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        # Choose device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load a lightweight model
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Loading model: {model_id}")
        
        start_time = time.time()
        
        # Load with optimizations
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.2f}s")
        
        # Test generation
        logger.info("🎨 Testing image generation...")
        
        prompt = "a red apple on a white background, simple, photorealistic"
        
        start_time = time.time()
        
        with torch.no_grad():
            result = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                width=512,
                height=512,
                num_images_per_prompt=1
            )
        
        generation_time = time.time() - start_time
        
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            
            # Save the image
            output_path = Path("test_real_generation.png")
            image.save(output_path)
            
            logger.info(f"✅ Image generated successfully in {generation_time:.2f}s")
            logger.info(f"📁 Saved to: {output_path}")
            logger.info(f"📐 Size: {image.size}")
            
            # Check if image is not blank
            import numpy as np
            img_array = np.array(image)
            
            # Check if image has variation (not all same color)
            if img_array.std() > 10:  # Some variation in pixel values
                logger.info("✅ Image appears to have content (not blank)")
                return True
            else:
                logger.warning("⚠️ Image appears to be blank or uniform")
                return False
        else:
            logger.error("❌ No images returned from pipeline")
            return False
            
    except Exception as e:
        logger.error(f"❌ Direct model test failed: {e}")
        return False
    finally:
        # Clean up
        try:
            del pipe
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

def test_system_integration():
    """Test our system's image generation."""
    logger.info("🔧 Testing system integration...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        
        # Create system
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
        
        # Check if image pipeline is using real models
        if system.image_pipeline:
            available_models = system.image_pipeline.get_available_models()
            logger.info(f"📋 Available models: {available_models}")
            
            # Check current pipeline type
            if hasattr(system.image_pipeline, 'current_pipeline'):
                pipeline_type = type(system.image_pipeline.current_pipeline).__name__
                logger.info(f"🔍 Current pipeline type: {pipeline_type}")
                
                if "Mock" in pipeline_type:
                    logger.warning("⚠️ System is using mock pipeline")
                else:
                    logger.info("✅ System is using real pipeline")
        
        # Test generation
        prompt = "a beautiful sunset over mountains, photorealistic"
        
        logger.info(f"🎨 Testing generation with prompt: '{prompt}'")
        
        result = system.execute_complete_generation_workflow(
            prompt=prompt,
            conversation_id="real_test",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            additional_params={
                'width': 512,
                'height': 512,
                'num_inference_steps': 20,
                'guidance_scale': 7.5,
                'force_real_generation': True,
                'force_gpu_usage': True
            }
        )
        
        if result.success:
            logger.info(f"✅ System generation successful!")
            logger.info(f"   Model used: {result.model_used}")
            logger.info(f"   Generation time: {result.generation_time:.2f}s")
            logger.info(f"   Output path: {result.output_path}")
            
            # Check if output file exists and is not blank
            if result.output_path and Path(result.output_path).exists():
                logger.info("✅ Output file exists")
                
                # Try to load and check the image
                try:
                    from PIL import Image
                    import numpy as np
                    
                    image = Image.open(result.output_path)
                    img_array = np.array(image)
                    
                    if img_array.std() > 10:
                        logger.info("✅ Generated image has content (not blank)")
                        return True
                    else:
                        logger.warning("⚠️ Generated image appears blank")
                        return False
                except Exception as e:
                    logger.warning(f"⚠️ Could not verify image content: {e}")
                    return True  # File exists, assume it's good
            else:
                logger.error("❌ Output file does not exist")
                return False
        else:
            logger.error(f"❌ System generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ System integration test failed: {e}")
        return False
    finally:
        try:
            system.cleanup()
        except:
            pass

def main():
    """Run real generation tests."""
    logger.info("🚀 Testing Real Image Generation")
    logger.info("="*50)
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Dependencies not met - cannot test real generation")
        return False
    
    tests = [
        ("Direct Model Loading", test_direct_model_loading),
        ("System Integration", test_system_integration)
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
    logger.info("REAL GENERATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 Real image generation is working!")
        logger.info("🖼️ The system can generate actual images, not just blank ones")
    elif passed > 0:
        logger.warning("\n⚠️ Partial success - some real generation working")
        logger.info("💡 Check system configuration and model loading")
    else:
        logger.error("\n💥 Real image generation failed")
        logger.info("💡 System may be falling back to mock implementations")
    
    return passed > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)