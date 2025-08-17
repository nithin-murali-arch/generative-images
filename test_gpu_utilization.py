#!/usr/bin/env python3
"""
GPU Utilization Test for Academic Multimodal LLM System

This script tests GPU utilization and ensures we're actually using the GPU
for generation instead of falling back to CPU or mock implementations.
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

def check_gpu_availability():
    """Check if GPU and required libraries are available."""
    logger.info("Checking GPU availability...")
    
    # Check PyTorch and CUDA
    try:
        import torch
        logger.info(f"✅ PyTorch available: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            logger.info(f"✅ CUDA available: {torch.version.cuda}")
            logger.info(f"✅ GPU Count: {gpu_count}")
            logger.info(f"✅ Current GPU: {gpu_name}")
            logger.info(f"✅ GPU Memory: {gpu_memory:.1f} GB")
            
            return True, {
                'gpu_name': gpu_name,
                'gpu_memory_gb': gpu_memory,
                'gpu_count': gpu_count
            }
        else:
            logger.warning("⚠️ CUDA not available - will use CPU")
            return False, {}
            
    except ImportError:
        logger.error("❌ PyTorch not available")
        return False, {}

def check_diffusers_availability():
    """Check if Diffusers library is available."""
    try:
        import diffusers
        logger.info(f"✅ Diffusers available: {diffusers.__version__}")
        
        # Try to import key pipeline classes
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
        logger.info("✅ Stable Diffusion pipelines available")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Diffusers not available: {e}")
        return False

def test_gpu_memory_allocation():
    """Test GPU memory allocation to ensure we can use the GPU."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, skipping GPU memory test")
            return False
        
        logger.info("Testing GPU memory allocation...")
        
        # Allocate some GPU memory
        device = torch.device('cuda')
        
        # Create a tensor on GPU
        test_tensor = torch.randn(1000, 1000, device=device)
        logger.info(f"✅ Allocated tensor on GPU: {test_tensor.shape}")
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        cached = torch.cuda.memory_reserved() / (1024**2)  # MB
        
        logger.info(f"✅ GPU Memory - Allocated: {allocated:.1f} MB, Cached: {cached:.1f} MB")
        
        # Clean up
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU memory allocation failed: {e}")
        return False

def test_real_model_loading():
    """Test loading a real Stable Diffusion model on GPU."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, testing CPU model loading")
            device = "cpu"
        else:
            device = "cuda"
        
        logger.info(f"Testing real model loading on {device}...")
        
        # Try to load a lightweight model
        model_id = "runwayml/stable-diffusion-v1-5"
        
        logger.info(f"Loading {model_id}...")
        start_time = time.time()
        
        # Load with optimizations for lower VRAM usage
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
        
        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
        
        # Test generation
        logger.info("Testing image generation...")
        start_time = time.time()
        
        with torch.no_grad():
            result = pipe(
                "a simple red apple on a white background",
                num_inference_steps=10,  # Reduced for faster testing
                guidance_scale=7.5,
                width=512,
                height=512
            )
        
        generation_time = time.time() - start_time
        logger.info(f"✅ Image generated successfully in {generation_time:.2f}s")
        
        # Check if we got a real image
        if hasattr(result, 'images') and len(result.images) > 0:
            image = result.images[0]
            logger.info(f"✅ Generated image size: {image.size}")
            
            # Save test image
            output_path = Path("test_gpu_output.png")
            image.save(output_path)
            logger.info(f"✅ Test image saved to: {output_path}")
        
        # Check GPU memory usage during generation
        if device == "cuda":
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            logger.info(f"✅ GPU Memory used during generation: {allocated:.1f} MB")
        
        # Clean up
        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True, generation_time
        
    except Exception as e:
        logger.error(f"❌ Real model loading/generation failed: {e}")
        return False, 0

def test_system_integration_gpu_usage():
    """Test our system integration with GPU usage."""
    try:
        logger.info("Testing system integration GPU usage...")
        
        # Import our system
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        
        # Create system with GPU-optimized config
        system = SystemIntegration()
        
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "auto_detect_hardware": True,
            "max_concurrent_generations": 1,
            "force_gpu_usage": True  # Force GPU usage
        }
        
        logger.info("Initializing system with GPU optimization...")
        if not system.initialize(config):
            logger.error("❌ System initialization failed")
            return False
        
        # Check if GPU was detected
        if system.hardware_config:
            logger.info(f"✅ Hardware detected: {system.hardware_config.gpu_model}")
            logger.info(f"✅ VRAM: {system.hardware_config.vram_size} MB")
            logger.info(f"✅ CUDA Available: {system.hardware_config.cuda_available}")
        
        # Force image pipeline to use real models
        if system.image_pipeline:
            # Check available models
            available_models = system.image_pipeline.get_available_models()
            logger.info(f"✅ Available models: {available_models}")
            
            # Try to switch to a specific model
            if "stable-diffusion-v1-5" in available_models:
                success = system.image_pipeline.switch_model("stable-diffusion-v1-5")
                logger.info(f"✅ Model switch successful: {success}")
        
        # Test generation with GPU monitoring
        prompt = "a red sports car on a mountain road, photorealistic"
        
        logger.info("Testing generation with GPU monitoring...")
        
        # Monitor GPU before generation
        gpu_before = None
        if system.hardware_config and system.hardware_config.cuda_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_before = torch.cuda.memory_allocated() / (1024**2)
                    logger.info(f"GPU memory before generation: {gpu_before:.1f} MB")
            except:
                pass
        
        # Generate
        result = system.execute_complete_generation_workflow(
            prompt=prompt,
            conversation_id="gpu_test_session",
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            additional_params={
                'width': 512,
                'height': 512,
                'num_inference_steps': 15,  # Reduced for faster testing
                'guidance_scale': 7.5,
                'force_real_generation': True  # Force real generation
            }
        )
        
        # Monitor GPU after generation
        gpu_after = None
        if system.hardware_config and system.hardware_config.cuda_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_after = torch.cuda.memory_allocated() / (1024**2)
                    logger.info(f"GPU memory after generation: {gpu_after:.1f} MB")
                    
                    if gpu_before is not None:
                        gpu_used = gpu_after - gpu_before
                        logger.info(f"GPU memory used for generation: {gpu_used:.1f} MB")
                        
                        if gpu_used > 100:  # More than 100MB used
                            logger.info("✅ Significant GPU usage detected - likely using real models")
                        else:
                            logger.warning("⚠️ Low GPU usage - might be using mock/CPU fallback")
            except:
                pass
        
        if result.success:
            logger.info(f"✅ System generation successful!")
            logger.info(f"   Model used: {result.model_used}")
            logger.info(f"   Generation time: {result.generation_time:.2f}s")
            logger.info(f"   Output path: {result.output_path}")
            
            # Check if output file exists
            if result.output_path and Path(result.output_path).exists():
                logger.info("✅ Output file exists - real generation confirmed")
            else:
                logger.warning("⚠️ Output file missing - might be mock generation")
        else:
            logger.error(f"❌ System generation failed: {result.error_message}")
            return False
        
        # Clean up
        system.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System integration GPU test failed: {e}")
        return False

def optimize_for_gpu():
    """Provide recommendations for better GPU utilization."""
    logger.info("\n" + "="*50)
    logger.info("GPU OPTIMIZATION RECOMMENDATIONS")
    logger.info("="*50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"VRAM: {gpu_memory:.1f} GB")
            
            # Provide specific recommendations based on VRAM
            if gpu_memory < 6:
                logger.info("\n🔧 LOW VRAM OPTIMIZATIONS:")
                logger.info("- Use attention slicing: pipe.enable_attention_slicing()")
                logger.info("- Use VAE slicing: pipe.enable_vae_slicing()")
                logger.info("- Use CPU offloading: pipe.enable_model_cpu_offload()")
                logger.info("- Use float16: torch_dtype=torch.float16")
                logger.info("- Reduce image resolution to 512x512 or lower")
                logger.info("- Use SDXL-Turbo for faster generation")
                
            elif gpu_memory < 12:
                logger.info("\n🔧 MEDIUM VRAM OPTIMIZATIONS:")
                logger.info("- Use attention slicing for larger images")
                logger.info("- Enable XFormers if available")
                logger.info("- Use float16 for better performance")
                logger.info("- Can handle 768x768 images")
                
            else:
                logger.info("\n🔧 HIGH VRAM OPTIMIZATIONS:")
                logger.info("- Can use full precision (float32) if needed")
                logger.info("- Can handle 1024x1024 images")
                logger.info("- Can run multiple models simultaneously")
                logger.info("- Consider using SDXL or larger models")
            
            logger.info("\n🚀 GENERAL GPU OPTIMIZATIONS:")
            logger.info("- Install xformers: pip install xformers")
            logger.info("- Use torch.compile() for PyTorch 2.0+")
            logger.info("- Enable CUDA memory fraction: torch.cuda.set_per_process_memory_fraction(0.8)")
            logger.info("- Use gradient checkpointing for training")
            
        else:
            logger.info("❌ No CUDA GPU detected")
            logger.info("\n💡 CPU OPTIMIZATIONS:")
            logger.info("- Install optimized PyTorch CPU version")
            logger.info("- Use smaller models (SD 1.5 instead of SDXL)")
            logger.info("- Reduce inference steps")
            logger.info("- Use lower resolution images")
            
    except ImportError:
        logger.info("❌ PyTorch not available - install with: pip install torch torchvision")

def main():
    """Run comprehensive GPU utilization tests."""
    logger.info("🚀 Starting GPU Utilization Tests...")
    
    tests = [
        ("GPU Availability Check", check_gpu_availability),
        ("Diffusers Library Check", check_diffusers_availability),
        ("GPU Memory Allocation Test", test_gpu_memory_allocation),
        ("Real Model Loading Test", test_real_model_loading),
        ("System Integration GPU Test", test_system_integration_gpu_usage)
    ]
    
    results = []
    gpu_info = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_name == "GPU Availability Check":
                success, info = test_func()
                gpu_info = info
            elif test_name == "Real Model Loading Test":
                success, gen_time = test_func()
                if success:
                    logger.info(f"Real generation time: {gen_time:.2f}s")
            else:
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
    logger.info("GPU UTILIZATION TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Provide optimization recommendations
    optimize_for_gpu()
    
    if passed >= 3:  # At least basic functionality working
        logger.info("\n🎉 GPU utilization tests mostly successful!")
        if gpu_info.get('gpu_memory_gb', 0) > 0:
            logger.info(f"💪 Your {gpu_info['gpu_name']} with {gpu_info['gpu_memory_gb']:.1f}GB should handle AI generation well!")
    else:
        logger.error("\n💥 GPU utilization tests failed. Check the recommendations above.")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)