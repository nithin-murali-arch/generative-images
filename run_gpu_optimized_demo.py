#!/usr/bin/env python3
"""
GPU-Optimized Demo Runner

This script demonstrates the complete Academic Multimodal LLM system with
GPU optimization and real model usage.
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

def check_system_requirements():
    """Check if all system requirements are met."""
    logger.info("🔍 Checking system requirements...")
    
    requirements_met = True
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ CUDA not available - will use CPU")
    except ImportError:
        logger.error("❌ PyTorch not installed")
        requirements_met = False
    
    # Check Diffusers
    try:
        import diffusers
        logger.info(f"✅ Diffusers: {diffusers.__version__}")
    except ImportError:
        logger.error("❌ Diffusers not installed")
        requirements_met = False
    
    # Check other dependencies
    try:
        from PIL import Image
        logger.info("✅ PIL available")
    except ImportError:
        logger.error("❌ PIL not installed")
        requirements_met = False
    
    return requirements_met

def run_gpu_optimized_generation():
    """Run GPU-optimized image generation."""
    logger.info("🚀 Starting GPU-optimized generation demo...")
    
    try:
        # Import system components
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        from src.core.gpu_optimizer import get_gpu_optimizer
        
        # Initialize GPU optimizer
        gpu_optimizer = get_gpu_optimizer()
        
        # Show initial GPU status
        gpu_status = gpu_optimizer.monitor_gpu_usage()
        if gpu_status.get('gpu_available'):
            logger.info(f"🎮 GPU Status: {gpu_status['gpu_name']}")
            logger.info(f"   VRAM: {gpu_status['allocated_mb']:.1f}/{gpu_status['total_mb']:.1f} MB")
        else:
            logger.info("💻 Using CPU mode")
        
        # Create system with GPU-optimized config
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
        
        # Test different quality settings
        test_cases = [
            {
                "name": "Quick Test (512x512, 10 steps)",
                "prompt": "a red apple on a white background, simple, clean",
                "params": {
                    'width': 512,
                    'height': 512,
                    'num_inference_steps': 10,
                    'guidance_scale': 7.5,
                    'force_gpu_usage': True,
                    'precision': 'float16',
                    'memory_optimization': 'Attention Slicing (Balanced)'
                }
            },
            {
                "name": "High Quality Test (768x768, 25 steps)",
                "prompt": "a majestic mountain landscape with a crystal clear lake, photorealistic",
                "params": {
                    'width': 768,
                    'height': 768,
                    'num_inference_steps': 25,
                    'guidance_scale': 9.0,
                    'force_gpu_usage': True,
                    'precision': 'float16',
                    'memory_optimization': 'Attention Slicing (Balanced)'
                }
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}: {test_case['name']}")
            logger.info(f"{'='*60}")
            
            # Monitor GPU before generation
            gpu_before = gpu_optimizer.monitor_gpu_usage()
            
            # Generate
            start_time = time.time()
            result = system.execute_complete_generation_workflow(
                prompt=test_case['prompt'],
                conversation_id=f"gpu_demo_{i}",
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                additional_params=test_case['params']
            )
            generation_time = time.time() - start_time
            
            # Monitor GPU after generation
            gpu_after = gpu_optimizer.monitor_gpu_usage()
            
            if result.success:
                logger.info(f"✅ Generation successful!")
                logger.info(f"   Time: {result.generation_time:.2f}s")
                logger.info(f"   Model: {result.model_used}")
                logger.info(f"   Output: {result.output_path}")
                
                # Calculate GPU usage
                if gpu_before.get('gpu_available') and gpu_after.get('gpu_available'):
                    vram_used = gpu_after['allocated_mb'] - gpu_before['allocated_mb']
                    logger.info(f"   VRAM Used: {vram_used:.1f} MB")
                    
                    if vram_used > 100:
                        logger.info("   🎮 Significant GPU usage detected!")
                    else:
                        logger.warning("   ⚠️ Low GPU usage - might be using CPU fallback")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'time': result.generation_time,
                    'model': result.model_used,
                    'vram_used': vram_used if 'vram_used' in locals() else 0
                })
            else:
                logger.error(f"❌ Generation failed: {result.error_message}")
                results.append({
                    'test': test_case['name'],
                    'success': False,
                    'error': result.error_message
                })
            
            # Clear GPU cache between tests
            gpu_optimizer.clear_gpu_cache()
            time.sleep(2)  # Brief pause
        
        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("DEMO SUMMARY")
        logger.info(f"{'='*60}")
        
        successful_tests = [r for r in results if r['success']]
        
        if successful_tests:
            avg_time = sum(r['time'] for r in successful_tests) / len(successful_tests)
            total_vram = sum(r.get('vram_used', 0) for r in successful_tests)
            
            logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
            logger.info(f"📊 Average generation time: {avg_time:.2f}s")
            logger.info(f"🎮 Total VRAM used: {total_vram:.1f} MB")
            
            if total_vram > 500:
                logger.info("🚀 GPU is being utilized effectively!")
            else:
                logger.warning("⚠️ Low GPU usage detected - check GPU setup")
        else:
            logger.error("❌ No successful generations")
        
        # Get optimization recommendations
        if gpu_optimizer.gpu_available:
            recommendations = gpu_optimizer.get_optimization_recommendations(
                width=768, height=768, 
                current_config=gpu_optimizer.get_optimal_config(768, 768)
            )
            
            logger.info(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for rec in recommendations:
                logger.info(f"   • {rec}")
        
        # Clean up
        system.cleanup()
        
        return len(successful_tests) > 0
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def run_ui_demo():
    """Run the UI demo."""
    logger.info("🖥️ Starting UI demo...")
    
    try:
        # Import UI components
        from src.core.system_integration import SystemIntegration
        from src.ui.research_interface import ResearchInterface
        
        # Create system
        system = SystemIntegration()
        
        config = {
            "data_dir": "data",
            "models_dir": "models",
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "auto_detect_hardware": True,
            "max_concurrent_generations": 1,
            "ui_host": "127.0.0.1",
            "ui_port": 7860,
            "ui_share": False
        }
        
        logger.info("⚙️ Initializing system for UI...")
        if not system.initialize(config):
            logger.error("❌ System initialization failed")
            return False
        
        # Create UI
        ui = ResearchInterface(
            system_controller=system,
            experiment_tracker=system.experiment_tracker,
            compliance_engine=None
        )
        
        logger.info("🖥️ Initializing UI...")
        if not ui.initialize():
            logger.warning("⚠️ UI initialization failed (Gradio not available)")
            logger.info("💡 Install Gradio to use the web interface: pip install gradio")
            return False
        
        logger.info("🚀 Launching web interface...")
        logger.info("📱 Open your browser to: http://127.0.0.1:7860")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        # Launch UI
        ui.launch(
            share=config.get("ui_share", False),
            server_name=config.get("ui_host", "127.0.0.1"),
            server_port=config.get("ui_port", 7860)
        )
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 UI demo stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ UI demo failed: {e}")
        return False

def main():
    """Main demo runner."""
    logger.info("🎯 Academic Multimodal LLM System - GPU Optimized Demo")
    logger.info("="*60)
    
    # Check requirements
    if not check_system_requirements():
        logger.error("❌ System requirements not met")
        logger.info("\n💡 To install requirements:")
        logger.info("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        logger.info("   pip install diffusers transformers accelerate")
        logger.info("   pip install pillow gradio")
        return False
    
    # Ask user what to run
    print("\n🎮 What would you like to run?")
    print("1. GPU-Optimized Generation Demo (automated tests)")
    print("2. Web UI Demo (interactive interface)")
    print("3. Both")
    
    try:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            success = run_gpu_optimized_generation()
        elif choice == "2":
            success = run_ui_demo()
        elif choice == "3":
            logger.info("Running generation demo first...")
            success1 = run_gpu_optimized_generation()
            
            if success1:
                logger.info("\n" + "="*60)
                logger.info("Generation demo completed. Starting UI demo...")
                logger.info("="*60)
                success = run_ui_demo()
            else:
                success = False
        else:
            logger.error("❌ Invalid choice")
            return False
        
        if success:
            logger.info("\n🎉 Demo completed successfully!")
            return True
        else:
            logger.error("\n💥 Demo failed - check the logs above")
            return False
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        return True
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)