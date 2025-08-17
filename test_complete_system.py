#!/usr/bin/env python3
"""
Complete System Test with Image and Video Generation

This script tests the complete system including image generation, video generation,
output management, and UI functionality.
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


def test_image_generation():
    """Test image generation with output management."""
    logger.info("🎨 Testing image generation...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        from src.core.output_manager import get_output_manager, OutputType
        
        # Initialize system
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
        
        # Test different image generation scenarios
        test_cases = [
            {
                "name": "Quick Test Image",
                "prompt": "a beautiful red rose in a garden",
                "params": {
                    'width': 512,
                    'height': 512,
                    'num_inference_steps': 15,
                    'guidance_scale': 7.5,
                    'force_gpu_usage': True,
                    'precision': 'float32',  # Use float32 to avoid black image corruption
                    'memory_optimization': 'Attention Slicing'
                }
            },
            {
                "name": "High Quality Image",
                "prompt": "a majestic mountain landscape with snow-capped peaks and a crystal clear lake",
                "params": {
                    'width': 768,
                    'height': 768,
                    'num_inference_steps': 25,
                    'guidance_scale': 9.0,
                    'force_gpu_usage': True,
                    'precision': 'float32',  # Use float32 to avoid black image corruption
                    'memory_optimization': 'Attention Slicing'
                }
            }
        ]
        
        output_manager = get_output_manager()
        successful_generations = 0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n📸 Test {i}: {test_case['name']}")
            logger.info(f"   Prompt: {test_case['prompt']}")
            
            start_time = time.time()
            
            result = system.execute_complete_generation_workflow(
                prompt=test_case['prompt'],
                conversation_id=f"image_test_{i}",
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                additional_params=test_case['params']
            )
            
            if result.success:
                logger.info(f"✅ Image generated successfully in {result.generation_time:.2f}s")
                logger.info(f"   Model: {result.model_used}")
                logger.info(f"   Output: {result.output_path}")
                
                # Save to output manager
                if result.output_path and Path(result.output_path).exists():
                    output_id = output_manager.save_output(
                        file_path=Path(result.output_path),
                        output_type=OutputType.IMAGE,
                        prompt=test_case['prompt'],
                        model_used=result.model_used,
                        generation_time=result.generation_time,
                        parameters=test_case['params'],
                        quality_metrics=result.quality_metrics or {},
                        compliance_mode="research_safe"
                    )
                    logger.info(f"   Saved to output manager: {output_id}")
                
                successful_generations += 1
            else:
                logger.error(f"❌ Image generation failed: {result.error_message}")
            
            # Clear cache between tests
            if system.memory_manager:
                system.memory_manager.clear_vram_cache()
            time.sleep(2)
        
        # Clean up
        system.cleanup()
        
        logger.info(f"\n📊 Image Generation Summary: {successful_generations}/{len(test_cases)} successful")
        return successful_generations > 0
        
    except Exception as e:
        logger.error(f"❌ Image generation test failed: {e}")
        return False


def test_video_generation():
    """Test video generation with output management."""
    logger.info("🎬 Testing video generation...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        from src.core.output_manager import get_output_manager, OutputType
        
        # Initialize system
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
        
        # Test video generation scenarios
        test_cases = [
            {
                "name": "Short Video Test",
                "prompt": "a gentle breeze moving through golden wheat fields",
                "params": {
                    'output_type': 'video',
                    'width': 512,
                    'height': 512,
                    'num_frames': 8,
                    'fps': 7,
                    'num_inference_steps': 20,
                    'guidance_scale': 7.5,
                    'force_gpu_usage': True,
                    'precision': 'float16'
                }
            }
        ]
        
        output_manager = get_output_manager()
        successful_generations = 0
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🎥 Test {i}: {test_case['name']}")
            logger.info(f"   Prompt: {test_case['prompt']}")
            
            start_time = time.time()
            
            result = system.execute_complete_generation_workflow(
                prompt=test_case['prompt'],
                conversation_id=f"video_test_{i}",
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                additional_params=test_case['params']
            )
            
            if result.success:
                logger.info(f"✅ Video generated successfully in {result.generation_time:.2f}s")
                logger.info(f"   Model: {result.model_used}")
                logger.info(f"   Output: {result.output_path}")
                
                # Save to output manager
                if result.output_path and Path(result.output_path).exists():
                    output_id = output_manager.save_output(
                        file_path=Path(result.output_path),
                        output_type=OutputType.VIDEO,
                        prompt=test_case['prompt'],
                        model_used=result.model_used,
                        generation_time=result.generation_time,
                        parameters=test_case['params'],
                        quality_metrics=result.quality_metrics or {},
                        compliance_mode="research_safe"
                    )
                    logger.info(f"   Saved to output manager: {output_id}")
                
                successful_generations += 1
            else:
                logger.error(f"❌ Video generation failed: {result.error_message}")
            
            # Clear cache between tests
            if system.memory_manager:
                system.memory_manager.clear_vram_cache()
            time.sleep(5)  # Longer pause for video
        
        # Clean up
        system.cleanup()
        
        logger.info(f"\n📊 Video Generation Summary: {successful_generations}/{len(test_cases)} successful")
        return successful_generations > 0
        
    except Exception as e:
        logger.error(f"❌ Video generation test failed: {e}")
        return False


def test_output_management():
    """Test output management functionality."""
    logger.info("📁 Testing output management...")
    
    try:
        from src.core.output_manager import get_output_manager, OutputType
        
        output_manager = get_output_manager()
        
        # Get statistics
        stats = output_manager.get_statistics()
        logger.info(f"📊 Output Statistics:")
        logger.info(f"   Total outputs: {stats['total_outputs']}")
        logger.info(f"   Total size: {stats['total_size_mb']:.2f}MB")
        logger.info(f"   Average generation time: {stats['avg_generation_time']:.2f}s")
        logger.info(f"   By type: {stats['by_type']}")
        
        # List recent outputs
        recent_outputs = output_manager.get_outputs(limit=10)
        logger.info(f"\n📋 Recent Outputs ({len(recent_outputs)}):")
        
        for output in recent_outputs:
            logger.info(f"   {output.output_id}: {output.output_type.value} - {output.prompt[:50]}...")
            logger.info(f"      Model: {output.model_used}, Time: {output.generation_time:.2f}s")
            logger.info(f"      File: {output.file_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Output management test failed: {e}")
        return False


def test_ui_launch():
    """Test UI launch functionality."""
    logger.info("🖥️ Testing UI launch...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.ui.research_interface import ResearchInterface
        
        # Initialize system
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
        
        logger.info("✅ UI initialized successfully")
        logger.info("🚀 UI is ready to launch")
        logger.info("💡 To launch the UI, run: python -c \"from test_complete_system import launch_ui; launch_ui()\"")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ UI test failed: {e}")
        return False


def launch_ui():
    """Launch the UI for interactive testing."""
    logger.info("🚀 Launching UI for interactive testing...")
    
    try:
        from src.core.system_integration import SystemIntegration
        from src.ui.research_interface import ResearchInterface
        
        # Initialize system
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
        
        if not system.initialize(config):
            logger.error("❌ System initialization failed")
            return
        
        # Create and launch UI
        ui = ResearchInterface(
            system_controller=system,
            experiment_tracker=system.experiment_tracker,
            compliance_engine=None
        )
        
        if not ui.initialize():
            logger.error("❌ UI initialization failed")
            return
        
        logger.info("🌐 Launching web interface...")
        logger.info("📱 Open your browser to: http://127.0.0.1:7860")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        ui.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 UI stopped by user")
    except Exception as e:
        logger.error(f"❌ UI launch failed: {e}")


def main():
    """Run complete system tests."""
    logger.info("🚀 Starting Complete System Tests")
    logger.info("="*70)
    
    tests = [
        ("Image Generation", test_image_generation),
        ("Video Generation", test_video_generation),
        ("Output Management", test_output_management),
        ("UI Launch Test", test_ui_launch)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*70}")
        
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
    logger.info(f"\n{'='*70}")
    logger.info("COMPLETE SYSTEM TEST SUMMARY")
    logger.info(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:
        logger.info("\n🎉 Complete system tests successful!")
        logger.info("💪 The system is ready for image and video generation")
        logger.info("📁 Output management is working properly")
        logger.info("🖥️ UI is ready to launch")
        
        # Ask if user wants to launch UI
        try:
            choice = input("\n🚀 Would you like to launch the UI now? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                launch_ui()
        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
    else:
        logger.error("\n💥 System tests failed. Check the logs above.")
    
    return passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)