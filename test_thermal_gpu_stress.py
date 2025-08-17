#!/usr/bin/env python3
"""
Thermal-Aware GPU Stress Test

This script tests CPU/GPU utilization with thermal monitoring and safety.
It ensures GPU is actually used before falling back to CPU and implements
thermal protection to prevent overheating.
"""

import sys
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_gpu_functionality():
    """Test that GPU is actually functional before proceeding."""
    logger.info("🔍 Testing GPU functionality...")
    
    try:
        from src.core.gpu_optimizer import get_gpu_optimizer
        
        optimizer = get_gpu_optimizer()
        is_functional, message = optimizer.verify_gpu_functionality()
        
        logger.info(f"GPU Functionality Test: {message}")
        
        if is_functional:
            logger.info("✅ GPU is functional and ready for AI workloads")
            return True
        else:
            logger.warning(f"⚠️ GPU functionality issue: {message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ GPU functionality test failed: {e}")
        return False


def test_thermal_monitoring():
    """Test thermal monitoring system."""
    logger.info("🌡️ Testing thermal monitoring...")
    
    try:
        from src.core.thermal_monitor import get_thermal_monitor
        
        monitor = get_thermal_monitor()
        monitor.start_monitoring()
        
        # Wait for initial readings
        time.sleep(3)
        
        temps = monitor.get_current_temperatures()
        
        logger.info("🌡️ Current System Temperatures:")
        if temps['cpu_temp']:
            logger.info(f"   CPU: {temps['cpu_temp']:.1f}°C")
        else:
            logger.warning("   CPU: Temperature not available")
        
        if temps['gpu_temp']:
            logger.info(f"   GPU: {temps['gpu_temp']:.1f}°C")
        else:
            logger.info("   GPU: Temperature not available")
        
        logger.info(f"   Thermal State: {temps['thermal_state']}")
        logger.info(f"   Throttled: {temps['is_throttled']}")
        
        # Check if system is safe to proceed
        if temps['is_throttled']:
            logger.warning("⚠️ System is currently throttled due to temperature")
            logger.info("🧊 Waiting for system to cool down...")
            
            if monitor.wait_for_cooling(max_wait_time=60):
                logger.info("✅ System cooled down, safe to proceed")
            else:
                logger.error("❌ System did not cool down in time")
                return False
        
        monitor.stop_monitoring()
        return True
        
    except Exception as e:
        logger.error(f"❌ Thermal monitoring test failed: {e}")
        return False


def test_gpu_stress_with_thermal_protection():
    """Test GPU stress with thermal protection."""
    logger.info("🔥 Testing GPU stress with thermal protection...")
    
    try:
        from src.core.thermal_monitor import get_thermal_monitor
        from src.pipelines.image_generation import ImageGenerationPipeline
        from src.hardware.detector import HardwareDetector
        from src.core.interfaces import GenerationRequest, ComplianceMode, StyleConfig
        
        # Initialize thermal monitoring
        thermal_monitor = get_thermal_monitor()
        thermal_monitor.start_monitoring()
        
        # Initialize hardware and pipeline
        detector = HardwareDetector()
        hardware_config = detector.detect_hardware()
        
        pipeline = ImageGenerationPipeline()
        if not pipeline.initialize(hardware_config):
            logger.error("❌ Failed to initialize image pipeline")
            return False
        
        logger.info(f"🖥️ Hardware: {hardware_config.gpu_model}")
        logger.info(f"💾 VRAM: {hardware_config.vram_size}MB")
        logger.info(f"🚀 CUDA: {hardware_config.cuda_available}")
        
        # Test cases with increasing GPU load
        test_cases = [
            {
                "name": "GPU Warm-up Test",
                "prompt": "a simple geometric shape",
                "width": 512,
                "height": 512,
                "steps": 10,
                "expected_gpu_usage": "low"
            },
            {
                "name": "GPU Load Test",
                "prompt": "a detailed photorealistic landscape with mountains and lakes",
                "width": 768,
                "height": 768,
                "steps": 25,
                "expected_gpu_usage": "medium"
            },
            {
                "name": "GPU Stress Test",
                "prompt": "a highly detailed photorealistic portrait with complex lighting and textures",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "expected_gpu_usage": "high"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🎯 Test {i}: {test_case['name']}")
            
            # Check thermal state before starting
            temps_before = thermal_monitor.get_current_temperatures()
            logger.info(f"   Pre-test temperatures:")
            if temps_before['cpu_temp']:
                logger.info(f"     CPU: {temps_before['cpu_temp']:.1f}°C")
            if temps_before['gpu_temp']:
                logger.info(f"     GPU: {temps_before['gpu_temp']:.1f}°C")
            
            # Check if we need to wait for cooling
            if thermal_monitor.should_throttle_operations():
                logger.warning("🌡️ System too hot, waiting for cooling...")
                if not thermal_monitor.wait_for_cooling(max_wait_time=120):
                    logger.error("❌ System did not cool down, aborting test")
                    break
            
            # Monitor GPU memory before generation
            gpu_memory_before = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / (1024**2)
            except:
                pass
            
            # Create generation request with GPU enforcement
            style_config = StyleConfig(
                generation_params={
                    'width': test_case['width'],
                    'height': test_case['height'],
                    'num_inference_steps': test_case['steps'],
                    'guidance_scale': 7.5,
                    'force_gpu_usage': True,  # Force GPU usage
                    'precision': 'float16',   # Use GPU-optimized precision
                    'memory_optimization': 'Attention Slicing (Balanced)'
                }
            )
            
            request = GenerationRequest(
                prompt=test_case['prompt'],
                output_type=None,
                style_config=style_config,
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=hardware_config,
                context=None,
                additional_params={
                    'force_real_generation': True,  # Ensure real model usage
                    'force_gpu_usage': True
                }
            )
            
            # Generate with thermal monitoring
            logger.info(f"   🎨 Generating: {test_case['prompt'][:50]}...")
            logger.info(f"   📐 Resolution: {test_case['width']}x{test_case['height']}")
            logger.info(f"   🔢 Steps: {test_case['steps']}")
            
            start_time = time.time()
            result = pipeline.generate(request)
            generation_time = time.time() - start_time
            
            # Monitor GPU memory after generation
            gpu_memory_after = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / (1024**2)
            except:
                pass
            
            gpu_memory_used = gpu_memory_after - gpu_memory_before
            
            # Check thermal state after generation
            temps_after = thermal_monitor.get_current_temperatures()
            
            if result.success:
                logger.info(f"   ✅ Generation successful in {result.generation_time:.2f}s")
                logger.info(f"   🤖 Model used: {result.model_used}")
                logger.info(f"   💾 GPU Memory used: {gpu_memory_used:.1f}MB")
                logger.info(f"   📁 Output: {result.output_path}")
                
                # Temperature analysis
                logger.info(f"   🌡️ Post-generation temperatures:")
                if temps_after['cpu_temp']:
                    temp_change = temps_after['cpu_temp'] - (temps_before['cpu_temp'] or 0)
                    logger.info(f"     CPU: {temps_after['cpu_temp']:.1f}°C ({temp_change:+.1f}°C)")
                if temps_after['gpu_temp']:
                    temp_change = temps_after['gpu_temp'] - (temps_before['gpu_temp'] or 0)
                    logger.info(f"     GPU: {temps_after['gpu_temp']:.1f}°C ({temp_change:+.1f}°C)")
                
                # Verify GPU usage
                gpu_usage_detected = gpu_memory_used > 100  # More than 100MB used
                
                if gpu_usage_detected:
                    logger.info(f"   🎮 GPU Usage: CONFIRMED ({gpu_memory_used:.1f}MB)")
                else:
                    logger.warning(f"   ⚠️ GPU Usage: LOW or NOT DETECTED ({gpu_memory_used:.1f}MB)")
                    logger.warning(f"   💡 System may be using CPU fallback or mock generation")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'generation_time': result.generation_time,
                    'model_used': result.model_used,
                    'gpu_memory_used': gpu_memory_used,
                    'gpu_usage_confirmed': gpu_usage_detected,
                    'cpu_temp_after': temps_after['cpu_temp'],
                    'gpu_temp_after': temps_after['gpu_temp'],
                    'thermal_state': temps_after['thermal_state']
                })
                
            else:
                logger.error(f"   ❌ Generation failed: {result.error_message}")
                results.append({
                    'test': test_case['name'],
                    'success': False,
                    'error': result.error_message,
                    'gpu_memory_used': gpu_memory_used,
                    'cpu_temp_after': temps_after['cpu_temp'],
                    'gpu_temp_after': temps_after['gpu_temp']
                })
            
            # Clear GPU cache and wait between tests
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            # Wait for thermal stabilization
            logger.info("   ⏳ Waiting for thermal stabilization...")
            time.sleep(10)
        
        # Analysis
        logger.info(f"\n{'='*70}")
        logger.info("THERMAL-AWARE GPU STRESS TEST ANALYSIS")
        logger.info(f"{'='*70}")
        
        successful_tests = [r for r in results if r['success']]
        gpu_confirmed_tests = [r for r in successful_tests if r.get('gpu_usage_confirmed', False)]
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Total tests: {len(results)}")
        logger.info(f"   Successful: {len(successful_tests)}")
        logger.info(f"   GPU usage confirmed: {len(gpu_confirmed_tests)}")
        
        if gpu_confirmed_tests:
            logger.info(f"\n🎮 GPU Usage Analysis:")
            total_gpu_memory = sum(r['gpu_memory_used'] for r in gpu_confirmed_tests)
            avg_gpu_memory = total_gpu_memory / len(gpu_confirmed_tests)
            logger.info(f"   Average GPU memory per generation: {avg_gpu_memory:.1f}MB")
            logger.info(f"   Total GPU memory used: {total_gpu_memory:.1f}MB")
            
            logger.info(f"\n✅ GPU is being effectively utilized!")
        else:
            logger.warning(f"\n⚠️ No confirmed GPU usage detected!")
            logger.info(f"💡 Possible causes:")
            logger.info(f"   - Models not properly loaded")
            logger.info(f"   - CUDA not properly configured")
            logger.info(f"   - System falling back to CPU/mock generation")
        
        # Thermal analysis
        max_cpu_temp = max((r.get('cpu_temp_after') or 0) for r in results)
        max_gpu_temp = max((r.get('gpu_temp_after') or 0) for r in results)
        
        logger.info(f"\n🌡️ Thermal Analysis:")
        if max_cpu_temp > 0:
            logger.info(f"   Peak CPU temperature: {max_cpu_temp:.1f}°C")
            if max_cpu_temp > 80:
                logger.warning(f"   ⚠️ CPU ran hot during testing")
            elif max_cpu_temp > 65:
                logger.info(f"   🔥 CPU warmed up but stayed within safe limits")
            else:
                logger.info(f"   ❄️ CPU stayed cool during testing")
        
        if max_gpu_temp > 0:
            logger.info(f"   Peak GPU temperature: {max_gpu_temp:.1f}°C")
            if max_gpu_temp > 85:
                logger.warning(f"   ⚠️ GPU ran hot during testing")
            elif max_gpu_temp > 75:
                logger.info(f"   🔥 GPU warmed up but stayed within safe limits")
            else:
                logger.info(f"   ❄️ GPU stayed cool during testing")
        
        thermal_monitor.stop_monitoring()
        pipeline.cleanup()
        
        return len(gpu_confirmed_tests) > 0  # Success if we confirmed GPU usage
        
    except Exception as e:
        logger.error(f"❌ GPU stress test failed: {e}")
        return False


def main():
    """Run comprehensive thermal-aware GPU stress tests."""
    logger.info("🚀 Starting Thermal-Aware GPU Stress Tests")
    logger.info("="*70)
    
    tests = [
        ("GPU Functionality Test", test_gpu_functionality),
        ("Thermal Monitoring Test", test_thermal_monitoring),
        ("GPU Stress with Thermal Protection", test_gpu_stress_with_thermal_protection)
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
    logger.info("THERMAL-AWARE GPU STRESS TEST SUMMARY")
    logger.info(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        logger.info("🎮 GPU is being effectively utilized")
        logger.info("🌡️ Thermal protection is working properly")
        logger.info("⚖️ System intelligently balances performance and safety")
    elif passed >= 2:
        logger.info("\n✅ Most tests passed!")
        logger.info("💡 System is functional with some limitations")
    else:
        logger.error("\n💥 Multiple tests failed")
        logger.info("🔧 Check system setup, GPU drivers, and thermal sensors")
    
    return passed >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)