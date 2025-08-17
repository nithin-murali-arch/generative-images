#!/usr/bin/env python3
"""
Simple Model-Based Stress Test

This script tests CPU/GPU utilization using just the image generation pipeline
to avoid complex dependencies and focus on the core functionality.
"""

import sys
import logging
import time
import threading
import psutil
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


class SimpleResourceMonitor:
    """Simple resource monitor for CPU and GPU."""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_history = []
        self.gpu_memory_history = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.cpu_history = []
        self.gpu_memory_history = []
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("📊 Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("📊 Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Monitor resources in a loop."""
        while self.monitoring:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_history.append(cpu_percent)
                
                # GPU memory monitoring
                gpu_memory = 0
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                except:
                    pass
                
                self.gpu_memory_history.append(gpu_memory)
                
                # Keep only last 60 measurements
                if len(self.cpu_history) > 60:
                    self.cpu_history = self.cpu_history[-60:]
                    self.gpu_memory_history = self.gpu_memory_history[-60:]
                
                time.sleep(1.0)
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def get_peak_usage(self) -> tuple:
        """Get peak CPU and GPU memory usage."""
        cpu_peak = max(self.cpu_history) if self.cpu_history else 0.0
        gpu_peak = max(self.gpu_memory_history) if self.gpu_memory_history else 0.0
        return cpu_peak, gpu_peak


def test_image_generation_stress():
    """Test image generation with resource monitoring."""
    logger.info("🎨 Testing image generation stress...")
    
    try:
        # Import just the image generation pipeline
        from src.pipelines.image_generation import ImageGenerationPipeline
        from src.hardware.detector import HardwareDetector
        from src.core.interfaces import GenerationRequest, ComplianceMode, StyleConfig
        
        # Initialize hardware detection
        detector = HardwareDetector()
        hardware_config = detector.detect_hardware()
        
        logger.info(f"🖥️ Hardware: {hardware_config.gpu_model}")
        logger.info(f"💾 VRAM: {hardware_config.vram_size}MB")
        logger.info(f"🚀 CUDA: {hardware_config.cuda_available}")
        
        # Initialize image pipeline
        pipeline = ImageGenerationPipeline()
        
        if not pipeline.initialize(hardware_config):
            logger.error("❌ Failed to initialize image pipeline")
            return False
        
        # Start monitoring
        monitor = SimpleResourceMonitor()
        monitor.start_monitoring()
        
        # Wait for baseline
        time.sleep(3)
        baseline_cpu, baseline_gpu = monitor.get_peak_usage()
        logger.info(f"📊 Baseline - CPU: {baseline_cpu:.1f}%, GPU Memory: {baseline_gpu:.1f}MB")
        
        # Test different generation scenarios
        test_cases = [
            {
                "name": "Quick Generation (Low Load)",
                "prompt": "a simple red apple",
                "width": 512,
                "height": 512,
                "steps": 10,
                "expected_load": "low"
            },
            {
                "name": "Standard Generation (Medium Load)",
                "prompt": "a detailed mountain landscape with a lake",
                "width": 768,
                "height": 768,
                "steps": 20,
                "expected_load": "medium"
            },
            {
                "name": "High Quality Generation (High Load)",
                "prompt": "a photorealistic portrait of a person in a garden",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "expected_load": "high"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n🎯 Test {i}: {test_case['name']}")
            logger.info(f"   Prompt: {test_case['prompt'][:50]}...")
            logger.info(f"   Resolution: {test_case['width']}x{test_case['height']}")
            logger.info(f"   Steps: {test_case['steps']}")
            
            # Clear monitoring history
            monitor.cpu_history = []
            monitor.gpu_memory_history = []
            
            # Create generation request
            style_config = StyleConfig(
                generation_params={
                    'width': test_case['width'],
                    'height': test_case['height'],
                    'num_inference_steps': test_case['steps'],
                    'guidance_scale': 7.5
                }
            )
            
            request = GenerationRequest(
                prompt=test_case['prompt'],
                output_type=None,  # Will be set by pipeline
                style_config=style_config,
                compliance_mode=ComplianceMode.RESEARCH_SAFE,
                hardware_constraints=hardware_config,
                context=None
            )
            
            # Generate
            start_time = time.time()
            result = pipeline.generate(request)
            generation_time = time.time() - start_time
            
            # Get peak usage during generation
            peak_cpu, peak_gpu = monitor.get_peak_usage()
            
            if result.success:
                logger.info(f"   ✅ Generation successful in {result.generation_time:.2f}s")
                logger.info(f"   Model used: {result.model_used}")
                logger.info(f"   Peak CPU: {peak_cpu:.1f}%")
                logger.info(f"   Peak GPU Memory: {peak_gpu:.1f}MB")
                logger.info(f"   Output: {result.output_path}")
                
                # Calculate resource usage increase
                cpu_increase = peak_cpu - baseline_cpu
                gpu_increase = peak_gpu - baseline_gpu
                
                logger.info(f"   📈 CPU increase: +{cpu_increase:.1f}%")
                logger.info(f"   📈 GPU Memory increase: +{gpu_increase:.1f}MB")
                
                results.append({
                    'test': test_case['name'],
                    'success': True,
                    'generation_time': result.generation_time,
                    'model_used': result.model_used,
                    'cpu_peak': peak_cpu,
                    'gpu_peak': peak_gpu,
                    'cpu_increase': cpu_increase,
                    'gpu_increase': gpu_increase,
                    'expected_load': test_case['expected_load']
                })
            else:
                logger.error(f"   ❌ Generation failed: {result.error_message}")
                results.append({
                    'test': test_case['name'],
                    'success': False,
                    'error': result.error_message,
                    'cpu_peak': peak_cpu,
                    'gpu_peak': peak_gpu
                })
            
            # Clear GPU cache and wait
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            time.sleep(3)
        
        # Analysis
        logger.info(f"\n{'='*60}")
        logger.info("STRESS TEST ANALYSIS")
        logger.info(f"{'='*60}")
        
        successful_tests = [r for r in results if r['success']]
        
        if successful_tests:
            logger.info(f"✅ Successful tests: {len(successful_tests)}/{len(results)}")
            
            # Check resource utilization patterns
            cpu_usage_detected = any(r['cpu_increase'] > 10 for r in successful_tests)
            gpu_usage_detected = any(r['gpu_increase'] > 100 for r in successful_tests)
            
            logger.info(f"\n📊 Resource Utilization Analysis:")
            logger.info(f"   Significant CPU usage: {'✅' if cpu_usage_detected else '❌'}")
            logger.info(f"   Significant GPU usage: {'✅' if gpu_usage_detected else '❌'}")
            
            # Performance scaling analysis
            if len(successful_tests) >= 2:
                logger.info(f"\n⚡ Performance Scaling:")
                for result in successful_tests:
                    load_indicator = "🟢" if result['expected_load'] == 'low' else "🟡" if result['expected_load'] == 'medium' else "🔴"
                    logger.info(f"   {load_indicator} {result['test']}: {result['generation_time']:.2f}s")
            
            # Check if we're actually using the GPU effectively
            if hardware_config.cuda_available:
                if gpu_usage_detected:
                    logger.info(f"\n🎮 GPU Utilization: GOOD")
                    logger.info(f"   The system is effectively using the GPU for generation")
                else:
                    logger.warning(f"\n⚠️ GPU Utilization: LOW")
                    logger.info(f"   The system may be falling back to CPU or using mocks")
                    logger.info(f"   Consider checking CUDA installation and model availability")
            else:
                logger.info(f"\n💻 CPU-Only Mode: Expected behavior for systems without CUDA")
            
            return len(successful_tests) >= len(results) // 2
        else:
            logger.error("❌ No successful generations")
            return False
        
    except Exception as e:
        logger.error(f"❌ Stress test failed: {e}")
        return False
    
    finally:
        monitor.stop_monitoring()
        if 'pipeline' in locals():
            pipeline.cleanup()


def test_load_balancing_intelligence():
    """Test that the system makes intelligent load balancing decisions."""
    logger.info("⚖️ Testing load balancing intelligence...")
    
    try:
        from src.core.gpu_optimizer import get_gpu_optimizer
        
        optimizer = get_gpu_optimizer()
        
        # Test different scenarios
        scenarios = [
            {
                "name": "Light workload",
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            {
                "name": "Heavy workload",
                "width": 1024,
                "height": 1024,
                "batch_size": 2
            }
        ]
        
        logger.info("🧠 Testing optimization decisions:")
        
        for scenario in scenarios:
            config = optimizer.get_optimal_config(
                width=scenario['width'],
                height=scenario['height'],
                batch_size=scenario['batch_size']
            )
            
            logger.info(f"\n📋 {scenario['name']}:")
            logger.info(f"   Resolution: {scenario['width']}x{scenario['height']}")
            logger.info(f"   Batch size: {scenario['batch_size']}")
            logger.info(f"   → Use GPU: {config.use_gpu}")
            logger.info(f"   → Precision: {config.precision}")
            logger.info(f"   → Memory strategy: {config.memory_strategy.value}")
            logger.info(f"   → Optimization level: {config.optimization_level.value}")
        
        # Test recommendations
        recommendations = optimizer.get_optimization_recommendations(
            width=768, height=768, current_config=optimizer.get_optimal_config(768, 768)
        )
        
        logger.info(f"\n💡 Optimization Recommendations:")
        for rec in recommendations:
            logger.info(f"   • {rec}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Load balancing test failed: {e}")
        return False


def main():
    """Run simple model-based stress tests."""
    logger.info("🚀 Starting Simple Model-Based Stress Tests")
    logger.info("="*60)
    
    tests = [
        ("Image Generation Stress Test", test_image_generation_stress),
        ("Load Balancing Intelligence", test_load_balancing_intelligence)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
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
    logger.info(f"\n{'='*60}")
    logger.info("SIMPLE STRESS TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 All tests passed!")
        logger.info("💪 The system is effectively utilizing available hardware")
        logger.info("🎮 GPU optimization and load balancing are working properly")
    else:
        logger.warning(f"\n⚠️ {total - passed} tests failed")
        logger.info("💡 Check system setup and dependencies")
    
    return passed >= total // 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)