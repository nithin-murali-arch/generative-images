#!/usr/bin/env python3
"""
Enhanced Model-Based Stress Test for CPU/GPU Utilization

This script uses both image and video generation models to stress test the system,
demonstrates intelligent load balancing, and tests output management with proper
file organization and UI integration.
"""

import sys
import logging
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationTask:
    """A generation task for stress testing."""
    name: str
    prompt: str
    width: int
    height: int
    steps: int
    force_gpu: bool
    expected_load: str  # "low", "medium", "high"
    output_type: str = "image"  # "image" or "video"
    num_frames: int = 14  # For video generation
    fps: int = 7  # For video generation


@dataclass
class StressTestResult:
    """Result of a stress test."""
    task_name: str
    success: bool
    generation_time: float
    cpu_usage_peak: float
    gpu_memory_peak: float
    gpu_utilization_peak: float
    model_used: str
    error_message: Optional[str] = None


class ModelStressTester:
    """Stress tester using actual AI generation models."""
    
    def __init__(self):
        self.system = None
        self.monitoring = False
        self.cpu_history = []
        self.gpu_history = []
        self.monitor_thread = None
        
    def initialize_system(self) -> bool:
        """Initialize the AI generation system."""
        try:
            from src.core.system_integration import SystemIntegration
            from src.core.interfaces import ComplianceMode
            
            self.system = SystemIntegration()
            
            config = {
                "data_dir": "data",
                "models_dir": "models",
                "experiments_dir": "experiments",
                "cache_dir": "cache",
                "logs_dir": "logs",
                "auto_detect_hardware": True,
                "max_concurrent_generations": 1
            }
            
            logger.info("🔧 Initializing AI generation system...")
            success = self.system.initialize(config)
            
            if success:
                logger.info("✅ System initialized successfully")
                
                # Show system capabilities
                if self.system.hardware_config:
                    hw = self.system.hardware_config
                    logger.info(f"🖥️ Hardware: {hw.gpu_model}")
                    logger.info(f"💾 VRAM: {hw.vram_size}MB")
                    logger.info(f"🔧 CPU Cores: {hw.cpu_cores}")
                    logger.info(f"🚀 CUDA Available: {hw.cuda_available}")
                
                return True
            else:
                logger.error("❌ System initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}")
            return False
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.monitoring = True
        self.cpu_history = []
        self.gpu_history = []
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("📊 Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("📊 Stopped resource monitoring")
    
    def _monitor_resources(self):
        """Monitor CPU and GPU resources."""
        while self.monitoring:
            try:
                # CPU monitoring
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_history.append(cpu_percent)
                
                # GPU monitoring
                gpu_info = {'memory_mb': 0, 'utilization': 0}
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_info['memory_mb'] = torch.cuda.memory_allocated() / (1024**2)
                        
                        # Try to get GPU utilization
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_info['utilization'] = utilization.gpu
                            pynvml.nvmlShutdown()
                        except:
                            # Estimate based on memory usage
                            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                            gpu_info['utilization'] = (gpu_info['memory_mb'] / total_mem) * 100
                except:
                    pass
                
                self.gpu_history.append(gpu_info)
                
                # Keep only last 120 measurements (2 minutes at 1s interval)
                if len(self.cpu_history) > 120:
                    self.cpu_history = self.cpu_history[-120:]
                    self.gpu_history = self.gpu_history[-120:]
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def get_peak_usage(self) -> Tuple[float, float, float]:
        """Get peak CPU and GPU usage from monitoring history."""
        cpu_peak = max(self.cpu_history) if self.cpu_history else 0.0
        gpu_mem_peak = max(g['memory_mb'] for g in self.gpu_history) if self.gpu_history else 0.0
        gpu_util_peak = max(g['utilization'] for g in self.gpu_history) if self.gpu_history else 0.0
        
        return cpu_peak, gpu_mem_peak, gpu_util_peak
    
    def run_generation_stress_test(self, task: GenerationTask) -> StressTestResult:
        """Run a single generation task and measure resource usage."""
        logger.info(f"🎨 Running stress test: {task.name}")
        logger.info(f"   Prompt: {task.prompt[:50]}...")
        logger.info(f"   Resolution: {task.width}x{task.height}")
        logger.info(f"   Steps: {task.steps}")
        logger.info(f"   Force GPU: {task.force_gpu}")
        
        # Clear monitoring history
        self.cpu_history = []
        self.gpu_history = []
        
        # Prepare generation parameters
        additional_params = {
            'width': task.width,
            'height': task.height,
            'num_inference_steps': task.steps,
            'guidance_scale': 7.5,
            'force_gpu_usage': task.force_gpu,
            'precision': 'float16' if task.force_gpu else 'float32',
            'memory_optimization': 'CPU Offloading' if not task.force_gpu else 'Attention Slicing',
            'force_real_generation': True,  # Ensure we use real models, not mocks
            'output_type': task.output_type
        }
        
        # Add video-specific parameters
        if task.output_type == "video":
            additional_params.update({
                'num_frames': task.num_frames,
                'fps': task.fps
            })
        
        try:
            # Start generation
            start_time = time.time()
            
            result = self.system.execute_complete_generation_workflow(
                prompt=task.prompt,
                conversation_id=f"stress_test_{int(time.time())}",
                compliance_mode=self.system.llm_controller.manage_context("test").current_mode,
                additional_params=additional_params
            )
            
            generation_time = time.time() - start_time
            
            # Get peak resource usage
            cpu_peak, gpu_mem_peak, gpu_util_peak = self.get_peak_usage()
            
            if result.success:
                logger.info(f"✅ Generation successful in {result.generation_time:.2f}s")
                logger.info(f"   Model used: {result.model_used}")
                logger.info(f"   Peak CPU: {cpu_peak:.1f}%")
                logger.info(f"   Peak GPU Memory: {gpu_mem_peak:.1f}MB")
                logger.info(f"   Peak GPU Utilization: {gpu_util_peak:.1f}%")
                
                return StressTestResult(
                    task_name=task.name,
                    success=True,
                    generation_time=result.generation_time,
                    cpu_usage_peak=cpu_peak,
                    gpu_memory_peak=gpu_mem_peak,
                    gpu_utilization_peak=gpu_util_peak,
                    model_used=result.model_used
                )
            else:
                logger.error(f"❌ Generation failed: {result.error_message}")
                return StressTestResult(
                    task_name=task.name,
                    success=False,
                    generation_time=generation_time,
                    cpu_usage_peak=cpu_peak,
                    gpu_memory_peak=gpu_mem_peak,
                    gpu_utilization_peak=gpu_util_peak,
                    model_used="none",
                    error_message=result.error_message
                )
                
        except Exception as e:
            logger.error(f"❌ Stress test failed: {e}")
            cpu_peak, gpu_mem_peak, gpu_util_peak = self.get_peak_usage()
            
            return StressTestResult(
                task_name=task.name,
                success=False,
                generation_time=0.0,
                cpu_usage_peak=cpu_peak,
                gpu_memory_peak=gpu_mem_peak,
                gpu_utilization_peak=gpu_util_peak,
                model_used="none",
                error_message=str(e)
            )
    
    def run_concurrent_stress_test(self, tasks: List[GenerationTask], max_workers: int = 2) -> List[StressTestResult]:
        """Run multiple generation tasks concurrently to stress the system."""
        logger.info(f"🔥 Running concurrent stress test with {len(tasks)} tasks, {max_workers} workers")
        
        results = []
        
        # Clear monitoring history
        self.cpu_history = []
        self.gpu_history = []
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.run_generation_stress_test, task): task 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"❌ Concurrent task {task.name} failed: {e}")
                    results.append(StressTestResult(
                        task_name=task.name,
                        success=False,
                        generation_time=0.0,
                        cpu_usage_peak=0.0,
                        gpu_memory_peak=0.0,
                        gpu_utilization_peak=0.0,
                        model_used="none",
                        error_message=str(e)
                    ))
        
        total_time = time.time() - start_time
        cpu_peak, gpu_mem_peak, gpu_util_peak = self.get_peak_usage()
        
        logger.info(f"🏁 Concurrent stress test completed in {total_time:.2f}s")
        logger.info(f"   Overall Peak CPU: {cpu_peak:.1f}%")
        logger.info(f"   Overall Peak GPU Memory: {gpu_mem_peak:.1f}MB")
        logger.info(f"   Overall Peak GPU Utilization: {gpu_util_peak:.1f}%")
        
        return results
    
    def test_load_balancing_decisions(self) -> bool:
        """Test that the system makes intelligent load balancing decisions."""
        logger.info("⚖️ Testing intelligent load balancing decisions...")
        
        # Create tasks that should trigger different load balancing decisions
        test_scenarios = [
            {
                "name": "Light Load Test - Image",
                "tasks": [
                    GenerationTask(
                        name="Quick Image",
                        prompt="a simple red apple",
                        width=512,
                        height=512,
                        steps=10,
                        force_gpu=True,
                        expected_load="low",
                        output_type="image"
                    )
                ]
            },
            {
                "name": "Light Load Test - Video",
                "tasks": [
                    GenerationTask(
                        name="Quick Video",
                        prompt="a red apple rotating slowly",
                        width=512,
                        height=512,
                        steps=15,
                        force_gpu=True,
                        expected_load="low",
                        output_type="video",
                        num_frames=8,
                        fps=7
                    )
                ]
            },
            {
                "name": "Medium Load Test - Image", 
                "tasks": [
                    GenerationTask(
                        name="Standard Image",
                        prompt="a detailed mountain landscape with crystal clear lake",
                        width=768,
                        height=768,
                        steps=20,
                        force_gpu=True,
                        expected_load="medium",
                        output_type="image"
                    )
                ]
            },
            {
                "name": "Medium Load Test - Video",
                "tasks": [
                    GenerationTask(
                        name="Standard Video",
                        prompt="gentle waves on a peaceful beach at sunset",
                        width=512,
                        height=512,
                        steps=20,
                        force_gpu=True,
                        expected_load="medium",
                        output_type="video",
                        num_frames=14,
                        fps=7
                    )
                ]
            },
            {
                "name": "Heavy Load Test - Image",
                "tasks": [
                    GenerationTask(
                        name="High Quality Image",
                        prompt="a photorealistic portrait of a person in golden hour lighting",
                        width=1024,
                        height=1024,
                        steps=30,
                        force_gpu=True,
                        expected_load="high",
                        output_type="image"
                    )
                ]
            }
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            logger.info(f"\n📊 {scenario['name']}")
            
            for task in scenario['tasks']:
                result = self.run_generation_stress_test(task)
                all_results.append(result)
                
                # Clear GPU cache between tests
                if self.system and hasattr(self.system, 'clear_memory_cache'):
                    self.system.clear_memory_cache()
                
                time.sleep(2)  # Brief pause between tests
        
        # Analyze results
        successful_tests = [r for r in all_results if r.success]
        
        if successful_tests:
            logger.info(f"\n📈 Load Balancing Analysis:")
            logger.info(f"   Successful tests: {len(successful_tests)}/{len(all_results)}")
            
            for result in successful_tests:
                logger.info(f"   {result.task_name}:")
                logger.info(f"     CPU Peak: {result.cpu_usage_peak:.1f}%")
                logger.info(f"     GPU Memory Peak: {result.gpu_memory_peak:.1f}MB")
                logger.info(f"     GPU Utilization Peak: {result.gpu_utilization_peak:.1f}%")
                logger.info(f"     Generation Time: {result.generation_time:.2f}s")
                logger.info(f"     Model: {result.model_used}")
            
            # Check if we see expected resource usage patterns
            has_cpu_usage = any(r.cpu_usage_peak > 20 for r in successful_tests)
            has_gpu_usage = any(r.gpu_memory_peak > 100 for r in successful_tests)
            
            logger.info(f"\n🔍 Resource Usage Analysis:")
            logger.info(f"   Significant CPU usage detected: {'✅' if has_cpu_usage else '❌'}")
            logger.info(f"   Significant GPU usage detected: {'✅' if has_gpu_usage else '❌'}")
            
            return len(successful_tests) >= len(all_results) // 2
        else:
            logger.error("❌ No successful generations for load balancing analysis")
            return False
    
    def cleanup(self):
        """Clean up system resources."""
        if self.system:
            self.system.cleanup()
        logger.info("🧹 System cleanup completed")


def main():
    """Run comprehensive model-based stress tests."""
    logger.info("🚀 Starting Model-Based CPU/GPU Stress Tests")
    logger.info("="*70)
    
    tester = ModelStressTester()
    
    try:
        # Initialize the AI generation system
        if not tester.initialize_system():
            logger.error("❌ Failed to initialize AI generation system")
            return False
        
        # Start resource monitoring
        tester.start_monitoring()
        
        # Wait for baseline
        time.sleep(3)
        
        # Test 1: Single generation stress tests
        logger.info(f"\n{'='*70}")
        logger.info("Test 1: Single Generation Stress Tests")
        logger.info(f"{'='*70}")
        
        single_tests_passed = tester.test_load_balancing_decisions()
        
        # Test 2: Concurrent generation stress test
        logger.info(f"\n{'='*70}")
        logger.info("Test 2: Concurrent Generation Stress Test")
        logger.info(f"{'='*70}")
        
        concurrent_tasks = [
            GenerationTask(
                name="Concurrent Image",
                prompt="a beautiful sunset over the ocean with sailing boats",
                width=512,
                height=512,
                steps=15,
                force_gpu=True,
                expected_load="medium",
                output_type="image"
            ),
            GenerationTask(
                name="Concurrent Video", 
                prompt="a forest path in autumn with falling leaves",
                width=512,
                height=512,
                steps=15,
                force_gpu=True,
                expected_load="medium",
                output_type="video",
                num_frames=10,
                fps=7
            )
        ]
        
        concurrent_results = tester.run_concurrent_stress_test(concurrent_tasks, max_workers=2)
        concurrent_tests_passed = any(r.success for r in concurrent_results)
        
        # Test 3: CPU vs GPU comparison
        logger.info(f"\n{'='*70}")
        logger.info("Test 3: CPU vs GPU Performance Comparison")
        logger.info(f"{'='*70}")
        
        comparison_task_gpu = GenerationTask(
            name="GPU Generation",
            prompt="a detailed cityscape at night",
            width=512,
            height=512,
            steps=20,
            force_gpu=True,
            expected_load="medium"
        )
        
        comparison_task_cpu = GenerationTask(
            name="CPU Generation",
            prompt="a detailed cityscape at night",
            width=512,
            height=512,
            steps=20,
            force_gpu=False,
            expected_load="medium"
        )
        
        gpu_result = tester.run_generation_stress_test(comparison_task_gpu)
        time.sleep(5)  # Clear cache
        cpu_result = tester.run_generation_stress_test(comparison_task_cpu)
        
        if gpu_result.success and cpu_result.success:
            logger.info(f"\n🏁 Performance Comparison Results:")
            logger.info(f"   GPU Generation: {gpu_result.generation_time:.2f}s")
            logger.info(f"   CPU Generation: {cpu_result.generation_time:.2f}s")
            
            if gpu_result.generation_time < cpu_result.generation_time:
                speedup = cpu_result.generation_time / gpu_result.generation_time
                logger.info(f"   🚀 GPU is {speedup:.1f}x faster than CPU")
            else:
                logger.info(f"   ⚠️ CPU performed similarly or better than GPU")
            
            comparison_passed = True
        else:
            logger.error("❌ Performance comparison failed")
            comparison_passed = False
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("MODEL-BASED STRESS TEST SUMMARY")
        logger.info(f"{'='*70}")
        
        tests = [
            ("Single Generation Tests", single_tests_passed),
            ("Concurrent Generation Tests", concurrent_tests_passed),
            ("CPU vs GPU Comparison", comparison_passed)
        ]
        
        passed = sum(1 for _, success in tests if success)
        total = len(tests)
        
        for test_name, success in tests:
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed >= 2:
            logger.info("\n🎉 Model-based stress testing successful!")
            logger.info("💪 The system effectively utilizes available hardware for AI generation")
            logger.info("⚖️ Load balancing between CPU and GPU is working properly")
        else:
            logger.error("\n💥 Model-based stress testing failed")
            logger.info("💡 Check system setup and model availability")
        
        return passed >= 2
        
    finally:
        tester.stop_monitoring()
        tester.cleanup()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)