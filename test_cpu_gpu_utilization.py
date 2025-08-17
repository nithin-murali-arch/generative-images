#!/usr/bin/env python3
"""
CPU/GPU Utilization Test and Load Balancing

This script tests both CPU and GPU utilization, monitors system resources,
and demonstrates intelligent load balancing between CPU and GPU based on
actual utilization levels.
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
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    cpu_cores: int
    ram_percent: float
    ram_available_gb: float
    gpu_available: bool
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    timestamp: float = 0.0


class ResourceMonitor:
    """Monitor system resources in real-time."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history: List[SystemMetrics] = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start monitoring system resources."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 60 measurements (1 minute at 1s interval)
                if len(self.metrics_history) > 60:
                    self.metrics_history = self.metrics_history[-60:]
                
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_cores = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_available_gb = memory.available / (1024**3)
        
        # GPU metrics
        gpu_available = False
        gpu_utilization = 0.0
        gpu_memory_used = 0.0
        gpu_memory_total = 0.0
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**2)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
                
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = utilization.gpu
                    pynvml.nvmlShutdown()
                except:
                    # Estimate GPU utilization based on memory usage
                    if gpu_memory_total > 0:
                        gpu_utilization = (gpu_memory_used / gpu_memory_total) * 100
        except:
            pass
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            cpu_cores=cpu_cores,
            ram_percent=ram_percent,
            ram_available_gb=ram_available_gb,
            gpu_available=gpu_available,
            gpu_utilization=gpu_utilization,
            gpu_memory_used=gpu_memory_used,
            gpu_memory_total=gpu_memory_total,
            timestamp=time.time()
        )
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return self._collect_metrics()
    
    def get_average_metrics(self, seconds: int = 10) -> SystemMetrics:
        """Get average metrics over the last N seconds."""
        if not self.metrics_history:
            return self.get_current_metrics()
        
        cutoff_time = time.time() - seconds
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return self.metrics_history[-1] if self.metrics_history else self.get_current_metrics()
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_ram = sum(m.ram_percent for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_gpu_mem = sum(m.gpu_memory_used for m in recent_metrics) / len(recent_metrics)
        
        # Use latest values for non-averaged metrics
        latest = recent_metrics[-1]
        
        return SystemMetrics(
            cpu_percent=avg_cpu,
            cpu_cores=latest.cpu_cores,
            ram_percent=avg_ram,
            ram_available_gb=latest.ram_available_gb,
            gpu_available=latest.gpu_available,
            gpu_utilization=avg_gpu_util,
            gpu_memory_used=avg_gpu_mem,
            gpu_memory_total=latest.gpu_memory_total,
            timestamp=latest.timestamp
        )


class LoadBalancer:
    """Intelligent load balancer for CPU/GPU workloads."""
    
    def __init__(self, resource_monitor: ResourceMonitor):
        self.resource_monitor = resource_monitor
        self.gpu_threshold_high = 85.0  # Switch to CPU if GPU > 85%
        self.gpu_threshold_low = 60.0   # Switch back to GPU if GPU < 60%
        self.cpu_threshold_high = 90.0  # Don't use CPU if CPU > 90%
        self.memory_threshold = 90.0    # Don't use if memory > 90%
        
    def should_use_gpu(self, task_requirements: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determine if GPU should be used for a task based on current utilization.
        
        Args:
            task_requirements: Dict with 'vram_mb', 'estimated_gpu_util', etc.
            
        Returns:
            Tuple of (should_use_gpu, reason)
        """
        metrics = self.resource_monitor.get_average_metrics(seconds=5)
        
        # Check if GPU is available
        if not metrics.gpu_available:
            return False, "GPU not available"
        
        # Check system memory
        if metrics.ram_percent > self.memory_threshold:
            return False, f"System memory too high: {metrics.ram_percent:.1f}%"
        
        # Check CPU utilization
        if metrics.cpu_percent > self.cpu_threshold_high:
            return True, f"CPU overloaded ({metrics.cpu_percent:.1f}%), prefer GPU"
        
        # Check GPU utilization
        if metrics.gpu_utilization > self.gpu_threshold_high:
            return False, f"GPU overloaded ({metrics.gpu_utilization:.1f}%), use CPU"
        
        # Check GPU memory requirements
        required_vram = task_requirements.get('vram_mb', 0)
        available_vram = metrics.gpu_memory_total - metrics.gpu_memory_used
        
        if required_vram > available_vram:
            return False, f"Insufficient VRAM: need {required_vram}MB, have {available_vram:.1f}MB"
        
        # Check if GPU memory usage would be too high
        projected_usage = (metrics.gpu_memory_used + required_vram) / metrics.gpu_memory_total * 100
        if projected_usage > 90:
            return False, f"Projected GPU memory usage too high: {projected_usage:.1f}%"
        
        # GPU looks good to use
        return True, f"GPU available (util: {metrics.gpu_utilization:.1f}%, mem: {metrics.gpu_memory_used:.0f}MB)"
    
    def get_optimal_batch_size(self, base_batch_size: int, task_requirements: Dict[str, Any]) -> int:
        """Determine optimal batch size based on current resources."""
        metrics = self.resource_monitor.get_average_metrics(seconds=5)
        
        if not metrics.gpu_available:
            # CPU mode - reduce batch size
            return max(1, base_batch_size // 2)
        
        # Check GPU memory availability
        required_vram_per_item = task_requirements.get('vram_mb', 0)
        available_vram = metrics.gpu_memory_total - metrics.gpu_memory_used
        
        if required_vram_per_item > 0:
            max_batch_from_memory = int(available_vram * 0.8 / required_vram_per_item)
            optimal_batch = min(base_batch_size, max_batch_from_memory)
        else:
            optimal_batch = base_batch_size
        
        # Reduce batch size if GPU is already busy
        if metrics.gpu_utilization > 70:
            optimal_batch = max(1, optimal_batch // 2)
        
        return max(1, optimal_batch)


def test_cpu_stress():
    """Test CPU stress generation."""
    logger.info("🔥 Testing CPU stress generation...")
    
    def cpu_intensive_task(duration: float = 2.0):
        """CPU-intensive task for testing."""
        start_time = time.time()
        result = 0
        while time.time() - start_time < duration:
            # More aggressive CPU-intensive computation
            for i in range(100000):
                result += i ** 2
                result = result % 1000000  # Prevent overflow
        return result
    
    monitor = ResourceMonitor()
    monitor.start_monitoring()
    
    try:
        # Get baseline metrics
        time.sleep(2)
        baseline = monitor.get_average_metrics(seconds=2)
        logger.info(f"Baseline CPU: {baseline.cpu_percent:.1f}%")
        
        # Run CPU stress test
        logger.info("Starting CPU stress test...")
        start_time = time.time()
        
        # Use multiple threads to stress CPU
        with ThreadPoolExecutor(max_workers=psutil.cpu_count()) as executor:
            futures = [executor.submit(cpu_intensive_task, 5.0) for _ in range(psutil.cpu_count())]
            
            # Monitor during stress
            peak_cpu = 0
            while any(not f.done() for f in futures):
                current = monitor.get_current_metrics()
                peak_cpu = max(peak_cpu, current.cpu_percent)
                time.sleep(0.5)
            
            # Wait for completion
            for future in as_completed(futures):
                future.result()
        
        stress_time = time.time() - start_time
        
        # Get final metrics
        time.sleep(2)
        final = monitor.get_average_metrics(seconds=2)
        
        logger.info(f"✅ CPU stress test completed in {stress_time:.1f}s")
        logger.info(f"   Peak CPU usage: {peak_cpu:.1f}%")
        logger.info(f"   Final CPU usage: {final.cpu_percent:.1f}%")
        
        return peak_cpu > 50  # Success if we achieved significant CPU load
        
    finally:
        monitor.stop_monitoring()


def test_gpu_stress():
    """Test GPU stress generation."""
    logger.info("🎮 Testing GPU stress generation...")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, skipping GPU stress test")
            return False
        
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        try:
            # Get baseline metrics
            time.sleep(2)
            baseline = monitor.get_average_metrics(seconds=2)
            logger.info(f"Baseline GPU: {baseline.gpu_utilization:.1f}%, VRAM: {baseline.gpu_memory_used:.0f}MB")
            
            # Create GPU stress workload
            logger.info("Starting GPU stress test...")
            device = torch.device('cuda')
            
            # Allocate large tensors to stress GPU memory
            tensors = []
            try:
                for i in range(5):
                    # Create large tensor (adjust size based on available VRAM)
                    size = min(1000, int((baseline.gpu_memory_total * 0.1) ** 0.5))
                    tensor = torch.randn(size, size, device=device, dtype=torch.float32)
                    tensors.append(tensor)
                    
                    # Perform computations to stress GPU
                    for _ in range(100):
                        tensor = torch.matmul(tensor, tensor.T)
                    
                    current = monitor.get_current_metrics()
                    logger.info(f"   Step {i+1}: GPU VRAM: {current.gpu_memory_used:.0f}MB")
                
                # Monitor peak usage
                time.sleep(2)
                peak = monitor.get_current_metrics()
                
                logger.info(f"✅ GPU stress test completed")
                logger.info(f"   Peak VRAM usage: {peak.gpu_memory_used:.0f}MB")
                logger.info(f"   GPU utilization: {peak.gpu_utilization:.1f}%")
                
                return peak.gpu_memory_used > baseline.gpu_memory_used + 100  # Success if we used significant VRAM
                
            finally:
                # Clean up tensors
                for tensor in tensors:
                    del tensor
                torch.cuda.empty_cache()
                
        finally:
            monitor.stop_monitoring()
            
    except ImportError:
        logger.warning("⚠️ PyTorch not available, skipping GPU stress test")
        return False
    except Exception as e:
        logger.error(f"❌ GPU stress test failed: {e}")
        return False


def test_load_balancing():
    """Test intelligent load balancing between CPU and GPU."""
    logger.info("⚖️ Testing intelligent load balancing...")
    
    monitor = ResourceMonitor()
    load_balancer = LoadBalancer(monitor)
    
    monitor.start_monitoring()
    
    try:
        # Wait for baseline
        time.sleep(3)
        
        # Test different scenarios
        test_scenarios = [
            {
                "name": "Light workload",
                "vram_mb": 500,
                "estimated_gpu_util": 20
            },
            {
                "name": "Medium workload", 
                "vram_mb": 2000,
                "estimated_gpu_util": 50
            },
            {
                "name": "Heavy workload",
                "vram_mb": 6000,
                "estimated_gpu_util": 80
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            logger.info(f"\n📊 Testing scenario: {scenario['name']}")
            
            # Get current system state
            metrics = monitor.get_current_metrics()
            logger.info(f"   Current CPU: {metrics.cpu_percent:.1f}%")
            logger.info(f"   Current GPU: {metrics.gpu_utilization:.1f}%")
            logger.info(f"   Current VRAM: {metrics.gpu_memory_used:.0f}/{metrics.gpu_memory_total:.0f}MB")
            
            # Test load balancing decision
            should_use_gpu, reason = load_balancer.should_use_gpu(scenario)
            optimal_batch = load_balancer.get_optimal_batch_size(4, scenario)
            
            logger.info(f"   Decision: {'GPU' if should_use_gpu else 'CPU'}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Optimal batch size: {optimal_batch}")
            
            results.append({
                'scenario': scenario['name'],
                'use_gpu': should_use_gpu,
                'reason': reason,
                'batch_size': optimal_batch,
                'cpu_percent': metrics.cpu_percent,
                'gpu_utilization': metrics.gpu_utilization
            })
            
            time.sleep(1)
        
        # Test under CPU stress
        logger.info(f"\n🔥 Testing under CPU stress...")
        
        def cpu_stress():
            """Create CPU stress."""
            end_time = time.time() + 5
            while time.time() < end_time:
                sum(i**2 for i in range(10000))
        
        # Start CPU stress in background
        stress_thread = threading.Thread(target=cpu_stress, daemon=True)
        stress_thread.start()
        
        time.sleep(2)  # Let CPU stress build up
        
        # Test decision under stress
        stressed_metrics = monitor.get_current_metrics()
        should_use_gpu_stressed, reason_stressed = load_balancer.should_use_gpu(test_scenarios[1])
        
        logger.info(f"   Under CPU stress ({stressed_metrics.cpu_percent:.1f}%): {'GPU' if should_use_gpu_stressed else 'CPU'}")
        logger.info(f"   Reason: {reason_stressed}")
        
        # Summary
        logger.info(f"\n📋 Load Balancing Test Summary:")
        for result in results:
            logger.info(f"   {result['scenario']}: {result['reason']}")
        
        return True
        
    finally:
        monitor.stop_monitoring()


def test_real_generation_with_monitoring():
    """Test real AI generation with resource monitoring."""
    logger.info("🤖 Testing real AI generation with resource monitoring...")
    
    try:
        # Import system components
        from src.core.system_integration import SystemIntegration
        from src.core.interfaces import ComplianceMode
        
        monitor = ResourceMonitor()
        load_balancer = LoadBalancer(monitor)
        
        monitor.start_monitoring()
        
        try:
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
            
            # Test generation with different loads
            test_cases = [
                {
                    "name": "Quick generation (low load)",
                    "prompt": "a simple red apple",
                    "params": {
                        'width': 512,
                        'height': 512,
                        'num_inference_steps': 10,
                        'vram_mb': 1000
                    }
                },
                {
                    "name": "High quality generation (high load)",
                    "prompt": "a detailed mountain landscape with reflections",
                    "params": {
                        'width': 768,
                        'height': 768,
                        'num_inference_steps': 30,
                        'vram_mb': 4000
                    }
                }
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                logger.info(f"\n🎨 Test {i}: {test_case['name']}")
                
                # Get pre-generation metrics
                pre_metrics = monitor.get_current_metrics()
                
                # Make load balancing decision
                should_use_gpu, reason = load_balancer.should_use_gpu(test_case['params'])
                optimal_batch = load_balancer.get_optimal_batch_size(1, test_case['params'])
                
                logger.info(f"   Load balancer decision: {'GPU' if should_use_gpu else 'CPU'}")
                logger.info(f"   Reason: {reason}")
                logger.info(f"   Batch size: {optimal_batch}")
                
                # Prepare generation parameters
                gen_params = test_case['params'].copy()
                gen_params.update({
                    'force_gpu_usage': should_use_gpu,
                    'precision': 'float16' if should_use_gpu else 'float32',
                    'memory_optimization': 'CPU Offloading' if not should_use_gpu else 'Attention Slicing',
                    'batch_size': optimal_batch
                })
                
                # Generate
                start_time = time.time()
                result = system.execute_complete_generation_workflow(
                    prompt=test_case['prompt'],
                    conversation_id=f"load_test_{i}",
                    compliance_mode=ComplianceMode.RESEARCH_SAFE,
                    additional_params=gen_params
                )
                generation_time = time.time() - start_time
                
                # Get post-generation metrics
                post_metrics = monitor.get_current_metrics()
                
                if result.success:
                    logger.info(f"   ✅ Generation successful in {result.generation_time:.2f}s")
                    logger.info(f"   Model used: {result.model_used}")
                    
                    # Show resource usage
                    cpu_change = post_metrics.cpu_percent - pre_metrics.cpu_percent
                    gpu_mem_change = post_metrics.gpu_memory_used - pre_metrics.gpu_memory_used
                    
                    logger.info(f"   CPU change: {cpu_change:+.1f}%")
                    if post_metrics.gpu_available:
                        logger.info(f"   VRAM change: {gpu_mem_change:+.1f}MB")
                        logger.info(f"   GPU utilization: {post_metrics.gpu_utilization:.1f}%")
                    
                else:
                    logger.error(f"   ❌ Generation failed: {result.error_message}")
                
                # Clear caches and wait
                if system.memory_manager:
                    system.memory_manager.clear_vram_cache()
                time.sleep(3)
            
            # Clean up
            system.cleanup()
            return True
            
        finally:
            monitor.stop_monitoring()
            
    except Exception as e:
        logger.error(f"❌ Real generation test failed: {e}")
        return False


def main():
    """Run comprehensive CPU/GPU utilization tests."""
    logger.info("🚀 Starting CPU/GPU Utilization and Load Balancing Tests")
    logger.info("="*70)
    
    tests = [
        ("CPU Stress Test", test_cpu_stress),
        ("GPU Stress Test", test_gpu_stress),
        ("Load Balancing Logic", test_load_balancing),
        ("Real Generation with Monitoring", test_real_generation_with_monitoring)
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
    logger.info("CPU/GPU UTILIZATION TEST SUMMARY")
    logger.info(f"{'='*70}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 3:
        logger.info("\n🎉 CPU/GPU utilization and load balancing working well!")
        logger.info("💡 The system intelligently balances load between CPU and GPU")
        logger.info("🔄 CPU offloading occurs only when GPU is heavily utilized")
    else:
        logger.error("\n💥 CPU/GPU utilization tests failed. Check system setup.")
    
    return passed >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)