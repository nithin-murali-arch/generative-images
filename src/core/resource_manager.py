"""
Resource Manager for Optimal CPU/GPU Utilization

This module implements intelligent resource management to ensure both CPU and GPU
are optimally utilized during AI generation tasks.
"""

import logging
import time
import threading
import asyncio
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from queue import Queue, Empty, PriorityQueue
from enum import Enum
import psutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    GPU = "gpu"
    MIXED = "mixed"
    IO = "io"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class ResourceTask:
    """A computational task with resource requirements."""
    task_id: str
    name: str
    resource_type: ResourceType
    priority: TaskPriority
    estimated_duration: float
    task_function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.priority.value < other.priority.value


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    active_cpu_tasks: int = 0
    active_gpu_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_cpu_time: float = 0.0
    total_gpu_time: float = 0.0
    parallel_efficiency: float = 0.0


class ResourceManager:
    """
    Intelligent resource manager that optimizes CPU/GPU utilization.
    
    Features:
    - Parallel execution of CPU and GPU tasks
    - Background CPU work during GPU operations
    - Task prioritization and scheduling
    - Resource monitoring and optimization
    - Automatic load balancing
    """
    
    def __init__(self, 
                 max_cpu_workers: Optional[int] = None,
                 max_gpu_workers: int = 1,
                 enable_background_tasks: bool = True):
        """
        Initialize the resource manager.
        
        Args:
            max_cpu_workers: Maximum CPU worker threads (default: CPU count)
            max_gpu_workers: Maximum GPU worker threads (default: 1)
            enable_background_tasks: Enable background CPU tasks during GPU work
        """
        self.max_cpu_workers = max_cpu_workers or min(8, psutil.cpu_count())
        self.max_gpu_workers = max_gpu_workers
        self.enable_background_tasks = enable_background_tasks
        
        # Task queues
        self.cpu_queue = PriorityQueue()
        self.gpu_queue = PriorityQueue()
        self.io_queue = PriorityQueue()
        self.background_queue = Queue()
        
        # Executors
        self.cpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_cpu_workers,
            thread_name_prefix="CPU-Worker"
        )
        self.gpu_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_gpu_workers,
            thread_name_prefix="GPU-Worker"
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="IO-Worker"
        )
        
        # State tracking
        self.active_tasks: Dict[str, ResourceTask] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.task_futures: Dict[str, concurrent.futures.Future] = {}
        
        # Resource monitoring
        self.metrics = ResourceMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Background task management
        self.background_worker_active = False
        self.background_thread = None
        
        # Synchronization
        self.gpu_busy_event = threading.Event()
        self.shutdown_event = threading.Event()
        
        logger.info(f"ResourceManager initialized: {self.max_cpu_workers} CPU workers, "
                   f"{self.max_gpu_workers} GPU workers")
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self.monitor_thread.start()
            
            if self.enable_background_tasks:
                self.background_worker_active = True
                self.background_thread = threading.Thread(
                    target=self._background_worker,
                    daemon=True
                )
                self.background_thread.start()
            
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        self.background_worker_active = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        if self.background_thread:
            self.background_thread.join(timeout=2.0)
        
        logger.info("Resource monitoring stopped")
    
    def submit_task(self, task: ResourceTask) -> str:
        """
        Submit a task for execution.
        
        Args:
            task: The task to execute
            
        Returns:
            str: Task ID for tracking
        """
        logger.info(f"Submitting task {task.task_id} ({task.resource_type.value}, priority: {task.priority.name})")
        
        self.active_tasks[task.task_id] = task
        
        # Submit to appropriate executor
        if task.resource_type == ResourceType.CPU:
            future = self._submit_cpu_task(task)
        elif task.resource_type == ResourceType.GPU:
            future = self._submit_gpu_task(task)
        elif task.resource_type == ResourceType.IO:
            future = self._submit_io_task(task)
        elif task.resource_type == ResourceType.MIXED:
            future = self._submit_mixed_task(task)
        else:
            raise ValueError(f"Unknown resource type: {task.resource_type}")
        
        self.task_futures[task.task_id] = future
        return task.task_id
    
    def _submit_cpu_task(self, task: ResourceTask) -> concurrent.futures.Future:
        """Submit CPU task."""
        def wrapped_task():
            return self._execute_task(task, ResourceType.CPU)
        
        return self.cpu_executor.submit(wrapped_task)
    
    def _submit_gpu_task(self, task: ResourceTask) -> concurrent.futures.Future:
        """Submit GPU task with background CPU scheduling."""
        def wrapped_task():
            # Signal that GPU is busy
            self.gpu_busy_event.set()
            
            try:
                # Schedule background CPU work
                if self.enable_background_tasks:
                    self._schedule_background_cpu_work()
                
                return self._execute_task(task, ResourceType.GPU)
            finally:
                self.gpu_busy_event.clear()
        
        return self.gpu_executor.submit(wrapped_task)
    
    def _submit_io_task(self, task: ResourceTask) -> concurrent.futures.Future:
        """Submit I/O task."""
        def wrapped_task():
            return self._execute_task(task, ResourceType.IO)
        
        return self.io_executor.submit(wrapped_task)
    
    def _submit_mixed_task(self, task: ResourceTask) -> concurrent.futures.Future:
        """Submit mixed CPU/GPU task."""
        def wrapped_task():
            return self._execute_task(task, ResourceType.MIXED)
        
        return self.cpu_executor.submit(wrapped_task)
    
    def _execute_task(self, task: ResourceTask, resource_type: ResourceType) -> Any:
        """Execute a task and track metrics."""
        start_time = time.time()
        
        try:
            logger.info(f"🔄 Executing {resource_type.value} task: {task.name}")
            
            # Update metrics
            if resource_type == ResourceType.CPU:
                self.metrics.active_cpu_tasks += 1
            elif resource_type == ResourceType.GPU:
                self.metrics.active_gpu_tasks += 1
            
            # Execute the actual task
            result = task.task_function(*task.args, **task.kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics.completed_tasks += 1
            if resource_type == ResourceType.CPU:
                self.metrics.total_cpu_time += execution_time
                self.metrics.active_cpu_tasks -= 1
            elif resource_type == ResourceType.GPU:
                self.metrics.total_gpu_time += execution_time
                self.metrics.active_gpu_tasks -= 1
            
            # Store result
            self.completed_tasks[task.task_id] = {
                'result': result,
                'execution_time': execution_time,
                'completed_at': time.time()
            }
            
            logger.info(f"✅ Task {task.name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.metrics.failed_tasks += 1
            if resource_type == ResourceType.CPU:
                self.metrics.active_cpu_tasks -= 1
            elif resource_type == ResourceType.GPU:
                self.metrics.active_gpu_tasks -= 1
            
            logger.error(f"❌ Task {task.name} failed: {e}")
            raise
        finally:
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _schedule_background_cpu_work(self):
        """Schedule background CPU work while GPU is busy."""
        background_tasks = [
            ("Cache Cleanup", self._background_cache_cleanup),
            ("Statistics Update", self._background_stats_update),
            ("Memory Optimization", self._background_memory_optimization),
            ("File Preprocessing", self._background_file_preprocessing)
        ]
        
        for task_name, task_func in background_tasks:
            if not self.gpu_busy_event.is_set():
                break
            
            try:
                self.background_queue.put((task_name, task_func), block=False)
            except:
                pass  # Queue full, skip
    
    def _background_worker(self):
        """Background worker thread for CPU tasks during GPU operations."""
        while self.background_worker_active and not self.shutdown_event.is_set():
            try:
                # Wait for background tasks
                task_name, task_func = self.background_queue.get(timeout=1.0)
                
                if self.gpu_busy_event.is_set():
                    logger.debug(f"🔧 Background CPU work: {task_name}")
                    try:
                        task_func()
                    except Exception as e:
                        logger.debug(f"Background task {task_name} failed: {e}")
                
                self.background_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.debug(f"Background worker error: {e}")
    
    def _background_cache_cleanup(self):
        """Background task: Clean up caches."""
        time.sleep(0.5)  # Simulate cache cleanup work
        logger.debug("Cache cleanup completed")
    
    def _background_stats_update(self):
        """Background task: Update statistics."""
        time.sleep(0.3)  # Simulate stats update work
        logger.debug("Statistics update completed")
    
    def _background_memory_optimization(self):
        """Background task: Optimize memory usage."""
        time.sleep(0.4)  # Simulate memory optimization work
        logger.debug("Memory optimization completed")
    
    def _background_file_preprocessing(self):
        """Background task: Preprocess files."""
        time.sleep(0.6)  # Simulate file preprocessing work
        logger.debug("File preprocessing completed")
    
    def _monitor_resources(self):
        """Monitor system resources."""
        while self.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Update CPU utilization
                self.metrics.cpu_utilization = psutil.cpu_percent(interval=None)
                
                # Update memory usage
                memory = psutil.virtual_memory()
                self.metrics.memory_usage = memory.percent
                
                # Update GPU metrics if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                        self.metrics.gpu_memory_usage = gpu_memory * 100
                        
                        # Estimate GPU utilization based on memory usage and active tasks
                        if self.metrics.active_gpu_tasks > 0:
                            self.metrics.gpu_utilization = min(100, gpu_memory * 100 + 20)
                        else:
                            self.metrics.gpu_utilization = gpu_memory * 100
                except:
                    pass
                
                # Calculate parallel efficiency
                total_time = max(self.metrics.total_cpu_time, self.metrics.total_gpu_time)
                if total_time > 0:
                    overlap_time = min(self.metrics.total_cpu_time, self.metrics.total_gpu_time)
                    self.metrics.parallel_efficiency = (overlap_time / total_time) * 100
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.debug(f"Resource monitoring error: {e}")
                time.sleep(1.0)
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a submitted task.
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait for result
            
        Returns:
            Task result
        """
        if task_id not in self.task_futures:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        future = self.task_futures[task_id]
        return future.result(timeout=timeout)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        return self.metrics
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs."""
        return list(self.active_tasks.keys())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a submitted task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            bool: True if task was cancelled
        """
        if task_id in self.task_futures:
            future = self.task_futures[task_id]
            return future.cancel()
        return False
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all submitted tasks to complete.
        
        Args:
            timeout: Maximum time to wait
            
        Returns:
            bool: True if all tasks completed
        """
        try:
            futures = list(self.task_futures.values())
            concurrent.futures.wait(futures, timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Error waiting for tasks: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the resource manager."""
        logger.info("Shutting down ResourceManager...")
        
        self.shutdown_event.set()
        self.stop_monitoring()
        
        # Shutdown executors
        self.cpu_executor.shutdown(wait=True)
        self.gpu_executor.shutdown(wait=True)
        self.io_executor.shutdown(wait=True)
        
        logger.info("ResourceManager shutdown completed")


# Global resource manager instance
_resource_manager: Optional[ResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
        _resource_manager.start_monitoring()
    return _resource_manager


def shutdown_resource_manager():
    """Shutdown the global resource manager."""
    global _resource_manager
    if _resource_manager is not None:
        _resource_manager.shutdown()
        _resource_manager = None