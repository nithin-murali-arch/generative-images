"""
Memory management for AI workloads.

This module provides memory management capabilities for GPU and CPU memory
to prevent out-of-memory errors during AI generation tasks.
"""

import logging
import gc
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .interfaces import HardwareConfig

logger = logging.getLogger(__name__)

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    total_mb: int
    used_mb: int
    free_mb: int
    utilization: float


class MemoryManager:
    """
    Manages memory allocation and cleanup for AI workloads.
    
    Provides GPU and CPU memory monitoring, cleanup, and optimization
    to prevent out-of-memory errors during generation tasks.
    """
    
    def __init__(self, hardware_config: HardwareConfig):
        """Initialize memory manager with hardware configuration."""
        self.hardware_config = hardware_config
        self.vram_size_mb = hardware_config.vram_size
        self.ram_size_mb = hardware_config.ram_size
        
        # Memory thresholds (as fraction of total)
        self.vram_warning_threshold = 0.8
        self.vram_critical_threshold = 0.9
        self.ram_warning_threshold = 0.85
        self.ram_critical_threshold = 0.95
        
        logger.info(f"MemoryManager initialized: {self.vram_size_mb}MB VRAM, {self.ram_size_mb}MB RAM")
    
    def get_gpu_memory_usage(self) -> Optional[MemoryUsage]:
        """Get current GPU memory usage."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None
        
        try:
            # Get memory stats from PyTorch
            allocated = torch.cuda.memory_allocated() // (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() // (1024 * 1024)    # MB
            total = self.vram_size_mb
            
            return MemoryUsage(
                total_mb=total,
                used_mb=reserved,
                free_mb=total - reserved,
                utilization=reserved / total if total > 0 else 0.0
            )
        except Exception as e:
            logger.error(f"Failed to get GPU memory usage: {e}")
            return None
    
    def get_cpu_memory_usage(self) -> Optional[MemoryUsage]:
        """Get current CPU memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return MemoryUsage(
                total_mb=memory.total // (1024 * 1024),
                used_mb=memory.used // (1024 * 1024),
                free_mb=memory.available // (1024 * 1024),
                utilization=memory.percent / 100.0
            )
        except ImportError:
            logger.warning("psutil not available - CPU memory monitoring disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to get CPU memory usage: {e}")
            return None
    
    def cleanup_gpu_memory(self) -> bool:
        """Clean up GPU memory."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        
        try:
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("GPU memory cleanup completed")
            return True
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")
            return False
    
    def cleanup_cpu_memory(self) -> bool:
        """Clean up CPU memory."""
        try:
            # Force garbage collection
            gc.collect()
            
            logger.debug("CPU memory cleanup completed")
            return True
        except Exception as e:
            logger.error(f"CPU memory cleanup failed: {e}")
            return False
    
    def cleanup_all_memory(self) -> Dict[str, bool]:
        """Clean up all memory (GPU and CPU)."""
        results = {
            "gpu": self.cleanup_gpu_memory(),
            "cpu": self.cleanup_cpu_memory()
        }
        
        logger.info(f"Memory cleanup results: {results}")
        return results
    
    def check_memory_safety(self) -> Dict[str, Any]:
        """Check if memory usage is safe for AI workloads."""
        gpu_usage = self.get_gpu_memory_usage()
        cpu_usage = self.get_cpu_memory_usage()
        
        warnings = []
        critical = []
        
        # Check GPU memory
        if gpu_usage:
            if gpu_usage.utilization >= self.vram_critical_threshold:
                critical.append(f"GPU memory critical: {gpu_usage.utilization:.1%}")
            elif gpu_usage.utilization >= self.vram_warning_threshold:
                warnings.append(f"GPU memory high: {gpu_usage.utilization:.1%}")
        
        # Check CPU memory
        if cpu_usage:
            if cpu_usage.utilization >= self.ram_critical_threshold:
                critical.append(f"CPU memory critical: {cpu_usage.utilization:.1%}")
            elif cpu_usage.utilization >= self.ram_warning_threshold:
                warnings.append(f"CPU memory high: {cpu_usage.utilization:.1%}")
        
        return {
            "safe": len(critical) == 0,
            "warnings": warnings,
            "critical": critical,
            "gpu_usage": gpu_usage,
            "cpu_usage": cpu_usage
        }
    
    def get_recommended_batch_size(self, base_batch_size: int = 1) -> int:
        """Get recommended batch size based on available memory."""
        gpu_usage = self.get_gpu_memory_usage()
        
        if not gpu_usage:
            return base_batch_size
        
        # Reduce batch size if memory is constrained
        if gpu_usage.utilization >= 0.7:
            return max(1, base_batch_size // 2)
        elif gpu_usage.utilization >= 0.5:
            return max(1, int(base_batch_size * 0.75))
        else:
            return base_batch_size
    
    def optimize_model_loading(self, model_size_mb: int) -> Dict[str, Any]:
        """Optimize model loading based on available memory."""
        gpu_usage = self.get_gpu_memory_usage()
        
        # Default optimization settings
        optimizations = {
            "cpu_offload": False,
            "sequential_cpu_offload": False,
            "attention_slicing": False,
            "enable_xformers": True,
            "fp16": True
        }
        
        if gpu_usage:
            available_vram = gpu_usage.free_mb
            
            # If model won't fit in VRAM, enable offloading
            if model_size_mb > available_vram * 0.8:
                optimizations["cpu_offload"] = True
                logger.info(f"Enabling CPU offload: model {model_size_mb}MB > available {available_vram}MB")
            
            # If VRAM is very limited, enable sequential offloading
            if model_size_mb > available_vram * 0.9:
                optimizations["sequential_cpu_offload"] = True
                logger.info("Enabling sequential CPU offload for very limited VRAM")
            
            # Enable attention slicing for memory efficiency
            if gpu_usage.utilization > 0.6:
                optimizations["attention_slicing"] = True
                logger.info("Enabling attention slicing for memory efficiency")
        
        return optimizations
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        gpu_usage = self.get_gpu_memory_usage()
        cpu_usage = self.get_cpu_memory_usage()
        safety = self.check_memory_safety()
        
        return {
            "gpu": {
                "available": gpu_usage is not None,
                "usage": gpu_usage.__dict__ if gpu_usage else None
            },
            "cpu": {
                "available": cpu_usage is not None,
                "usage": cpu_usage.__dict__ if cpu_usage else None
            },
            "safety": safety,
            "recommended_batch_size": self.get_recommended_batch_size()
        }


# Global instance
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager(hardware_config: Optional[HardwareConfig] = None) -> Optional[MemoryManager]:
    """Get the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None and hardware_config is not None:
        _memory_manager = MemoryManager(hardware_config)
    return _memory_manager

def create_memory_manager(hardware_config: HardwareConfig) -> MemoryManager:
    """Create a new memory manager instance."""
    return MemoryManager(hardware_config)