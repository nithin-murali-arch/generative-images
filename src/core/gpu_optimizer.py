"""
GPU Optimization Module

This module provides GPU optimization utilities to ensure maximum GPU utilization
for image and video generation tasks.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU optimization disabled")


class OptimizationLevel(Enum):
    """GPU optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    NONE = "none"
    ATTENTION_SLICING = "attention_slicing"
    VAE_SLICING = "vae_slicing"
    CPU_OFFLOADING = "cpu_offloading"
    SEQUENTIAL_OFFLOADING = "sequential_offloading"


@dataclass
class GPUOptimizationConfig:
    """Configuration for GPU optimization."""
    use_gpu: bool = True
    precision: str = "float16"  # "float16" or "float32"
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    memory_strategy: MemoryStrategy = MemoryStrategy.ATTENTION_SLICING
    enable_xformers: bool = True
    enable_torch_compile: bool = False
    memory_fraction: float = 0.9
    batch_size: int = 1


class GPUOptimizer:
    """
    GPU optimization utility for AI generation pipelines.
    
    Provides methods to optimize GPU usage, memory management, and performance
    for different hardware configurations.
    """
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.gpu_info = self._get_gpu_info()
        self.optimization_cache: Dict[str, Any] = {}
        
        logger.info(f"GPUOptimizer initialized - GPU Available: {self.gpu_available}")
        if self.gpu_info:
            logger.info(f"GPU: {self.gpu_info['name']} ({self.gpu_info['memory_gb']:.1f}GB)")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for computation."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - GPU optimization disabled")
            return False
        
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            
            if not cuda_available:
                # Check if this is a CPU-only PyTorch installation
                if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
                    logger.warning("PyTorch installed without CUDA support - GPU unavailable")
                    logger.info("💡 To enable GPU support, install PyTorch with CUDA:")
                    logger.info("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                else:
                    logger.warning("CUDA not available - check GPU drivers and CUDA installation")
            
            return cuda_available
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information."""
        if not self.gpu_available or not TORCH_AVAILABLE:
            return None
        
        try:
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return None
            
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_properties = torch.cuda.get_device_properties(current_device)
            memory_gb = gpu_properties.total_memory / (1024**3)
            
            return {
                'name': gpu_name,
                'memory_gb': memory_gb,
                'device_count': device_count,
                'current_device': current_device,
                'compute_capability': f"{gpu_properties.major}.{gpu_properties.minor}"
            }
        except Exception as e:
            logger.warning(f"Error getting GPU info: {e}")
            return None
    
    def get_optimal_config(self, 
                          width: int, 
                          height: int, 
                          batch_size: int = 1,
                          user_preferences: Optional[Dict[str, Any]] = None,
                          current_gpu_utilization: Optional[float] = None,
                          force_gpu_first: bool = True) -> GPUOptimizationConfig:
        """
        Get optimal GPU configuration for given parameters.
        
        Args:
            width: Image width
            height: Image height
            batch_size: Batch size for generation
            user_preferences: User-specified preferences
            
        Returns:
            GPUOptimizationConfig: Optimal configuration
        """
        config = GPUOptimizationConfig()
        
        if not self.gpu_available:
            if force_gpu_first:
                logger.warning("GPU not available but force_gpu_first=True")
                logger.info("💡 This usually means PyTorch was installed without CUDA support")
                logger.info("   The system will fall back to optimized CPU generation")
            
            config.use_gpu = False
            config.precision = "float32"
            config.optimization_level = OptimizationLevel.NONE
            config.memory_strategy = MemoryStrategy.NONE
            return config
        
        # Calculate memory requirements
        pixel_count = width * height * batch_size
        estimated_vram_gb = self._estimate_vram_usage(pixel_count, config.precision)
        
        # Get current GPU utilization if available
        current_utilization = current_gpu_utilization or self._get_current_gpu_utilization()
        
        if self.gpu_info:
            available_memory = self.gpu_info['memory_gb'] * config.memory_fraction
            
            # Check if GPU is heavily utilized (> 85%) - use attention slicing instead of offloading
            if current_utilization and current_utilization > 85.0:
                logger.info(f"GPU heavily utilized ({current_utilization:.1f}%), using attention slicing")
                config.optimization_level = OptimizationLevel.AGGRESSIVE
                config.memory_strategy = MemoryStrategy.ATTENTION_SLICING  # Always use attention slicing
                config.precision = "float16"
            
            # Check if we need to optimize due to memory constraints
            elif estimated_vram_gb > available_memory:
                logger.info(f"Insufficient VRAM ({estimated_vram_gb:.1f}GB needed, {available_memory:.1f}GB available)")
                logger.info("Using attention slicing to reduce memory usage")
                config.optimization_level = OptimizationLevel.AGGRESSIVE
                
                # Always use attention slicing - avoid CPU offloading to prevent black images
                config.memory_strategy = MemoryStrategy.ATTENTION_SLICING
                logger.info("Using attention slicing to fit in available VRAM")
                
                config.precision = "float16"
                
                # Reduce batch size if necessary
                if batch_size > 1:
                    config.batch_size = 1
                    logger.info("Reduced batch size to 1 due to memory constraints")
            
            # Use moderate optimization if approaching limits
            elif estimated_vram_gb > available_memory * 0.7 or (current_utilization and current_utilization > 60.0):
                config.optimization_level = OptimizationLevel.BASIC
                config.memory_strategy = MemoryStrategy.ATTENTION_SLICING
                config.precision = "float16"
            
            else:
                # Can use minimal optimization - GPU has plenty of resources
                config.optimization_level = OptimizationLevel.BASIC
                config.memory_strategy = MemoryStrategy.NONE
                config.precision = "float16"  # Still use float16 for speed
        
        # Apply user preferences
        if user_preferences:
            if 'force_gpu_usage' in user_preferences:
                config.use_gpu = user_preferences['force_gpu_usage']
            
            if 'precision' in user_preferences:
                if 'float16' in user_preferences['precision']:
                    config.precision = 'float16'
                else:
                    config.precision = 'float32'
            
            if 'disable_cpu_offload' in user_preferences and user_preferences['disable_cpu_offload']:
                # Force GPU-only mode - no CPU offloading
                if config.memory_strategy in [MemoryStrategy.CPU_OFFLOADING, MemoryStrategy.SEQUENTIAL_OFFLOADING]:
                    config.memory_strategy = MemoryStrategy.ATTENTION_SLICING
                    logger.info("Disabled CPU offloading - using attention slicing instead")
            
            if 'memory_optimization' in user_preferences:
                memory_opt = user_preferences['memory_optimization']
                if 'None' in memory_opt:
                    config.memory_strategy = MemoryStrategy.NONE
                elif 'Attention' in memory_opt:
                    config.memory_strategy = MemoryStrategy.ATTENTION_SLICING
                # Remove CPU offloading option to prevent black images
                # elif 'CPU' in memory_opt:
                #     config.memory_strategy = MemoryStrategy.CPU_OFFLOADING
        
        logger.info(f"Optimal config: GPU={config.use_gpu}, Precision={config.precision}, "
                   f"Memory Strategy={config.memory_strategy.value}")
        
        return config
    
    def _estimate_vram_usage(self, pixel_count: int, precision: str) -> float:
        """Estimate VRAM usage in GB for given parameters."""
        # Base model size (approximate for SD 1.5)
        base_model_gb = 3.5 if precision == "float16" else 7.0
        
        # Additional memory for image processing
        bytes_per_pixel = 2 if precision == "float16" else 4
        image_memory_gb = (pixel_count * bytes_per_pixel * 4) / (1024**3)  # 4x for intermediate tensors
        
        # Add some overhead
        overhead_gb = 1.0
        
        total_gb = base_model_gb + image_memory_gb + overhead_gb
        
        logger.debug(f"Estimated VRAM usage: {total_gb:.2f}GB "
                    f"(Model: {base_model_gb}GB, Image: {image_memory_gb:.2f}GB, Overhead: {overhead_gb}GB)")
        
        return total_gb
    
    def _get_current_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        if not self.gpu_available or not TORCH_AVAILABLE:
            return None
        
        try:
            # Try to get GPU utilization using nvidia-ml-py
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                pynvml.nvmlShutdown()
                return float(gpu_util)
            except ImportError:
                # Fallback: estimate based on memory usage
                import torch
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    return (allocated / total) * 100.0
                return None
        except Exception as e:
            logger.debug(f"Could not get GPU utilization: {e}")
            return None
    
    def apply_optimizations(self, pipeline, config: GPUOptimizationConfig) -> None:
        """
        Apply optimizations to a diffusion pipeline.
        
        Args:
            pipeline: Diffusion pipeline to optimize
            config: Optimization configuration
        """
        if not pipeline or not TORCH_AVAILABLE:
            return
        
        logger.info(f"Applying GPU optimizations: {config.optimization_level.value}")
        
        try:
            # Move to appropriate device
            device = 'cuda' if config.use_gpu and self.gpu_available else 'cpu'
            
            # Set precision first
            if config.precision == "float16" and device == 'cuda':
                pipeline = pipeline.to(torch_dtype=torch.float16)
            else:
                pipeline = pipeline.to(torch_dtype=torch.float32)
            
            # Only move to device if not using CPU offloading strategies
            # CPU offloading handles device placement internally
            if config.memory_strategy not in [MemoryStrategy.CPU_OFFLOADING, MemoryStrategy.SEQUENTIAL_OFFLOADING]:
                pipeline = pipeline.to(device)
                logger.debug(f"Moved pipeline to {device}")
            else:
                logger.debug(f"Skipping device move - using {config.memory_strategy.value}")
            
            # Apply CPU offloading strategies first (before device placement)
            if config.memory_strategy == MemoryStrategy.CPU_OFFLOADING:
                try:
                    if hasattr(pipeline, 'enable_model_cpu_offload'):
                        pipeline.enable_model_cpu_offload()
                        logger.debug("Enabled CPU offloading")
                except Exception as e:
                    logger.warning(f"Failed to enable CPU offloading: {e}")
            
            elif config.memory_strategy == MemoryStrategy.SEQUENTIAL_OFFLOADING:
                try:
                    if hasattr(pipeline, 'enable_sequential_cpu_offload'):
                        pipeline.enable_sequential_cpu_offload()
                        logger.debug("Enabled sequential CPU offloading")
                except Exception as e:
                    logger.warning(f"Failed to enable sequential CPU offloading: {e}")
            
            # Apply other memory optimizations
            elif config.memory_strategy == MemoryStrategy.ATTENTION_SLICING:
                if hasattr(pipeline, 'enable_attention_slicing'):
                    pipeline.enable_attention_slicing()
                    logger.debug("Enabled attention slicing")
            
            elif config.memory_strategy == MemoryStrategy.VAE_SLICING:
                if hasattr(pipeline, 'enable_vae_slicing'):
                    pipeline.enable_vae_slicing()
                    logger.debug("Enabled VAE slicing")
            
            # Enable XFormers if available and requested
            if config.enable_xformers and device == 'cuda':
                try:
                    # Check if xformers is actually available
                    import xformers
                    if hasattr(pipeline, 'enable_xformers_memory_efficient_attention'):
                        pipeline.enable_xformers_memory_efficient_attention()
                        logger.debug("Enabled XFormers memory efficient attention")
                    else:
                        logger.debug("Pipeline doesn't support XFormers")
                except ImportError:
                    logger.debug("XFormers not available - skipping optimization")
                except Exception as e:
                    logger.warning(f"Failed to enable XFormers: {e}")
            
            # Enable torch.compile if requested (PyTorch 2.0+)
            if config.enable_torch_compile and hasattr(torch, 'compile'):
                try:
                    pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
                    logger.debug("Enabled torch.compile optimization")
                except Exception as e:
                    logger.warning(f"Failed to enable torch.compile: {e}")
            
            # Set memory fraction
            if device == 'cuda' and config.memory_fraction < 1.0:
                try:
                    torch.cuda.set_per_process_memory_fraction(config.memory_fraction)
                    logger.debug(f"Set CUDA memory fraction to {config.memory_fraction}")
                except Exception as e:
                    logger.warning(f"Failed to set memory fraction: {e}")
            
        except Exception as e:
            logger.error(f"Error applying GPU optimizations: {e}")
    
    def monitor_gpu_usage(self) -> Dict[str, Any]:
        """Monitor current GPU usage."""
        if not self.gpu_available or not TORCH_AVAILABLE:
            return {'gpu_available': False}
        
        try:
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            
            if self.gpu_info:
                total = self.gpu_info['memory_gb'] * 1024  # MB
                free = total - allocated
                utilization = (allocated / total) * 100
            else:
                total = 0
                free = 0
                utilization = 0
            
            return {
                'gpu_available': True,
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'total_mb': total,
                'free_mb': free,
                'utilization_percent': utilization,
                'gpu_name': self.gpu_info['name'] if self.gpu_info else 'Unknown'
            }
        except Exception as e:
            logger.warning(f"Error monitoring GPU usage: {e}")
            return {'gpu_available': True, 'error': str(e)}
    
    def clear_gpu_cache(self) -> bool:
        """Clear GPU memory cache."""
        if not self.gpu_available or not TORCH_AVAILABLE:
            return False
        
        try:
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing GPU cache: {e}")
            return False
    
    def get_optimization_recommendations(self, 
                                       width: int, 
                                       height: int, 
                                       current_config: GPUOptimizationConfig) -> List[str]:
        """Get optimization recommendations for better performance."""
        recommendations = []
        
        if not self.gpu_available:
            recommendations.append("Install CUDA-compatible PyTorch for GPU acceleration")
            recommendations.append("Consider using smaller image resolutions for faster CPU generation")
            return recommendations
        
        if not self.gpu_info:
            return recommendations
        
        pixel_count = width * height
        estimated_vram = self._estimate_vram_usage(pixel_count, current_config.precision)
        available_memory = self.gpu_info['memory_gb'] * 0.9
        
        if estimated_vram > available_memory:
            recommendations.append(f"Reduce image resolution from {width}x{height} to save VRAM")
            recommendations.append("Enable CPU offloading to reduce VRAM usage")
            recommendations.append("Use float16 precision instead of float32")
        
        elif estimated_vram > available_memory * 0.7:
            recommendations.append("Enable attention slicing for better memory efficiency")
            recommendations.append("Consider using float16 precision for faster generation")
        
        else:
            recommendations.append("Your current settings should work well with your GPU")
            recommendations.append("Consider increasing batch size for multiple images")
        
        # General recommendations
        if self.gpu_info['memory_gb'] >= 8:
            recommendations.append("Install xformers for better memory efficiency: pip install xformers")
        
        if current_config.precision == "float32":
            recommendations.append("Switch to float16 for 2x faster generation with minimal quality loss")
        
        return recommendations


# Global GPU optimizer instance
    def verify_gpu_functionality(self) -> Tuple[bool, str]:
        """
        Verify that GPU is actually functional for AI workloads.
        
        Returns:
            Tuple of (is_functional, message)
        """
        if not self.gpu_available or not TORCH_AVAILABLE:
            return False, "GPU or PyTorch not available"
        
        try:
            import torch
            
            # Test basic GPU operations
            device = torch.device('cuda')
            
            # Test tensor creation and operations
            test_tensor = torch.randn(100, 100, device=device, dtype=torch.float16)
            result = torch.matmul(test_tensor, test_tensor.T)
            
            # Test memory allocation
            memory_before = torch.cuda.memory_allocated()
            large_tensor = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            memory_after = torch.cuda.memory_allocated()
            
            # Clean up
            del test_tensor, result, large_tensor
            torch.cuda.empty_cache()
            
            memory_used = (memory_after - memory_before) / (1024**2)  # MB
            
            if memory_used > 1.0:  # At least 1MB was allocated
                return True, f"GPU functional - allocated {memory_used:.1f}MB successfully"
            else:
                return False, "GPU operations succeeded but no significant memory usage detected"
                
        except Exception as e:
            return False, f"GPU functionality test failed: {str(e)}"


# Global GPU optimizer instance
gpu_optimizer = GPUOptimizer()


def get_gpu_optimizer() -> GPUOptimizer:
    """Get the global GPU optimizer instance."""
    return gpu_optimizer