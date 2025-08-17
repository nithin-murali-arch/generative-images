"""
Memory management engine for VRAM optimization and model switching.

This module implements the IMemoryManager interface to provide VRAM optimization
strategies including attention slicing, CPU offloading, and cache clearing for
different hardware configurations.
"""

import gc
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.interfaces import IMemoryManager, HardwareConfig, MemoryError

logger = logging.getLogger(__name__)

# Try to import torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - memory management will be limited")


class OptimizationStrategy(Enum):
    """Memory optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    MINIMAL = "minimal"


@dataclass
class MemoryStatus:
    """Current memory usage status."""
    total_vram_mb: int
    used_vram_mb: int
    free_vram_mb: int
    total_ram_mb: int
    used_ram_mb: int
    free_ram_mb: int
    gpu_utilization_percent: float
    memory_fragmentation_percent: float


@dataclass
class ModelMemoryInfo:
    """Memory information for a loaded model."""
    model_name: str
    vram_usage_mb: int
    ram_usage_mb: int
    load_time: float
    last_used: float
    optimization_flags: Dict[str, Any]


class MemoryManager(IMemoryManager):
    """
    Memory management engine with VRAM optimization strategies.
    
    Provides automatic memory optimization, model switching with cleanup,
    and adaptive strategies based on hardware capabilities.
    """
    
    def __init__(self, hardware_config: HardwareConfig):
        self.hardware_config = hardware_config
        self.strategy = self._determine_strategy(hardware_config)
        self.loaded_models: Dict[str, ModelMemoryInfo] = {}
        self.memory_callbacks: List[Callable] = []
        self.cleanup_threshold = 0.85  # 85% VRAM usage triggers cleanup
        self.monitoring_enabled = True
        self._lock = threading.Lock()
        
        # Initialize optimization settings
        self.optimization_settings = self._initialize_optimization_settings()
        
        logger.info(f"MemoryManager initialized with {self.strategy.value} strategy")
        logger.info(f"Hardware: {hardware_config.gpu_model} ({hardware_config.vram_size}MB VRAM)")
    
    def optimize_model_loading(self, model_name: str) -> Dict[str, Any]:
        """
        Optimize model loading for available hardware.
        
        Args:
            model_name: Name of the model to optimize
            
        Returns:
            Dict containing optimization parameters for model loading
        """
        with self._lock:
            logger.info(f"Optimizing model loading for: {model_name}")
            
            # Check if we need to free memory first
            if self._should_cleanup_memory():
                self._cleanup_unused_models()
            
            # Get base optimization settings
            optimization_params = self.optimization_settings.copy()
            
            # Model-specific optimizations
            model_optimizations = self._get_model_specific_optimizations(model_name)
            optimization_params.update(model_optimizations)
            
            # Hardware-specific adjustments
            hardware_adjustments = self._get_hardware_adjustments()
            optimization_params.update(hardware_adjustments)
            
            logger.info(f"Optimization parameters for {model_name}: {optimization_params}")
            return optimization_params
    
    def manage_model_switching(self, current_model: str, next_model: str) -> None:
        """
        Efficiently switch between models with memory cleanup.
        
        Args:
            current_model: Name of currently loaded model
            next_model: Name of model to load next
        """
        with self._lock:
            logger.info(f"Switching models: {current_model} -> {next_model}")
            
            start_time = time.time()
            
            # Unload current model if it exists
            if current_model and current_model in self.loaded_models:
                self._unload_model(current_model)
            
            # Clear VRAM cache
            self.clear_vram_cache()
            
            # Update model tracking
            if next_model:
                # Estimate memory requirements for next model
                estimated_vram = self._estimate_model_vram(next_model)
                
                # Ensure we have enough free memory
                if not self._ensure_free_memory(estimated_vram):
                    raise MemoryError(f"Insufficient VRAM for {next_model} "
                                    f"(requires ~{estimated_vram}MB)")
                
                # Track the new model (will be updated when actually loaded)
                self.loaded_models[next_model] = ModelMemoryInfo(
                    model_name=next_model,
                    vram_usage_mb=estimated_vram,
                    ram_usage_mb=0,  # Will be updated
                    load_time=time.time(),
                    last_used=time.time(),
                    optimization_flags={}
                )
            
            switch_time = time.time() - start_time
            logger.info(f"Model switch completed in {switch_time:.2f}s")
    
    def clear_vram_cache(self) -> None:
        """Clear VRAM cache to free memory."""
        logger.debug("Clearing VRAM cache")
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Force garbage collection
            gc.collect()
            
            # Additional cleanup for specific libraries
            self._clear_diffusers_cache()
            self._clear_transformers_cache()
        
        # System garbage collection
        gc.collect()
        
        logger.debug("VRAM cache cleared")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dict containing detailed memory status information
        """
        status = {
            'strategy': self.strategy.value,
            'hardware': {
                'gpu_model': self.hardware_config.gpu_model,
                'total_vram_mb': self.hardware_config.vram_size,
                'total_ram_mb': self.hardware_config.ram_size
            },
            'loaded_models': len(self.loaded_models),
            'model_details': {}
        }
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)  # MB
            
            status.update({
                'vram_allocated_mb': memory_allocated,
                'vram_reserved_mb': memory_reserved,
                'vram_free_mb': self.hardware_config.vram_size - memory_reserved,
                'vram_utilization_percent': (memory_reserved / self.hardware_config.vram_size) * 100
            })
        else:
            status.update({
                'vram_allocated_mb': 0,
                'vram_reserved_mb': 0,
                'vram_free_mb': self.hardware_config.vram_size,
                'vram_utilization_percent': 0
            })
        
        # Add model details
        for model_name, model_info in self.loaded_models.items():
            status['model_details'][model_name] = {
                'vram_usage_mb': model_info.vram_usage_mb,
                'ram_usage_mb': model_info.ram_usage_mb,
                'last_used': model_info.last_used,
                'load_time': model_info.load_time
            }
        
        return status
    
    def register_memory_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for memory status changes."""
        self.memory_callbacks.append(callback)
    
    def set_cleanup_threshold(self, threshold: float) -> None:
        """Set VRAM usage threshold for automatic cleanup."""
        if 0.5 <= threshold <= 0.95:
            self.cleanup_threshold = threshold
            logger.info(f"Cleanup threshold set to {threshold * 100}%")
        else:
            raise ValueError("Cleanup threshold must be between 0.5 and 0.95")
    
    def enable_cpu_offloading(self, model_name: str) -> Dict[str, Any]:
        """
        Enable CPU offloading for a model.
        
        Args:
            model_name: Name of the model to configure for CPU offloading
            
        Returns:
            Dict containing CPU offloading configuration
        """
        logger.info(f"Enabling CPU offloading for {model_name}")
        
        offload_config = {
            'device_map': 'auto',
            'offload_folder': 'offload_cache',
            'offload_state_dict': True
        }
        
        # Strategy-specific configurations
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            offload_config.update({
                'sequential_cpu_offload': True,
                'enable_sequential_cpu_offload': True,
                'cpu_offload': True
            })
        elif self.strategy == OptimizationStrategy.BALANCED:
            offload_config.update({
                'sequential_cpu_offload': False,
                'cpu_offload': True
            })
        
        return offload_config
    
    def enable_attention_slicing(self, model_name: str) -> Dict[str, Any]:
        """
        Enable attention slicing for memory reduction.
        
        Args:
            model_name: Name of the model to configure
            
        Returns:
            Dict containing attention slicing configuration
        """
        logger.info(f"Enabling attention slicing for {model_name}")
        
        slice_config = {
            'enable_attention_slicing': True,
            'attention_slice_size': None  # Auto-determine
        }
        
        # Determine slice size based on VRAM
        if self.hardware_config.vram_size < 6000:
            slice_config['attention_slice_size'] = 1  # Most aggressive
        elif self.hardware_config.vram_size < 12000:
            slice_config['attention_slice_size'] = 2  # Moderate
        else:
            slice_config['attention_slice_size'] = 4  # Conservative
        
        return slice_config
    
    def enable_vae_optimizations(self, model_name: str) -> Dict[str, Any]:
        """
        Enable VAE optimizations for memory reduction.
        
        Args:
            model_name: Name of the model to configure
            
        Returns:
            Dict containing VAE optimization configuration
        """
        logger.info(f"Enabling VAE optimizations for {model_name}")
        
        vae_config = {}
        
        if self.strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.BALANCED]:
            vae_config.update({
                'enable_vae_slicing': True,
                'enable_vae_tiling': True
            })
        
        return vae_config
    
    def _determine_strategy(self, hardware_config: HardwareConfig) -> OptimizationStrategy:
        """Determine optimization strategy based on hardware."""
        vram_mb = hardware_config.vram_size
        
        if vram_mb < 6000:
            return OptimizationStrategy.AGGRESSIVE
        elif vram_mb < 12000:
            return OptimizationStrategy.BALANCED
        else:
            return OptimizationStrategy.MINIMAL
    
    def _initialize_optimization_settings(self) -> Dict[str, Any]:
        """Initialize base optimization settings."""
        settings = {
            'torch_dtype': 'float16' if TORCH_AVAILABLE else None,
            'use_safetensors': True
        }
        
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            settings.update({
                'low_cpu_mem_usage': True,
                'use_auth_token': None,
                'torch_dtype': 'float32',  # Use float32 to avoid black image corruption
                'enable_attention_slicing': True,
                'enable_vae_slicing': True,
                'enable_vae_tiling': True,
                'cpu_offload': False,  # Disable CPU offloading - let GPU do the work
                'sequential_cpu_offload': False  # Disable sequential offloading
            })
        elif self.strategy == OptimizationStrategy.BALANCED:
            settings.update({
                'low_cpu_mem_usage': True,
                'torch_dtype': 'float32',  # Use float32 to avoid black image corruption
                'enable_vae_tiling': True,
                'cpu_offload': False,
                'sequential_cpu_offload': False
            })
        else:  # MINIMAL
            settings.update({
                'torch_dtype': 'float32',  # Use float32 to avoid black image corruption
                'cpu_offload': False,
                'sequential_cpu_offload': False
            })
        
        return settings
    
    def _get_model_specific_optimizations(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific optimization parameters."""
        optimizations = {}
        
        # Model-specific settings
        if 'flux' in model_name.lower():
            optimizations.update({
                'torch_dtype': 'bfloat16',  # FLUX prefers bfloat16
                'variant': None
            })
        elif 'sdxl' in model_name.lower():
            optimizations.update({
                'torch_dtype': 'float16'
            })
        elif 'stable-diffusion' in model_name.lower():
            optimizations.update({
                'torch_dtype': 'float16'
            })
        
        return optimizations
    
    def _get_hardware_adjustments(self) -> Dict[str, Any]:
        """Get hardware-specific adjustments."""
        adjustments = {}
        
        # CUDA-specific optimizations
        if self.hardware_config.cuda_available and TORCH_AVAILABLE:
            adjustments.update({
                'device': 'cuda',
                'enable_xformers': False  # Disable XFormers to avoid warnings
            })
        else:
            adjustments.update({
                'device': 'cpu'
            })
        
        return adjustments
    
    def _should_cleanup_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        
        memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)  # MB
        utilization = memory_reserved / self.hardware_config.vram_size
        
        return utilization > self.cleanup_threshold
    
    def _cleanup_unused_models(self) -> None:
        """Clean up unused models to free memory."""
        logger.info("Cleaning up unused models")
        
        current_time = time.time()
        models_to_remove = []
        
        # Find models that haven't been used recently
        for model_name, model_info in self.loaded_models.items():
            time_since_use = current_time - model_info.last_used
            if time_since_use > 300:  # 5 minutes
                models_to_remove.append(model_name)
        
        # Remove old models
        for model_name in models_to_remove:
            self._unload_model(model_name)
        
        # Clear caches
        self.clear_vram_cache()
        
        logger.info(f"Cleaned up {len(models_to_remove)} unused models")
    
    def _unload_model(self, model_name: str) -> None:
        """Unload a specific model from memory."""
        if model_name in self.loaded_models:
            logger.info(f"Unloading model: {model_name}")
            del self.loaded_models[model_name]
            
            # Force garbage collection
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _estimate_model_vram(self, model_name: str) -> int:
        """Estimate VRAM requirements for a model."""
        # Model VRAM estimates (in MB)
        model_estimates = {
            'stable-diffusion-v1-5': 3500,
            'sdxl-turbo': 7000,
            'flux.1-schnell': 20000,
            'stable-video-diffusion': 12000,
            'llama-3.1-8b': 16000,
            'phi-3-mini': 2500
        }
        
        # Find matching model
        for known_model, vram_req in model_estimates.items():
            if known_model in model_name.lower():
                return vram_req
        
        # Default estimate based on model type
        if 'video' in model_name.lower():
            return 12000
        elif 'xl' in model_name.lower() or 'large' in model_name.lower():
            return 8000
        else:
            return 4000
    
    def _ensure_free_memory(self, required_mb: int) -> bool:
        """Ensure sufficient free memory is available."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return True  # Assume CPU has enough memory
        
        memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
        free_memory = self.hardware_config.vram_size - memory_reserved
        
        if free_memory < required_mb:
            logger.warning(f"Insufficient free VRAM: {free_memory}MB < {required_mb}MB")
            
            # Try cleanup
            self._cleanup_unused_models()
            
            # Check again
            memory_reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
            free_memory = self.hardware_config.vram_size - memory_reserved
            
            if free_memory < required_mb:
                return False
        
        return True
    
    def _clear_diffusers_cache(self) -> None:
        """Clear diffusers-specific caches."""
        try:
            # Clear diffusers cache if available
            import diffusers
            if hasattr(diffusers.utils, 'clear_cache'):
                diffusers.utils.clear_cache()
        except ImportError:
            pass
    
    def _clear_transformers_cache(self) -> None:
        """Clear transformers-specific caches."""
        try:
            # Clear transformers cache if available
            import transformers
            if hasattr(transformers, 'clean_up_tokenization'):
                transformers.clean_up_tokenization()
        except ImportError:
            pass
    
    def update_model_usage(self, model_name: str, vram_usage_mb: int = None) -> None:
        """Update model usage statistics."""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].last_used = time.time()
            if vram_usage_mb is not None:
                self.loaded_models[model_name].vram_usage_mb = vram_usage_mb
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get optimization recommendations for current hardware."""
        recommendations = {
            'strategy': self.strategy.value,
            'optimizations': []
        }
        
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            recommendations['optimizations'].extend([
                'Enable attention slicing for all models',
                'Use CPU offloading for large models',
                'Enable VAE slicing and tiling',
                'Use lower resolution (512x512) for generation',
                'Consider using smaller models (SD 1.5 instead of SDXL)'
            ])
        elif self.strategy == OptimizationStrategy.BALANCED:
            recommendations['optimizations'].extend([
                'Enable VAE tiling for memory efficiency',
                'Use mixed precision (FP16) for all models',
                'Monitor VRAM usage and cleanup unused models',
                'Consider attention slicing for very large models'
            ])
        else:  # MINIMAL
            recommendations['optimizations'].extend([
                'Use mixed precision (FP16) for faster inference',
                'Enable XFormers for memory-efficient attention',
                'Can run multiple models simultaneously',
                'Higher resolutions and batch sizes supported'
            ])
        
        return recommendations