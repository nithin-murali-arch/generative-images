"""
Training Parameter Optimization for Different Hardware Configurations

This module provides automatic optimization of training parameters based on
hardware capabilities, memory constraints, and performance targets.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from ..core.interfaces import HardwareConfig
from ..hardware.profiles import HardwareProfileManager
from .lora_training import LoRAConfig

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Training optimization strategies."""
    SPEED_FOCUSED = "speed_focused"      # Prioritize training speed
    MEMORY_FOCUSED = "memory_focused"    # Prioritize memory efficiency
    QUALITY_FOCUSED = "quality_focused"  # Prioritize training quality
    BALANCED = "balanced"                # Balance all factors


@dataclass
class TrainingConstraints:
    """Hardware and performance constraints for training."""
    max_vram_mb: int
    max_batch_size: int
    max_resolution: int
    target_training_time_hours: Optional[float] = None
    min_quality_threshold: Optional[float] = None
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True


@dataclass
class OptimizedConfig:
    """Optimized training configuration."""
    lora_config: LoRAConfig
    estimated_vram_usage_mb: float
    estimated_training_time_hours: float
    optimization_notes: Dict[str, str]
    performance_predictions: Dict[str, Any]


class TrainingOptimizer:
    """
    Automatic training parameter optimization based on hardware capabilities.
    
    Analyzes hardware configuration and automatically adjusts training parameters
    for optimal performance within memory and time constraints.
    """
    
    def __init__(self):
        self.profile_manager = HardwareProfileManager()
        
        # Base parameter ranges for optimization
        self.param_ranges = {
            'rank': [4, 8, 16, 32, 64],
            'alpha': [16.0, 32.0, 64.0, 128.0],
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [1, 2, 4, 8],
            'resolution': [512, 640, 768, 1024],
            'num_epochs': [5, 10, 15, 20, 30]
        }
        
        # VRAM usage estimates (MB per parameter combination)
        self.vram_estimates = self._initialize_vram_estimates()
    
    def optimize_config(self, hardware_config: HardwareConfig,
                       strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                       constraints: Optional[TrainingConstraints] = None) -> OptimizedConfig:
        """
        Optimize training configuration for hardware and strategy.
        
        Args:
            hardware_config: Hardware configuration
            strategy: Optimization strategy
            constraints: Additional constraints
            
        Returns:
            OptimizedConfig with optimized parameters
        """
        logger.info(f"Optimizing training config for {hardware_config.gpu_model} with {strategy.value} strategy")
        
        # Get hardware profile
        profile = self.profile_manager.get_profile(hardware_config)
        
        # Create constraints if not provided
        if constraints is None:
            constraints = self._create_default_constraints(hardware_config, profile)
        
        # Optimize parameters based on strategy
        if strategy == OptimizationStrategy.SPEED_FOCUSED:
            config = self._optimize_for_speed(hardware_config, constraints)
        elif strategy == OptimizationStrategy.MEMORY_FOCUSED:
            config = self._optimize_for_memory(hardware_config, constraints)
        elif strategy == OptimizationStrategy.QUALITY_FOCUSED:
            config = self._optimize_for_quality(hardware_config, constraints)
        else:  # BALANCED
            config = self._optimize_balanced(hardware_config, constraints)
        
        # Estimate performance
        vram_usage = self._estimate_vram_usage(config, hardware_config)
        training_time = self._estimate_training_time(config, hardware_config)
        
        # Generate optimization notes
        notes = self._generate_optimization_notes(config, hardware_config, strategy)
        
        # Create performance predictions
        predictions = self._create_performance_predictions(config, hardware_config)
        
        optimized_config = OptimizedConfig(
            lora_config=config,
            estimated_vram_usage_mb=vram_usage,
            estimated_training_time_hours=training_time,
            optimization_notes=notes,
            performance_predictions=predictions
        )
        
        logger.info(f"Optimization completed: {vram_usage:.0f}MB VRAM, {training_time:.1f}h training time")
        return optimized_config
    
    def validate_config(self, config: LoRAConfig, 
                       hardware_config: HardwareConfig) -> Dict[str, Any]:
        """
        Validate training configuration against hardware constraints.
        
        Args:
            config: LoRA configuration to validate
            hardware_config: Hardware configuration
            
        Returns:
            Validation result with warnings and recommendations
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'recommendations': [],
            'estimated_vram_mb': 0.0,
            'estimated_time_hours': 0.0
        }
        
        # Estimate resource usage
        vram_usage = self._estimate_vram_usage(config, hardware_config)
        training_time = self._estimate_training_time(config, hardware_config)
        
        validation_result['estimated_vram_mb'] = vram_usage
        validation_result['estimated_time_hours'] = training_time
        
        # Check VRAM constraints
        available_vram = hardware_config.vram_size * 0.9  # Leave 10% buffer
        if vram_usage > available_vram:
            validation_result['valid'] = False
            validation_result['warnings'].append(
                f"Estimated VRAM usage ({vram_usage:.0f}MB) exceeds available VRAM ({available_vram:.0f}MB)"
            )
            validation_result['recommendations'].append("Reduce batch size, resolution, or LoRA rank")
        
        # Check batch size
        if config.batch_size > 4 and hardware_config.vram_size < 12000:
            validation_result['warnings'].append("Large batch size may cause memory issues on this hardware")
            validation_result['recommendations'].append("Consider reducing batch size to 1-2")
        
        # Check resolution
        if config.resolution > 768 and hardware_config.vram_size < 8000:
            validation_result['warnings'].append("High resolution training may be slow on this hardware")
            validation_result['recommendations'].append("Consider using 512x512 resolution")
        
        # Check training time
        if training_time > 24:
            validation_result['warnings'].append(f"Estimated training time is very long ({training_time:.1f} hours)")
            validation_result['recommendations'].append("Consider reducing epochs or using faster settings")
        
        return validation_result
    
    def suggest_improvements(self, config: LoRAConfig, 
                           hardware_config: HardwareConfig,
                           target_metric: str = "balanced") -> Dict[str, Any]:
        """
        Suggest improvements to existing configuration.
        
        Args:
            config: Current LoRA configuration
            hardware_config: Hardware configuration
            target_metric: Target metric to optimize ("speed", "memory", "quality", "balanced")
            
        Returns:
            Suggestions for improvement
        """
        suggestions = {
            'current_config': config.__dict__,
            'improvements': [],
            'alternative_configs': []
        }
        
        # Analyze current config
        current_vram = self._estimate_vram_usage(config, hardware_config)
        current_time = self._estimate_training_time(config, hardware_config)
        
        # Generate improvement suggestions
        if target_metric in ["speed", "balanced"]:
            if config.batch_size == 1 and hardware_config.vram_size > 8000:
                suggestions['improvements'].append({
                    'parameter': 'batch_size',
                    'current': config.batch_size,
                    'suggested': 2,
                    'reason': 'Increase batch size to improve training speed',
                    'impact': 'Faster training, slightly more VRAM usage'
                })
            
            if config.mixed_precision == "no":
                suggestions['improvements'].append({
                    'parameter': 'mixed_precision',
                    'current': config.mixed_precision,
                    'suggested': 'fp16',
                    'reason': 'Enable mixed precision for faster training',
                    'impact': 'Faster training, reduced VRAM usage'
                })
        
        if target_metric in ["memory", "balanced"]:
            if config.gradient_checkpointing == False:
                suggestions['improvements'].append({
                    'parameter': 'gradient_checkpointing',
                    'current': config.gradient_checkpointing,
                    'suggested': True,
                    'reason': 'Enable gradient checkpointing to reduce memory usage',
                    'impact': 'Lower VRAM usage, slightly slower training'
                })
            
            if config.rank > 16 and current_vram > hardware_config.vram_size * 0.8:
                suggestions['improvements'].append({
                    'parameter': 'rank',
                    'current': config.rank,
                    'suggested': config.rank // 2,
                    'reason': 'Reduce LoRA rank to save memory',
                    'impact': 'Lower VRAM usage, potentially lower quality'
                })
        
        if target_metric in ["quality", "balanced"]:
            if config.learning_rate > 1e-4:
                suggestions['improvements'].append({
                    'parameter': 'learning_rate',
                    'current': config.learning_rate,
                    'suggested': 1e-4,
                    'reason': 'Lower learning rate for more stable training',
                    'impact': 'Better convergence, longer training time'
                })
            
            if config.num_epochs < 15 and current_time < 12:
                suggestions['improvements'].append({
                    'parameter': 'num_epochs',
                    'current': config.num_epochs,
                    'suggested': config.num_epochs + 5,
                    'reason': 'Increase epochs for better quality',
                    'impact': 'Better quality, longer training time'
                })
        
        # Generate alternative configurations
        strategies = [OptimizationStrategy.SPEED_FOCUSED, OptimizationStrategy.MEMORY_FOCUSED, OptimizationStrategy.QUALITY_FOCUSED]
        for strategy in strategies:
            if strategy.value != target_metric:
                alt_config = self.optimize_config(hardware_config, strategy)
                suggestions['alternative_configs'].append({
                    'strategy': strategy.value,
                    'config': alt_config.lora_config.__dict__,
                    'estimated_vram_mb': alt_config.estimated_vram_usage_mb,
                    'estimated_time_hours': alt_config.estimated_training_time_hours
                })
        
        return suggestions
    
    def _create_default_constraints(self, hardware_config: HardwareConfig,
                                  profile) -> TrainingConstraints:
        """Create default constraints based on hardware profile."""
        optimization_settings = profile.optimizations
        
        return TrainingConstraints(
            max_vram_mb=int(hardware_config.vram_size * 0.9),  # Leave 10% buffer
            max_batch_size=optimization_settings.get('batch_size', 1),
            max_resolution=optimization_settings.get('max_resolution', 512),
            enable_mixed_precision=optimization_settings.get('mixed_precision', True),
            enable_gradient_checkpointing=optimization_settings.get('gradient_checkpointing', True)
        )
    
    def _optimize_for_speed(self, hardware_config: HardwareConfig,
                           constraints: TrainingConstraints) -> LoRAConfig:
        """Optimize configuration for training speed."""
        config = LoRAConfig()
        
        # Prioritize speed optimizations
        config.batch_size = min(constraints.max_batch_size, 4)
        config.resolution = min(constraints.max_resolution, 640)  # Lower resolution for speed
        config.num_epochs = 10  # Fewer epochs
        config.learning_rate = 5e-4  # Higher learning rate
        config.rank = 16  # Moderate rank
        config.alpha = 32.0
        
        # Enable speed optimizations
        config.mixed_precision = "fp16" if constraints.enable_mixed_precision else "no"
        config.gradient_checkpointing = False  # Disable for speed
        config.use_8bit_adam = True  # Faster optimizer
        config.enable_xformers = True
        
        return config
    
    def _optimize_for_memory(self, hardware_config: HardwareConfig,
                            constraints: TrainingConstraints) -> LoRAConfig:
        """Optimize configuration for memory efficiency."""
        config = LoRAConfig()
        
        # Prioritize memory efficiency
        config.batch_size = 1  # Minimum batch size
        config.resolution = min(constraints.max_resolution, 512)  # Lower resolution
        config.rank = 8  # Lower rank
        config.alpha = 16.0
        config.learning_rate = 1e-4
        config.num_epochs = 15  # More epochs to compensate for smaller batch
        
        # Enable memory optimizations
        config.mixed_precision = "fp16" if constraints.enable_mixed_precision else "no"
        config.gradient_checkpointing = True
        config.use_8bit_adam = True
        config.enable_cpu_offload = hardware_config.vram_size < 6000
        config.enable_xformers = True
        
        return config
    
    def _optimize_for_quality(self, hardware_config: HardwareConfig,
                             constraints: TrainingConstraints) -> LoRAConfig:
        """Optimize configuration for training quality."""
        config = LoRAConfig()
        
        # Prioritize quality
        config.batch_size = min(constraints.max_batch_size, 2)
        config.resolution = min(constraints.max_resolution, 768)  # Higher resolution if possible
        config.rank = 32 if hardware_config.vram_size > 12000 else 16  # Higher rank if VRAM allows
        config.alpha = 64.0
        config.learning_rate = 1e-4  # Lower, more stable learning rate
        config.num_epochs = 20  # More epochs
        
        # Quality-focused settings
        config.mixed_precision = "fp16" if constraints.enable_mixed_precision else "no"
        config.gradient_checkpointing = constraints.enable_gradient_checkpointing
        config.use_8bit_adam = False  # Use full precision optimizer
        config.enable_xformers = True
        config.dropout = 0.05  # Lower dropout
        
        return config
    
    def _optimize_balanced(self, hardware_config: HardwareConfig,
                          constraints: TrainingConstraints) -> LoRAConfig:
        """Optimize configuration for balanced performance."""
        config = LoRAConfig()
        
        # Balanced settings
        config.batch_size = min(constraints.max_batch_size, 2 if hardware_config.vram_size > 8000 else 1)
        config.resolution = min(constraints.max_resolution, 640)
        config.rank = 16
        config.alpha = 32.0
        config.learning_rate = 1e-4
        config.num_epochs = 15
        
        # Balanced optimizations
        config.mixed_precision = "fp16" if constraints.enable_mixed_precision else "no"
        config.gradient_checkpointing = constraints.enable_gradient_checkpointing
        config.use_8bit_adam = True
        config.enable_xformers = True
        config.enable_cpu_offload = hardware_config.vram_size < 6000
        
        return config
    
    def _estimate_vram_usage(self, config: LoRAConfig, 
                            hardware_config: HardwareConfig) -> float:
        """Estimate VRAM usage for configuration."""
        # Base model VRAM usage (approximate)
        base_vram = {
            'stable-diffusion-v1-5': 3500,
            'sdxl-turbo': 7000
        }
        
        # Estimate based on configuration
        model_vram = base_vram.get('stable-diffusion-v1-5', 3500)  # Default to SD 1.5
        
        # LoRA overhead
        lora_vram = config.rank * config.rank * 4 * 0.001  # Approximate LoRA parameter overhead
        
        # Batch size scaling
        batch_scaling = config.batch_size * 1.2
        
        # Resolution scaling
        resolution_scaling = (config.resolution / 512) ** 2
        
        # Mixed precision reduction
        precision_factor = 0.7 if config.mixed_precision == "fp16" else 1.0
        
        # Gradient checkpointing reduction
        checkpoint_factor = 0.8 if config.gradient_checkpointing else 1.0
        
        total_vram = (model_vram + lora_vram) * batch_scaling * resolution_scaling * precision_factor * checkpoint_factor
        
        return total_vram
    
    def _estimate_training_time(self, config: LoRAConfig,
                               hardware_config: HardwareConfig) -> float:
        """Estimate training time in hours."""
        # Base time per step (seconds) based on hardware
        base_times = {
            'gtx_1650': 2.0,
            'rtx_3070': 0.8,
            'rtx_4090': 0.3
        }
        
        # Get normalized GPU name
        gpu_normalized = self.profile_manager._normalize_gpu_name(hardware_config.gpu_model)
        base_time = base_times.get(gpu_normalized, 1.5)  # Default to mid-range
        
        # Estimate steps per epoch (assuming 1000 images in dataset)
        dataset_size = 1000
        steps_per_epoch = dataset_size // config.batch_size
        total_steps = steps_per_epoch * config.num_epochs
        
        # Resolution scaling
        resolution_factor = (config.resolution / 512) ** 1.5
        
        # Mixed precision speedup
        precision_factor = 0.7 if config.mixed_precision == "fp16" else 1.0
        
        # XFormers speedup
        xformers_factor = 0.8 if config.enable_xformers else 1.0
        
        # Gradient checkpointing slowdown
        checkpoint_factor = 1.2 if config.gradient_checkpointing else 1.0
        
        total_time_seconds = (total_steps * base_time * resolution_factor * 
                            precision_factor * xformers_factor * checkpoint_factor)
        
        return total_time_seconds / 3600  # Convert to hours
    
    def _generate_optimization_notes(self, config: LoRAConfig,
                                   hardware_config: HardwareConfig,
                                   strategy: OptimizationStrategy) -> Dict[str, str]:
        """Generate optimization notes explaining choices."""
        notes = {}
        
        notes['strategy'] = f"Configuration optimized for {strategy.value}"
        notes['hardware'] = f"Optimized for {hardware_config.gpu_model} with {hardware_config.vram_size}MB VRAM"
        
        if config.batch_size == 1:
            notes['batch_size'] = "Batch size set to 1 for memory efficiency"
        elif config.batch_size > 2:
            notes['batch_size'] = "Larger batch size for improved training speed"
        
        if config.resolution < 640:
            notes['resolution'] = "Lower resolution for faster training and memory efficiency"
        elif config.resolution > 768:
            notes['resolution'] = "Higher resolution for better quality (requires more VRAM)"
        
        if config.mixed_precision == "fp16":
            notes['mixed_precision'] = "FP16 mixed precision enabled for speed and memory efficiency"
        
        if config.gradient_checkpointing:
            notes['gradient_checkpointing'] = "Gradient checkpointing enabled to reduce memory usage"
        
        if config.enable_cpu_offload:
            notes['cpu_offload'] = "CPU offloading enabled for low VRAM systems"
        
        return notes
    
    def _create_performance_predictions(self, config: LoRAConfig,
                                      hardware_config: HardwareConfig) -> Dict[str, Any]:
        """Create performance predictions for configuration."""
        vram_usage = self._estimate_vram_usage(config, hardware_config)
        training_time = self._estimate_training_time(config, hardware_config)
        
        # Calculate efficiency metrics
        vram_efficiency = (hardware_config.vram_size - vram_usage) / hardware_config.vram_size
        time_efficiency = "fast" if training_time < 6 else "medium" if training_time < 12 else "slow"
        
        # Quality prediction based on parameters
        quality_score = (config.rank / 32) * 0.4 + (config.num_epochs / 20) * 0.3 + (config.resolution / 1024) * 0.3
        quality_level = "high" if quality_score > 0.7 else "medium" if quality_score > 0.4 else "basic"
        
        return {
            'vram_usage_mb': vram_usage,
            'vram_efficiency': vram_efficiency,
            'training_time_hours': training_time,
            'time_efficiency': time_efficiency,
            'quality_prediction': quality_level,
            'quality_score': quality_score,
            'recommended_for': self._get_use_case_recommendations(config, hardware_config)
        }
    
    def _get_use_case_recommendations(self, config: LoRAConfig,
                                    hardware_config: HardwareConfig) -> List[str]:
        """Get use case recommendations for configuration."""
        recommendations = []
        
        if config.rank >= 32 and config.num_epochs >= 15:
            recommendations.append("High-quality style transfer")
            recommendations.append("Character/object training")
        
        if config.batch_size >= 2 and config.resolution >= 640:
            recommendations.append("Fast prototyping")
            recommendations.append("Batch processing")
        
        if config.enable_cpu_offload or hardware_config.vram_size < 6000:
            recommendations.append("Low-resource experimentation")
            recommendations.append("Educational use")
        
        if config.resolution >= 768 and config.rank >= 16:
            recommendations.append("Production-quality training")
            recommendations.append("Commercial applications")
        
        return recommendations
    
    def _initialize_vram_estimates(self) -> Dict[str, float]:
        """Initialize VRAM usage estimates for different parameter combinations."""
        # This would be expanded with empirical measurements
        return {
            'base_sd15': 3500,
            'base_sdxl': 7000,
            'lora_rank_4': 50,
            'lora_rank_16': 200,
            'lora_rank_32': 400,
            'batch_size_multiplier': 1.2,
            'resolution_512': 1.0,
            'resolution_768': 2.25,
            'resolution_1024': 4.0
        }