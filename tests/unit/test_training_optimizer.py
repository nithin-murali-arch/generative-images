"""
Unit tests for training parameter optimization.

Tests automatic optimization of training parameters based on hardware
capabilities and optimization strategies.
"""

import pytest
from unittest.mock import Mock, patch

from src.core.interfaces import HardwareConfig, ComplianceMode
from src.pipelines.training_optimizer import (
    TrainingOptimizer, OptimizationStrategy, TrainingConstraints,
    OptimizedConfig
)
from src.pipelines.lora_training import LoRAConfig


class TestTrainingConstraints:
    """Test training constraints."""
    
    def test_constraints_initialization(self):
        """Test training constraints initialization."""
        constraints = TrainingConstraints(
            max_vram_mb=8192,
            max_batch_size=4,
            max_resolution=768,
            target_training_time_hours=6.0
        )
        
        assert constraints.max_vram_mb == 8192
        assert constraints.max_batch_size == 4
        assert constraints.max_resolution == 768
        assert constraints.target_training_time_hours == 6.0
        assert constraints.enable_mixed_precision is True
        assert constraints.enable_gradient_checkpointing is True


class TestTrainingOptimizer:
    """Test training parameter optimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create training optimizer."""
        return TrainingOptimizer()
    
    @pytest.fixture
    def low_end_hardware(self):
        """Create low-end hardware configuration."""
        return HardwareConfig(
            gpu_model="GTX 1650",
            vram_size=4096,
            cuda_available=True,
            cpu_cores=4,
            ram_size=16384
        )
    
    @pytest.fixture
    def mid_range_hardware(self):
        """Create mid-range hardware configuration."""
        return HardwareConfig(
            gpu_model="RTX 3070",
            vram_size=8192,
            cuda_available=True,
            cpu_cores=8,
            ram_size=32768
        )
    
    @pytest.fixture
    def high_end_hardware(self):
        """Create high-end hardware configuration."""
        return HardwareConfig(
            gpu_model="RTX 4090",
            vram_size=24576,
            cuda_available=True,
            cpu_cores=16,
            ram_size=65536
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.profile_manager is not None
        assert 'rank' in optimizer.param_ranges
        assert 'alpha' in optimizer.param_ranges
        assert 'learning_rate' in optimizer.param_ranges
        assert optimizer.vram_estimates is not None
    
    def test_optimize_for_speed_low_end(self, optimizer, low_end_hardware):
        """Test speed optimization for low-end hardware."""
        config = optimizer.optimize_config(
            low_end_hardware, 
            OptimizationStrategy.SPEED_FOCUSED
        )
        
        assert isinstance(config, OptimizedConfig)
        assert config.lora_config.batch_size <= 2  # Limited by VRAM
        assert config.lora_config.resolution <= 640  # Lower resolution for speed
        assert config.lora_config.mixed_precision == "fp16"  # Speed optimization
        assert config.lora_config.use_8bit_adam is True  # Faster optimizer
        assert config.estimated_vram_usage_mb <= low_end_hardware.vram_size
    
    def test_optimize_for_speed_high_end(self, optimizer, high_end_hardware):
        """Test speed optimization for high-end hardware."""
        config = optimizer.optimize_config(
            high_end_hardware,
            OptimizationStrategy.SPEED_FOCUSED
        )
        
        assert config.lora_config.batch_size >= 2  # Can use larger batches
        assert config.lora_config.gradient_checkpointing is False  # Disabled for speed
        assert config.lora_config.mixed_precision == "fp16"
        assert config.estimated_vram_usage_mb <= high_end_hardware.vram_size * 0.9
    
    def test_optimize_for_memory_low_end(self, optimizer, low_end_hardware):
        """Test memory optimization for low-end hardware."""
        config = optimizer.optimize_config(
            low_end_hardware,
            OptimizationStrategy.MEMORY_FOCUSED
        )
        
        assert config.lora_config.batch_size == 1  # Minimum for memory
        assert config.lora_config.resolution <= 512  # Lower resolution
        assert config.lora_config.rank <= 8  # Lower rank for memory
        assert config.lora_config.gradient_checkpointing is True  # Memory optimization
        assert config.lora_config.enable_cpu_offload is True  # For low VRAM
        assert config.estimated_vram_usage_mb <= low_end_hardware.vram_size
    
    def test_optimize_for_quality_high_end(self, optimizer, high_end_hardware):
        """Test quality optimization for high-end hardware."""
        config = optimizer.optimize_config(
            high_end_hardware,
            OptimizationStrategy.QUALITY_FOCUSED
        )
        
        assert config.lora_config.rank >= 16  # Higher rank for quality
        assert config.lora_config.alpha >= 32.0  # Higher alpha
        assert config.lora_config.num_epochs >= 15  # More epochs
        assert config.lora_config.learning_rate <= 1e-4  # Stable learning rate
        assert config.lora_config.use_8bit_adam is False  # Full precision
        assert config.lora_config.dropout <= 0.1  # Lower dropout
    
    def test_optimize_balanced(self, optimizer, mid_range_hardware):
        """Test balanced optimization."""
        config = optimizer.optimize_config(
            mid_range_hardware,
            OptimizationStrategy.BALANCED
        )
        
        # Should be reasonable middle-ground settings
        assert 1 <= config.lora_config.batch_size <= 2
        assert 512 <= config.lora_config.resolution <= 768
        assert 8 <= config.lora_config.rank <= 32
        assert 10 <= config.lora_config.num_epochs <= 20
        assert config.lora_config.mixed_precision == "fp16"
        assert config.estimated_vram_usage_mb <= mid_range_hardware.vram_size * 0.9
    
    def test_validate_config_valid(self, optimizer, mid_range_hardware):
        """Test configuration validation - valid case."""
        config = LoRAConfig(
            batch_size=1,
            resolution=512,
            rank=16,
            num_epochs=10
        )
        
        result = optimizer.validate_config(config, mid_range_hardware)
        
        assert result['valid'] is True
        assert len(result['warnings']) == 0
        assert result['estimated_vram_mb'] > 0
        assert result['estimated_time_hours'] > 0
    
    def test_validate_config_vram_exceeded(self, optimizer, low_end_hardware):
        """Test configuration validation - VRAM exceeded."""
        config = LoRAConfig(
            batch_size=8,  # Too large for low-end hardware
            resolution=1024,  # Too high resolution
            rank=64  # Too high rank
        )
        
        result = optimizer.validate_config(config, low_end_hardware)
        
        assert result['valid'] is False
        assert any("VRAM usage" in warning for warning in result['warnings'])
        assert any("batch size" in rec for rec in result['recommendations'])
    
    def test_validate_config_warnings(self, optimizer, low_end_hardware):
        """Test configuration validation warnings."""
        config = LoRAConfig(
            batch_size=4,  # Large for low VRAM
            resolution=768,  # High for low VRAM
            num_epochs=50  # Very long training
        )
        
        result = optimizer.validate_config(config, low_end_hardware)
        
        # Should have warnings but might still be valid
        assert len(result['warnings']) > 0
        assert len(result['recommendations']) > 0
    
    def test_suggest_improvements_speed(self, optimizer, mid_range_hardware):
        """Test improvement suggestions for speed."""
        config = LoRAConfig(
            batch_size=1,
            mixed_precision="no",
            gradient_checkpointing=False
        )
        
        suggestions = optimizer.suggest_improvements(
            config, mid_range_hardware, "speed"
        )
        
        assert 'improvements' in suggestions
        assert 'alternative_configs' in suggestions
        
        # Should suggest speed improvements
        improvements = suggestions['improvements']
        improvement_params = [imp['parameter'] for imp in improvements]
        
        # Might suggest batch size increase or mixed precision
        assert len(improvements) > 0
    
    def test_suggest_improvements_memory(self, optimizer, low_end_hardware):
        """Test improvement suggestions for memory."""
        config = LoRAConfig(
            rank=32,  # High rank
            gradient_checkpointing=False,  # Memory inefficient
            batch_size=2
        )
        
        suggestions = optimizer.suggest_improvements(
            config, low_end_hardware, "memory"
        )
        
        improvements = suggestions['improvements']
        
        # Should suggest memory optimizations
        assert len(improvements) > 0
        
        # Check for memory-related suggestions
        improvement_params = [imp['parameter'] for imp in improvements]
        memory_params = ['gradient_checkpointing', 'rank', 'batch_size']
        
        assert any(param in improvement_params for param in memory_params)
    
    def test_suggest_improvements_quality(self, optimizer, high_end_hardware):
        """Test improvement suggestions for quality."""
        config = LoRAConfig(
            learning_rate=1e-3,  # High learning rate
            num_epochs=5,  # Few epochs
            rank=4  # Low rank
        )
        
        suggestions = optimizer.suggest_improvements(
            config, high_end_hardware, "quality"
        )
        
        improvements = suggestions['improvements']
        
        # Should suggest quality improvements
        assert len(improvements) > 0
        
        # Check for quality-related suggestions
        improvement_params = [imp['parameter'] for imp in improvements]
        quality_params = ['learning_rate', 'num_epochs', 'rank']
        
        assert any(param in improvement_params for param in quality_params)
    
    def test_vram_estimation(self, optimizer, mid_range_hardware):
        """Test VRAM usage estimation."""
        config = LoRAConfig(
            batch_size=2,
            resolution=768,
            rank=16
        )
        
        vram_usage = optimizer._estimate_vram_usage(config, mid_range_hardware)
        
        assert vram_usage > 0
        assert isinstance(vram_usage, float)
        
        # Higher settings should use more VRAM
        high_config = LoRAConfig(
            batch_size=4,
            resolution=1024,
            rank=32
        )
        
        high_vram_usage = optimizer._estimate_vram_usage(high_config, mid_range_hardware)
        assert high_vram_usage > vram_usage
    
    def test_training_time_estimation(self, optimizer, mid_range_hardware):
        """Test training time estimation."""
        config = LoRAConfig(
            num_epochs=10,
            batch_size=2,
            resolution=512
        )
        
        training_time = optimizer._estimate_training_time(config, mid_range_hardware)
        
        assert training_time > 0
        assert isinstance(training_time, float)
        
        # More epochs should take longer
        long_config = LoRAConfig(
            num_epochs=20,
            batch_size=2,
            resolution=512
        )
        
        long_training_time = optimizer._estimate_training_time(long_config, mid_range_hardware)
        assert long_training_time > training_time
    
    def test_create_default_constraints(self, optimizer, mid_range_hardware):
        """Test default constraints creation."""
        with patch.object(optimizer.profile_manager, 'get_profile') as mock_get_profile:
            mock_profile = Mock()
            mock_profile.optimizations = {
                'batch_size': 2,
                'max_resolution': 768,
                'mixed_precision': True,
                'gradient_checkpointing': True
            }
            mock_get_profile.return_value = mock_profile
            
            constraints = optimizer._create_default_constraints(mid_range_hardware, mock_profile)
            
            assert constraints.max_vram_mb == int(mid_range_hardware.vram_size * 0.9)
            assert constraints.max_batch_size == 2
            assert constraints.max_resolution == 768
            assert constraints.enable_mixed_precision is True
            assert constraints.enable_gradient_checkpointing is True
    
    def test_optimization_notes_generation(self, optimizer, mid_range_hardware):
        """Test optimization notes generation."""
        config = LoRAConfig(
            batch_size=1,
            resolution=512,
            mixed_precision="fp16",
            gradient_checkpointing=True
        )
        
        notes = optimizer._generate_optimization_notes(
            config, mid_range_hardware, OptimizationStrategy.BALANCED
        )
        
        assert 'strategy' in notes
        assert 'hardware' in notes
        assert 'balanced' in notes['strategy']
        assert 'RTX 3070' in notes['hardware']
        
        # Should have notes about specific settings
        if config.batch_size == 1:
            assert 'batch_size' in notes
        if config.mixed_precision == "fp16":
            assert 'mixed_precision' in notes
    
    def test_performance_predictions(self, optimizer, mid_range_hardware):
        """Test performance predictions."""
        config = LoRAConfig(
            rank=16,
            num_epochs=15,
            resolution=640
        )
        
        predictions = optimizer._create_performance_predictions(config, mid_range_hardware)
        
        assert 'vram_usage_mb' in predictions
        assert 'vram_efficiency' in predictions
        assert 'training_time_hours' in predictions
        assert 'time_efficiency' in predictions
        assert 'quality_prediction' in predictions
        assert 'quality_score' in predictions
        assert 'recommended_for' in predictions
        
        assert predictions['vram_efficiency'] >= 0
        assert predictions['vram_efficiency'] <= 1
        assert predictions['quality_score'] >= 0
        assert predictions['quality_score'] <= 1
        assert isinstance(predictions['recommended_for'], list)
    
    def test_use_case_recommendations(self, optimizer, high_end_hardware):
        """Test use case recommendations."""
        # High quality config
        high_quality_config = LoRAConfig(
            rank=32,
            num_epochs=20,
            resolution=768,
            batch_size=2
        )
        
        recommendations = optimizer._get_use_case_recommendations(
            high_quality_config, high_end_hardware
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should recommend high-quality use cases
        rec_text = ' '.join(recommendations).lower()
        assert any(term in rec_text for term in ['quality', 'production', 'commercial'])
    
    def test_alternative_configs_generation(self, optimizer, mid_range_hardware):
        """Test alternative configuration generation."""
        config = LoRAConfig()
        
        suggestions = optimizer.suggest_improvements(
            config, mid_range_hardware, "balanced"
        )
        
        alt_configs = suggestions['alternative_configs']
        
        # Should have alternative strategies
        assert len(alt_configs) > 0
        
        strategies = [alt['strategy'] for alt in alt_configs]
        expected_strategies = ['speed_focused', 'memory_focused', 'quality_focused']
        
        # Should have at least some alternative strategies
        assert any(strategy in strategies for strategy in expected_strategies)
        
        # Each alternative should have required fields
        for alt in alt_configs:
            assert 'strategy' in alt
            assert 'config' in alt
            assert 'estimated_vram_mb' in alt
            assert 'estimated_time_hours' in alt


class TestIntegration:
    """Integration tests for training optimizer."""
    
    def test_full_optimization_pipeline(self):
        """Test full optimization pipeline."""
        optimizer = TrainingOptimizer()
        
        hardware_configs = [
            HardwareConfig("GTX 1650", 4096, True, 4, 16384),
            HardwareConfig("RTX 3070", 8192, True, 8, 32768),
            HardwareConfig("RTX 4090", 24576, True, 16, 65536)
        ]
        
        strategies = [
            OptimizationStrategy.SPEED_FOCUSED,
            OptimizationStrategy.MEMORY_FOCUSED,
            OptimizationStrategy.QUALITY_FOCUSED,
            OptimizationStrategy.BALANCED
        ]
        
        for hardware in hardware_configs:
            for strategy in strategies:
                config = optimizer.optimize_config(hardware, strategy)
                
                # Verify optimization result
                assert isinstance(config, OptimizedConfig)
                assert isinstance(config.lora_config, LoRAConfig)
                assert config.estimated_vram_usage_mb > 0
                assert config.estimated_training_time_hours > 0
                assert isinstance(config.optimization_notes, dict)
                assert isinstance(config.performance_predictions, dict)
                
                # Verify VRAM constraint
                assert config.estimated_vram_usage_mb <= hardware.vram_size * 0.95
                
                # Verify configuration is valid
                validation = optimizer.validate_config(config.lora_config, hardware)
                assert validation['valid'] or len(validation['warnings']) > 0  # Should be valid or have warnings
    
    def test_constraint_based_optimization(self):
        """Test optimization with custom constraints."""
        optimizer = TrainingOptimizer()
        
        hardware = HardwareConfig("RTX 3070", 8192, True, 8, 32768)
        
        # Strict memory constraints
        memory_constraints = TrainingConstraints(
            max_vram_mb=6000,  # Stricter than hardware
            max_batch_size=1,
            max_resolution=512,
            target_training_time_hours=4.0
        )
        
        config = optimizer.optimize_config(
            hardware, 
            OptimizationStrategy.BALANCED,
            memory_constraints
        )
        
        # Should respect constraints
        assert config.lora_config.batch_size <= memory_constraints.max_batch_size
        assert config.lora_config.resolution <= memory_constraints.max_resolution
        assert config.estimated_vram_usage_mb <= memory_constraints.max_vram_mb