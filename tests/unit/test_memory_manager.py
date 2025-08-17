"""
Unit tests for memory management engine.

Tests VRAM optimization strategies, model switching, cache clearing,
and memory monitoring functionality.
"""

import pytest
import sys
import time
import unittest.mock as mock
from unittest.mock import patch, MagicMock

from src.hardware.memory_manager import MemoryManager, OptimizationStrategy, MemoryStatus, ModelMemoryInfo
from src.core.interfaces import HardwareConfig, MemoryError


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.hardware_config_low = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        
        self.hardware_config_high = HardwareConfig(
            vram_size=24576, gpu_model='RTX 4090', cpu_cores=16,
            ram_size=32768, cuda_available=True, optimization_level='minimal'
        )
    
    def test_initialization_aggressive_strategy(self):
        """Test memory manager initialization with aggressive strategy."""
        manager = MemoryManager(self.hardware_config_low)
        
        assert manager.strategy == OptimizationStrategy.AGGRESSIVE
        assert manager.hardware_config == self.hardware_config_low
        assert manager.cleanup_threshold == 0.85
        assert len(manager.loaded_models) == 0
    
    def test_initialization_minimal_strategy(self):
        """Test memory manager initialization with minimal strategy."""
        manager = MemoryManager(self.hardware_config_high)
        
        assert manager.strategy == OptimizationStrategy.MINIMAL
        assert manager.hardware_config == self.hardware_config_high
    
    def test_optimize_model_loading_aggressive(self):
        """Test model loading optimization with aggressive strategy."""
        manager = MemoryManager(self.hardware_config_low)
        
        params = manager.optimize_model_loading('stable-diffusion-v1-5')
        
        assert params['enable_attention_slicing'] is True
        assert params['cpu_offload'] is True
        assert params['sequential_cpu_offload'] is True
        assert params['enable_vae_slicing'] is True
        assert params['torch_dtype'] == 'float16'
    
    def test_optimize_model_loading_minimal(self):
        """Test model loading optimization with minimal strategy."""
        manager = MemoryManager(self.hardware_config_high)
        
        params = manager.optimize_model_loading('flux.1-schnell')
        
        assert params['cpu_offload'] is False
        assert params['sequential_cpu_offload'] is False
        assert params['torch_dtype'] == 'bfloat16'  # FLUX-specific
    
    def test_model_specific_optimizations(self):
        """Test model-specific optimization parameters."""
        manager = MemoryManager(self.hardware_config_low)
        
        # Test FLUX optimizations
        flux_params = manager._get_model_specific_optimizations('flux.1-schnell')
        assert flux_params['torch_dtype'] == 'bfloat16'
        
        # Test SDXL optimizations
        sdxl_params = manager._get_model_specific_optimizations('sdxl-turbo')
        assert sdxl_params['torch_dtype'] == 'float16'
        assert sdxl_params['variant'] == 'fp16'
        
        # Test SD 1.5 optimizations
        sd_params = manager._get_model_specific_optimizations('stable-diffusion-v1-5')
        assert sd_params['torch_dtype'] == 'float16'
    
    def test_manage_model_switching(self):
        """Test model switching with memory cleanup."""
        manager = MemoryManager(self.hardware_config_low)
        
        # Add a mock model
        manager.loaded_models['old_model'] = ModelMemoryInfo(
            model_name='old_model',
            vram_usage_mb=3000,
            ram_usage_mb=1000,
            load_time=time.time(),
            last_used=time.time(),
            optimization_flags={}
        )
        
        with patch.object(manager, 'clear_vram_cache') as mock_clear, \
             patch.object(manager, '_ensure_free_memory', return_value=True) as mock_ensure:
            
            manager.manage_model_switching('old_model', 'new_model')
            
            # Verify old model was removed
            assert 'old_model' not in manager.loaded_models
            
            # Verify new model was added
            assert 'new_model' in manager.loaded_models
            
            # Verify cache was cleared
            mock_clear.assert_called_once()
            mock_ensure.assert_called_once()
    
    def test_manage_model_switching_insufficient_memory(self):
        """Test model switching with insufficient memory."""
        manager = MemoryManager(self.hardware_config_low)
        
        with patch.object(manager, '_ensure_free_memory', return_value=False):
            with pytest.raises(MemoryError):
                manager.manage_model_switching('old_model', 'new_model')
    
    def test_clear_vram_cache_with_torch(self):
        """Test VRAM cache clearing with PyTorch available."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', True), \
             patch('src.hardware.memory_manager.torch', mock_torch), \
             patch('gc.collect') as mock_gc:
            
            manager = MemoryManager(self.hardware_config_low)
            manager.clear_vram_cache()
            
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.synchronize.assert_called_once()
            mock_gc.assert_called()
    
    @patch('src.hardware.memory_manager.TORCH_AVAILABLE', False)
    @patch('gc.collect')
    def test_clear_vram_cache_without_torch(self, mock_gc):
        """Test VRAM cache clearing without PyTorch."""
        manager = MemoryManager(self.hardware_config_low)
        
        manager.clear_vram_cache()
        
        mock_gc.assert_called()
    
    def test_get_memory_status_with_torch(self):
        """Test memory status retrieval with PyTorch."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock()
        mock_torch.cuda.memory_allocated.return_value = 2147483648  # 2GB in bytes
        mock_torch.cuda.memory_reserved.return_value = 3221225472   # 3GB in bytes
        
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', True), \
             patch('src.hardware.memory_manager.torch', mock_torch):
            
            manager = MemoryManager(self.hardware_config_low)
            
            # Add a mock model
            manager.loaded_models['test_model'] = ModelMemoryInfo(
                model_name='test_model',
                vram_usage_mb=2048,
                ram_usage_mb=1024,
                load_time=time.time(),
                last_used=time.time(),
                optimization_flags={}
            )
            
            status = manager.get_memory_status()
            
            assert status['strategy'] == 'aggressive'
            assert status['hardware']['gpu_model'] == 'GTX 1650'
            assert status['hardware']['total_vram_mb'] == 4096
            assert status['loaded_models'] == 1
            assert status['vram_allocated_mb'] == 2048  # 2GB
            assert status['vram_reserved_mb'] == 3072   # 3GB
            assert 'test_model' in status['model_details']
    
    def test_get_memory_status_without_torch(self):
        """Test memory status retrieval without PyTorch."""
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', False):
            manager = MemoryManager(self.hardware_config_low)
            
            status = manager.get_memory_status()
            
            assert status['vram_allocated_mb'] == 0
            assert status['vram_reserved_mb'] == 0
            assert status['vram_free_mb'] == 4096
            assert status['vram_utilization_percent'] == 0
    
    def test_enable_cpu_offloading_aggressive(self):
        """Test CPU offloading configuration for aggressive strategy."""
        manager = MemoryManager(self.hardware_config_low)
        
        config = manager.enable_cpu_offloading('test_model')
        
        assert config['device_map'] == 'auto'
        assert config['sequential_cpu_offload'] is True
        assert config['cpu_offload'] is True
    
    def test_enable_cpu_offloading_minimal(self):
        """Test CPU offloading configuration for minimal strategy."""
        manager = MemoryManager(self.hardware_config_high)
        
        config = manager.enable_cpu_offloading('test_model')
        
        assert config['device_map'] == 'auto'
        assert 'sequential_cpu_offload' not in config or config['sequential_cpu_offload'] is False
    
    def test_enable_attention_slicing(self):
        """Test attention slicing configuration."""
        manager = MemoryManager(self.hardware_config_low)
        
        config = manager.enable_attention_slicing('test_model')
        
        assert config['enable_attention_slicing'] is True
        assert config['attention_slice_size'] == 1  # Most aggressive for low VRAM
    
    def test_enable_attention_slicing_high_vram(self):
        """Test attention slicing configuration for high VRAM."""
        manager = MemoryManager(self.hardware_config_high)
        
        config = manager.enable_attention_slicing('test_model')
        
        assert config['enable_attention_slicing'] is True
        assert config['attention_slice_size'] == 4  # Conservative for high VRAM
    
    def test_enable_vae_optimizations(self):
        """Test VAE optimization configuration."""
        manager = MemoryManager(self.hardware_config_low)
        
        config = manager.enable_vae_optimizations('test_model')
        
        assert config['enable_vae_slicing'] is True
        assert config['enable_vae_tiling'] is True
    
    def test_set_cleanup_threshold_valid(self):
        """Test setting valid cleanup threshold."""
        manager = MemoryManager(self.hardware_config_low)
        
        manager.set_cleanup_threshold(0.75)
        assert manager.cleanup_threshold == 0.75
    
    def test_set_cleanup_threshold_invalid(self):
        """Test setting invalid cleanup threshold."""
        manager = MemoryManager(self.hardware_config_low)
        
        with pytest.raises(ValueError):
            manager.set_cleanup_threshold(0.3)  # Too low
        
        with pytest.raises(ValueError):
            manager.set_cleanup_threshold(0.98)  # Too high
    
    def test_estimate_model_vram(self):
        """Test VRAM estimation for different models."""
        manager = MemoryManager(self.hardware_config_low)
        
        # Test known models
        assert manager._estimate_model_vram('stable-diffusion-v1-5') == 3500
        assert manager._estimate_model_vram('sdxl-turbo') == 7000
        assert manager._estimate_model_vram('flux.1-schnell') == 20000
        assert manager._estimate_model_vram('stable-video-diffusion') == 12000
        
        # Test unknown models
        assert manager._estimate_model_vram('unknown-video-model') == 12000  # Video default
        assert manager._estimate_model_vram('unknown-xl-model') == 8000      # XL default
        assert manager._estimate_model_vram('unknown-model') == 4000         # General default
    
    def test_should_cleanup_memory(self):
        """Test memory cleanup threshold detection."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_reserved.return_value = 3221225472  # 3GB
        
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', True), \
             patch('src.hardware.memory_manager.torch', mock_torch):
            
            manager = MemoryManager(self.hardware_config_low)  # 4GB VRAM
            
            # 3GB used out of 4GB = 75% usage, below 85% threshold
            assert manager._should_cleanup_memory() is False
            
            # Simulate higher usage
            mock_torch.cuda.memory_reserved.return_value = 3758096384  # 3.5GB = 87.5% usage
            assert manager._should_cleanup_memory() is True
    
    def test_ensure_free_memory_sufficient(self):
        """Test ensuring sufficient free memory when available."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_reserved.return_value = 2147483648  # 2GB
        
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', True), \
             patch('src.hardware.memory_manager.torch', mock_torch):
            
            manager = MemoryManager(self.hardware_config_low)  # 4GB VRAM
            
            # 2GB used, 2GB free, requiring 1GB
            assert manager._ensure_free_memory(1024) is True
    
    def test_ensure_free_memory_insufficient(self):
        """Test ensuring sufficient free memory when insufficient."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_reserved.return_value = 3758096384  # 3.5GB
        
        with patch('src.hardware.memory_manager.TORCH_AVAILABLE', True), \
             patch('src.hardware.memory_manager.torch', mock_torch):
            
            manager = MemoryManager(self.hardware_config_low)  # 4GB VRAM
            
            with patch.object(manager, '_cleanup_unused_models') as mock_cleanup:
                # 3.5GB used, 0.5GB free, requiring 1GB
                result = manager._ensure_free_memory(1024)
                
                # Should attempt cleanup
                mock_cleanup.assert_called_once()
                
                # Still insufficient after cleanup
                assert result is False
    
    def test_cleanup_unused_models(self):
        """Test cleanup of unused models."""
        manager = MemoryManager(self.hardware_config_low)
        
        current_time = time.time()
        
        # Add models with different usage times
        manager.loaded_models['recent_model'] = ModelMemoryInfo(
            model_name='recent_model',
            vram_usage_mb=2000,
            ram_usage_mb=500,
            load_time=current_time,
            last_used=current_time - 60,  # 1 minute ago
            optimization_flags={}
        )
        
        manager.loaded_models['old_model'] = ModelMemoryInfo(
            model_name='old_model',
            vram_usage_mb=2000,
            ram_usage_mb=500,
            load_time=current_time,
            last_used=current_time - 400,  # 6+ minutes ago
            optimization_flags={}
        )
        
        with patch.object(manager, 'clear_vram_cache') as mock_clear:
            manager._cleanup_unused_models()
            
            # Recent model should remain
            assert 'recent_model' in manager.loaded_models
            
            # Old model should be removed
            assert 'old_model' not in manager.loaded_models
            
            # Cache should be cleared
            mock_clear.assert_called_once()
    
    def test_update_model_usage(self):
        """Test updating model usage statistics."""
        manager = MemoryManager(self.hardware_config_low)
        
        # Add a model
        initial_time = time.time() - 100
        manager.loaded_models['test_model'] = ModelMemoryInfo(
            model_name='test_model',
            vram_usage_mb=2000,
            ram_usage_mb=500,
            load_time=initial_time,
            last_used=initial_time,
            optimization_flags={}
        )
        
        # Update usage
        manager.update_model_usage('test_model', 2500)
        
        # Verify updates
        model_info = manager.loaded_models['test_model']
        assert model_info.vram_usage_mb == 2500
        assert model_info.last_used > initial_time
    
    def test_get_optimization_recommendations_aggressive(self):
        """Test optimization recommendations for aggressive strategy."""
        manager = MemoryManager(self.hardware_config_low)
        
        recommendations = manager.get_optimization_recommendations()
        
        assert recommendations['strategy'] == 'aggressive'
        assert 'Enable attention slicing for all models' in recommendations['optimizations']
        assert 'Use CPU offloading for large models' in recommendations['optimizations']
    
    def test_get_optimization_recommendations_minimal(self):
        """Test optimization recommendations for minimal strategy."""
        manager = MemoryManager(self.hardware_config_high)
        
        recommendations = manager.get_optimization_recommendations()
        
        assert recommendations['strategy'] == 'minimal'
        assert 'Can run multiple models simultaneously' in recommendations['optimizations']
        assert 'Higher resolutions and batch sizes supported' in recommendations['optimizations']
    
    def test_register_memory_callback(self):
        """Test registering memory status callbacks."""
        manager = MemoryManager(self.hardware_config_low)
        
        callback_called = False
        
        def test_callback(status):
            nonlocal callback_called
            callback_called = True
        
        manager.register_memory_callback(test_callback)
        
        assert len(manager.memory_callbacks) == 1
        assert manager.memory_callbacks[0] == test_callback


if __name__ == '__main__':
    pytest.main([__file__])