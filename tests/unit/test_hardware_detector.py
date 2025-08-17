"""
Unit tests for hardware detection module.

Tests GPU, CPU, and memory detection functionality as well as
hardware profile matching and optimization strategy selection.
"""

import pytest
import sys
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import platform
import psutil

from src.hardware.detector import HardwareDetector
from src.hardware.profiles import HardwareProfileManager, OptimizationProfile
from src.core.interfaces import HardwareConfig, HardwareError


class TestHardwareDetector:
    """Test cases for HardwareDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = HardwareDetector()
    
    def test_detect_hardware_success(self):
        """Test successful hardware detection."""
        with patch.object(self.detector, '_detect_gpu') as mock_gpu, \
             patch.object(self.detector, '_detect_cpu') as mock_cpu, \
             patch.object(self.detector, '_detect_memory') as mock_memory:
            
            # Mock return values
            mock_gpu.return_value = {
                'name': 'RTX 3070',
                'vram_mb': 8192,
                'cuda_available': True,
                'driver_version': '525.60.11'
            }
            mock_cpu.return_value = {
                'cores': 8,
                'logical_cores': 16,
                'frequency': 3600,
                'name': 'Intel Core i7-10700K'
            }
            mock_memory.return_value = {
                'total_mb': 16384,
                'available_mb': 12288,
                'swap_mb': 4096,
                'usage_percent': 25.0
            }
            
            # Test detection
            config = self.detector.detect_hardware()
            
            # Verify results
            assert isinstance(config, HardwareConfig)
            assert config.vram_size == 8192
            assert config.gpu_model == 'RTX 3070'
            assert config.cpu_cores == 8
            assert config.ram_size == 16384
            assert config.cuda_available is True
            assert config.optimization_level == 'balanced'
    
    def test_detect_hardware_failure(self):
        """Test hardware detection failure handling."""
        with patch.object(self.detector, '_detect_gpu', side_effect=Exception("GPU detection failed")):
            with pytest.raises(HardwareError):
                self.detector.detect_hardware()
    
    def test_get_optimization_strategy(self):
        """Test optimization strategy determination."""
        # Test aggressive optimization (low VRAM)
        config_low = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        assert self.detector.get_optimization_strategy(config_low) == 'aggressive'
        
        # Test balanced optimization (medium VRAM)
        config_medium = HardwareConfig(
            vram_size=8192, gpu_model='RTX 3070', cpu_cores=8,
            ram_size=16384, cuda_available=True, optimization_level='balanced'
        )
        assert self.detector.get_optimization_strategy(config_medium) == 'balanced'
        
        # Test minimal optimization (high VRAM)
        config_high = HardwareConfig(
            vram_size=24576, gpu_model='RTX 4090', cpu_cores=16,
            ram_size=32768, cuda_available=True, optimization_level='minimal'
        )
        assert self.detector.get_optimization_strategy(config_high) == 'minimal'
    
    def test_validate_requirements_success(self):
        """Test successful requirements validation."""
        config = HardwareConfig(
            vram_size=8192, gpu_model='RTX 3070', cpu_cores=8,
            ram_size=16384, cuda_available=True, optimization_level='balanced'
        )
        
        requirements = {
            'min_vram_mb': 6000,
            'min_ram_mb': 8192,
            'min_cpu_cores': 4,
            'requires_cuda': True
        }
        
        assert self.detector.validate_requirements(config, requirements) is True
    
    def test_validate_requirements_failure(self):
        """Test requirements validation failure."""
        config = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=False, optimization_level='aggressive'
        )
        
        # Test insufficient VRAM
        requirements = {'min_vram_mb': 8000}
        assert self.detector.validate_requirements(config, requirements) is False
        
        # Test missing CUDA
        requirements = {'requires_cuda': True}
        assert self.detector.validate_requirements(config, requirements) is False
        
        # Test insufficient CPU cores
        requirements = {'min_cpu_cores': 8}
        assert self.detector.validate_requirements(config, requirements) is False
    
    @patch('subprocess.run')
    def test_detect_nvidia_gpu_with_nvidia_smi(self, mock_run):
        """Test NVIDIA GPU detection using nvidia-smi."""
        # Mock nvidia-smi output
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3070, 8192, 525.60.11\n"
        mock_run.return_value = mock_result
        
        # Mock pynvml import failure
        with patch('builtins.__import__', side_effect=ImportError("pynvml not available")):
            gpu_info = self.detector._detect_nvidia_gpu()
        
        assert gpu_info is not None
        assert gpu_info['name'] == 'NVIDIA GeForce RTX 3070'
        assert gpu_info['vram_mb'] == 8192
        assert gpu_info['cuda_available'] is True
        assert gpu_info['driver_version'] == '525.60.11'
    
    def test_detect_nvidia_gpu_with_pynvml(self):
        """Test NVIDIA GPU detection using pynvml."""
        # Mock pynvml module
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = MagicMock()
        mock_pynvml.nvmlDeviceGetName.return_value = b'NVIDIA GeForce RTX 4090'
        
        mock_mem_info = MagicMock()
        mock_mem_info.total = 25769803776  # 24GB in bytes
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = b'525.60.11'
        
        # Mock the import
        with patch.dict('sys.modules', {'pynvml': mock_pynvml}):
            gpu_info = self.detector._detect_nvidia_gpu()
        
        assert gpu_info is not None
        assert gpu_info['name'] == 'NVIDIA GeForce RTX 4090'
        assert gpu_info['vram_mb'] == 24576  # 24GB in MB
        assert gpu_info['cuda_available'] is True
        assert gpu_info['driver_version'] == '525.60.11'
    
    def test_detect_nvidia_gpu_not_found(self):
        """Test NVIDIA GPU detection when no GPU is found."""
        with patch('builtins.__import__', side_effect=ImportError("pynvml not available")), \
             patch('subprocess.run', side_effect=FileNotFoundError("nvidia-smi not found")):
            
            gpu_info = self.detector._detect_nvidia_gpu()
            assert gpu_info is None
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    @patch('platform.processor')
    def test_detect_cpu(self, mock_processor, mock_freq, mock_count):
        """Test CPU detection."""
        mock_count.side_effect = [8, 16]  # physical, logical cores
        mock_processor.return_value = 'Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz'
        
        mock_freq_info = MagicMock()
        mock_freq_info.max = 3800.0
        mock_freq_info.current = 3600.0
        mock_freq.return_value = mock_freq_info
        
        cpu_info = self.detector._detect_cpu()
        
        assert cpu_info['cores'] == 8
        assert cpu_info['logical_cores'] == 16
        assert cpu_info['frequency'] == 3800.0
        assert 'Intel' in cpu_info['name']
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_detect_memory(self, mock_swap, mock_virtual):
        """Test memory detection."""
        mock_virtual_mem = MagicMock()
        mock_virtual_mem.total = 17179869184  # 16GB in bytes
        mock_virtual_mem.available = 12884901888  # 12GB available
        mock_virtual_mem.percent = 25.0
        mock_virtual.return_value = mock_virtual_mem
        
        mock_swap_mem = MagicMock()
        mock_swap_mem.total = 4294967296  # 4GB swap
        mock_swap.return_value = mock_swap_mem
        
        memory_info = self.detector._detect_memory()
        
        assert memory_info['total_mb'] == 16384
        assert memory_info['available_mb'] == 12288
        assert memory_info['swap_mb'] == 4096
        assert memory_info['usage_percent'] == 25.0
    
    def test_determine_optimization_level(self):
        """Test optimization level determination."""
        # Test aggressive optimization
        gpu_info = {'vram_mb': 4096}
        memory_info = {'total_mb': 8192}
        level = self.detector._determine_optimization_level(gpu_info, memory_info)
        assert level == 'aggressive'
        
        # Test balanced optimization
        gpu_info = {'vram_mb': 8192}
        memory_info = {'total_mb': 16384}
        level = self.detector._determine_optimization_level(gpu_info, memory_info)
        assert level == 'balanced'
        
        # Test minimal optimization
        gpu_info = {'vram_mb': 24576}
        memory_info = {'total_mb': 32768}
        level = self.detector._determine_optimization_level(gpu_info, memory_info)
        assert level == 'minimal'
    
    def test_get_detailed_info(self):
        """Test detailed hardware information retrieval."""
        with patch.object(self.detector, '_detect_gpu') as mock_gpu, \
             patch.object(self.detector, '_detect_cpu') as mock_cpu, \
             patch.object(self.detector, '_detect_memory') as mock_memory:
            
            mock_gpu.return_value = {'name': 'RTX 3070', 'vram_mb': 8192}
            mock_cpu.return_value = {'cores': 8, 'name': 'Intel i7'}
            mock_memory.return_value = {'total_mb': 16384}
            
            info = self.detector.get_detailed_info()
            
            assert 'gpu' in info
            assert 'cpu' in info
            assert 'memory' in info
            assert 'platform' in info
            assert info['gpu']['name'] == 'RTX 3070'
            assert info['cpu']['cores'] == 8


class TestHardwareProfileManager:
    """Test cases for HardwareProfileManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profile_manager = HardwareProfileManager()
    
    def test_get_profile_exact_match(self):
        """Test getting profile with exact GPU match."""
        config = HardwareConfig(
            vram_size=8192, gpu_model='RTX 3070', cpu_cores=8,
            ram_size=16384, cuda_available=True, optimization_level='balanced'
        )
        
        profile = self.profile_manager.get_profile(config)
        
        assert profile.name == "RTX 3070 (8GB VRAM)"
        assert profile.optimization_level == "balanced"
        assert 'sdxl-turbo' in profile.recommended_models
    
    def test_get_profile_vram_match(self):
        """Test getting profile with VRAM range match."""
        config = HardwareConfig(
            vram_size=4096, gpu_model='Unknown GPU', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        
        profile = self.profile_manager.get_profile(config)
        
        assert profile.optimization_level == "aggressive"
        assert profile.min_vram_mb <= 4096 <= profile.max_vram_mb
    
    def test_get_profile_adaptive(self):
        """Test adaptive profile creation for unknown hardware."""
        config = HardwareConfig(
            vram_size=12288, gpu_model='Unknown Future GPU', cpu_cores=12,
            ram_size=24576, cuda_available=True, optimization_level='balanced'
        )
        
        profile = self.profile_manager.get_profile(config)
        
        assert "Adaptive Profile" in profile.name
        assert profile.optimization_level == "balanced"
        assert profile.min_vram_mb <= 12288 <= profile.max_vram_mb
    
    def test_get_model_recommendations(self):
        """Test model recommendations for different hardware."""
        # Low-end hardware
        config_low = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        recommendations = self.profile_manager.get_model_recommendations(config_low)
        assert 'stable-diffusion-v1-5' in recommendations
        assert 'flux.1-schnell' not in recommendations
        
        # High-end hardware
        config_high = HardwareConfig(
            vram_size=24576, gpu_model='RTX 4090', cpu_cores=16,
            ram_size=32768, cuda_available=True, optimization_level='minimal'
        )
        recommendations = self.profile_manager.get_model_recommendations(config_high)
        assert 'flux.1-schnell' in recommendations
    
    def test_get_optimization_settings(self):
        """Test optimization settings retrieval."""
        config = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        
        settings = self.profile_manager.get_optimization_settings(config)
        
        assert settings['attention_slicing'] is True
        assert settings['cpu_offload'] is True
        assert settings['batch_size'] == 1
        assert settings['max_resolution'] == 512
    
    def test_validate_model_compatibility(self):
        """Test model compatibility validation."""
        # Low VRAM config
        config_low = HardwareConfig(
            vram_size=4096, gpu_model='GTX 1650', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        
        # Should be compatible with SD 1.5
        assert self.profile_manager.validate_model_compatibility(config_low, 'stable-diffusion-v1-5') is True
        
        # Should not be compatible with FLUX
        assert self.profile_manager.validate_model_compatibility(config_low, 'flux.1-schnell') is False
        
        # High VRAM config
        config_high = HardwareConfig(
            vram_size=24576, gpu_model='RTX 4090', cpu_cores=16,
            ram_size=32768, cuda_available=True, optimization_level='minimal'
        )
        
        # Should be compatible with all models
        assert self.profile_manager.validate_model_compatibility(config_high, 'flux.1-schnell') is True
        assert self.profile_manager.validate_model_compatibility(config_high, 'stable-diffusion-v1-5') is True
    
    def test_normalize_gpu_name(self):
        """Test GPU name normalization."""
        # Test various GPU name formats
        assert self.profile_manager._normalize_gpu_name('NVIDIA GeForce RTX 3070') == 'rtx_3070'
        assert self.profile_manager._normalize_gpu_name('RTX 4090') == 'rtx_4090'
        assert self.profile_manager._normalize_gpu_name('GTX 1650 SUPER') == 'gtx_1650'
        assert self.profile_manager._normalize_gpu_name('GeForce GTX 1080 Ti') == 'gtx_1080'
    
    def test_create_adaptive_profile_low_vram(self):
        """Test adaptive profile creation for low VRAM."""
        config = HardwareConfig(
            vram_size=3072, gpu_model='Low-end GPU', cpu_cores=4,
            ram_size=8192, cuda_available=True, optimization_level='aggressive'
        )
        
        profile = self.profile_manager._create_adaptive_profile(config)
        
        assert profile.optimization_level == 'aggressive'
        assert profile.optimizations['attention_slicing'] is True
        assert profile.optimizations['cpu_offload'] is True
        assert profile.optimizations['low_vram_mode'] is True
        assert 'phi-3-mini' in profile.recommended_models
    
    def test_create_adaptive_profile_high_vram(self):
        """Test adaptive profile creation for high VRAM."""
        config = HardwareConfig(
            vram_size=20480, gpu_model='High-end GPU', cpu_cores=16,
            ram_size=32768, cuda_available=True, optimization_level='minimal'
        )
        
        profile = self.profile_manager._create_adaptive_profile(config)
        
        assert profile.optimization_level == 'minimal'
        assert profile.optimizations['attention_slicing'] is False
        assert profile.optimizations['cpu_offload'] is False
        assert profile.optimizations['low_vram_mode'] is False
        assert 'flux.1-schnell' in profile.recommended_models


if __name__ == '__main__':
    pytest.main([__file__])