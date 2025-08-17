"""
Hardware detection implementation for GPU, CPU, and memory detection.

This module implements the IHardwareDetector interface to automatically detect
available hardware resources and create appropriate configuration profiles.
"""

import platform
import subprocess
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from ..core.interfaces import IHardwareDetector, HardwareConfig, HardwareError
from .profiles import HardwareProfileManager

logger = logging.getLogger(__name__)


class HardwareDetector(IHardwareDetector):
    """
    Hardware detection implementation that identifies GPU, CPU, and memory resources.
    
    Supports NVIDIA GPUs with CUDA, AMD GPUs with ROCm, and CPU-only configurations.
    Automatically determines optimal settings based on detected hardware.
    """
    
    def __init__(self):
        self.profile_manager = HardwareProfileManager()
        self._gpu_info_cache: Optional[Dict[str, Any]] = None
        self._system_info_cache: Optional[Dict[str, Any]] = None
    
    def detect_hardware(self) -> HardwareConfig:
        """
        Detect available hardware and create configuration.
        
        Returns:
            HardwareConfig: Complete hardware configuration with optimization settings
            
        Raises:
            HardwareError: If hardware detection fails or no compatible hardware found
        """
        try:
            logger.info("Starting hardware detection...")
            
            # Detect GPU information
            gpu_info = self._detect_gpu()
            
            # Detect CPU and memory
            cpu_info = self._detect_cpu()
            memory_info = self._detect_memory()
            
            # Create hardware configuration
            hardware_config = HardwareConfig(
                vram_size=gpu_info.get('vram_mb', 0),
                gpu_model=gpu_info.get('name', 'Unknown'),
                cpu_cores=cpu_info.get('cores', 1),
                ram_size=memory_info.get('total_mb', 0),
                cuda_available=gpu_info.get('cuda_available', False),
                optimization_level=self._determine_optimization_level(gpu_info, memory_info)
            )
            
            logger.info(f"Hardware detected: {hardware_config.gpu_model} "
                       f"({hardware_config.vram_size}MB VRAM), "
                       f"{hardware_config.cpu_cores} CPU cores, "
                       f"{hardware_config.ram_size}MB RAM")
            
            return hardware_config
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            raise HardwareError(f"Failed to detect hardware: {e}")
    
    def get_optimization_strategy(self, hardware_config: HardwareConfig) -> str:
        """
        Determine optimal strategy for given hardware.
        
        Args:
            hardware_config: Hardware configuration to analyze
            
        Returns:
            str: Optimization strategy ("aggressive", "balanced", "minimal")
        """
        vram_mb = hardware_config.vram_size
        
        if vram_mb < 6000:  # Less than 6GB VRAM
            return "aggressive"
        elif vram_mb < 12000:  # 6-12GB VRAM
            return "balanced"
        else:  # 12GB+ VRAM
            return "minimal"
    
    def validate_requirements(self, hardware_config: HardwareConfig, 
                            requirements: Dict[str, Any]) -> bool:
        """
        Validate if hardware meets minimum requirements.
        
        Args:
            hardware_config: Current hardware configuration
            requirements: Dictionary of minimum requirements
            
        Returns:
            bool: True if requirements are met, False otherwise
        """
        min_vram = requirements.get('min_vram_mb', 0)
        min_ram = requirements.get('min_ram_mb', 0)
        min_cpu_cores = requirements.get('min_cpu_cores', 1)
        requires_cuda = requirements.get('requires_cuda', False)
        
        if hardware_config.vram_size < min_vram:
            logger.warning(f"Insufficient VRAM: {hardware_config.vram_size}MB < {min_vram}MB")
            return False
        
        if hardware_config.ram_size < min_ram:
            logger.warning(f"Insufficient RAM: {hardware_config.ram_size}MB < {min_ram}MB")
            return False
        
        if hardware_config.cpu_cores < min_cpu_cores:
            logger.warning(f"Insufficient CPU cores: {hardware_config.cpu_cores} < {min_cpu_cores}")
            return False
        
        if requires_cuda and not hardware_config.cuda_available:
            logger.warning("CUDA required but not available")
            return False
        
        return True
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """
        Detect GPU information including VRAM and CUDA availability.
        
        Returns:
            Dict containing GPU name, VRAM size, and CUDA availability
        """
        if self._gpu_info_cache is not None:
            return self._gpu_info_cache
        
        gpu_info = {
            'name': 'CPU Only',
            'vram_mb': 0,
            'cuda_available': False,
            'driver_version': None
        }
        
        try:
            # Try to detect NVIDIA GPU first
            nvidia_info = self._detect_nvidia_gpu()
            if nvidia_info:
                gpu_info.update(nvidia_info)
                self._gpu_info_cache = gpu_info
                return gpu_info
            
            # Try to detect AMD GPU
            amd_info = self._detect_amd_gpu()
            if amd_info:
                gpu_info.update(amd_info)
                self._gpu_info_cache = gpu_info
                return gpu_info
            
            # Try Intel GPU detection
            intel_info = self._detect_intel_gpu()
            if intel_info:
                gpu_info.update(intel_info)
            
        except Exception as e:
            logger.warning(f"GPU detection encountered error: {e}")
        
        self._gpu_info_cache = gpu_info
        return gpu_info
    
    def _detect_nvidia_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect NVIDIA GPU using nvidia-ml-py or nvidia-smi."""
        try:
            # Try nvidia-ml-py first (more reliable)
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None
            
            # Get first GPU (primary)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_mb = mem_info.total // (1024 * 1024)
            
            # Get driver version
            driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            
            pynvml.nvmlShutdown()
            
            return {
                'name': name,
                'vram_mb': vram_mb,
                'cuda_available': True,
                'driver_version': driver_version
            }
            
        except ImportError:
            logger.debug("pynvml not available, trying nvidia-smi")
            
        except Exception as e:
            logger.debug(f"pynvml detection failed: {e}")
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        vram_mb = int(parts[1].strip())
                        driver_version = parts[2].strip() if len(parts) > 2 else None
                        
                        return {
                            'name': name,
                            'vram_mb': vram_mb,
                            'cuda_available': True,
                            'driver_version': driver_version
                        }
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.debug(f"nvidia-smi detection failed: {e}")
        
        return None
    
    def _detect_amd_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect AMD GPU using rocm-smi or system information."""
        try:
            # Try rocm-smi
            result = subprocess.run([
                'rocm-smi', '--showproductname', '--showmeminfo', 'vram'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse rocm-smi output (simplified)
                lines = result.stdout.strip().split('\n')
                gpu_name = "AMD GPU"
                vram_mb = 0
                
                for line in lines:
                    if "Product Name" in line:
                        gpu_name = line.split(':')[-1].strip()
                    elif "VRAM Total Memory" in line:
                        # Extract memory size (usually in MB or GB)
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                size = int(part)
                                if i + 1 < len(parts) and parts[i + 1].lower() == 'gb':
                                    vram_mb = size * 1024
                                elif i + 1 < len(parts) and parts[i + 1].lower() == 'mb':
                                    vram_mb = size
                                break
                
                return {
                    'name': gpu_name,
                    'vram_mb': vram_mb,
                    'cuda_available': False,  # AMD uses ROCm
                    'driver_version': None
                }
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("rocm-smi not available")
        
        return None
    
    def _detect_intel_gpu(self) -> Optional[Dict[str, Any]]:
        """Detect Intel GPU (basic detection)."""
        try:
            # Basic Intel GPU detection (limited capabilities)
            if platform.system() == "Windows":
                # Try Windows GPU detection
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 'get', 'name,AdapterRAM'
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines[1:]:  # Skip header
                        if 'Intel' in line and line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                return {
                                    'name': 'Intel Integrated GPU',
                                    'vram_mb': 0,  # Shared memory
                                    'cuda_available': False,
                                    'driver_version': None
                                }
        
        except Exception:
            pass
        
        return None
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information."""
        try:
            cpu_info = {
                'cores': psutil.cpu_count(logical=False) or 1,
                'logical_cores': psutil.cpu_count(logical=True) or 1,
                'frequency': None,
                'name': platform.processor() or 'Unknown CPU'
            }
            
            # Try to get CPU frequency
            try:
                freq_info = psutil.cpu_freq()
                if freq_info:
                    cpu_info['frequency'] = freq_info.max or freq_info.current
            except Exception:
                pass
            
            return cpu_info
            
        except Exception as e:
            logger.warning(f"CPU detection failed: {e}")
            return {
                'cores': 1,
                'logical_cores': 1,
                'frequency': None,
                'name': 'Unknown CPU'
            }
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect system memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'total_mb': memory.total // (1024 * 1024),
                'available_mb': memory.available // (1024 * 1024),
                'swap_mb': swap.total // (1024 * 1024),
                'usage_percent': memory.percent
            }
            
        except Exception as e:
            logger.warning(f"Memory detection failed: {e}")
            return {
                'total_mb': 8192,  # Default 8GB
                'available_mb': 6144,
                'swap_mb': 0,
                'usage_percent': 50.0
            }
    
    def _determine_optimization_level(self, gpu_info: Dict[str, Any], 
                                    memory_info: Dict[str, Any]) -> str:
        """Determine optimization level based on detected hardware."""
        vram_mb = gpu_info.get('vram_mb', 0)
        ram_mb = memory_info.get('total_mb', 0)
        
        # Consider both VRAM and system RAM
        if vram_mb < 6000 or ram_mb < 8192:
            return "aggressive"
        elif vram_mb < 12000 or ram_mb < 16384:
            return "balanced"
        else:
            return "minimal"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Get detailed hardware information for diagnostics."""
        return {
            'gpu': self._detect_gpu(),
            'cpu': self._detect_cpu(),
            'memory': self._detect_memory(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'machine': platform.machine(),
                'python_version': platform.python_version()
            }
        }