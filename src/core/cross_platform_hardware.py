"""
Cross-platform hardware detection without heavy dependencies.

This module provides hardware detection that works across Windows, Linux, and macOS
without requiring PyTorch or other heavy AI libraries to be installed.
"""

import logging
import platform
import subprocess
import os
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrossPlatformHardwareConfig:
    """Cross-platform hardware configuration."""
    # Basic system info
    system: str
    platform_name: str
    architecture: str
    
    # CPU info
    cpu_brand: str
    cpu_cores: int
    cpu_threads: int
    
    # Memory info
    ram_total_mb: int
    ram_available_mb: int
    
    # GPU info
    gpu_model: str
    vram_size: int  # MB
    cuda_available: bool
    is_dedicated_gpu: bool
    
    # Derived properties
    optimization_level: str
    hardware_tier: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'system': self.system,
            'platform': self.platform_name,
            'architecture': self.architecture,
            'cpu_brand': self.cpu_brand,
            'cpu_cores': self.cpu_cores,
            'cpu_threads': self.cpu_threads,
            'ram_total_mb': self.ram_total_mb,
            'ram_available_mb': self.ram_available_mb,
            'gpu_model': self.gpu_model,
            'vram_size': self.vram_size,
            'cuda_available': self.cuda_available,
            'optimization_level': self.optimization_level,
            'hardware_tier': self.hardware_tier
        }


class CrossPlatformHardwareDetector:
    """
    Cross-platform hardware detector that works without heavy dependencies.
    
    Detects:
    - Operating system and architecture
    - CPU information (brand, cores, threads)
    - RAM amount and availability
    - GPU information (model, VRAM, CUDA support)
    - Determines appropriate optimization settings
    """
    
    def __init__(self):
        self.system = platform.system()
        logger.info(f"CrossPlatformHardwareDetector initialized for {self.system}")
    
    def detect_hardware(self) -> CrossPlatformHardwareConfig:
        """Detect hardware configuration across platforms."""
        try:
            # Basic system info
            system_info = self._get_system_info()
            
            # CPU info
            cpu_info = self._get_cpu_info()
            
            # Memory info
            memory_info = self._get_memory_info()
            
            # GPU info
            gpu_info = self._get_gpu_info()
            
            # Create configuration
            config = CrossPlatformHardwareConfig(
                system=system_info['system'],
                platform_name=system_info['platform'],
                architecture=system_info['architecture'],
                cpu_brand=cpu_info['brand'],
                cpu_cores=cpu_info['cores'],
                cpu_threads=cpu_info['threads'],
                ram_total_mb=memory_info['total_mb'],
                ram_available_mb=memory_info['available_mb'],
                gpu_model=gpu_info['model'],
                vram_size=gpu_info['vram_mb'],
                cuda_available=gpu_info['cuda_available'],
                is_dedicated_gpu=gpu_info.get('is_dedicated', False),
                optimization_level=self._determine_optimization_level(gpu_info['vram_mb']),
                hardware_tier=self._determine_hardware_tier(gpu_info['vram_mb'])
            )
            
            logger.info(f"Hardware detected: {config.gpu_model} with {config.vram_size}MB VRAM")
            return config
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            return self._get_fallback_config()
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get basic system information."""
        return {
            'system': platform.system(),
            'platform': platform.platform(),
            'architecture': platform.machine()
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information across platforms."""
        cpu_info = {
            'brand': 'Unknown CPU',
            'cores': 4,  # Default fallback
            'threads': 4
        }
        
        try:
            # Try to get CPU count
            import multiprocessing
            cpu_info['threads'] = multiprocessing.cpu_count()
            
            # Try to get physical cores
            try:
                import psutil
                cpu_info['cores'] = psutil.cpu_count(logical=False) or cpu_info['threads']
            except ImportError:
                cpu_info['cores'] = cpu_info['threads'] // 2  # Estimate
            
        except Exception:
            pass
        
        # Platform-specific CPU brand detection
        if self.system == "Windows":
            cpu_info['brand'] = self._get_windows_cpu_brand()
        elif self.system == "Linux":
            cpu_info['brand'] = self._get_linux_cpu_brand()
        elif self.system == "Darwin":  # macOS
            cpu_info['brand'] = self._get_macos_cpu_brand()
        
        return cpu_info
    
    def _get_windows_cpu_brand(self) -> str:
        """Get CPU brand on Windows."""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip()
        except Exception:
            pass
        return "Unknown Windows CPU"
    
    def _get_linux_cpu_brand(self) -> str:
        """Get CPU brand on Linux."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        except Exception:
            pass
        return "Unknown Linux CPU"
    
    def _get_macos_cpu_brand(self) -> str:
        """Get CPU brand on macOS."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "Unknown macOS CPU"
    
    def _get_memory_info(self) -> Dict[str, int]:
        """Get memory information across platforms."""
        memory_info = {
            'total_mb': 8000,  # Default 8GB
            'available_mb': 6000  # Default 6GB available
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_info['total_mb'] = memory.total // (1024 * 1024)
            memory_info['available_mb'] = memory.available // (1024 * 1024)
        except ImportError:
            # Platform-specific fallbacks
            if self.system == "Windows":
                memory_info = self._get_windows_memory()
            elif self.system == "Linux":
                memory_info = self._get_linux_memory()
            elif self.system == "Darwin":
                memory_info = self._get_macos_memory()
        
        return memory_info
    
    def _get_windows_memory(self) -> Dict[str, int]:
        """Get memory info on Windows."""
        try:
            result = subprocess.run(
                ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    total_bytes = int(lines[1].strip())
                    total_mb = total_bytes // (1024 * 1024)
                    return {
                        'total_mb': total_mb,
                        'available_mb': int(total_mb * 0.75)  # Estimate 75% available
                    }
        except Exception:
            pass
        return {'total_mb': 8000, 'available_mb': 6000}
    
    def _get_linux_memory(self) -> Dict[str, int]:
        """Get memory info on Linux."""
        try:
            with open("/proc/meminfo", "r") as f:
                meminfo = {}
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        # Extract numeric value (remove kB, etc.)
                        value_kb = int(''.join(filter(str.isdigit, value)))
                        meminfo[key.strip()] = value_kb
                
                total_mb = meminfo.get("MemTotal", 8000000) // 1024
                available_mb = meminfo.get("MemAvailable", total_mb * 3 // 4) // 1024
                
                return {
                    'total_mb': total_mb,
                    'available_mb': available_mb
                }
        except Exception:
            pass
        return {'total_mb': 8000, 'available_mb': 6000}
    
    def _get_macos_memory(self) -> Dict[str, int]:
        """Get memory info on macOS."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                total_bytes = int(result.stdout.strip())
                total_mb = total_bytes // (1024 * 1024)
                return {
                    'total_mb': total_mb,
                    'available_mb': int(total_mb * 0.75)  # Estimate 75% available
                }
        except Exception:
            pass
        return {'total_mb': 8000, 'available_mb': 6000}
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information across platforms."""
        gpu_info = {
            'model': 'Unknown GPU',
            'vram_mb': 0,
            'cuda_available': False,
            'is_dedicated': False
        }
        
        # Try PyTorch detection first (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                props = torch.cuda.get_device_properties(0)
                gpu_info['model'] = props.name
                gpu_info['vram_mb'] = props.total_memory // (1024 * 1024)
                # CUDA GPUs are typically dedicated
                gpu_info['is_dedicated'] = True
                return gpu_info
        except ImportError:
            pass
        
        # Platform-specific GPU detection
        if self.system == "Windows":
            gpu_info.update(self._get_windows_gpu())
        elif self.system == "Linux":
            gpu_info.update(self._get_linux_gpu())
        elif self.system == "Darwin":
            gpu_info.update(self._get_macos_gpu())
        
        return gpu_info
    
    def _get_windows_gpu(self) -> Dict[str, Any]:
        """Get GPU info on Windows."""
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True, text=True, timeout=15
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if line and "NVIDIA" in line.upper():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                vram_bytes = int(parts[0]) if parts[0].isdigit() else 0
                                name = ' '.join(parts[1:])
                                is_dedicated = "NVIDIA" in name.upper() or "AMD" in name.upper() or "Radeon RX" in name.upper()
                                # Check for integrated GPU keywords
                                integrated_keywords = ["Intel", "UHD", "Iris", "Vega", "APU", "Integrated"]
                                if any(keyword in name for keyword in integrated_keywords):
                                    is_dedicated = False
                                
                                return {
                                    'model': name,
                                    'vram_mb': vram_bytes // (1024 * 1024) if vram_bytes > 0 else 4000,
                                    'cuda_available': "NVIDIA" in name.upper(),
                                    'is_dedicated': is_dedicated
                                }
                            except:
                                continue
        except Exception:
            pass
        
        return {'model': 'Windows GPU', 'vram_mb': 4000, 'cuda_available': False, 'is_dedicated': False}
    
    def _get_linux_gpu(self) -> Dict[str, Any]:
        """Get GPU info on Linux."""
        # Try nvidia-smi first
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        name = parts[0]
                        try:
                            vram_mb = int(parts[1])
                        except:
                            vram_mb = 4000
                        
                        return {
                            'model': name,
                            'vram_mb': vram_mb,
                            'cuda_available': True,
                            'is_dedicated': True  # NVIDIA GPUs are dedicated
                        }
        except Exception:
            pass
        
        # Try lspci as fallback
        try:
            result = subprocess.run(
                ["lspci", "-nn"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA compatible controller' in line and 'NVIDIA' in line.upper():
                        gpu_name = line.split(':', 2)[-1].strip()
                        return {
                            'model': gpu_name,
                            'vram_mb': 4000,  # Can't detect VRAM from lspci
                            'cuda_available': True,
                            'is_dedicated': True  # NVIDIA GPUs are dedicated
                        }
        except Exception:
            pass
        
        return {'model': 'Linux GPU', 'vram_mb': 4000, 'cuda_available': False, 'is_dedicated': False}
    
    def _get_macos_gpu(self) -> Dict[str, Any]:
        """Get GPU info on macOS."""
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=15
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for display_data in data.get("SPDisplaysDataType", []):
                    name = display_data.get("sppci_model", "Unknown GPU")
                    vram_str = display_data.get("sppci_vram", "0 MB")
                    
                    # Parse VRAM
                    vram_mb = 0
                    if "MB" in vram_str:
                        try:
                            vram_mb = int(vram_str.replace("MB", "").strip())
                        except:
                            pass
                    elif "GB" in vram_str:
                        try:
                            vram_gb = float(vram_str.replace("GB", "").strip())
                            vram_mb = int(vram_gb * 1024)
                        except:
                            pass
                    
                    # Apple Silicon M1/M2/M3 are integrated but powerful
                    is_dedicated = "M1" in name or "M2" in name or "M3" in name
                    
                    return {
                        'model': name,
                        'vram_mb': vram_mb or 4000,  # Default if can't detect
                        'cuda_available': False,  # macOS doesn't support CUDA
                        'is_dedicated': is_dedicated
                    }
        except Exception:
            pass
        
        return {'model': 'macOS GPU', 'vram_mb': 4000, 'cuda_available': False, 'is_dedicated': False}
    
    def _determine_optimization_level(self, vram_mb: int) -> str:
        """Determine optimization level based on VRAM."""
        if vram_mb >= 16000:
            return "minimal"
        elif vram_mb >= 8000:
            return "balanced"
        elif vram_mb >= 4000:
            return "aggressive"
        else:
            return "maximum"
    
    def _determine_hardware_tier(self, vram_mb: int) -> str:
        """Determine hardware tier based on VRAM."""
        if vram_mb >= 20000:
            return "high_end"
        elif vram_mb >= 8000:
            return "mid_tier"
        elif vram_mb >= 4000:
            return "budget"
        else:
            return "cpu_only"
    
    def _get_fallback_config(self) -> CrossPlatformHardwareConfig:
        """Get fallback configuration when detection fails."""
        return CrossPlatformHardwareConfig(
            system=self.system,
            platform_name=platform.platform(),
            architecture=platform.machine(),
            cpu_brand="Unknown CPU",
            cpu_cores=4,
            cpu_threads=4,
            ram_total_mb=8000,
            ram_available_mb=6000,
            gpu_model="Unknown GPU",
            vram_size=4000,  # Conservative default
            cuda_available=False,
            is_dedicated_gpu=False,
            optimization_level="balanced",
            hardware_tier="budget"
        )


# Convenience function
def detect_cross_platform_hardware() -> CrossPlatformHardwareConfig:
    """Detect hardware configuration across platforms."""
    detector = CrossPlatformHardwareDetector()
    return detector.detect_hardware()


if __name__ == "__main__":
    # Test the detector
    detector = CrossPlatformHardwareDetector()
    config = detector.detect_hardware()
    
    print("Cross-Platform Hardware Detection Results:")
    print("=" * 50)
    print(f"System: {config.system} ({config.architecture})")
    print(f"CPU: {config.cpu_brand}")
    print(f"Cores: {config.cpu_cores} physical, {config.cpu_threads} threads")
    print(f"RAM: {config.ram_available_mb}/{config.ram_total_mb} MB")
    print(f"GPU: {config.gpu_model}")
    print(f"VRAM: {config.vram_size} MB")
    print(f"CUDA: {'Available' if config.cuda_available else 'Not Available'}")
    print(f"Hardware Tier: {config.hardware_tier}")
    print(f"Optimization Level: {config.optimization_level}")
    
    # Save to JSON
    import json
    with open("hardware_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    print(f"\nConfiguration saved to hardware_config.json")