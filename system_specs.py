#!/usr/bin/env python3
"""
Cross-platform system specifications detector.

This script detects and displays comprehensive system information including:
- Operating system and version
- CPU information and cores
- RAM amount and usage
- GPU information and VRAM
- Python environment details
- AI framework availability
"""

import sys
import os
import platform
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def get_os_info() -> Dict[str, Any]:
    """Get operating system information."""
    try:
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "platform": platform.platform(),
            "node": platform.node()
        }
    except Exception as e:
        return {"error": str(e)}

def get_cpu_info() -> Dict[str, Any]:
    """Get CPU information."""
    try:
        import psutil
        
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "Unknown",
            "current_frequency": f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "Unknown",
            "cpu_usage": f"{psutil.cpu_percent(interval=1):.1f}%"
        }
        
        # Try to get CPU brand on different platforms
        if platform.system() == "Windows":
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], 
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) > 1:
                        cpu_info["brand"] = lines[1].strip()
            except:
                pass
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info["brand"] = line.split(":")[1].strip()
                            break
            except:
                pass
        elif platform.system() == "Darwin":  # macOS
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    cpu_info["brand"] = result.stdout.strip()
            except:
                pass
        
        return cpu_info
        
    except ImportError:
        return {"error": "psutil not available - install with: pip install psutil"}
    except Exception as e:
        return {"error": str(e)}

def get_memory_info() -> Dict[str, Any]:
    """Get memory information."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        return {
            "total_ram_gb": f"{memory.total / (1024**3):.1f} GB",
            "available_ram_gb": f"{memory.available / (1024**3):.1f} GB",
            "used_ram_gb": f"{memory.used / (1024**3):.1f} GB",
            "ram_usage_percent": f"{memory.percent:.1f}%",
            "total_swap_gb": f"{swap.total / (1024**3):.1f} GB",
            "used_swap_gb": f"{swap.used / (1024**3):.1f} GB",
            "swap_usage_percent": f"{swap.percent:.1f}%"
        }
        
    except ImportError:
        return {"error": "psutil not available - install with: pip install psutil"}
    except Exception as e:
        return {"error": str(e)}

def get_gpu_info() -> Dict[str, Any]:
    """Get GPU information across platforms."""
    gpu_info = {
        "gpus": [],
        "cuda_available": False,
        "cuda_version": None,
        "primary_gpu": None,
        "total_vram_mb": 0
    }
    
    # Try PyTorch CUDA detection
    try:
        import torch
        gpu_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_data = {
                    "id": i,
                    "name": props.name,
                    "vram_mb": props.total_memory // (1024 * 1024),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count
                }
                gpu_info["gpus"].append(gpu_data)
                
                if i == 0:  # Primary GPU
                    gpu_info["primary_gpu"] = gpu_data
                    gpu_info["total_vram_mb"] = gpu_data["vram_mb"]
    except ImportError:
        pass
    except Exception as e:
        gpu_info["torch_error"] = str(e)
    
    # Try nvidia-ml-py for more detailed NVIDIA info
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
            
            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
            except:
                gpu_util = None
                mem_util = None
            
            gpu_data = {
                "id": i,
                "name": name,
                "vram_total_mb": mem_info.total // (1024 * 1024),
                "vram_used_mb": mem_info.used // (1024 * 1024),
                "vram_free_mb": mem_info.free // (1024 * 1024),
                "temperature_c": temp,
                "gpu_utilization_percent": gpu_util,
                "memory_utilization_percent": mem_util
            }
            
            # Update or add to GPU list
            found = False
            for j, existing_gpu in enumerate(gpu_info["gpus"]):
                if existing_gpu["name"] == name:
                    gpu_info["gpus"][j].update(gpu_data)
                    found = True
                    break
            
            if not found:
                gpu_info["gpus"].append(gpu_data)
            
            if i == 0:
                gpu_info["primary_gpu"] = gpu_data
                gpu_info["total_vram_mb"] = gpu_data["vram_total_mb"]
                
    except ImportError:
        pass
    except Exception as e:
        gpu_info["pynvml_error"] = str(e)
    
    # Platform-specific GPU detection fallbacks
    if not gpu_info["gpus"]:
        if platform.system() == "Windows":
            gpu_info.update(_get_windows_gpu_info())
        elif platform.system() == "Linux":
            gpu_info.update(_get_linux_gpu_info())
        elif platform.system() == "Darwin":
            gpu_info.update(_get_macos_gpu_info())
    
    return gpu_info

def _get_windows_gpu_info() -> Dict[str, Any]:
    """Get GPU info on Windows using wmic."""
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            gpus = []
            
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            vram_bytes = int(parts[0]) if parts[0].isdigit() else 0
                            name = ' '.join(parts[1:])
                            gpus.append({
                                "name": name,
                                "vram_mb": vram_bytes // (1024 * 1024) if vram_bytes > 0 else 0,
                                "source": "wmic"
                            })
                        except:
                            continue
            
            return {"gpus": gpus}
    except Exception as e:
        return {"windows_error": str(e)}
    
    return {}

def _get_linux_gpu_info() -> Dict[str, Any]:
    """Get GPU info on Linux using various methods."""
    gpus = []
    
    # Try lspci
    try:
        result = subprocess.run(
            ["lspci", "-nn"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'VGA compatible controller' in line or 'Display controller' in line:
                    # Extract GPU name
                    if ':' in line:
                        gpu_name = line.split(':', 2)[-1].strip()
                        gpus.append({
                            "name": gpu_name,
                            "vram_mb": 0,  # Can't get VRAM from lspci easily
                            "source": "lspci"
                        })
    except:
        pass
    
    # Try nvidia-smi for NVIDIA GPUs
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 2:
                        name = parts[0]
                        try:
                            vram_mb = int(parts[1])
                        except:
                            vram_mb = 0
                        
                        gpus.append({
                            "name": name,
                            "vram_mb": vram_mb,
                            "source": "nvidia-smi"
                        })
    except:
        pass
    
    return {"gpus": gpus}

def _get_macos_gpu_info() -> Dict[str, Any]:
    """Get GPU info on macOS using system_profiler."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            gpus = []
            
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
                
                gpus.append({
                    "name": name,
                    "vram_mb": vram_mb,
                    "source": "system_profiler"
                })
            
            return {"gpus": gpus}
    except Exception as e:
        return {"macos_error": str(e)}
    
    return {}

def get_python_info() -> Dict[str, Any]:
    """Get Python environment information."""
    try:
        return {
            "version": sys.version,
            "version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "executable": sys.executable,
            "platform": sys.platform,
            "path": sys.path[:3],  # First 3 paths
            "prefix": sys.prefix,
            "base_prefix": getattr(sys, 'base_prefix', sys.prefix),
            "in_virtualenv": sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
        }
    except Exception as e:
        return {"error": str(e)}

def get_ai_frameworks_info() -> Dict[str, Any]:
    """Check availability of AI frameworks."""
    frameworks = {}
    
    # PyTorch
    try:
        import torch
        frameworks["pytorch"] = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except ImportError:
        frameworks["pytorch"] = {"available": False, "error": "Not installed"}
    except Exception as e:
        frameworks["pytorch"] = {"available": False, "error": str(e)}
    
    # Transformers
    try:
        import transformers
        frameworks["transformers"] = {
            "available": True,
            "version": transformers.__version__
        }
    except ImportError:
        frameworks["transformers"] = {"available": False, "error": "Not installed"}
    except Exception as e:
        frameworks["transformers"] = {"available": False, "error": str(e)}
    
    # Diffusers
    try:
        import diffusers
        frameworks["diffusers"] = {
            "available": True,
            "version": diffusers.__version__
        }
    except ImportError:
        frameworks["diffusers"] = {"available": False, "error": "Not installed"}
    except Exception as e:
        frameworks["diffusers"] = {"available": False, "error": str(e)}
    
    # Gradio
    try:
        import gradio
        frameworks["gradio"] = {
            "available": True,
            "version": gradio.__version__
        }
    except ImportError:
        frameworks["gradio"] = {"available": False, "error": "Not installed"}
    except Exception as e:
        frameworks["gradio"] = {"available": False, "error": str(e)}
    
    # XFormers (optional optimization)
    try:
        import xformers
        frameworks["xformers"] = {
            "available": True,
            "version": xformers.__version__
        }
    except ImportError:
        frameworks["xformers"] = {"available": False, "error": "Not installed (optional)"}
    except Exception as e:
        frameworks["xformers"] = {"available": False, "error": str(e)}
    
    return frameworks

def get_disk_info() -> Dict[str, Any]:
    """Get disk space information."""
    try:
        import psutil
        
        disk_info = {}
        
        # Get disk usage for current directory
        current_disk = psutil.disk_usage('.')
        disk_info["current_drive"] = {
            "total_gb": f"{current_disk.total / (1024**3):.1f} GB",
            "used_gb": f"{current_disk.used / (1024**3):.1f} GB",
            "free_gb": f"{current_disk.free / (1024**3):.1f} GB",
            "usage_percent": f"{(current_disk.used / current_disk.total) * 100:.1f}%"
        }
        
        # Get all disk partitions
        partitions = psutil.disk_partitions()
        disk_info["partitions"] = []
        
        for partition in partitions:
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_info["partitions"].append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": f"{usage.total / (1024**3):.1f} GB",
                    "free_gb": f"{usage.free / (1024**3):.1f} GB"
                })
            except PermissionError:
                # Skip inaccessible partitions
                continue
        
        return disk_info
        
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}

def print_system_specs():
    """Print comprehensive system specifications."""
    print("🖥️  SYSTEM SPECIFICATIONS")
    print("=" * 60)
    
    # Operating System
    print("\n📋 OPERATING SYSTEM")
    print("-" * 30)
    os_info = get_os_info()
    if "error" not in os_info:
        print(f"System: {os_info['system']} {os_info['release']}")
        print(f"Version: {os_info['version']}")
        print(f"Architecture: {os_info['machine']} ({os_info['architecture'][0]})")
        print(f"Platform: {os_info['platform']}")
        print(f"Node: {os_info['node']}")
    else:
        print(f"❌ Error: {os_info['error']}")
    
    # CPU Information
    print("\n🔧 CPU INFORMATION")
    print("-" * 30)
    cpu_info = get_cpu_info()
    if "error" not in cpu_info:
        if "brand" in cpu_info:
            print(f"Processor: {cpu_info['brand']}")
        print(f"Physical Cores: {cpu_info['physical_cores']}")
        print(f"Logical Cores: {cpu_info['logical_cores']}")
        print(f"Max Frequency: {cpu_info['max_frequency']}")
        print(f"Current Usage: {cpu_info['cpu_usage']}")
    else:
        print(f"❌ Error: {cpu_info['error']}")
    
    # Memory Information
    print("\n💾 MEMORY INFORMATION")
    print("-" * 30)
    memory_info = get_memory_info()
    if "error" not in memory_info:
        print(f"Total RAM: {memory_info['total_ram_gb']}")
        print(f"Available RAM: {memory_info['available_ram_gb']}")
        print(f"Used RAM: {memory_info['used_ram_gb']} ({memory_info['ram_usage_percent']})")
        if float(memory_info['total_swap_gb'].split()[0]) > 0:
            print(f"Swap: {memory_info['used_swap_gb']} / {memory_info['total_swap_gb']}")
    else:
        print(f"❌ Error: {memory_info['error']}")
    
    # GPU Information
    print("\n🎮 GPU INFORMATION")
    print("-" * 30)
    gpu_info = get_gpu_info()
    
    if gpu_info["cuda_available"]:
        print(f"✅ CUDA Available: Version {gpu_info['cuda_version']}")
    else:
        print("❌ CUDA Not Available")
    
    if gpu_info["gpus"]:
        for i, gpu in enumerate(gpu_info["gpus"]):
            print(f"\nGPU {i}: {gpu['name']}")
            if "vram_total_mb" in gpu:
                print(f"  VRAM: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB ({gpu['vram_free_mb']} MB free)")
            elif gpu.get("vram_mb", 0) > 0:
                print(f"  VRAM: {gpu['vram_mb']} MB")
            
            if gpu.get("temperature_c"):
                print(f"  Temperature: {gpu['temperature_c']}°C")
            if gpu.get("gpu_utilization_percent") is not None:
                print(f"  GPU Usage: {gpu['gpu_utilization_percent']}%")
            if gpu.get("compute_capability"):
                print(f"  Compute Capability: {gpu['compute_capability']}")
    else:
        print("❌ No GPUs detected or GPU detection failed")
        if "torch_error" in gpu_info:
            print(f"   PyTorch Error: {gpu_info['torch_error']}")
        if "pynvml_error" in gpu_info:
            print(f"   NVIDIA-ML Error: {gpu_info['pynvml_error']}")
    
    # Disk Information
    print("\n💿 DISK INFORMATION")
    print("-" * 30)
    disk_info = get_disk_info()
    if "error" not in disk_info:
        print(f"Current Drive: {disk_info['current_drive']['free_gb']} free / {disk_info['current_drive']['total_gb']} total")
        print(f"Usage: {disk_info['current_drive']['usage_percent']}")
        
        if len(disk_info["partitions"]) > 1:
            print("\nAll Partitions:")
            for partition in disk_info["partitions"][:5]:  # Show first 5
                print(f"  {partition['device']}: {partition['free_gb']} free ({partition['fstype']})")
    else:
        print(f"❌ Error: {disk_info['error']}")
    
    # Python Environment
    print("\n🐍 PYTHON ENVIRONMENT")
    print("-" * 30)
    python_info = get_python_info()
    if "error" not in python_info:
        print(f"Version: {python_info['version_info']}")
        print(f"Executable: {python_info['executable']}")
        print(f"Virtual Environment: {'Yes' if python_info['in_virtualenv'] else 'No'}")
        print(f"Platform: {python_info['platform']}")
    else:
        print(f"❌ Error: {python_info['error']}")
    
    # AI Frameworks
    print("\n🤖 AI FRAMEWORKS")
    print("-" * 30)
    frameworks = get_ai_frameworks_info()
    
    for name, info in frameworks.items():
        if info["available"]:
            version = info.get("version", "Unknown")
            print(f"✅ {name.capitalize()}: {version}")
            
            if name == "pytorch" and info.get("cuda_available"):
                print(f"   CUDA: {info['cuda_version']} ({info['device_count']} devices)")
        else:
            error = info.get("error", "Unknown error")
            if "optional" in error.lower():
                print(f"⚪ {name.capitalize()}: {error}")
            else:
                print(f"❌ {name.capitalize()}: {error}")
    
    # AI Generation Recommendations
    print("\n🎨 AI GENERATION RECOMMENDATIONS")
    print("-" * 30)
    
    total_vram = gpu_info.get("total_vram_mb", 0)
    
    if total_vram >= 20000:
        tier = "🔴 High-End"
        models = "All models (SD 1.5, SDXL, FLUX.1)"
        performance = "Excellent (2-6 seconds per image)"
    elif total_vram >= 8000:
        tier = "🟡 Mid-Tier"
        models = "SD 1.5, SDXL Turbo"
        performance = "Good (3-8 seconds per image)"
    elif total_vram >= 4000:
        tier = "💚 Budget"
        models = "SD 1.5"
        performance = "Fair (5-15 seconds per image)"
    else:
        tier = "⚪ CPU-Only"
        models = "Tiny SD (CPU mode)"
        performance = "Slow (30+ seconds per image)"
    
    print(f"Hardware Tier: {tier}")
    print(f"Recommended Models: {models}")
    print(f"Expected Performance: {performance}")
    
    if not gpu_info["cuda_available"]:
        print("\n⚠️  CUDA not available - AI generation will use CPU (much slower)")
        print("   Install CUDA-enabled PyTorch for GPU acceleration")
    
    if not frameworks["gradio"]["available"]:
        print("\n⚠️  Gradio not installed - UI will not work")
        print("   Install with: pip install gradio")
    
    print("\n" + "=" * 60)
    print("🚀 Run 'python app.py' to start the AI Content Generator!")

def save_specs_to_file(filename: str = "system_specs.json"):
    """Save system specifications to a JSON file."""
    specs = {
        "timestamp": str(platform.uname()),
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "disk": get_disk_info(),
        "python": get_python_info(),
        "ai_frameworks": get_ai_frameworks_info()
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(specs, f, indent=2, default=str)
        print(f"\n💾 System specs saved to {filename}")
    except Exception as e:
        print(f"\n❌ Failed to save specs: {e}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Display system specifications for AI content generation")
    parser.add_argument("--save", "-s", help="Save specs to JSON file", metavar="filename")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.json:
        # Output as JSON
        specs = {
            "os": get_os_info(),
            "cpu": get_cpu_info(),
            "memory": get_memory_info(),
            "gpu": get_gpu_info(),
            "disk": get_disk_info(),
            "python": get_python_info(),
            "ai_frameworks": get_ai_frameworks_info()
        }
        print(json.dumps(specs, indent=2, default=str))
    else:
        # Pretty print
        print_system_specs()
    
    if args.save:
        save_specs_to_file(args.save)

if __name__ == "__main__":
    main()