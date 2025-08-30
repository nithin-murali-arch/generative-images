#!/usr/bin/env python3
"""
Cross-platform launcher for AI Content Generator.

This script automatically detects the operating system and provides
platform-specific setup and launch instructions.
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def detect_platform():
    """Detect the current platform."""
    system = platform.system().lower()
    
    if system == "windows":
        return "windows"
    elif system == "linux":
        return "linux"
    elif system == "darwin":
        return "macos"
    else:
        return "unknown"

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"{version.major}.{version.minor}.{version.micro}"
    return True, f"{version.major}.{version.minor}.{version.micro}"

def check_dependencies():
    """Check if core dependencies are available."""
    dependencies = {
        "torch": "PyTorch",
        "gradio": "Gradio", 
        "diffusers": "Diffusers",
        "transformers": "Transformers",
        "PIL": "Pillow"
    }
    
    available = {}
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            available[module] = True
        except ImportError:
            available[module] = False
            missing.append(name)
    
    return available, missing

def detect_gpu():
    """Detect if NVIDIA GPU is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def get_installation_command(platform_name):
    """Get the appropriate installation command for the platform."""
    has_gpu = detect_gpu()
    
    if platform_name == "windows":
        if has_gpu:
            pytorch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            gpu_note = "# NVIDIA GPU detected - installing CUDA support"
        else:
            pytorch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            gpu_note = "# No NVIDIA GPU detected - installing CPU-only version"
        
        return [
            "# Windows Installation:",
            gpu_note,
            "python -m pip install --upgrade pip",
            pytorch_cmd,
            "pip install -r requirements.txt"
        ]
    elif platform_name == "linux":
        if has_gpu:
            pytorch_cmd = "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            gpu_note = "# NVIDIA GPU detected - installing CUDA support"
        else:
            pytorch_cmd = "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            gpu_note = "# No NVIDIA GPU detected - installing CPU-only version"
        
        return [
            "# Linux Installation:",
            gpu_note,
            "python3 -m pip install --upgrade pip",
            pytorch_cmd,
            "pip3 install -r requirements.txt"
        ]
    elif platform_name == "macos":
        arch = platform.machine().lower()
        if "arm" in arch or "m1" in arch or "m2" in arch:
            pytorch_cmd = "pip3 install torch torchvision"
            gpu_note = "# Apple Silicon detected - installing with MPS support"
        else:
            pytorch_cmd = "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            gpu_note = "# Intel Mac detected - installing CPU-only version"
        
        return [
            "# macOS Installation:",
            gpu_note,
            "python3 -m pip install --upgrade pip",
            pytorch_cmd,
            "pip3 install -r requirements.txt"
        ]
    else:
        return ["# Unknown platform - use pip to install requirements.txt"]

def print_header():
    """Print application header."""
    print("🎨" + "=" * 58 + "🎨")
    print("   AI CONTENT GENERATOR - CROSS-PLATFORM LAUNCHER")
    print("🎨" + "=" * 58 + "🎨")
    print()

def print_platform_info(platform_name):
    """Print platform-specific information."""
    platform_icons = {
        "windows": "🪟",
        "linux": "🐧", 
        "macos": "🍎",
        "unknown": "❓"
    }
    
    icon = platform_icons.get(platform_name, "❓")
    print(f"{icon} Platform: {platform_name.title()}")
    print(f"🖥️  System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    
    # Check GPU availability
    has_gpu = detect_gpu()
    if has_gpu:
        print("🎮 GPU: NVIDIA GPU detected (CUDA support available)")
    else:
        print("💻 GPU: No NVIDIA GPU detected (CPU-only mode)")
    
    print()

def print_python_info():
    """Print Python environment information."""
    compatible, version = check_python_version()
    
    if compatible:
        print(f"✅ Python: {version} (Compatible)")
    else:
        print(f"❌ Python: {version} (Requires 3.8+)")
        return False
    
    # Check virtual environment
    in_venv = sys.prefix != getattr(sys, 'base_prefix', sys.prefix)
    if in_venv:
        print(f"✅ Virtual Environment: Active ({sys.prefix})")
    else:
        print("⚠️  Virtual Environment: Not active (recommended)")
    
    print()
    return compatible

def print_dependency_status():
    """Print dependency availability status."""
    available, missing = check_dependencies()
    
    print("📦 Dependencies Status:")
    
    for module, status in available.items():
        name = {
            "torch": "PyTorch",
            "gradio": "Gradio",
            "diffusers": "Diffusers", 
            "transformers": "Transformers",
            "PIL": "Pillow"
        }.get(module, module)
        
        if status:
            print(f"   ✅ {name}")
        else:
            print(f"   ❌ {name}")
    
    print()
    return len(missing) == 0, missing

def show_installation_instructions(platform_name, missing_deps):
    """Show installation instructions for missing dependencies."""
    print("📥 INSTALLATION INSTRUCTIONS")
    print("-" * 40)
    
    commands = get_installation_command(platform_name)
    for command in commands:
        print(command)
    
    print()
    print("💡 Alternative: Use platform-specific launchers:")
    
    if platform_name == "windows":
        print("   run_windows.bat")
    elif platform_name == "linux":
        print("   ./run_linux.sh")
    elif platform_name == "macos":
        print("   ./run_linux.sh  (works on macOS too)")
    
    print()

def launch_application(platform_name):
    """Launch the application using the appropriate method."""
    print("🚀 LAUNCHING APPLICATION")
    print("-" * 40)
    
    # Try to use platform-specific launcher first
    if platform_name == "windows" and Path("run_windows.bat").exists():
        print("Using Windows launcher...")
        try:
            subprocess.run(["run_windows.bat"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Windows launcher failed, trying direct launch...")
    
    elif platform_name in ["linux", "macos"] and Path("run_linux.sh").exists():
        print("Using Linux/Unix launcher...")
        try:
            subprocess.run(["./run_linux.sh"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Linux launcher failed, trying direct launch...")
    
    # Fallback to direct Python launch
    print("Using direct Python launch...")
    try:
        python_cmd = "python" if platform_name == "windows" else "python3"
        subprocess.run([python_cmd, "app.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Launch failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ app.py not found in current directory")
        return False

def show_system_specs():
    """Show system specifications."""
    print("🖥️  SYSTEM SPECIFICATIONS")
    print("-" * 40)
    
    try:
        python_cmd = "python" if platform.system() == "Windows" else "python3"
        result = subprocess.run([python_cmd, "system_specs.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("❌ Failed to get system specs")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("⏱️  System specs check timed out")
    except FileNotFoundError:
        print("❌ system_specs.py not found")
    except Exception as e:
        print(f"❌ Error getting system specs: {e}")

def main():
    """Main launcher function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-platform AI Content Generator launcher")
    parser.add_argument("--specs", "-s", action="store_true", help="Show system specifications only")
    parser.add_argument("--check", "-c", action="store_true", help="Check dependencies only")
    parser.add_argument("--force", "-f", action="store_true", help="Force launch even with missing dependencies")
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Detect platform
    platform_name = detect_platform()
    print_platform_info(platform_name)
    
    # Show system specs if requested
    if args.specs:
        show_system_specs()
        return
    
    # Check Python compatibility
    if not print_python_info():
        print("❌ Python version incompatible. Please install Python 3.8 or higher.")
        return
    
    # Check dependencies
    deps_ok, missing_deps = print_dependency_status()
    
    if args.check:
        # Just check dependencies and exit
        if deps_ok:
            print("✅ All dependencies are available!")
        else:
            print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
            show_installation_instructions(platform_name, missing_deps)
        return
    
    # Handle missing dependencies
    if not deps_ok and not args.force:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print()
        show_installation_instructions(platform_name, missing_deps)
        
        # Ask user if they want to continue anyway
        try:
            response = input("Continue anyway? Some features may not work [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("👋 Setup cancelled. Install dependencies and try again.")
                return
        except KeyboardInterrupt:
            print("\n👋 Setup cancelled.")
            return
    
    # Launch the application
    print()
    success = launch_application(platform_name)
    
    if success:
        print("\n✅ Application launched successfully!")
    else:
        print("\n❌ Failed to launch application.")
        print("\n💡 Try manual launch:")
        python_cmd = "python" if platform_name == "windows" else "python3"
        print(f"   {python_cmd} app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Launcher interrupted.")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        print("💡 Try running 'python app.py' directly")