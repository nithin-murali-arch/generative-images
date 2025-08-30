# 🌍 Cross-Platform Setup Guide

## 🎯 Platform Support

The AI Content Generator now supports **Windows**, **Linux (Ubuntu/Debian/CentOS/Arch)**, and **macOS** with automatic hardware detection and platform-specific optimizations.

## 🚀 Quick Start by Platform

### 🪟 **Windows**

#### Option 1: One-Click Launch (Recommended)
```cmd
# Double-click or run in Command Prompt
run_windows.bat
```

#### Option 2: Manual Setup
```cmd
# Check system specs
python system_specs.py

# Install dependencies
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Launch application
python app.py
```

### 🐧 **Linux (Ubuntu/Debian/CentOS/Arch)**

#### Option 1: One-Click Launch (Recommended)
```bash
# Make executable and run
chmod +x run_linux.sh
./run_linux.sh
```

#### Option 2: Manual Setup
```bash
# Check system specs
python3 system_specs.py

# Install dependencies
python3 -m pip install --upgrade pip

# For NVIDIA GPUs (CUDA support)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU-only or AMD GPUs
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install -r requirements.txt

# Launch application
python3 app.py
```

### 🍎 **macOS**

#### Option 1: Using the Linux Script (Works on macOS)
```bash
# Make executable and run
chmod +x run_linux.sh
./run_linux.sh
```

#### Option 2: Manual Setup
```bash
# Check system specs
python3 system_specs.py

# Install dependencies
python3 -m pip install --upgrade pip
pip3 install torch torchvision  # MPS support for Apple Silicon
pip3 install -r requirements.txt

# Launch application
python3 app.py
```

## 🔧 **Cross-Platform Launcher**

Use the universal launcher that automatically detects your platform:

```bash
# Check system specifications
python3 launch.py --specs

# Check dependencies only
python3 launch.py --check

# Launch with automatic platform detection
python3 launch.py

# Force launch even with missing dependencies
python3 launch.py --force
```

## 📊 **System Requirements by Platform**

### 🪟 **Windows Requirements**
- **OS**: Windows 10/11 (64-bit)
- **Python**: 3.8+ (from python.org or Microsoft Store)
- **GPU**: NVIDIA GTX 1050+ (recommended), CPU-only supported
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 20GB+ free space

### 🐧 **Linux Requirements**
- **OS**: Ubuntu 18.04+, Debian 10+, CentOS 7+, Arch Linux
- **Python**: 3.8+ (usually pre-installed)
- **GPU**: NVIDIA GTX 1050+ with CUDA drivers, AMD GPUs supported via CPU mode
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 20GB+ free space

### 🍎 **macOS Requirements**
- **OS**: macOS 10.15+ (Catalina or newer)
- **Python**: 3.8+ (install via Homebrew or python.org)
- **GPU**: Apple Silicon (M1/M2) with MPS support, Intel Macs use CPU
- **RAM**: 8GB+ (16GB+ recommended for Apple Silicon)
- **Storage**: 20GB+ free space

## 🛠️ **Platform-Specific Installation**

### 🪟 **Windows Detailed Setup**

1. **Install Python**:
   ```cmd
   # Download from python.org or use Microsoft Store
   # Ensure "Add to PATH" is checked during installation
   ```

2. **Install CUDA (for NVIDIA GPUs)**:
   ```cmd
   # Download CUDA 12.1 from NVIDIA website
   # Or let PyTorch handle CUDA installation
   ```

3. **Install Dependencies**:
   ```cmd
   # Create virtual environment (recommended)
   python -m venv ai_generator
   ai_generator\Scripts\activate

   # Install PyTorch with CUDA
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install other packages
   pip install -r requirements.txt
   ```

### 🐧 **Linux Detailed Setup**

1. **Update System**:
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt upgrade
   sudo apt install python3 python3-pip python3-venv

   # CentOS/RHEL
   sudo yum update
   sudo yum install python3 python3-pip

   # Arch Linux
   sudo pacman -Syu
   sudo pacman -S python python-pip
   ```

2. **Install NVIDIA Drivers (for NVIDIA GPUs)**:
   ```bash
   # Ubuntu (automatic)
   sudo ubuntu-drivers autoinstall

   # Or manual NVIDIA driver installation
   # Download from nvidia.com
   ```

3. **Install Dependencies**:
   ```bash
   # Create virtual environment
   python3 -m venv ai_generator
   source ai_generator/bin/activate

   # Install PyTorch (CUDA or CPU)
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

   # Install other packages
   pip3 install -r requirements.txt
   ```

### 🍎 **macOS Detailed Setup**

1. **Install Python**:
   ```bash
   # Using Homebrew (recommended)
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   brew install python

   # Or download from python.org
   ```

2. **Install Dependencies**:
   ```bash
   # Create virtual environment
   python3 -m venv ai_generator
   source ai_generator/bin/activate

   # Install PyTorch with MPS support (Apple Silicon)
   pip3 install torch torchvision

   # Install other packages
   pip3 install -r requirements.txt
   ```

## 🔍 **Hardware Detection Features**

The system automatically detects:

### 🖥️ **System Information**
- Operating system and version
- CPU brand, cores, and threads
- Total and available RAM
- Disk space and partitions

### 🎮 **GPU Detection**
- **Windows**: Uses WMI (Windows Management Instrumentation)
- **Linux**: Uses `nvidia-smi`, `lspci`, and `/proc` filesystem
- **macOS**: Uses `system_profiler` and system calls
- **Cross-platform**: PyTorch CUDA detection when available

### 🎯 **Automatic Optimization**
Based on detected hardware:
- **Model recommendations** (SD 1.5, SDXL, FLUX.1)
- **Memory optimization** settings
- **Performance expectations**
- **Auto-download** of appropriate models

## 🚨 **Troubleshooting**

### ❌ **Common Issues**

#### "Python not found"
```bash
# Windows
# Install Python from python.org, ensure "Add to PATH" is checked

# Linux
sudo apt install python3  # Ubuntu/Debian
sudo yum install python3   # CentOS/RHEL

# macOS
brew install python
```

#### "CUDA not available"
```bash
# Check NVIDIA driver
nvidia-smi  # Should show GPU info

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

#### "Permission denied" (Linux/macOS)
```bash
# Make scripts executable
chmod +x run_linux.sh
chmod +x launch.py

# Or run with python directly
python3 launch.py
```

#### "Module not found"
```bash
# Ensure virtual environment is activated
source ai_generator/bin/activate  # Linux/macOS
ai_generator\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### 🔧 **Platform-Specific Fixes**

#### 🪟 **Windows Issues**
- **Long path names**: Enable long path support in Windows settings
- **Antivirus blocking**: Add Python and project folder to antivirus exclusions
- **PowerShell execution policy**: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

#### 🐧 **Linux Issues**
- **Permission errors**: Use `sudo` for system-wide installations or stick to virtual environments
- **Missing libraries**: Install development packages: `sudo apt install build-essential python3-dev`
- **CUDA version mismatch**: Check `nvidia-smi` output and install matching PyTorch version

#### 🍎 **macOS Issues**
- **Xcode command line tools**: Install with `xcode-select --install`
- **Apple Silicon compatibility**: Ensure you're using ARM64 versions of packages
- **Homebrew path**: Add Homebrew to PATH: `echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc`

## 📈 **Performance by Platform**

### 🏆 **Expected Performance**

| Platform | Hardware | Model | Resolution | Speed |
|----------|----------|-------|------------|-------|
| Windows | RTX 4090 | FLUX.1 | 1024x1024 | 3-6s |
| Windows | RTX 3070 | SDXL | 1024x1024 | 8-12s |
| Linux | RTX 4080 | FLUX.1 | 1024x1024 | 4-7s |
| Linux | GTX 1660 | SD 1.5 | 512x512 | 5-8s |
| macOS M2 | Apple GPU | SD 1.5 | 512x512 | 15-25s |
| macOS Intel | CPU | SD 1.5 | 512x512 | 60-120s |

### 🎯 **Optimization Tips**

#### 🪟 **Windows Optimization**
- Close unnecessary applications to free VRAM
- Use Game Mode for better GPU priority
- Ensure adequate cooling for sustained performance

#### 🐧 **Linux Optimization**
- Use `nvidia-smi -l 1` to monitor GPU usage
- Adjust GPU power limits with `nvidia-smi -pl <watts>`
- Use `htop` to monitor CPU and RAM usage

#### 🍎 **macOS Optimization**
- Enable "High Performance" mode in Energy Saver
- Close other GPU-intensive applications
- Use Activity Monitor to check memory pressure

## 🎉 **Ready to Generate!**

Once setup is complete:

1. **Launch**: Use platform-specific launcher or `python app.py`
2. **Check hardware**: System automatically detects and optimizes
3. **Start creating**: Use Easy mode for quick results, Advanced for full control
4. **Enjoy**: Generate amazing images and videos with AI!

## 📞 **Support**

If you encounter issues:

1. **Check system specs**: `python system_specs.py`
2. **Verify dependencies**: `python launch.py --check`
3. **Try different launcher**: Use cross-platform `launch.py`
4. **Check logs**: Look for error messages in console output
5. **Update drivers**: Ensure GPU drivers are up to date

The system is designed to work across all platforms with automatic fallbacks and clear error messages to help you get up and running quickly! 🚀