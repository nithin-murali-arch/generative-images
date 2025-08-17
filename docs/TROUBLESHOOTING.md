# Troubleshooting and Maintenance Guide

## Common Issues and Solutions

### 1. System Startup Issues

#### Import Errors
**Problem**: `ImportError: attempted relative import beyond top-level package`

**Symptoms**:
- System fails to start with import errors
- Python path issues when running from different directories

**Solutions**:
1. **Use the correct launcher**:
   ```bash
   py -3.13 launch_real.py
   ```

2. **Check Python version**:
   ```bash
   py -3.13 --version
   ```

3. **Verify dependencies**:
   ```bash
   py -3.13 -m pip list | findstr -i torch
   py -3.13 -m pip list | findstr -i gradio
   py -3.13 -m pip list | findstr -i diffusers
   ```

4. **Fix import paths**:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent / "src"))
   ```

#### Syntax Errors
**Problem**: `SyntaxError: expected an indented block after 'try' statement`

**Symptoms**:
- Python syntax errors in core files
- System fails to parse Python code

**Solutions**:
1. **Check file syntax**:
   ```bash
   py -3.13 -m py_compile src/core/gpu_optimizer.py
   ```

2. **Fix indentation issues**:
   - Ensure proper indentation after `try:` statements
   - Check for missing colons
   - Verify bracket matching

3. **Use a linter**:
   ```bash
   py -3.13 -m flake8 src/
   ```

### 2. GPU and Memory Issues

#### CUDA Out of Memory
**Problem**: `RuntimeError: CUDA out of memory`

**Symptoms**:
- Generation fails with memory errors
- System becomes unresponsive
- High VRAM usage

**Solutions**:
1. **Reduce generation parameters**:
   - Lower resolution (512x512 instead of 768x768)
   - Reduce steps (20 instead of 50)
   - Lower guidance scale (7.5 instead of 10)

2. **Enable memory optimizations**:
   ```python
   # In hardware profiles
   "enable_attention_slicing": True,
   "enable_vae_slicing": True,
   "enable_vae_tiling": True
   ```

3. **Clear GPU memory**:
   - Use the "Clear VRAM Cache" button in the UI
   - Restart the system
   - Close other GPU-intensive applications

4. **Set memory limits**:
   ```bash
   set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
   ```

#### Black/Corrupted Images
**Problem**: Generated images are black or corrupted (270 bytes)

**Symptoms**:
- Images appear completely black
- File sizes are very small
- Runtime warnings about invalid values

**Solutions**:
1. **Use float32 precision**:
   ```python
   torch_dtype=torch.float32  # Instead of float16
   ```

2. **Check model loading**:
   - Ensure models are downloaded completely
   - Verify model integrity
   - Check disk space

3. **Update diffusers library**:
   ```bash
   py -3.13 -m pip install --upgrade diffusers
   ```

### 3. Model Loading Issues

#### Model Download Failures
**Problem**: `Failed to load model: [model_name]`

**Symptoms**:
- Models fail to download
- Generation requests fail
- Network timeout errors

**Solutions**:
1. **Check internet connection**:
   ```bash
   ping huggingface.co
   ```

2. **Verify model availability**:
   - Check Hugging Face model pages
   - Verify model names are correct
   - Check for model updates

3. **Manual download**:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("runwayml/stable-diffusion-v1-5")
   ```

4. **Use local models**:
   - Download models manually
   - Place in `models/` directory
   - Update configuration to use local paths

#### Model Compatibility Issues
**Problem**: Models fail to load due to version incompatibility

**Symptoms**:
- Version mismatch errors
- Pipeline initialization failures
- CUDA compatibility issues

**Solutions**:
1. **Check PyTorch version**:
   ```bash
   py -3.13 -c "import torch; print(torch.__version__)"
   ```

2. **Verify CUDA compatibility**:
   ```bash
   py -3.13 -c "import torch; print(torch.version.cuda)"
   ```

3. **Update dependencies**:
   ```bash
   py -3.13 -m pip install --upgrade torch torchvision torchaudio
   py -3.13 -m pip install --upgrade diffusers transformers
   ```

### 4. UI and Interface Issues

#### Gradio Interface Not Loading
**Problem**: Web interface fails to start or is inaccessible

**Symptoms**:
- Browser shows connection refused
- Port already in use errors
- Interface starts but doesn't respond

**Solutions**:
1. **Check port availability**:
   ```bash
   netstat -an | findstr :7861
   ```

2. **Use different port**:
   ```python
   interface.launch(server_port=7862)
   ```

3. **Check firewall settings**:
   - Allow Python through Windows Firewall
   - Check antivirus software
   - Verify network permissions

4. **Restart the system**:
   ```bash
   # Kill existing processes
   taskkill /f /im python.exe
   # Restart
   py -3.13 launch_real.py
   ```

#### UI Elements Not Working
**Problem**: Buttons, dropdowns, or other UI elements are unresponsive

**Symptoms**:
- Click events don't trigger
- Model dropdown is empty
- Generation buttons don't work

**Solutions**:
1. **Check JavaScript console**:
   - Open browser developer tools
   - Look for JavaScript errors
   - Check network requests

2. **Verify model loading**:
   - Check system status in UI
   - Verify models are available
   - Check hardware detection

3. **Refresh the interface**:
   - Reload the page
   - Restart the Gradio server
   - Clear browser cache

### 5. Performance Issues

#### Slow Generation
**Problem**: Image or video generation takes too long

**Symptoms**:
- Generation times >2 minutes for images
- Video generation >10 minutes
- System becomes unresponsive during generation

**Solutions**:
1. **Optimize generation parameters**:
   - Reduce resolution
   - Lower number of steps
   - Use faster schedulers

2. **Enable hardware optimizations**:
   ```python
   # In hardware profiles
   "enable_xformers": True,
   "enable_attention_slicing": False,
   "torch_dtype": "float16"
   ```

3. **Check system resources**:
   - Monitor GPU usage
   - Check CPU utilization
   - Verify memory availability

4. **Update drivers**:
   - Update NVIDIA drivers
   - Update CUDA toolkit
   - Restart system after updates

#### High Memory Usage
**Problem**: System uses excessive RAM or VRAM

**Symptoms**:
- High memory usage even when idle
- Memory not freed after generation
- System becomes slow over time

**Solutions**:
1. **Enable automatic cleanup**:
   ```python
   # In RealImageGenerator
   self.memory_cleanup_interval = 30  # Clean every 30 seconds
   ```

2. **Force memory cleanup**:
   ```python
   torch.cuda.empty_cache()
   gc.collect()
   ```

3. **Unload unused models**:
   - Switch models to trigger cleanup
   - Use the memory cleanup button
   - Restart system periodically

### 6. API Issues

#### API Endpoints Not Responding
**Problem**: REST API calls fail or timeout

**Symptoms**:
- HTTP 500 errors
- Request timeouts
- Connection refused errors

**Solutions**:
1. **Check API server status**:
   ```bash
   curl http://localhost:8000/status
   ```

2. **Verify server configuration**:
   ```python
   # In api/server.py
   app = FastAPI(title="AI System API")
   ```

3. **Check port conflicts**:
   ```bash
   netstat -an | findstr :8000
   ```

4. **Restart API server**:
   ```bash
   py -3.13 main.py --mode api --port 8000
   ```

#### Authentication Issues
**Problem**: API requests are rejected due to authentication

**Symptoms**:
- 401 Unauthorized errors
- Authentication token errors
- Access denied messages

**Solutions**:
1. **Check authentication configuration**:
   - Verify API key settings
   - Check token validity
   - Confirm user permissions

2. **Use correct headers**:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/status
   ```

3. **Disable authentication for testing**:
   ```python
   # Comment out auth middleware temporarily
   # app.add_middleware(AuthMiddleware)
   ```

## Maintenance Procedures

### 1. Regular Maintenance

#### Daily Tasks
- Check system logs for errors
- Monitor GPU memory usage
- Verify model availability
- Test basic generation functionality

#### Weekly Tasks
- Review experiment logs
- Clean up old output files
- Update model versions
- Check system performance metrics

#### Monthly Tasks
- Full system restart
- Update dependencies
- Backup configuration files
- Review and optimize hardware profiles

### 2. Log Management

#### Log Locations
```
logs/
├── system.log          # System-level logs
├── generation.log      # Generation operation logs
├── api.log            # API request logs
├── error.log          # Error-specific logs
└── experiments.log    # Experiment tracking logs
```

#### Log Rotation
```python
import logging
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'logs/system.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

#### Log Analysis
```bash
# Find errors in logs
findstr /i "error" logs\*.log

# Check recent activity
findstr /i "2025-08-17" logs\system.log

# Monitor real-time logs
Get-Content logs\system.log -Wait
```

### 3. Performance Monitoring

#### Key Metrics to Track
- **Generation Time**: Average time per image/video
- **Memory Usage**: Peak and average VRAM usage
- **Success Rate**: Percentage of successful generations
- **Error Frequency**: Common error types and frequency
- **Resource Utilization**: CPU, GPU, and memory usage

#### Monitoring Commands
```bash
# GPU status
nvidia-smi

# Memory usage
py -3.13 -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"

# System resources
Get-Process python | Select-Object ProcessName, CPU, WorkingSet
```

#### Performance Optimization
1. **Hardware-specific tuning**:
   - Adjust batch sizes
   - Optimize memory allocation
   - Use appropriate precision

2. **Model optimization**:
   - Enable attention slicing
   - Use VAE slicing
   - Implement model caching

3. **System optimization**:
   - Regular memory cleanup
   - Optimize disk I/O
   - Monitor background processes

### 4. Backup and Recovery

#### Configuration Backup
```bash
# Backup configuration files
xcopy config\*.* backup\config\ /E /Y

# Backup hardware profiles
xcopy src\hardware\profiles.py backup\hardware\ /Y

# Backup system configuration
copy config\system_config.json backup\
```

#### Model Backup
```bash
# Backup downloaded models
xcopy models\*.* backup\models\ /E /Y

# Backup model cache
xcopy cache\*.* backup\cache\ /E /Y
```

#### Recovery Procedures
1. **Configuration recovery**:
   ```bash
   copy backup\config\*.* config\
   copy backup\hardware\profiles.py src\hardware\
   ```

2. **Model recovery**:
   ```bash
   xcopy backup\models\*.* models\ /E /Y
   xcopy backup\cache\*.* cache\ /E /Y
   ```

3. **System recovery**:
   ```bash
   # Restore from backup
   copy backup\system_config.json config\
   
   # Restart system
   py -3.13 launch_real.py
   ```

### 5. Troubleshooting Checklist

#### System Won't Start
- [ ] Check Python version (3.13+)
- [ ] Verify all dependencies installed
- [ ] Check for syntax errors in source files
- [ ] Verify file paths and permissions
- [ ] Check system logs for errors

#### Generation Fails
- [ ] Verify GPU drivers and CUDA
- [ ] Check available VRAM
- [ ] Verify model availability
- [ ] Check generation parameters
- [ ] Review error logs

#### Performance Issues
- [ ] Monitor GPU memory usage
- [ ] Check CPU utilization
- [ ] Verify hardware optimization profiles
- [ ] Review generation parameters
- [ ] Check for background processes

#### UI Problems
- [ ] Verify Gradio installation
- [ ] Check port availability
- [ ] Verify browser compatibility
- [ ] Check JavaScript console
- [ ] Restart Gradio server

## Getting Help

### 1. Self-Diagnosis
- Check the troubleshooting checklist above
- Review system logs for error messages
- Test with minimal configuration
- Verify hardware compatibility

### 2. Community Support
- Check GitHub issues for similar problems
- Review system documentation
- Search for error messages online
- Check PyTorch and Diffusers documentation

### 3. Professional Support
- Contact system administrators
- Consult with AI/ML specialists
- Check hardware vendor support
- Consider professional services

### 4. Reporting Issues
When reporting issues, include:
- System specifications (OS, Python version, GPU)
- Error messages and logs
- Steps to reproduce the problem
- Expected vs. actual behavior
- System configuration files 