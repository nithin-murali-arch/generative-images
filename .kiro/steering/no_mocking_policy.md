---
inclusion: always
---

# NO MOCKING POLICY - CRITICAL SAFETY REQUIREMENT

## 🚨 ABSOLUTE REQUIREMENTS

### ❌ NEVER MOCK THESE SYSTEMS
- **Hardware detection** (CPU, GPU, RAM, VRAM)
- **Thermal monitoring** (CPU/GPU temperatures)
- **Memory management** (RAM/VRAM usage)
- **System resource monitoring**
- **Platform detection** (OS, architecture)

### 🔥 THERMAL SAFETY IS MANDATORY
- **ALL temperature readings MUST be real**
- **NO fallback temperatures or estimates**
- **System MUST halt if thermal sensors fail**
- **45°C cooling target is NON-NEGOTIABLE**
- **Emergency shutdown at 90°C is REQUIRED**

### 💻 HARDWARE DETECTION REQUIREMENTS
- **Real VRAM detection only** - no estimates or defaults
- **Actual GPU model identification required**
- **CPU core count must be accurate**
- **RAM detection must be precise**

## 🛡️ SAFETY IMPLEMENTATIONS

### Thermal Monitoring
```python
# REQUIRED: Real thermal monitoring
from src.core.thermal_monitor import get_thermal_monitor, ensure_thermal_safety

# BEFORE any AI workload
if not ensure_thermal_safety():
    raise RuntimeError("System too hot - cannot proceed")
```

### Hardware Detection
```python
# REQUIRED: Real hardware detection
from src.core.cross_platform_hardware import detect_cross_platform_hardware

config = detect_cross_platform_hardware()
if config.vram_size == 0:
    raise RuntimeError("Cannot detect VRAM - unsafe to proceed")
```

### Error Handling
```python
# REQUIRED: Fail fast on detection errors
try:
    hardware_config = detect_hardware()
except Exception as e:
    logger.error(f"Hardware detection failed: {e}")
    raise RuntimeError("Cannot proceed without hardware detection")
    # NO fallbacks, NO defaults, NO mocks
```

## 🚫 FORBIDDEN PATTERNS

### ❌ Mock Hardware
```python
# NEVER DO THIS
class MockHardwareDetector:
    def detect_hardware(self):
        return {"vram": 8000}  # DANGEROUS!
```

### ❌ Fallback Temperatures
```python
# NEVER DO THIS
def get_temperature():
    try:
        return read_real_temperature()
    except:
        return 45.0  # DANGEROUS FALLBACK!
```

### ❌ Default VRAM Values
```python
# NEVER DO THIS
vram = detect_vram() or 8000  # DANGEROUS DEFAULT!
```

## ✅ REQUIRED PATTERNS

### ✅ Strict Hardware Detection
```python
def detect_hardware_strict():
    config = detect_cross_platform_hardware()
    
    # Validate all critical values
    if config.vram_size <= 0:
        raise RuntimeError("Invalid VRAM detection")
    if config.gpu_model == "Unknown GPU":
        raise RuntimeError("GPU model detection failed")
    if config.ram_total_mb <= 0:
        raise RuntimeError("RAM detection failed")
    
    return config
```

### ✅ Mandatory Thermal Checks
```python
def before_ai_workload():
    monitor = get_thermal_monitor()
    
    # Start monitoring if not active
    if not monitor.is_monitoring:
        monitor.start_monitoring()
        time.sleep(3.0)  # Allow sensor readings
    
    # Verify we have thermal data
    readings = monitor.get_current_readings()
    if not readings:
        raise RuntimeError("No thermal sensors detected - cannot proceed safely")
    
    # Check temperatures
    if not monitor.is_safe_for_ai_workload():
        logger.warning("System too hot - waiting for cooling")
        if not monitor.wait_for_cooling():
            raise RuntimeError("System failed to cool - aborting for safety")
```

### ✅ Proper Error Logging
```python
def safe_hardware_operation():
    try:
        return perform_hardware_detection()
    except Exception as e:
        logger.error(f"CRITICAL: Hardware operation failed: {e}")
        logger.error(f"System: {platform.system()}")
        logger.error(f"Architecture: {platform.machine()}")
        raise RuntimeError(f"Hardware operation failed: {e}")
```

## 🔧 IMPLEMENTATION CHECKLIST

### Before Any AI Generation:
- [ ] Real hardware detection completed
- [ ] Thermal monitoring active
- [ ] All temperatures < 45°C
- [ ] VRAM accurately detected
- [ ] GPU model identified
- [ ] No mock objects in use

### During AI Generation:
- [ ] Continuous thermal monitoring
- [ ] Automatic throttling if temps rise
- [ ] Emergency shutdown at 90°C
- [ ] Memory usage tracking
- [ ] Resource cleanup on errors

### Error Conditions:
- [ ] Log all hardware detection failures
- [ ] Log all thermal sensor failures
- [ ] Fail fast - no silent fallbacks
- [ ] Clear error messages to user
- [ ] Safe system shutdown if needed

## 🎯 COMPLIANCE VERIFICATION

### Code Review Requirements:
1. **No mock classes** in production code paths
2. **No default hardware values** without detection
3. **No temperature estimates** or fallbacks
4. **Proper error handling** with logging
5. **Thermal safety checks** before AI workloads

### Testing Requirements:
1. **Test on real hardware only**
2. **Verify thermal sensor access**
3. **Test emergency shutdown scenarios**
4. **Validate hardware detection accuracy**
5. **Confirm error handling behavior**

## 🚨 VIOLATION CONSEQUENCES

Violating this policy can result in:
- **Hardware damage** from overheating
- **System instability** from resource exhaustion
- **Data loss** from unexpected shutdowns
- **Fire hazard** from thermal runaway

**THERE ARE NO EXCEPTIONS TO THIS POLICY**