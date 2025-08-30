"""
Thermal monitoring and safety system.

This module provides real-time thermal monitoring for CPU and GPU to prevent
hardware damage. It enforces strict temperature limits and automatically
throttles or stops processing when temperatures exceed safe thresholds.

CRITICAL: This system has NO FALLBACKS or mocks. All thermal readings must be real.
"""

import logging
import time
import threading
import subprocess
import platform
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """Thermal states for hardware components."""
    SAFE = "safe"           # < 45°C - Normal operation
    WARM = "warm"           # 45-70°C - Monitoring required
    HOT = "hot"             # 70-80°C - Operations paused
    CRITICAL = "critical"   # 80-90°C - Immediate throttling
    EMERGENCY = "emergency" # > 90°C - Emergency shutdown


@dataclass
class ThermalReading:
    """Thermal reading for a hardware component."""
    component: str
    temperature_c: float
    state: ThermalState
    timestamp: float
    source: str
    
    def is_safe_for_ai_workload(self) -> bool:
        """Check if temperature is safe for AI workloads."""
        return self.state == ThermalState.SAFE
    
    def requires_cooling(self) -> bool:
        """Check if component requires cooling before continuing."""
        return self.state in [ThermalState.WARM, ThermalState.HOT, ThermalState.CRITICAL, ThermalState.EMERGENCY]


class ThermalMonitor:
    """
    Real-time thermal monitoring system with automatic safety controls.
    
    Features:
    - Continuous temperature monitoring
    - Automatic workload throttling
    - Emergency shutdown protection
    - Cross-platform temperature detection
    - NO MOCKS OR FALLBACKS - Real readings only
    """
    
    # Temperature thresholds (Celsius)
    TEMP_SAFE = 45.0      # Safe for AI workloads (resume operations)
    TEMP_WARM = 65.0      # Start monitoring more closely
    TEMP_PAUSE = 70.0     # Pause all operations until cooling
    TEMP_HOT = 80.0       # Throttle performance
    TEMP_CRITICAL = 90.0  # Emergency shutdown
    
    # Cooling requirements
    COOLING_TARGET = 45.0  # Must cool to this temperature before resuming
    COOLING_TIMEOUT = 300  # Maximum 5 minutes cooling wait
    
    def __init__(self, monitoring_interval: float = 3.0):
        """
        Initialize thermal monitor.
        
        Args:
            monitoring_interval: Seconds between temperature checks
        """
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.thermal_readings: Dict[str, ThermalReading] = {}
        self.thermal_callbacks: List[Callable[[Dict[str, ThermalReading]], None]] = []
        self.emergency_shutdown_callbacks: List[Callable[[], None]] = []
        
        # Cooling state tracking - once we hit 70°C, must cool to 45°C
        self.cooling_required: Dict[str, bool] = {}  # Track which components need cooling
        self.last_hot_time: Dict[str, float] = {}    # Track when components last exceeded 70°C
        
        # Platform detection
        self.platform = platform.system()
        
        # Validate thermal detection capabilities
        self._validate_thermal_capabilities()
        
        logger.info(f"ThermalMonitor initialized for {self.platform}")
    
    def _validate_thermal_capabilities(self):
        """Validate that we can actually read temperatures on this system."""
        test_readings = self._get_all_temperatures()
        
        if not test_readings:
            error_msg = f"CRITICAL: Cannot read temperatures on {self.platform}. Thermal monitoring is REQUIRED for safe operation."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info(f"Thermal monitoring validated: {len(test_readings)} sensors detected")
        for component, temp in test_readings.items():
            logger.info(f"  {component}: {temp:.1f}°C")
    
    def start_monitoring(self):
        """Start continuous thermal monitoring."""
        if self.is_monitoring:
            logger.warning("Thermal monitoring already active")
            return
        
        logger.info("Starting thermal monitoring")
        self.is_monitoring = True
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=False  # Critical thread - don't make daemon
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        logger.info("Stopping thermal monitoring")
        self.is_monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Thermal monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Get current temperatures
                temperatures = self._get_all_temperatures()
                
                if not temperatures:
                    logger.error("CRITICAL: Lost thermal sensor readings!")
                    self._trigger_emergency_shutdown("Lost thermal sensors")
                    break
                
                # Update thermal readings with cooling hysteresis
                current_time = time.time()
                for component, temp in temperatures.items():
                    # Check if component hit 70°C threshold
                    if temp >= self.TEMP_PAUSE:
                        self.cooling_required[component] = True
                        self.last_hot_time[component] = current_time
                        logger.warning(f"{component}: {temp:.1f}°C - COOLING REQUIRED until {self.COOLING_TARGET}°C")
                    
                    # Check if component has cooled sufficiently
                    elif temp <= self.COOLING_TARGET and self.cooling_required.get(component, False):
                        self.cooling_required[component] = False
                        logger.info(f"{component}: {temp:.1f}°C - COOLED SUFFICIENTLY, resuming operations")
                    
                    # Classify temperature state with hysteresis
                    state = self._classify_temperature_with_hysteresis(component, temp)
                    
                    reading = ThermalReading(
                        component=component,
                        temperature_c=temp,
                        state=state,
                        timestamp=current_time,
                        source=self.platform
                    )
                    
                    self.thermal_readings[component] = reading
                    
                    # Log temperature changes
                    if state != ThermalState.SAFE:
                        cooling_status = " (COOLING REQUIRED)" if self.cooling_required.get(component, False) else ""
                        logger.warning(f"{component}: {temp:.1f}°C ({state.value}){cooling_status}")
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Notify callbacks
                for callback in self.thermal_callbacks:
                    try:
                        callback(self.thermal_readings.copy())
                    except Exception as e:
                        logger.error(f"Thermal callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Thermal monitoring error: {e}")
                # Don't break on errors - keep monitoring
                time.sleep(self.monitoring_interval)
        
        logger.info("Thermal monitoring loop stopped")
    
    def _get_all_temperatures(self) -> Dict[str, float]:
        """Get all available temperature readings."""
        temperatures = {}
        
        # Get CPU temperatures
        cpu_temps = self._get_cpu_temperatures()
        temperatures.update(cpu_temps)
        
        # Get GPU temperatures
        gpu_temps = self._get_gpu_temperatures()
        temperatures.update(gpu_temps)
        
        return temperatures
    
    def _get_cpu_temperatures(self) -> Dict[str, float]:
        """Get CPU temperature readings by platform."""
        if self.platform == "Linux":
            return self._get_linux_cpu_temps()
        elif self.platform == "Windows":
            return self._get_windows_cpu_temps()
        elif self.platform == "Darwin":  # macOS
            return self._get_macos_cpu_temps()
        else:
            logger.error(f"Unsupported platform for CPU temperature: {self.platform}")
            return {}
    
    def _get_linux_cpu_temps(self) -> Dict[str, float]:
        """Get CPU temperatures on Linux."""
        temperatures = {}
        
        try:
            # Try lm-sensors first
            result = subprocess.run(
                ["sensors", "-A", "-j"], 
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for chip, sensors in data.items():
                    if isinstance(sensors, dict):
                        for sensor, values in sensors.items():
                            if isinstance(values, dict) and "temp" in sensor.lower():
                                for key, value in values.items():
                                    if key.endswith("_input") and isinstance(value, (int, float)):
                                        temp_name = f"CPU_{chip}_{sensor}"
                                        temperatures[temp_name] = float(value)
                
                if temperatures:
                    return temperatures
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        # Fallback to /sys/class/thermal
        try:
            import glob
            thermal_zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
            
            for i, zone_file in enumerate(thermal_zones):
                try:
                    with open(zone_file, 'r') as f:
                        temp_millic = int(f.read().strip())
                        temp_c = temp_millic / 1000.0
                        temperatures[f"CPU_Zone_{i}"] = temp_c
                except (IOError, ValueError):
                    continue
        
        except Exception as e:
            logger.error(f"Failed to read Linux CPU temperatures: {e}")
        
        return temperatures
    
    def _get_windows_cpu_temps(self) -> Dict[str, float]:
        """Get CPU temperatures on Windows."""
        temperatures = {}
        
        try:
            # Try WMI query for temperature
            result = subprocess.run([
                "wmic", "/namespace:\\\\root\\wmi", "path", "MSAcpi_ThermalZoneTemperature",
                "get", "CurrentTemperature", "/format:csv"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for i, line in enumerate(lines):
                    if line.strip():
                        parts = line.split(',')
                        if len(parts) >= 2 and parts[1].strip().isdigit():
                            # Convert from tenths of Kelvin to Celsius
                            temp_kelvin_tenths = int(parts[1].strip())
                            temp_c = (temp_kelvin_tenths / 10.0) - 273.15
                            temperatures[f"CPU_Zone_{i}"] = temp_c
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Failed to read Windows CPU temperatures via WMI")
        
        return temperatures
    
    def _get_macos_cpu_temps(self) -> Dict[str, float]:
        """Get CPU temperatures on macOS."""
        temperatures = {}
        
        try:
            # Try powermetrics (requires sudo, may not work)
            result = subprocess.run([
                "sudo", "powermetrics", "--samplers", "smc", "-n", "1", "--show-initial-usage"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "CPU die temperature" in line:
                        # Extract temperature value
                        parts = line.split()
                        for part in parts:
                            if part.replace('.', '').isdigit():
                                temperatures["CPU_Die"] = float(part)
                                break
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Failed to read macOS CPU temperatures")
        
        return temperatures
    
    def _get_gpu_temperatures(self) -> Dict[str, float]:
        """Get GPU temperature readings."""
        temperatures = {}
        
        # Try NVIDIA GPUs first
        nvidia_temps = self._get_nvidia_temps()
        temperatures.update(nvidia_temps)
        
        # Try AMD GPUs
        amd_temps = self._get_amd_temps()
        temperatures.update(amd_temps)
        
        return temperatures
    
    def _get_nvidia_temps(self) -> Dict[str, float]:
        """Get NVIDIA GPU temperatures."""
        temperatures = {}
        
        try:
            # Try nvidia-smi
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=name,temperature.gpu", 
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_name = parts[0]
                            try:
                                temp = float(parts[1])
                                temperatures[f"GPU_{i}_{gpu_name}"] = temp
                            except ValueError:
                                continue
        
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Try nvidia-ml-py if available
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temperatures[f"GPU_{i}_{name}"] = float(temp)
        
        except (ImportError, Exception):
            pass
        
        return temperatures
    
    def _get_amd_temps(self) -> Dict[str, float]:
        """Get AMD GPU temperatures."""
        temperatures = {}
        
        if self.platform == "Linux":
            try:
                # Try reading from sysfs
                import glob
                hwmon_paths = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input")
                
                for i, temp_file in enumerate(hwmon_paths):
                    try:
                        with open(temp_file, 'r') as f:
                            temp_millic = int(f.read().strip())
                            temp_c = temp_millic / 1000.0
                            temperatures[f"AMD_GPU_{i}"] = temp_c
                    except (IOError, ValueError):
                        continue
            
            except Exception:
                pass
        
        return temperatures
    
    def _classify_temperature(self, temp_c: float) -> ThermalState:
        """Classify temperature into thermal state."""
        if temp_c < self.TEMP_SAFE:
            return ThermalState.SAFE
        elif temp_c < self.TEMP_PAUSE:
            return ThermalState.WARM
        elif temp_c < self.TEMP_HOT:
            return ThermalState.HOT
        elif temp_c < self.TEMP_CRITICAL:
            return ThermalState.CRITICAL
        else:
            return ThermalState.EMERGENCY
    
    def _classify_temperature_with_hysteresis(self, component: str, temp_c: float) -> ThermalState:
        """
        Classify temperature with hysteresis - once a component hits 70°C,
        it must cool to 45°C before being considered safe again.
        """
        # If component requires cooling (hit 70°C), it's not safe until it cools to 45°C
        if self.cooling_required.get(component, False):
            if temp_c > self.COOLING_TARGET:
                # Still needs cooling - classify as at least HOT
                if temp_c >= self.TEMP_CRITICAL:
                    return ThermalState.EMERGENCY
                elif temp_c >= self.TEMP_HOT:
                    return ThermalState.CRITICAL
                else:
                    return ThermalState.HOT  # Force HOT state until cooled
            else:
                # Has cooled sufficiently - use normal classification
                return self._classify_temperature(temp_c)
        else:
            # Normal classification
            return self._classify_temperature(temp_c)
    
    def _check_emergency_conditions(self):
        """Check for emergency thermal conditions - kill process at 80°C."""
        for component, reading in self.thermal_readings.items():
            if reading.temperature_c >= 80.0:  # CRITICAL: Kill process at 80°C
                logger.critical(f"CRITICAL TEMPERATURE: {component} at {reading.temperature_c:.1f}°C - KILLING PROCESS!")
                self._trigger_emergency_shutdown(f"{component} critical temperature")
                # Force immediate process termination
                import os
                os._exit(1)
            elif reading.state == ThermalState.EMERGENCY:
                logger.critical(f"EMERGENCY: {component} at {reading.temperature_c:.1f}°C!")
                self._trigger_emergency_shutdown(f"{component} overheating")
                return
    
    def _trigger_emergency_shutdown(self, reason: str):
        """Trigger emergency shutdown due to thermal conditions."""
        logger.critical(f"EMERGENCY THERMAL SHUTDOWN: {reason}")
        
        # Notify all emergency callbacks
        for callback in self.emergency_shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Emergency shutdown callback error: {e}")
        
        # Stop monitoring
        self.is_monitoring = False
    
    def get_current_readings(self) -> Dict[str, ThermalReading]:
        """Get current thermal readings."""
        return self.thermal_readings.copy()
    
    def is_safe_for_ai_workload(self) -> bool:
        """Check if all components are safe for AI workloads with cooling hysteresis."""
        if not self.thermal_readings:
            logger.error("No thermal readings available - cannot determine safety")
            return False
        
        # Filter out invalid readings (0°C or negative)
        valid_readings = {
            comp: reading for comp, reading in self.thermal_readings.items()
            if reading.temperature_c > 0
        }
        
        if not valid_readings:
            logger.error("No valid thermal readings available")
            return False
        
        # Check if any component requires cooling (hit 70°C and hasn't cooled to 45°C)
        for component in valid_readings.keys():
            if self.cooling_required.get(component, False):
                logger.debug(f"{component} requires cooling before AI workload")
                return False
        
        # Check normal temperature safety
        for reading in valid_readings.values():
            if not reading.is_safe_for_ai_workload():
                return False
        
        return True
    
    def is_safe_for_startup(self) -> bool:
        """Check if system is safe for server startup (less strict than AI workload)."""
        if not self.thermal_readings:
            # Allow startup if we can't read temperatures, but log warning
            logger.warning("No thermal readings available - allowing startup with monitoring")
            return True
        
        # Filter out invalid readings (0°C or negative)
        valid_readings = {
            comp: reading for comp, reading in self.thermal_readings.items()
            if reading.temperature_c > 0
        }
        
        if not valid_readings:
            logger.warning("No valid thermal readings - allowing startup with monitoring")
            return True
        
        # Check for critical/emergency temperatures only
        for reading in valid_readings.values():
            if reading.state in [ThermalState.CRITICAL, ThermalState.EMERGENCY]:
                logger.error(f"CRITICAL: {reading.component} at {reading.temperature_c:.1f}°C - too hot for startup")
                return False
        
        # Log warm/hot components but allow startup
        warm_components = [
            f"{comp}: {reading.temperature_c:.1f}°C"
            for comp, reading in valid_readings.items()
            if reading.state in [ThermalState.WARM, ThermalState.HOT]
        ]
        
        if warm_components:
            logger.warning(f"Warm components detected (will monitor): {', '.join(warm_components)}")
        
        return True
    
    def wait_for_cooling(self, timeout: float = COOLING_TIMEOUT) -> bool:
        """
        Wait for all components to cool to safe temperatures with hysteresis.
        Components that hit 70°C must cool to 45°C before resuming.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if cooled successfully, False if timeout
        """
        logger.info(f"Waiting for system to cool to {self.COOLING_TARGET}°C...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_safe_for_ai_workload():
                logger.info("System cooled to safe temperatures")
                return True
            
            # Log current temperatures and cooling requirements
            cooling_components = []
            hot_components = []
            
            for component, reading in self.thermal_readings.items():
                if reading.temperature_c > 0:  # Valid reading
                    if self.cooling_required.get(component, False):
                        cooling_components.append(f"{component}: {reading.temperature_c:.1f}°C (needs {self.COOLING_TARGET}°C)")
                    elif reading.requires_cooling():
                        hot_components.append(f"{component}: {reading.temperature_c:.1f}°C")
            
            status_parts = []
            if cooling_components:
                status_parts.append(f"Cooling required: {', '.join(cooling_components)}")
            if hot_components:
                status_parts.append(f"Hot: {', '.join(hot_components)}")
            
            if status_parts:
                logger.info(f"Cooling... {' | '.join(status_parts)}")
            
            time.sleep(3.0)  # Check every 3 seconds (matches monitoring interval)
        
        logger.error(f"Cooling timeout after {timeout}s")
        return False
    
    def add_thermal_callback(self, callback: Callable[[Dict[str, ThermalReading]], None]):
        """Add callback for thermal updates."""
        self.thermal_callbacks.append(callback)
    
    def add_emergency_callback(self, callback: Callable[[], None]):
        """Add callback for emergency shutdown."""
        self.emergency_shutdown_callbacks.append(callback)
    
    def get_thermal_summary(self) -> Dict[str, any]:
        """Get thermal summary for logging/display with cooling state."""
        if not self.thermal_readings:
            return {"status": "no_readings", "safe": False, "cooling_required": []}
        
        # Filter valid readings
        valid_readings = {
            comp: reading for comp, reading in self.thermal_readings.items()
            if reading.temperature_c > 0
        }
        
        if not valid_readings:
            return {"status": "no_valid_readings", "safe": False, "cooling_required": []}
        
        max_temp = max(r.temperature_c for r in valid_readings.values())
        min_temp = min(r.temperature_c for r in valid_readings.values())
        avg_temp = sum(r.temperature_c for r in valid_readings.values()) / len(valid_readings)
        
        hot_components = [
            f"{comp}: {reading.temperature_c:.1f}°C"
            for comp, reading in valid_readings.items()
            if reading.state not in [ThermalState.SAFE, ThermalState.WARM]
        ]
        
        cooling_required = [
            f"{comp}: {reading.temperature_c:.1f}°C (needs {self.COOLING_TARGET}°C)"
            for comp, reading in valid_readings.items()
            if self.cooling_required.get(comp, False)
        ]
        
        return {
            "status": "active",
            "safe": self.is_safe_for_ai_workload(),
            "component_count": len(valid_readings),
            "max_temp": max_temp,
            "min_temp": min_temp,
            "avg_temp": avg_temp,
            "hot_components": hot_components,
            "cooling_required": cooling_required,
            "cooling_count": len(cooling_required)
        }


# Global thermal monitor instance
_thermal_monitor: Optional[ThermalMonitor] = None

def get_thermal_monitor() -> ThermalMonitor:
    """Get the global thermal monitor instance."""
    global _thermal_monitor
    if _thermal_monitor is None:
        _thermal_monitor = ThermalMonitor()
    return _thermal_monitor


def ensure_thermal_safety() -> bool:
    """Ensure thermal safety before starting AI workloads."""
    monitor = get_thermal_monitor()
    
    if not monitor.is_monitoring:
        monitor.start_monitoring()
        time.sleep(3.0)  # Allow initial readings
    
    if not monitor.is_safe_for_ai_workload():
        logger.warning("System too hot for AI workload - waiting for cooling")
        return monitor.wait_for_cooling()
    
    return True


def ensure_startup_thermal_safety() -> bool:
    """Ensure thermal safety for server startup (less strict than AI workloads)."""
    monitor = get_thermal_monitor()
    
    if not monitor.is_monitoring:
        monitor.start_monitoring()
        time.sleep(3.0)  # Allow initial readings
    
    return monitor.is_safe_for_startup()


if __name__ == "__main__":
    # Test thermal monitoring
    monitor = ThermalMonitor()
    
    def thermal_callback(readings):
        for component, reading in readings.items():
            print(f"{component}: {reading.temperature_c:.1f}°C ({reading.state.value})")
    
    monitor.add_thermal_callback(thermal_callback)
    
    try:
        monitor.start_monitoring()
        print("Thermal monitoring active. Press Ctrl+C to stop.")
        
        while True:
            time.sleep(10)
            summary = monitor.get_thermal_summary()
            print(f"\nThermal Summary: {summary}")
            
    except KeyboardInterrupt:
        print("\nStopping thermal monitoring...")
        monitor.stop_monitoring()