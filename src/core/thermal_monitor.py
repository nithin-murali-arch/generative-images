"""
Thermal Monitoring and Safety System

This module monitors CPU and GPU temperatures to prevent overheating and
implements automatic throttling and cooling periods when temperatures
exceed safe thresholds.
"""

import logging
import time
import threading
import statistics
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)

# Try to import temperature monitoring libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - temperature monitoring limited")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logger.warning("pynvml not available - GPU temperature monitoring disabled")


class ThermalState(Enum):
    """Thermal states for the system."""
    NORMAL = "normal"          # < 60°C
    WARM = "warm"              # 60-70°C
    HOT = "hot"                # 70-80°C
    CRITICAL = "critical"      # 80-90°C
    EMERGENCY = "emergency"    # > 90°C


@dataclass
class TemperatureReading:
    """Temperature reading with timestamp."""
    cpu_temp: Optional[float]
    gpu_temp: Optional[float]
    timestamp: float
    thermal_state: ThermalState


@dataclass
class ThermalThresholds:
    """Temperature thresholds for thermal management."""
    cpu_normal: float = 60.0      # Normal operation threshold
    cpu_warm: float = 65.0        # Start monitoring more closely
    cpu_hot: float = 75.0         # Reduce performance
    cpu_critical: float = 85.0    # Emergency throttling
    cpu_emergency: float = 95.0   # Immediate shutdown
    
    gpu_normal: float = 70.0      # Normal operation threshold
    gpu_warm: float = 75.0        # Start monitoring more closely
    gpu_hot: float = 80.0         # Reduce performance
    gpu_critical: float = 85.0    # Emergency throttling
    gpu_emergency: float = 90.0   # Immediate shutdown
    
    # Cooling thresholds (when to resume normal operation)
    cpu_resume: float = 40.0      # Resume after cooling to this temp
    gpu_resume: float = 50.0      # Resume after cooling to this temp
    
    # Monitoring settings
    monitoring_interval: float = 2.0    # Check every 2 seconds
    history_window: int = 15            # Keep 15 readings (30 seconds at 2s interval)


class ThermalMonitor:
    """
    Thermal monitoring system that tracks CPU and GPU temperatures
    and implements safety measures to prevent overheating.
    """
    
    def __init__(self, thresholds: Optional[ThermalThresholds] = None):
        self.thresholds = thresholds or ThermalThresholds()
        self.monitoring = False
        self.monitor_thread = None
        
        # Temperature history (deque for efficient sliding window)
        self.temperature_history: deque = deque(maxlen=self.thresholds.history_window)
        
        # Current state
        self.current_state = ThermalState.NORMAL
        self.is_throttled = False
        self.cooling_mode = False
        
        # Callbacks for thermal events
        self.thermal_callbacks: List[Callable] = []
        
        # Initialize GPU monitoring if available
        self.gpu_handle = None
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                if pynvml.nvmlDeviceGetCount() > 0:
                    self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    logger.info("GPU temperature monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU temperature monitoring: {e}")
        
        logger.info("ThermalMonitor initialized")
    
    def start_monitoring(self):
        """Start thermal monitoring."""
        if self.monitoring:
            logger.warning("Thermal monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("🌡️ Started thermal monitoring")
    
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("🌡️ Stopped thermal monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get temperature readings
                reading = self._get_temperature_reading()
                self.temperature_history.append(reading)
                
                # Analyze thermal state
                new_state = self._analyze_thermal_state(reading)
                
                # Handle state changes
                if new_state != self.current_state:
                    self._handle_thermal_state_change(self.current_state, new_state, reading)
                    self.current_state = new_state
                
                # Check if we need to throttle or resume
                self._check_throttling_conditions(reading)
                
                # Notify callbacks
                for callback in self.thermal_callbacks:
                    try:
                        callback(reading, self.current_state, self.is_throttled)
                    except Exception as e:
                        logger.warning(f"Thermal callback failed: {e}")
                
                time.sleep(self.thresholds.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in thermal monitoring loop: {e}")
                time.sleep(self.thresholds.monitoring_interval)
    
    def _get_temperature_reading(self) -> TemperatureReading:
        """Get current temperature readings."""
        cpu_temp = self._get_cpu_temperature()
        gpu_temp = self._get_gpu_temperature()
        
        # Determine thermal state based on highest temperature
        thermal_state = self._determine_thermal_state(cpu_temp, gpu_temp)
        
        return TemperatureReading(
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            timestamp=time.time(),
            thermal_state=thermal_state
        )
    
    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            # Try different methods to get CPU temperature
            temps = psutil.sensors_temperatures()
            
            # Look for common CPU temperature sensors
            cpu_temp_keys = ['coretemp', 'cpu_thermal', 'acpi', 'k10temp', 'zenpower']
            
            for key in cpu_temp_keys:
                if key in temps:
                    # Get the first temperature reading (usually package temp)
                    if temps[key]:
                        return temps[key][0].current
            
            # Fallback: try any available temperature sensor
            for sensor_name, sensor_list in temps.items():
                if sensor_list and 'cpu' in sensor_name.lower():
                    return sensor_list[0].current
            
            # If no CPU-specific sensor found, use the first available
            for sensor_name, sensor_list in temps.items():
                if sensor_list:
                    return sensor_list[0].current
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get CPU temperature: {e}")
            return None
    
    def _get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature."""
        if not self.gpu_handle:
            return None
        
        try:
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except Exception as e:
            logger.debug(f"Failed to get GPU temperature: {e}")
            return None
    
    def _determine_thermal_state(self, cpu_temp: Optional[float], gpu_temp: Optional[float]) -> ThermalState:
        """Determine thermal state based on temperatures."""
        max_temp = 0.0
        
        if cpu_temp is not None:
            max_temp = max(max_temp, cpu_temp)
        
        if gpu_temp is not None:
            max_temp = max(max_temp, gpu_temp)
        
        if max_temp == 0.0:
            return ThermalState.NORMAL
        
        # Use CPU thresholds as they're typically more conservative
        if max_temp >= self.thresholds.cpu_emergency:
            return ThermalState.EMERGENCY
        elif max_temp >= self.thresholds.cpu_critical:
            return ThermalState.CRITICAL
        elif max_temp >= self.thresholds.cpu_hot:
            return ThermalState.HOT
        elif max_temp >= self.thresholds.cpu_warm:
            return ThermalState.WARM
        else:
            return ThermalState.NORMAL
    
    def _analyze_thermal_state(self, reading: TemperatureReading) -> ThermalState:
        """Analyze thermal state using historical data."""
        if len(self.temperature_history) < 3:
            return reading.thermal_state
        
        # Get average temperature over the last 15 seconds (7-8 readings at 2s interval)
        recent_readings = list(self.temperature_history)[-8:]
        
        cpu_temps = [r.cpu_temp for r in recent_readings if r.cpu_temp is not None]
        gpu_temps = [r.gpu_temp for r in recent_readings if r.gpu_temp is not None]
        
        avg_cpu_temp = statistics.mean(cpu_temps) if cpu_temps else 0.0
        avg_gpu_temp = statistics.mean(gpu_temps) if gpu_temps else 0.0
        
        # Use average temperatures for state determination
        return self._determine_thermal_state(avg_cpu_temp, avg_gpu_temp)
    
    def _handle_thermal_state_change(self, old_state: ThermalState, new_state: ThermalState, reading: TemperatureReading):
        """Handle thermal state changes."""
        logger.info(f"🌡️ Thermal state changed: {old_state.value} → {new_state.value}")
        
        if reading.cpu_temp:
            logger.info(f"   CPU: {reading.cpu_temp:.1f}°C")
        if reading.gpu_temp:
            logger.info(f"   GPU: {reading.gpu_temp:.1f}°C")
        
        # Log appropriate warnings
        if new_state == ThermalState.WARM:
            logger.warning("⚠️ System warming up - monitoring closely")
        elif new_state == ThermalState.HOT:
            logger.warning("🔥 System running hot - consider reducing workload")
        elif new_state == ThermalState.CRITICAL:
            logger.error("🚨 CRITICAL TEMPERATURE - Implementing emergency throttling")
        elif new_state == ThermalState.EMERGENCY:
            logger.error("🆘 EMERGENCY TEMPERATURE - Immediate shutdown required")
    
    def _check_throttling_conditions(self, reading: TemperatureReading):
        """Check if we need to throttle or resume operations."""
        # Check if we need to start throttling
        if not self.is_throttled and not self.cooling_mode:
            # Check average temperature over last 15 seconds
            if len(self.temperature_history) >= 8:  # At least 15 seconds of data
                recent_readings = list(self.temperature_history)[-8:]
                cpu_temps = [r.cpu_temp for r in recent_readings if r.cpu_temp is not None]
                
                if cpu_temps:
                    avg_cpu_temp = statistics.mean(cpu_temps)
                    
                    # Trigger throttling if average CPU temp > 65°C over 15 seconds
                    if avg_cpu_temp >= 65.0:
                        logger.warning(f"🔥 CPU temperature too high (avg: {avg_cpu_temp:.1f}°C over 15s)")
                        logger.warning("⏸️ PAUSING operations for thermal protection")
                        self.is_throttled = True
                        self.cooling_mode = True
                        return
        
        # Check if we can resume operations
        elif self.is_throttled and self.cooling_mode:
            if reading.cpu_temp is not None and reading.cpu_temp <= self.thresholds.cpu_resume:
                logger.info(f"❄️ CPU cooled down to {reading.cpu_temp:.1f}°C")
                logger.info("▶️ RESUMING operations")
                self.is_throttled = False
                self.cooling_mode = False
    
    def should_throttle_operations(self) -> bool:
        """Check if operations should be throttled due to temperature."""
        return self.is_throttled
    
    def wait_for_cooling(self, max_wait_time: float = 300.0) -> bool:
        """
        Wait for system to cool down if it's overheating.
        
        Args:
            max_wait_time: Maximum time to wait in seconds
            
        Returns:
            bool: True if system cooled down, False if timeout
        """
        if not self.should_throttle_operations():
            return True
        
        logger.info(f"🧊 Waiting for system to cool down (max {max_wait_time}s)...")
        start_time = time.time()
        
        while self.should_throttle_operations() and (time.time() - start_time) < max_wait_time:
            current_reading = self._get_temperature_reading()
            
            if current_reading.cpu_temp:
                logger.info(f"   Current CPU temp: {current_reading.cpu_temp:.1f}°C (target: ≤{self.thresholds.cpu_resume:.1f}°C)")
            
            time.sleep(5.0)  # Check every 5 seconds during cooling
        
        if self.should_throttle_operations():
            logger.error(f"⏰ Cooling timeout after {max_wait_time}s")
            return False
        else:
            logger.info("✅ System cooled down, ready to resume")
            return True
    
    def get_current_temperatures(self) -> Dict[str, Optional[float]]:
        """Get current temperature readings."""
        reading = self._get_temperature_reading()
        return {
            'cpu_temp': reading.cpu_temp,
            'gpu_temp': reading.gpu_temp,
            'thermal_state': reading.thermal_state.value,
            'is_throttled': self.is_throttled,
            'cooling_mode': self.cooling_mode
        }
    
    def add_thermal_callback(self, callback: Callable):
        """Add a callback for thermal events."""
        self.thermal_callbacks.append(callback)
    
    def get_thermal_history(self, minutes: int = 5) -> List[TemperatureReading]:
        """Get thermal history for the last N minutes."""
        cutoff_time = time.time() - (minutes * 60)
        return [r for r in self.temperature_history if r.timestamp >= cutoff_time]
    
    def cleanup(self):
        """Clean up thermal monitoring resources."""
        self.stop_monitoring()
        
        if PYNVML_AVAILABLE and self.gpu_handle:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        
        logger.info("ThermalMonitor cleanup completed")


# Global thermal monitor instance
_thermal_monitor: Optional[ThermalMonitor] = None


def get_thermal_monitor() -> ThermalMonitor:
    """Get the global thermal monitor instance."""
    global _thermal_monitor
    if _thermal_monitor is None:
        _thermal_monitor = ThermalMonitor()
    return _thermal_monitor


def start_thermal_monitoring():
    """Start global thermal monitoring."""
    monitor = get_thermal_monitor()
    monitor.start_monitoring()


def stop_thermal_monitoring():
    """Stop global thermal monitoring."""
    global _thermal_monitor
    if _thermal_monitor:
        _thermal_monitor.stop_monitoring()


def should_throttle_for_temperature() -> bool:
    """Check if operations should be throttled due to temperature."""
    monitor = get_thermal_monitor()
    return monitor.should_throttle_operations()


def wait_for_system_cooling(max_wait_time: float = 300.0) -> bool:
    """Wait for system to cool down if overheating."""
    monitor = get_thermal_monitor()
    return monitor.wait_for_cooling(max_wait_time)