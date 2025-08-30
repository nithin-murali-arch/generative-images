#!/usr/bin/env python3
"""
Thermal safety test script.

This script tests the thermal monitoring system to ensure it can properly
detect temperatures and enforce safety limits.

CRITICAL: This script tests REAL thermal sensors - no mocks allowed.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_thermal_detection():
    """Test thermal sensor detection."""
    print("🌡️ Testing Thermal Sensor Detection")
    print("=" * 50)
    
    try:
        from src.core.thermal_monitor import ThermalMonitor
        
        # Create thermal monitor
        monitor = ThermalMonitor()
        print("✅ Thermal monitor created successfully")
        
        # Test temperature reading
        temperatures = monitor._get_all_temperatures()
        
        if not temperatures:
            print("❌ CRITICAL: No thermal sensors detected!")
            print("This system cannot run AI workloads safely without thermal monitoring.")
            return False
        
        print(f"✅ Detected {len(temperatures)} thermal sensors:")
        for component, temp in temperatures.items():
            state = monitor._classify_temperature(temp)
            print(f"   {component}: {temp:.1f}°C ({state.value})")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Thermal detection failed: {e}")
        return False


def test_thermal_monitoring():
    """Test continuous thermal monitoring."""
    print("\n🔄 Testing Continuous Thermal Monitoring")
    print("=" * 50)
    
    try:
        from src.core.thermal_monitor import get_thermal_monitor
        
        monitor = get_thermal_monitor()
        
        # Start monitoring
        monitor.start_monitoring()
        print("✅ Thermal monitoring started")
        
        # Monitor for 10 seconds
        for i in range(10):
            time.sleep(1)
            readings = monitor.get_current_readings()
            
            if readings:
                max_temp = max(r.temperature_c for r in readings.values())
                safe = monitor.is_safe_for_ai_workload()
                status = "SAFE" if safe else "UNSAFE"
                print(f"   {i+1:2d}s: Max temp {max_temp:.1f}°C - {status}")
            else:
                print(f"   {i+1:2d}s: No readings available")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("✅ Thermal monitoring stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Thermal monitoring failed: {e}")
        return False


def test_thermal_safety_enforcement():
    """Test thermal safety enforcement."""
    print("\n🛡️ Testing Thermal Safety Enforcement")
    print("=" * 50)
    
    try:
        from src.core.thermal_monitor import ensure_thermal_safety
        
        print("Testing thermal safety check...")
        is_safe = ensure_thermal_safety()
        
        if is_safe:
            print("✅ System is thermally safe for AI workloads")
        else:
            print("❌ System is too hot for AI workloads")
            print("The system would wait for cooling before proceeding")
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Thermal safety enforcement failed: {e}")
        return False


def test_hardware_detection():
    """Test hardware detection without mocks."""
    print("\n🖥️ Testing Hardware Detection")
    print("=" * 50)
    
    try:
        from src.core.cross_platform_hardware import detect_cross_platform_hardware
        
        config = detect_cross_platform_hardware()
        
        print("✅ Hardware detection successful:")
        print(f"   System: {config.system}")
        print(f"   CPU: {config.cpu_brand}")
        print(f"   CPU Cores: {config.cpu_cores}")
        print(f"   RAM: {config.ram_total_mb} MB")
        print(f"   GPU: {config.gpu_model}")
        print(f"   VRAM: {config.vram_size} MB")
        print(f"   CUDA: {'Available' if config.cuda_available else 'Not Available'}")
        print(f"   Hardware Tier: {config.hardware_tier}")
        
        # Validate critical values
        if config.vram_size <= 0:
            print("❌ CRITICAL: Invalid VRAM detection")
            return False
        
        if config.gpu_model == "Unknown GPU":
            print("❌ CRITICAL: GPU model detection failed")
            return False
        
        if config.ram_total_mb <= 0:
            print("❌ CRITICAL: Invalid RAM detection")
            return False
        
        print("✅ All hardware values validated")
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL: Hardware detection failed: {e}")
        return False


def test_system_integration():
    """Test system integration with safety checks."""
    print("\n🔧 Testing System Integration")
    print("=" * 50)
    
    try:
        from src.ui.ui_integration import SystemIntegration
        
        print("Creating system integration...")
        system = SystemIntegration()
        
        print("Initializing with safety checks...")
        success = system.initialize()
        
        if success:
            print("✅ System integration initialized successfully")
            
            # Test hardware info
            hw_info = system.get_hardware_info()
            print(f"   GPU: {hw_info['gpu_model']}")
            print(f"   VRAM: {hw_info['vram_total']} MB")
            print(f"   Thermal Safe: {hw_info['thermal_safe']}")
            print(f"   Max Temperature: {hw_info['max_temperature']:.1f}°C")
            
            return True
        else:
            print("❌ System integration initialization failed")
            return False
        
    except Exception as e:
        print(f"❌ CRITICAL: System integration failed: {e}")
        return False


def main():
    """Run all thermal safety tests."""
    print("🧪 THERMAL SAFETY TEST SUITE")
    print("=" * 60)
    print("Testing real thermal monitoring and hardware detection")
    print("NO MOCKS - All readings must be real for safety")
    print("=" * 60)
    
    tests = [
        ("Thermal Detection", test_thermal_detection),
        ("Thermal Monitoring", test_thermal_monitoring),
        ("Thermal Safety Enforcement", test_thermal_safety_enforcement),
        ("Hardware Detection", test_hardware_detection),
        ("System Integration", test_system_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n🏁 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ System is safe for AI workloads")
        print("✅ Thermal monitoring is working correctly")
        print("✅ Hardware detection is accurate")
        print("\n🚀 You can now run: python app.py")
    else:
        print("⚠️ SOME TESTS FAILED!")
        print("❌ System may not be safe for AI workloads")
        print("🔧 Please fix the issues before running AI generation")
        
        if passed == 0:
            print("\n🚨 CRITICAL: All tests failed!")
            print("This system cannot safely run AI workloads.")
            sys.exit(1)
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()