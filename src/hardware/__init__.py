"""
Hardware detection and optimization module for the Academic Multimodal LLM Experiment System.

This module provides hardware detection capabilities and optimization strategies
for different GPU configurations, enabling the system to adapt to available
hardware resources automatically.
"""

from .detector import HardwareDetector
from .profiles import HardwareProfileManager
from .memory_manager import MemoryManager

__all__ = ['HardwareDetector', 'HardwareProfileManager', 'MemoryManager']