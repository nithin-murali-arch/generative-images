"""
User interface components for AI content generation.

This module provides the modern Gradio-based interface with hardware-aware
model selection, thermal safety integration, and cross-platform support.
"""

from .modern_interface import create_modern_interface as create_interface
from .ui_integration import SystemIntegration

__all__ = [
    'create_interface',
    'SystemIntegration'
]