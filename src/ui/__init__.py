"""
User interface components for AI content generation.

This module provides the modern Gradio-based interface with hardware-aware
model selection, thermal safety integration, and cross-platform support.
"""

from .modern_interface import create_interface
from .ui_integration import UIIntegration

__all__ = [
    'create_interface',
    'UIIntegration'
]