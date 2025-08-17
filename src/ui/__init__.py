"""
User interface components and research dashboard.

This module provides the Gradio-based research interface for the Academic
Multimodal LLM Experiment System, including generation controls, compliance
management, and experiment tracking.
"""

from .research_interface import (
    ResearchInterface,
    UIState,
    InterfaceTheme,
    create_research_interface,
    launch_research_interface
)

__all__ = [
    'ResearchInterface',
    'UIState', 
    'InterfaceTheme',
    'create_research_interface',
    'launch_research_interface'
]