"""
User interface components and research dashboard.

This module provides the Gradio-based research interface for the Academic
Multimodal LLM Experiment System, including generation controls, compliance
management, and experiment tracking.
"""

from .research_interface_simple import (
    ResearchInterface,
    UIState,
    ComplianceMode
)

__all__ = [
    'ResearchInterface',
    'UIState', 
    'ComplianceMode'
]