"""
Launcher script for the Academic Multimodal LLM Research Interface.

This script provides a simple way to start the research interface with
proper initialization and configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from ui.research_interface import create_research_interface, launch_research_interface
from core.interfaces import HardwareConfig, ComplianceMode

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('research_interface.log')
        ]
    )


def create_mock_system_controller():
    """Create a mock system controller for testing."""
    class MockSystemController:
        def __init__(self):
            self.hardware_config = HardwareConfig(
                vram_size=8192,
                gpu_model="RTX 3070",
                cpu_cores=8,
                ram_size=16384,
                cuda_available=True,
                optimization_level="balanced"
            )
        
        def generate_content(self, request):
            """Mock content generation."""
            logger.info(f"Mock generation request: {request.prompt[:50]}...")
            return None
        
        def get_available_models(self):
            """Get available models."""
            return ["stable-diffusion-v1-5", "sdxl-turbo", "stable-video-diffusion"]
        
        def get_system_status(self):
            """Get system status."""
            return {
                "status": "ready",
                "models_loaded": 2,
                "vram_usage": 2048
            }
    
    return MockSystemController()


def create_mock_experiment_tracker():
    """Create a mock experiment tracker for testing."""
    class MockExperimentTracker:
        def __init__(self):
            self.experiments = []
        
        def start_experiment(self, request):
            """Start a new experiment."""
            experiment_id = f"exp_{len(self.experiments) + 1:03d}"
            logger.info(f"Started experiment: {experiment_id}")
            return experiment_id
        
        def save_experiment(self, experiment_id, result, notes=None):
            """Save experiment result."""
            logger.info(f"Saved experiment: {experiment_id}")
            self.experiments.append({
                'id': experiment_id,
                'result': result,
                'notes': notes
            })
        
        def get_experiment_history(self):
            """Get experiment history."""
            return self.experiments
    
    return MockExperimentTracker()


def create_mock_compliance_engine():
    """Create a mock compliance engine for testing."""
    class MockComplianceEngine:
        def __init__(self):
            self.compliance_mode = ComplianceMode.RESEARCH_SAFE
        
        def validate_request(self, request):
            """Validate generation request."""
            logger.info(f"Validating request in {self.compliance_mode.value} mode")
            return True
        
        def get_dataset_stats(self):
            """Get dataset license statistics."""
            return {
                "public_domain": 1000,
                "creative_commons": 2000,
                "fair_use_research": 500,
                "copyrighted": 0,
                "unknown": 100
            }
        
        def get_attribution_info(self, content_items):
            """Get attribution information."""
            return "Attribution: Mock dataset for testing purposes"
    
    return MockComplianceEngine()


def main():
    """Main launcher function."""
    # Set up logging
    setup_logging("INFO")
    
    logger.info("Starting Academic Multimodal LLM Research Interface")
    
    try:
        # Create mock system components for testing
        system_controller = create_mock_system_controller()
        experiment_tracker = create_mock_experiment_tracker()
        compliance_engine = create_mock_compliance_engine()
        
        logger.info("Created mock system components")
        
        # Create and initialize the research interface
        interface = create_research_interface(
            system_controller=system_controller,
            experiment_tracker=experiment_tracker,
            compliance_engine=compliance_engine
        )
        
        if interface is None:
            logger.error("Failed to create research interface")
            sys.exit(1)
        
        logger.info("Research interface created successfully")
        
        # Launch the interface
        logger.info("Launching Gradio interface...")
        launch_research_interface(
            interface,
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )
        
    except KeyboardInterrupt:
        logger.info("Interface shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()