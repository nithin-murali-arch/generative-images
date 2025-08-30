"""
Shared dependencies for the API server.

This module contains shared dependencies and state management
to avoid circular imports between server modules.
"""

import logging
from typing import Optional

from ..core.llm_controller import LLMController
from ..pipelines.image_generation import ImageGenerationPipeline
from ..pipelines.video_generation import VideoGenerationPipeline
from ..core.cross_platform_hardware import detect_cross_platform_hardware
from ..data.experiment_tracker import ExperimentTracker
from ..core.interfaces import HardwareConfig

logger = logging.getLogger(__name__)


class APIState:
    """Global state management for the API server."""
    
    def __init__(self):
        self.system_integration = None  # Will be set by APIServer
        self.llm_controller: Optional[LLMController] = None
        self.image_pipeline: Optional[ImageGenerationPipeline] = None
        self.video_pipeline: Optional[VideoGenerationPipeline] = None
        self.hardware_config: Optional[HardwareConfig] = None
        self.experiment_tracker: Optional[ExperimentTracker] = None
        self.active_tasks: dict = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize all system components."""
        try:
            # If system integration is already set (by APIServer), use it
            if self.system_integration:
                logger.info("Using system integration from APIServer")
                self.llm_controller = self.system_integration.llm_controller
                self.image_pipeline = self.system_integration.image_pipeline
                self.video_pipeline = self.system_integration.video_pipeline
                self.hardware_config = self.system_integration.hardware_config
                self.experiment_tracker = self.system_integration.experiment_tracker
                self.is_initialized = True
                return
            
            logger.info("Initializing API server components...")
            
            # Detect hardware configuration
            hardware_detector = HardwareDetector()
            self.hardware_config = hardware_detector.detect_hardware()
            logger.info(f"Detected hardware: {self.hardware_config.gpu_model} with {self.hardware_config.vram_size}MB VRAM")
            
            # Initialize LLM controller
            self.llm_controller = LLMController(self.hardware_config)
            
            # Initialize generation pipelines
            self.image_pipeline = ImageGenerationPipeline()
            self.video_pipeline = VideoGenerationPipeline()
            
            # Initialize pipelines with hardware config
            image_init_success = self.image_pipeline.initialize(self.hardware_config)
            video_init_success = self.video_pipeline.initialize(self.hardware_config)
            
            if not image_init_success:
                logger.warning("Image pipeline initialization failed")
            if not video_init_success:
                logger.warning("Video pipeline initialization failed")
            
            # Initialize experiment tracker
            self.experiment_tracker = ExperimentTracker()
            
            self.is_initialized = True
            logger.info("API server components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize API server: {e}")
            raise


# Global API state instance
api_state = APIState()


# Dependency to ensure API is initialized
async def get_api_state() -> APIState:
    """Dependency to get initialized API state."""
    if not api_state.is_initialized:
        await api_state.initialize()
    return api_state