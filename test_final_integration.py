#!/usr/bin/env python3
"""
Final test for real model integration with realistic hardware.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.interfaces import HardwareConfig
from src.pipelines.image_generation import ImageGenerationPipeline
from src.pipelines.video_generation import VideoGenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_image_pipeline():
    """Test image generation pipeline."""
    logger.info("Testing image generation pipeline...")
    
    # Create hardware config based on actual system (CPU-only)
    hardware_config = HardwareConfig(
        vram_size=0,  # No dedicated GPU
        gpu_model="Integrated",
        cpu_cores=8,
        ram_size=16000,
        cuda_available=False,
        optimization_level="conservative"
    )
    
    # Initialize pipeline
    pipeline = ImageGenerationPipeline()
    success = pipeline.initialize(hardware_config)
    
    if success:
        available = pipeline.get_available_models()
        logger.info(f"Image models available: {available}")
        
        if available:
            # Test model validation
            model = available[0]
            validation = pipeline.validate_model_availability(model)
            logger.info(f"Image model {model}: Available={validation['available']}")
            
            # Test model loading
            load_success = pipeline.switch_model(model)
            logger.info(f"Image model loading: {load_success}")
        
        pipeline.cleanup()
        return True
    else:
        logger.error("Image pipeline initialization failed")
        return False

def test_video_pipeline():
    """Test video generation pipeline."""
    logger.info("Testing video generation pipeline...")
    
    # Create hardware config based on actual system (CPU-only)
    hardware_config = HardwareConfig(
        vram_size=0,  # No dedicated GPU
        gpu_model="Integrated",
        cpu_cores=8,
        ram_size=16000,
        cuda_available=False,
        optimization_level="conservative"
    )
    
    # Initialize pipeline
    pipeline = VideoGenerationPipeline()
    success = pipeline.initialize(hardware_config)
    
    if success:
        available = pipeline.get_available_models()
        logger.info(f"Video models available: {available}")
        
        if available:
            # Test model validation
            model = available[0]
            validation = pipeline.validate_video_model_availability(model)
            logger.info(f"Video model {model}: Available={validation['available']}")
            
            # Test model loading
            load_success = pipeline.switch_model(model)
            logger.info(f"Video model loading: {load_success}")
        
        pipeline.cleanup()
        return True
    else:
        logger.error("Video pipeline initialization failed")
        return False

def main():
    """Run comprehensive integration tests."""
    logger.info("Starting comprehensive real model integration tests...")
    
    # Test image pipeline
    image_success = test_image_pipeline()
    
    # Test video pipeline
    video_success = test_video_pipeline()
    
    # Summary
    if image_success and video_success:
        logger.info("✓ All real model integration tests passed!")
        return True
    else:
        logger.error("✗ Some integration tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)