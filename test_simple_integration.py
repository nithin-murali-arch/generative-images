#!/usr/bin/env python3
"""
Simple test for real model integration with realistic hardware.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.interfaces import HardwareConfig
from src.pipelines.image_generation import ImageGenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_realistic_hardware():
    """Test with realistic hardware configuration."""
    logger.info("Testing with realistic hardware configuration...")
    
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
    
    logger.info(f"Pipeline initialized: {success}")
    
    if success:
        # Check available models
        available = pipeline.get_available_models()
        logger.info(f"Available models: {available}")
        
        # Test model validation for first available model
        if available:
            model = available[0]
            validation = pipeline.validate_model_availability(model)
            logger.info(f"{model}: Available={validation['available']}")
            
            if validation['issues']:
                logger.info(f"  Issues: {validation['issues']}")
            
            if validation['recommendations']:
                logger.info(f"  Recommendations: {validation['recommendations']}")
        
        # Clean up
        pipeline.cleanup()
        logger.info("Test completed successfully")
        return True
    else:
        logger.error("Pipeline initialization failed")
        return False

if __name__ == "__main__":
    test_realistic_hardware()