#!/usr/bin/env python3
"""
Test script to verify video generation pipeline initialization and functionality.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_pipeline():
    """Test the video generation pipeline initialization."""
    try:
        logger.info("Testing video generation pipeline...")
        
        # Test imports
        from core.system_integration import SystemIntegration
        from core.interfaces import GenerationRequest, ComplianceMode
        
        logger.info("✓ Core modules imported successfully")
        
        # Initialize system integration
        config = {
            "data_dir": "data",
            "models_dir": "models", 
            "experiments_dir": "experiments",
            "cache_dir": "cache",
            "logs_dir": "logs",
            "auto_detect_hardware": True,
            "max_concurrent_generations": 1
        }
        
        system = SystemIntegration()
        logger.info("✓ SystemIntegration created")
        
        # Initialize the system
        if system.initialize(config):
            logger.info("✓ System integration initialized successfully")
        else:
            logger.error("✗ System integration initialization failed")
            return False
        
        # Check video pipeline
        if system.video_pipeline:
            logger.info(f"✓ Video pipeline available: {type(system.video_pipeline).__name__}")
            
            if system.video_pipeline.is_initialized:
                logger.info("✓ Video pipeline is initialized")
                
                # Check available models
                models = system.video_pipeline.get_available_models()
                logger.info(f"✓ Available video models: {models}")
                
                # Test a simple generation request
                request = GenerationRequest(
                    prompt="A simple test video",
                    negative_prompt="",
                    width=256,
                    height=256,
                    compliance_mode=ComplianceMode.RESEARCH_SAFE,
                    additional_params={
                        "frames": 8,
                        "fps": 8,
                        "model_name": "stable-video-diffusion"
                    }
                )
                
                logger.info("✓ Generation request created successfully")
                logger.info(f"Request details: {request}")
                
                return True
            else:
                logger.warning("⚠ Video pipeline exists but is not initialized")
                return False
        else:
            logger.error("✗ Video pipeline not available")
            return False
            
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_pipeline()
    if success:
        print("\n🎉 Video pipeline test completed successfully!")
    else:
        print("\n❌ Video pipeline test failed!")
        sys.exit(1) 