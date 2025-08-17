#!/usr/bin/env python3
"""
Test video generation fix.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_video_generation_request():
    """Test that video generation request creation works."""
    try:
        logger.info("Testing video generation request creation...")
        
        # Test imports
        from src.core.interfaces import GenerationRequest, StyleConfig, HardwareConfig, ConversationContext, ComplianceMode, OutputType
        
        logger.info("✓ Core interfaces imported successfully")
        
        # Test creating a video generation request
        style_config = StyleConfig(
            generation_params={
                "width": 512,
                "height": 512,
                "frames": 32,
                "fps": 20,
                "model_name": "stable-video-diffusion"
            }
        )
        
        hardware_config = HardwareConfig(
            vram_size=4096,  # 4GB for GTX 1650
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=8192,  # 8GB RAM
            cuda_available=True,
            optimization_level="balanced"
        )
        
        context = ConversationContext(
            conversation_id="test_video_gen",
            history=[],
            current_mode=ComplianceMode.RESEARCH_SAFE,
            user_preferences={}
        )
        
        # Create the request
        request = GenerationRequest(
            prompt="A unicorn walking in a meadow",
            negative_prompt="blurry, low quality",
            output_type=OutputType.VIDEO,
            style_config=style_config,
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=context,
            additional_params={
                "frames": 32,
                "fps": 20,
                "model_name": "stable-video-diffusion"
            }
        )
        
        logger.info("✓ Video generation request created successfully")
        logger.info(f"  Prompt: {request.prompt}")
        logger.info(f"  Negative prompt: {request.negative_prompt}")
        logger.info(f"  Output type: {request.output_type}")
        logger.info(f"  Compliance mode: {request.compliance_mode}")
        logger.info(f"  Hardware: {request.hardware_constraints.gpu_model}")
        logger.info(f"  Additional params: {request.additional_params}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_generation_request()
    if success:
        print("\n🎉 Video generation request test completed successfully!")
    else:
        print("\n❌ Video generation request test failed!")
        sys.exit(1) 