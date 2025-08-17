#!/usr/bin/env python3
"""
Test ExperimentResult fix.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_experiment_result_creation():
    """Test that ExperimentResult can be created correctly."""
    try:
        logger.info("Testing ExperimentResult creation...")
        
        # Test imports
        from src.core.interfaces import (
            GenerationRequest, GenerationResult, ExperimentResult, 
            StyleConfig, HardwareConfig, ConversationContext, 
            ComplianceMode, OutputType
        )
        
        logger.info("✓ Core interfaces imported successfully")
        
        # Create required objects
        style_config = StyleConfig(
            generation_params={
                "width": 512,
                "height": 512,
                "steps": 20
            }
        )
        
        hardware_config = HardwareConfig(
            vram_size=4096,
            gpu_model="GTX 1650",
            cpu_cores=4,
            ram_size=8192,
            cuda_available=True,
            optimization_level="balanced"
        )
        
        context = ConversationContext(
            conversation_id="test_experiment",
            history=[],
            current_mode=ComplianceMode.RESEARCH_SAFE,
            user_preferences={}
        )
        
        # Create GenerationRequest
        request = GenerationRequest(
            prompt="Test prompt",
            output_type=OutputType.IMAGE,
            style_config=style_config,
            compliance_mode=ComplianceMode.RESEARCH_SAFE,
            hardware_constraints=hardware_config,
            context=context,
            negative_prompt="Test negative prompt",
            additional_params={}
        )
        
        logger.info("✓ GenerationRequest created successfully")
        
        # Create GenerationResult
        result = GenerationResult(
            success=True,
            output_path=Path("test_output.png"),
            generation_time=5.0,
            model_used="stable-diffusion-v1-5",
            quality_metrics={"fid": 0.8},
            compliance_info={"license": "research_safe"}
        )
        
        logger.info("✓ GenerationResult created successfully")
        
        # Create ExperimentResult
        import datetime
        import uuid
        
        experiment_result = ExperimentResult(
            experiment_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(),
            request=request,
            result=result,
            notes="Test experiment"
        )
        
        logger.info("✓ ExperimentResult created successfully")
        logger.info(f"  Experiment ID: {experiment_result.experiment_id}")
        logger.info(f"  Timestamp: {experiment_result.timestamp}")
        logger.info(f"  Request prompt: {experiment_result.request.prompt}")
        logger.info(f"  Result success: {experiment_result.result.success}")
        logger.info(f"  Notes: {experiment_result.notes}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_experiment_result_creation()
    if success:
        print("\n🎉 ExperimentResult test completed successfully!")
    else:
        print("\n❌ ExperimentResult test failed!")
        sys.exit(1) 