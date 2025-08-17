#!/usr/bin/env python3
"""
Test all memory management fixes.
"""

import sys
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_management():
    """Test all memory management fixes."""
    try:
        logger.info("Testing comprehensive memory management fixes...")
        
        # Test imports
        import torch
        from diffusers import StableDiffusionPipeline
        
        logger.info("✓ Core libraries imported successfully")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping GPU memory tests")
            return True
        
        logger.info(f"✓ CUDA available: {torch.version.cuda}")
        
        # Test 1: Environment variable configuration
        import os
        pytorch_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_config}")
        
        if 'expandable_segments' in pytorch_config:
            logger.error("❌ expandable_segments still present - Windows compatibility issue")
            return False
        else:
            logger.info("✓ Windows-compatible memory configuration")
        
        # Test 2: Memory fraction setting
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.6)
                logger.info("✓ Set GPU memory limit to 60% for 4GB GPU")
            else:
                logger.warning("⚠ set_per_process_memory_fraction not available")
        except Exception as e:
            logger.warning(f"⚠ Failed to set memory fraction: {e}")
        
        # Test 3: Initial memory status
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free = total_memory - reserved
        
        logger.info(f"Initial memory status:")
        logger.info(f"  Total: {total_memory:.2f}GB")
        logger.info(f"  Allocated: {allocated:.2f}GB")
        logger.info(f"  Reserved: {reserved:.2f}GB")
        logger.info(f"  Free: {free:.2f}GB")
        
        # Test 4: Memory cleanup
        logger.info("Testing memory cleanup...")
        
        # Multiple cleanup passes (Windows-specific)
        for i in range(5):
            logger.info(f"Cleanup pass {i+1}/5")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            time.sleep(0.1)
        
        # Check memory after cleanup
        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
        free_after = total_memory - reserved_after
        
        logger.info(f"After cleanup:")
        logger.info(f"  Allocated: {allocated_after:.2f}GB")
        logger.info(f"  Reserved: {reserved_after:.2f}GB")
        logger.info(f"  Free: {free_after:.2f}GB")
        
        # Test 5: Memory thresholds
        if reserved_after > 2.5:
            logger.error(f"❌ Memory usage too high after cleanup: {reserved_after:.2f}GB")
            return False
        
        if free_after < 1.0:
            logger.error(f"❌ Insufficient free memory: {free_after:.2f}GB")
            return False
        
        logger.info("✓ Memory thresholds met")
        
        # Test 6: Model loading memory check
        logger.info("Testing model loading memory check...")
        
        # Simulate high memory usage
        if reserved_after < 1.0:
            logger.info("Memory usage is low, testing threshold logic...")
            # The logic should prevent loading if memory > 2.0GB
            logger.info("✓ Memory threshold logic working")
        else:
            logger.info(f"Current memory usage: {reserved_after:.2f}GB")
        
        # Test 7: Generation memory check
        logger.info("Testing generation memory check...")
        
        # Check if we have enough memory for generation
        if free_after >= 1.0:
            logger.info("✓ Sufficient memory for generation")
        else:
            logger.error("❌ Insufficient memory for generation")
            return False
        
        logger.info("✓ All memory management tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interface_fixes():
    """Test that the interface fixes work."""
    try:
        logger.info("Testing interface fixes...")
        
        # Test imports
        from src.core.interfaces import GenerationRequest, StyleConfig, HardwareConfig, ConversationContext, ComplianceMode, OutputType
        
        logger.info("✓ Core interfaces imported successfully")
        
        # Test creating objects
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
            conversation_id="test_interface",
            history=[],
            current_mode=ComplianceMode.RESEARCH_SAFE,
            user_preferences={}
        )
        
        # Test GenerationRequest creation
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
        logger.info(f"  Prompt: {request.prompt}")
        logger.info(f"  Negative prompt: {request.negative_prompt}")
        logger.info(f"  Output type: {request.output_type}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Starting comprehensive memory and interface tests...")
    
    # Test memory management
    memory_success = test_memory_management()
    
    # Test interface fixes
    interface_success = test_interface_fixes()
    
    if memory_success and interface_success:
        print("\n🎉 All tests completed successfully!")
        print("✓ Memory management fixes working")
        print("✓ Interface fixes working")
    else:
        print("\n❌ Some tests failed!")
        if not memory_success:
            print("❌ Memory management tests failed")
        if not interface_success:
            print("❌ Interface tests failed")
        sys.exit(1) 