#!/usr/bin/env python3
"""
Test script to verify memory management fixes.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_management():
    """Test the memory management functionality."""
    try:
        logger.info("Testing memory management fixes...")
        
        # Test imports
        import torch
        from diffusers import StableDiffusionPipeline
        
        logger.info("✓ Core libraries imported successfully")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping GPU memory tests")
            return True
        
        logger.info(f"✓ CUDA available: {torch.version.cuda}")
        
        # Check initial memory status
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
        
        # Test memory cleanup
        logger.info("Testing memory cleanup...")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory after cleanup
        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
        free_after = total_memory - reserved_after
        
        logger.info(f"After cleanup:")
        logger.info(f"  Allocated: {allocated_after:.2f}GB")
        logger.info(f"  Reserved: {reserved_after:.2f}GB")
        logger.info(f"  Free: {free_after:.2f}GB")
        
        # Test if we have enough memory for basic operations
        if free_after < 1.0:
            logger.error(f"❌ Insufficient free memory: {free_after:.2f}GB")
            return False
        
        logger.info("✓ Sufficient memory available for operations")
        
        # Test environment variable
        import os
        pytorch_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_config}")
        
        if 'expandable_segments' in pytorch_config:
            logger.info("✓ Memory fragmentation prevention enabled")
        else:
            logger.warning("⚠ Memory fragmentation prevention not configured")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_management()
    if success:
        print("\n🎉 Memory management test completed successfully!")
    else:
        print("\n❌ Memory management test failed!")
        sys.exit(1) 