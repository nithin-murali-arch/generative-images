#!/usr/bin/env python3
"""
Windows-specific memory management test.
"""

import sys
import logging
import os
from pathlib import Path

# Set Windows-compatible PyTorch memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_windows_memory_management():
    """Test Windows-compatible memory management."""
    try:
        logger.info("Testing Windows-compatible memory management...")
        
        # Test imports
        import torch
        from diffusers import StableDiffusionPipeline
        
        logger.info("✓ Core libraries imported successfully")
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - skipping GPU memory tests")
            return True
        
        logger.info(f"✓ CUDA available: {torch.version.cuda}")
        
        # Check platform
        import platform
        is_windows = platform.system() == "Windows"
        logger.info(f"Platform: {platform.system()}")
        
        # Check environment variable
        import os
        pytorch_config = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_config}")
        
        if is_windows:
            if 'expandable_segments' in pytorch_config:
                logger.warning("⚠ expandable_segments not supported on Windows")
            else:
                logger.info("✓ Windows-compatible memory configuration")
        
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
        
        # Test Windows-specific memory cleanup
        logger.info("Testing Windows-compatible memory cleanup...")
        
        # Multiple cleanup passes (Windows-specific)
        for i in range(5):
            logger.info(f"Cleanup pass {i+1}/5")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            import time
            time.sleep(0.1)
        
        # Check memory after cleanup
        allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
        free_after = total_memory - reserved_after
        
        logger.info(f"After Windows cleanup:")
        logger.info(f"  Allocated: {allocated_after:.2f}GB")
        logger.info(f"  Reserved: {reserved_after:.2f}GB")
        logger.info(f"  Free: {free_after:.2f}GB")
        
        # Test Windows-compatible thresholds
        if is_windows:
            # Windows needs more conservative memory management
            min_free = 1.5  # GB
            max_reserved = 2.5  # GB
            
            if free_after < min_free:
                logger.error(f"❌ Insufficient free memory for Windows: {free_after:.2f}GB (need {min_free}GB)")
                return False
            
            if reserved_after > max_reserved:
                logger.error(f"❌ Too much reserved memory for Windows: {reserved_after:.2f}GB (max {max_reserved}GB)")
                return False
            
            logger.info("✓ Windows memory thresholds met")
        
        # Test if we have enough memory for basic operations
        if free_after < 1.0:
            logger.error(f"❌ Insufficient free memory: {free_after:.2f}GB")
            return False
        
        logger.info("✓ Sufficient memory available for operations")
        
        # Test memory fraction setting (Windows-compatible)
        try:
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
                logger.info("✓ Set GPU memory limit to 80% for Windows compatibility")
            else:
                logger.warning("⚠ set_per_process_memory_fraction not available")
        except Exception as e:
            logger.warning(f"⚠ Failed to set memory fraction: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_windows_memory_management()
    if success:
        print("\n🎉 Windows memory management test completed successfully!")
    else:
        print("\n❌ Windows memory management test failed!")
        sys.exit(1) 