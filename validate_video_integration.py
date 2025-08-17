#!/usr/bin/env python3
"""Simple validation script for video model integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.pipelines.video_generation import VideoGenerationPipeline, VideoModel
    from src.core.interfaces import HardwareConfig
    
    # Test basic initialization
    hardware_config = HardwareConfig(
        vram_size=8000,
        gpu_model='RTX 3070',
        cpu_cores=8,
        ram_size=32000,
        cuda_available=True,
        optimization_level='balanced'
    )
    
    pipeline = VideoGenerationPipeline()
    success = pipeline.initialize(hardware_config)
    print(f'Video pipeline initialization: {success}')
    
    # Test model availability
    available_models = pipeline.get_available_models()
    print(f'Available video models: {available_models}')
    
    # Test model validation
    for model in [VideoModel.STABLE_VIDEO_DIFFUSION.value, VideoModel.TEXT2VIDEO_ZERO.value]:
        validation = pipeline.validate_video_model_availability(model)
        print(f'{model}: Available={validation["available"]}, Issues={len(validation["issues"])}')
    
    print('✓ Video model integration validation completed successfully!')
    
except Exception as e:
    print(f'✗ Video model integration validation failed: {e}')
    import traceback
    traceback.print_exc()