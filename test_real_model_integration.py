#!/usr/bin/env python3
"""
Test script for real model integration in image generation pipeline.

This script tests the actual model loading functionality with Stable Diffusion models.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with absolute imports
from src.core.interfaces import HardwareConfig, GenerationRequest, OutputType, StyleConfig, ComplianceMode, ConversationContext
from src.pipelines.image_generation import ImageGenerationPipeline, ImageModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_validation():
    """Test model validation functionality."""
    logger.info("Testing model validation...")
    
    # Create a test hardware config based on actual system (CPU-only)
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
    pipeline.initialize(hardware_config)
    
    # Test model validation for each model
    models_to_test = [
        ImageModel.STABLE_DIFFUSION_V1_5.value,
        ImageModel.SDXL_TURBO.value,
        ImageModel.FLUX_SCHNELL.value
    ]
    
    for model_name in models_to_test:
        logger.info(f"\nValidating model: {model_name}")
        
        # Get model info
        model_info = pipeline.get_model_info(model_name)
        if model_info:
            logger.info(f"  Model ID: {model_info['model_id']}")
            logger.info(f"  Min VRAM: {model_info['min_vram_mb']}MB")
            logger.info(f"  Cached: {model_info['is_cached']}")
        
        # Validate availability
        validation = pipeline.validate_model_availability(model_name)
        logger.info(f"  Available: {validation['available']}")
        
        if validation['issues']:
            logger.info(f"  Issues: {validation['issues']}")
        
        if validation['recommendations']:
            logger.info(f"  Recommendations: {validation['recommendations']}")

def test_model_loading():
    """Test actual model loading (SD 1.5 only for safety)."""
    logger.info("\nTesting model loading...")
    
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
    
    if not success:
        logger.error("Failed to initialize pipeline")
        return False
    
    # Try to load SD 1.5 (smallest model)
    model_name = ImageModel.STABLE_DIFFUSION_V1_5.value
    logger.info(f"Attempting to load {model_name}...")
    
    try:
        success = pipeline.switch_model(model_name)
        if success:
            logger.info(f"✓ Successfully loaded {model_name}")
            
            # Get available models
            available_models = pipeline.get_available_models()
            logger.info(f"Available models: {available_models}")
            
            return True
        else:
            logger.error(f"✗ Failed to load {model_name}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception loading {model_name}: {e}")
        return False
    
    finally:
        # Clean up
        pipeline.cleanup()

def test_image_generation():
    """Test image generation with loaded model."""
    logger.info("\nTesting image generation...")
    
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
    
    if not success:
        logger.error("Failed to initialize pipeline")
        return False
    
    # Create generation request
    style_config = StyleConfig(
        generation_params={
            'width': 512,
            'height': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5
        }
    )
    
    context = ConversationContext(
        conversation_id="test",
        history=[],
        current_mode=ComplianceMode.RESEARCH_SAFE,
        user_preferences={}
    )
    
    request = GenerationRequest(
        prompt="A beautiful landscape with mountains and a lake",
        output_type=OutputType.IMAGE,
        style_config=style_config,
        compliance_mode=ComplianceMode.RESEARCH_SAFE,
        hardware_constraints=hardware_config,
        context=context
    )
    
    try:
        logger.info("Generating test image...")
        result = pipeline.generate(request)
        
        if result.success:
            logger.info(f"✓ Image generated successfully!")
            logger.info(f"  Output path: {result.output_path}")
            logger.info(f"  Generation time: {result.generation_time:.2f}s")
            logger.info(f"  Model used: {result.model_used}")
            
            if result.quality_metrics:
                logger.info(f"  Quality metrics: {result.quality_metrics}")
            
            return True
        else:
            logger.error(f"✗ Image generation failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception during generation: {e}")
        return False
    
    finally:
        # Clean up
        pipeline.cleanup()

def main():
    """Run all tests."""
    logger.info("Starting real model integration tests...")
    
    try:
        # Test 1: Model validation
        test_model_validation()
        
        # Test 2: Model loading (may download models)
        logger.info("\n" + "="*50)
        logger.info("WARNING: The next test may download large model files!")
        logger.info("This could take several minutes and use several GB of disk space.")
        logger.info("="*50)
        
        user_input = input("\nProceed with model loading test? (y/N): ").strip().lower()
        if user_input == 'y':
            model_load_success = test_model_loading()
            
            if model_load_success:
                # Test 3: Image generation
                user_input = input("\nProceed with image generation test? (y/N): ").strip().lower()
                if user_input == 'y':
                    test_image_generation()
        
        logger.info("\nReal model integration tests completed!")
        
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()