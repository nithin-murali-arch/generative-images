"""
Test the UI's RealImageGenerator to ensure it works with float32
"""

import logging
import time
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ui_generator():
    """Test the UI's RealImageGenerator."""
    try:
        # Import the UI components
        from src.ui.research_interface_real import RealImageGenerator
        
        logger.info("Testing UI RealImageGenerator...")
        
        # Create generator
        generator = RealImageGenerator()
        logger.info("✅ RealImageGenerator created")
        
        # Check available models
        models = generator.get_available_models()
        logger.info(f"Available models: {models}")
        
        if not models:
            logger.error("❌ No models available")
            return False
        
        # Test model loading
        model_name = models[0]  # Use first available model
        logger.info(f"Testing model: {model_name}")
        
        if generator.load_model(model_name):
            logger.info("✅ Model loaded successfully")
        else:
            logger.error("❌ Failed to load model")
            return False
        
        # Test image generation
        prompt = "a beautiful red rose in a garden"
        logger.info(f"Testing generation with prompt: '{prompt}'")
        
        image, info = generator.generate_image(
            prompt=prompt,
            negative_prompt="",
            width=256,  # Small size for testing
            height=256,
            steps=15,
            guidance_scale=7.5,
            seed=None
        )
        
        if image:
            logger.info("✅ Image generated successfully!")
            logger.info(f"Image size: {image.size}, mode: {image.mode}")
            
            # Save test image
            test_path = "test_ui_generator.png"
            image.save(test_path)
            logger.info(f"Test image saved to: {test_path}")
            
            # Check file size
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                logger.info(f"File size: {file_size} bytes")
                
                if file_size > 10000:
                    logger.info("🎉 SUCCESS: UI generator is working with float32!")
                    return True
                else:
                    logger.error(f"❌ FAILED: File too small ({file_size} bytes)")
                    return False
            else:
                logger.error("❌ FAILED: File not created")
                return False
        else:
            logger.error(f"❌ Generation failed: {info}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("🔧 Testing UI RealImageGenerator")
    logger.info("=" * 50)
    
    success = test_ui_generator()
    
    if success:
        logger.info("🎉 SUCCESS: UI generator is working!")
    else:
        logger.error("💥 FAILED: UI generator has issues")
    
    logger.info("=" * 50) 