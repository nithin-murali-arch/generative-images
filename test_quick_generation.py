#!/usr/bin/env python3
"""
Quick test for improved image generation
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Quick test for improved image generation...")

try:
    from ui.research_interface_real import RealImageGenerator
    
    # Create generator
    generator = RealImageGenerator()
    print(f"Generator created on device: {generator.device}")
    
    # Load model
    if generator.load_model("stable-diffusion-v1-5"):
        print("Model loaded successfully")
        
        # Test with very simple prompt and small size
        prompt = "red circle"
        print(f"Testing with prompt: '{prompt}'")
        
        image, info = generator.generate_image(
            prompt=prompt,
            width=256,
            height=256,
            steps=10,
            guidance_scale=5.0
        )
        
        if image:
            print("✅ Image generated successfully!")
            print(f"Size: {image.size}, Mode: {image.mode}")
            
            # Save and check
            test_path = "quick_test.png"
            image.save(test_path)
            file_size = os.path.getsize(test_path)
            print(f"File saved: {test_path} ({file_size} bytes)")
            
            if file_size > 1000:
                print("✅ File size looks good!")
            else:
                print("⚠️  File size too small - may be corrupted")
        else:
            print(f"❌ Generation failed: {info}")
    else:
        print("❌ Failed to load model")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 