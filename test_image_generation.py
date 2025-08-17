#!/usr/bin/env python3
"""
Test script for real image generation
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing real image generation...")

try:
    from ui.research_interface_real import RealImageGenerator
    
    # Create generator
    generator = RealImageGenerator()
    print(f"Generator created on device: {generator.device}")
    
    # Check available models
    models = generator.get_available_models()
    print(f"Available models: {models}")
    
    if not models:
        print("No models available!")
        exit(1)
    
    # Load first model
    first_model = models[0]
    print(f"Loading model: {first_model}")
    
    if generator.load_model(first_model):
        print(f"Model {first_model} loaded successfully")
        
        # Test image generation with simple prompt
        print("Testing image generation...")
        prompt = "A simple red circle on white background"
        
        image, info = generator.generate_image(
            prompt=prompt,
            width=256,  # Small size for testing
            height=256,
            steps=10,   # Fewer steps for testing
            guidance_scale=7.5
        )
        
        if image:
            print(f"Image generated successfully!")
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
            print(f"Generation info: {info}")
            
            # Save test image
            test_path = "test_generated_image.png"
            image.save(test_path)
            print(f"Test image saved to: {test_path}")
            
            # Check if file exists and has content
            if os.path.exists(test_path):
                file_size = os.path.getsize(test_path)
                print(f"File size: {file_size} bytes")
                if file_size > 1000:
                    print("✅ Image generation working correctly!")
                else:
                    print("⚠️  Generated file is very small - may be corrupted")
            else:
                print("❌ Test image file not created")
        else:
            print(f"❌ Image generation failed: {info}")
    else:
        print(f"❌ Failed to load model {first_model}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 