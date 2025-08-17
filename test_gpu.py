#!/usr/bin/env python3
"""
Test GPU status and model availability
"""

import torch
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=== GPU Status Test ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Check VRAM
    device_props = torch.cuda.get_device_properties(0)
    vram_total = device_props.total_memory / (1024**3)
    print(f"VRAM total: {vram_total:.1f}GB")
else:
    print("CUDA not available - using CPU")

print("\n=== Testing RealImageGenerator ===")
try:
    from ui.research_interface_real import RealImageGenerator
    
    generator = RealImageGenerator()
    print(f"Device: {generator.device}")
    print(f"Available models: {generator.get_available_models()}")
    
    # Test model loading
    if generator.get_available_models():
        first_model = generator.get_available_models()[0]
        print(f"Testing model load: {first_model}")
        success = generator.load_model(first_model)
        print(f"Load success: {success}")
    else:
        print("No models available")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 