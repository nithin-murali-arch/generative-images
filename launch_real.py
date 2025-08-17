#!/usr/bin/env python3
"""
Real launcher for the research interface with actual AI image generation
"""

import sys
import os
from pathlib import Path

# Set PyTorch memory configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Starting REAL AI image generation launcher...")

try:
    # Import and create the real interface
    from ui.research_interface_real import ResearchInterface
    
    print("Creating real interface...")
    interface = ResearchInterface()
    
    print("Initializing interface...")
    if interface.initialize():
        print("Interface initialized successfully!")
        print("Launching Gradio interface with REAL AI generation on port 7861...")
        
        # Launch the interface on port 7861 to avoid conflicts
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7861
        )
    else:
        print("Failed to initialize interface")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 