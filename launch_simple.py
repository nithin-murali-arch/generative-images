#!/usr/bin/env python3
"""
Simple launcher for the research interface
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Starting simple launcher...")

try:
    # Import and create the interface
    from ui.research_interface_simple import ResearchInterface
    
    print("Creating interface...")
    interface = ResearchInterface()
    
    print("Initializing interface...")
    if interface.initialize():
        print("Interface initialized successfully!")
        print("Launching Gradio interface...")
        
        # Launch the interface
        interface.launch(
            share=False,
            server_name="127.0.0.1",
            server_port=7860
        )
    else:
        print("Failed to initialize interface")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 