#!/usr/bin/env python3
"""
Test script for the simplified research interface
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Testing simplified research interface...")

try:
    # Test import
    from ui.research_interface_simple import ResearchInterface, ComplianceMode
    print("✓ Import successful")
    
    # Test creation
    interface = ResearchInterface()
    print("✓ Interface created")
    
    # Test initialization
    if interface.initialize():
        print("✓ Interface initialized")
        
        # Test interface creation
        try:
            gr_interface = interface._create_interface()
            print("✓ Gradio interface created")
        except Exception as e:
            print(f"✗ Gradio interface creation failed: {e}")
    else:
        print("✗ Interface initialization failed")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc() 