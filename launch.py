#!/usr/bin/env python3
"""
Launcher script for Academic Multimodal LLM Experiment System
"""

import sys
import os
from pathlib import Path

# Get the directory containing this script
script_dir = Path(__file__).parent.absolute()

# Add the src directory to Python path
src_path = script_dir / "src"
sys.path.insert(0, str(src_path))

# Also add the current directory
sys.path.insert(0, str(script_dir))

print(f"Script directory: {script_dir}")
print(f"Added to Python path: {src_path}")
print(f"Current Python path: {sys.path[:3]}")

try:
    # Now try to import and run the main application
    print("\nStarting Academic Multimodal LLM Experiment System...")
    
    # Import the main function
    from main import main
    
    # Run the application
    main()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTrying alternative import method...")
    
    try:
        # Try running main.py directly
        import subprocess
        result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e2:
        print(f"Alternative method failed: {e2}")
        
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc() 