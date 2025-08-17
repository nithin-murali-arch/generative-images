#!/usr/bin/env python3
"""
Simple test script to check imports
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Current directory:", os.getcwd())
print("Python path:", sys.path[:3])

try:
    print("Testing imports...")
    from core.interfaces import ComplianceMode
    print("✓ core.interfaces imported successfully")
    
    from core.system_integration import SystemIntegration
    print("✓ core.system_integration imported successfully")
    
    from ui.research_interface import ResearchInterface
    print("✓ ui.research_interface imported successfully")
    
    print("\nAll imports successful! System is ready to run.")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc() 