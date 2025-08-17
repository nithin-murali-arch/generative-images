#!/usr/bin/env python3
"""
Simple test to check basic functionality
"""

import sys
import os

print("Python version:", sys.version)
print("Current directory:", os.getcwd())

# Test basic imports
try:
    import psutil
    print("✓ psutil imported successfully")
    print("psutil version:", psutil.__version__)
except ImportError as e:
    print("✗ psutil import failed:", e)

try:
    import torch
    print("✓ PyTorch imported successfully")
    print("PyTorch version:", torch.__version__)
except ImportError as e:
    print("✗ PyTorch import failed:", e)

try:
    import gradio
    print("✓ Gradio imported successfully")
    print("Gradio version:", gradio.__version__)
except ImportError as e:
    print("✗ Gradio import failed:", e)

# Test if we can access the src directory
src_path = os.path.join(os.getcwd(), "src")
if os.path.exists(src_path):
    print(f"✓ src directory exists: {src_path}")
    print("Contents:", os.listdir(src_path)[:5])
else:
    print(f"✗ src directory not found: {src_path}")

# Test basic file operations
try:
    with open("main.py", "r") as f:
        first_line = f.readline().strip()
        print(f"✓ main.py readable, first line: {first_line}")
except Exception as e:
    print(f"✗ Cannot read main.py: {e}") 