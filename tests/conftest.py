"""
conftest.py
===========
Pytest configuration for adding src/ to Python path.

This allows pytest to find modules in the src/ directory
without setting PYTHONPATH manually every time.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
