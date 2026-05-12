"""
conftest.py — pytest configuration
====================================
Ensures the project root is on sys.path so all src.* imports resolve correctly
regardless of where pytest is invoked from.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
