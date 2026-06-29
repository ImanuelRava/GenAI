"""
Pytest configuration for GenAI Research Platform
"""

import pytest
import sys
import os


backend_path = os.path.join(os.path.dirname(__file__), '..', 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)
