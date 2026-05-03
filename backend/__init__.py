"""
Backend Package for GenAI Research Platform
"""

from .app import app
from .config import config
from .errors import APIError, ValidationError
from .utils import sanitize_input

__version__ = '2.0.0'
__all__ = ['app', 'config', 'APIError', 'ValidationError', 'sanitize_input']
