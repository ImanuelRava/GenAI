"""
Core Package - Framework utilities for GenAI Research Platform
"""

from .config import config, LLM_PROVIDER_CONFIG
from .errors import APIError, ValidationError, NotFoundError, LLMError, register_error_handlers
from .cache import get_cache, CacheBackend
from .utils import sanitize_input, sanitize_filename, validate_doi, validate_api_key

__all__ = [
    'config', 'LLM_PROVIDER_CONFIG',
    'APIError', 'ValidationError', 'NotFoundError', 'LLMError', 'register_error_handlers',
    'get_cache', 'CacheBackend',
    'sanitize_input', 'sanitize_filename', 'validate_doi', 'validate_api_key',
]
