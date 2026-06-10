"""
Backend Package for GenAI Research Platform

NOTE: ``app`` is imported lazily to avoid triggering the full dependency
chain (routes, chat, modules) when only the package metadata is needed.
"""

__version__ = '2.2.0'


def __getattr__(name):
    """Lazy-load popular symbols on first access."""
    if name == 'app':
        from .app import app
        return app
    if name == 'config':
        from .core.config import config
        return config
    if name in ('APIError', 'ValidationError'):
        from .core.errors import APIError, ValidationError
        return APIError if name == 'APIError' else ValidationError
    if name == 'sanitize_input':
        from .core.utils import sanitize_input
        return sanitize_input
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['app', 'config', 'APIError', 'ValidationError', 'sanitize_input']
