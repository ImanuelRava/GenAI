"""
Error Handling Module for GenAI Research Platform
Provides standardized error handling and responses
"""

from typing import Optional, Dict, Any
from flask import jsonify, Response
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
    """
    Standardized API Error class for consistent error responses.
    
    Attributes:
        message: Human-readable error message
        status_code: HTTP status code
        payload: Additional error details
    """
    
    def __init__(
        self, 
        message: str, 
        status_code: int = 400, 
        payload: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON response"""
        error_dict = {
            'success': False,
            'error': self.message,
            'status_code': self.status_code
        }
        if self.payload:
            error_dict['details'] = self.payload
        return error_dict


class ValidationError(APIError):
    """Error for input validation failures"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        payload = kwargs.pop('payload', {})
        if field:
            payload['field'] = field
        super().__init__(message, status_code=400, payload=payload)


class AuthenticationError(APIError):
    """Error for authentication failures"""
    
    def __init__(self, message: str = "Authentication required", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(APIError):
    """Error for authorization failures"""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class NotFoundError(APIError):
    """Error for resource not found"""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class RateLimitError(APIError):
    """Error for rate limiting"""
    
    def __init__(self, message: str = "Rate limit exceeded", **kwargs):
        super().__init__(message, status_code=429, **kwargs)


class ExternalAPIError(APIError):
    """Error for external API failures"""
    
    def __init__(self, message: str = "External API error", **kwargs):
        super().__init__(message, status_code=502, **kwargs)


class LLMError(APIError):
    """Error for LLM-related failures"""
    
    def __init__(self, message: str = "LLM processing error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


def register_error_handlers(app):
    """
    Register error handlers with Flask app.
    
    Args:
        app: Flask application instance
    """
    
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> tuple:
        """Handle APIError exceptions"""
        logger.warning(f"API Error: {error.message} (Status: {error.status_code})")
        return jsonify(error.to_dict()), error.status_code
    
    @app.errorhandler(400)
    def handle_bad_request(error) -> tuple:
        """Handle 400 Bad Request errors"""
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'status_code': 400
        }), 400
    
    @app.errorhandler(404)
    def handle_not_found(error) -> tuple:
        """Handle 404 Not Found errors"""
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(413)
    def handle_file_too_large(error) -> tuple:
        """Handle file too large errors"""
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16MB',
            'status_code': 413
        }), 413
    
    @app.errorhandler(429)
    def handle_rate_limit(error) -> tuple:
        """Handle rate limit errors"""
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please try again later.',
            'status_code': 429
        }), 429
    
    @app.errorhandler(500)
    def handle_internal_error(error) -> tuple:
        """Handle internal server errors"""
        logger.error(f"Internal Server Error: {str(error)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An internal error occurred',
            'status_code': 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error) -> tuple:
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected Error: {str(error)}", exc_info=True)
        
        # Don't expose internal errors to users in production
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'status_code': 500
        }), 500


def success_response(data: Any, message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data
        message: Optional success message
        **kwargs: Additional response fields
        
    Returns:
        Response dictionary
    """
    response = {
        'success': True,
        'data': data
    }
    if message:
        response['message'] = message
    response.update(kwargs)
    return response
