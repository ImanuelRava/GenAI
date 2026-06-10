from typing import Optional, Dict, Any
from flask import jsonify
import logging

logger = logging.getLogger(__name__)


class APIError(Exception):
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
        error_dict = {
            'success': False,
            'error': self.message,
            'status_code': self.status_code
        }
        if self.payload:
            error_dict['details'] = self.payload
        return error_dict


class ValidationError(APIError):
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        payload = kwargs.pop('payload', {})
        if field:
            payload['field'] = field
        super().__init__(message, status_code=400, payload=payload)


class NotFoundError(APIError):
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class LLMError(APIError):
    def __init__(self, message: str = "LLM processing error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error: APIError) -> tuple:
        logger.warning(f"API Error: {error.message} (Status: {error.status_code})")
        return jsonify(error.to_dict()), error.status_code

    @app.errorhandler(400)
    def handle_bad_request(error) -> tuple:
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'status_code': 400
        }), 400

    @app.errorhandler(404)
    def handle_not_found(error) -> tuple:
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'status_code': 404
        }), 404

    @app.errorhandler(413)
    def handle_file_too_large(error) -> tuple:
        return jsonify({
            'success': False,
            'error': 'File too large. Maximum size is 16MB',
            'status_code': 413
        }), 413

    @app.errorhandler(429)
    def handle_rate_limit(error) -> tuple:
        return jsonify({
            'success': False,
            'error': 'Rate limit exceeded. Please try again later.',
            'status_code': 429
        }), 429

    @app.errorhandler(500)
    def handle_internal_error(error) -> tuple:
        logger.error(f"Internal Server Error: {str(error)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An internal error occurred',
            'status_code': 500
        }), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error) -> tuple:
        logger.error(f"Unexpected Error: {str(error)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'status_code': 500
        }), 500


def success_response(data: Any, message: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    response = {
        'success': True,
        'data': data
    }
    if message:
        response['message'] = message
    response.update(kwargs)
    return response
