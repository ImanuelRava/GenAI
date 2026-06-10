"""
Redox-Active Ligands chat blueprint — sync and async endpoints.
"""

import logging
from flask import Blueprint, request, jsonify

from core.errors import ValidationError
from llm.prompts import REDOX_SYSTEM_PROMPT
from llm import get_llm_response, get_llm_response_async

from .helpers import extract_chat_params, build_response_json

logger = logging.getLogger(__name__)

redox_bp = Blueprint('redox', __name__)


def register_redox_blueprint(app, limiter):
    """Register Redox chat routes on the Flask app with rate limiting."""

    @redox_bp.route('/api/redox/chat', methods=['POST'])
    @limiter.limit("20 per minute")
    def redox_chat():
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            logger.info(f"[RAL Bot] Provider: {provider}, Has API Key: {bool(api_key)}")
            response = get_llm_response(REDOX_SYSTEM_PROMPT, message,
                                         provider=provider, api_key=api_key, model=model)
            return build_response_json(response, provider)

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"RAL chat error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500

    @redox_bp.route('/api/redox/chat/async', methods=['POST'])
    @limiter.limit("20 per minute")
    async def redox_chat_async():
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            logger.info(f"[RAL Bot Async] Provider: {provider}, Has API Key: {bool(api_key)}")
            response = await get_llm_response_async(REDOX_SYSTEM_PROMPT, message,
                                                     provider=provider, api_key=api_key, model=model)
            return build_response_json(response, provider, is_async=True)

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"RAL async chat error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500

    app.register_blueprint(redox_bp)
