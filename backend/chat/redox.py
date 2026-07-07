"""
Redox-Active Ligands chat blueprint — sync and async endpoints.

Wire RAL RAG into the chat flow (mirrors the NiCOBot pattern).
"""

import logging
from flask import Blueprint, request, jsonify

from core.errors import ValidationError
from llm import get_llm_response, get_llm_response_async

from .helpers import (
    extract_chat_params,
    build_redox_system_prompt,
    build_response_json,
)
from .conversation import conversation_store

logger = logging.getLogger(__name__)

redox_bp = Blueprint('redox', __name__)


def register_redox_blueprint(app, limiter):
    """Register Redox chat routes on the Flask app with rate limiting."""

    @redox_bp.route('/api/redox/chat', methods=['POST'])
    @limiter.limit("20 per minute")
    def redox_chat():
        """RAL-Bot chat endpoint with RAG integration and conversation history."""
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(
                f"[RAL Bot] Provider: {provider}, "
                f"Has API Key: {bool(api_key)}, RAG: {use_rag}"
            )
            system_prompt, database_context = build_redox_system_prompt(
                message, use_rag
            )

            # Build full messages list with conversation history
            history = conversation_store.get_history(conversation_id)
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(history)
            messages.append({'role': 'user', 'content': message})

            response = get_llm_response(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(
                    conversation_id, 'assistant', response
                )

            return build_response_json(
                response, provider, database_context,
                conversation_id=conversation_id,
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"RAL chat error: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Error communicating with LLM: {str(e)}'
            }), 500

    @redox_bp.route('/api/redox/chat/async', methods=['POST'])
    @limiter.limit("20 per minute")
    async def redox_chat_async():
        """Async RAL-Bot chat endpoint with RAG integration and conversation history."""
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(
                f"[RAL Bot Async] Provider: {provider}, "
                f"Has API Key: {bool(api_key)}, RAG: {use_rag}"
            )
            system_prompt, database_context = build_redox_system_prompt(
                message, use_rag
            )

            # Build full messages list with conversation history
            history = conversation_store.get_history(conversation_id)
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(history)
            messages.append({'role': 'user', 'content': message})

            response = await get_llm_response_async(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(
                    conversation_id, 'assistant', response
                )

            return build_response_json(
                response, provider, database_context,
                is_async=True, conversation_id=conversation_id,
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"RAL async chat error: {e}", exc_info=True)
            return jsonify({
                'success': False,
                'error': f'Error communicating with LLM: {str(e)}'
            }), 500

    app.register_blueprint(redox_bp)