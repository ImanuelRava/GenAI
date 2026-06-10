"""
NiCOBot chat blueprint — sync and async endpoints with conversation history.

Fixes Critical Issue #2: conversation history is now correctly passed to the LLM
via the messages parameter of get_llm_response / get_llm_response_async.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from core.errors import ValidationError
from core.utils import sanitize_input

from llm.prompts import NICOBOT_SYSTEM_PROMPT, EXPLAIN_SYSTEM_PROMPT
from llm import get_llm_response, get_llm_response_async
from llm.knowledge_graph import generate_mock_knowledge_graph
from llm.helpers import generate_knowledge_graph, generate_knowledge_graph_async

from .helpers import (
    extract_chat_params,
    build_nicobot_system_prompt,
    build_response_json,
    get_explanation,
)
from .conversation import conversation_store

logger = logging.getLogger(__name__)

nicobot_bp = Blueprint('nicobot', __name__)

# NOTE: The limiter is passed in from app.py via register_nicobot_blueprint()


def register_nicobot_blueprint(app, limiter):
    """Register NiCOBot routes on the Flask app with rate limiting."""

    @nicobot_bp.route('/api/nicobot/chat', methods=['POST'])
    @limiter.limit("20 per minute")
    def nicobot_chat():
        """NiCOBot chat endpoint with RAG integration and conversation history.

        Supports optional ``conversation_id`` to maintain multi-turn context.
        """
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(f"[NiCOBot] Provider: {provider}, Has API Key: {bool(api_key)}, RAG: {use_rag}")
            system_prompt, database_context = build_nicobot_system_prompt(message, use_rag)

            # Build full messages list with conversation history
            history = conversation_store.get_history(conversation_id)
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(history)
            messages.append({'role': 'user', 'content': message})

            # Pass full messages to LLM for multi-turn support
            response = get_llm_response(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(conversation_id, 'assistant', response)

            return build_response_json(response, provider, database_context,
                                       conversation_id=conversation_id)

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"NiCOBot chat error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Error communicating with LLM: {str(e)}'}), 500

    @nicobot_bp.route('/api/nicobot/chat/async', methods=['POST'])
    @limiter.limit("20 per minute")
    async def nicobot_chat_async():
        """Async NiCOBot chat endpoint with RAG integration and conversation history."""
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(f"[NiCOBot Async] Provider: {provider}, Has API Key: {bool(api_key)}, RAG: {use_rag}")
            system_prompt, database_context = build_nicobot_system_prompt(message, use_rag)

            # Build full messages list with conversation history
            history = conversation_store.get_history(conversation_id)
            messages = [{'role': 'system', 'content': system_prompt}]
            messages.extend(history)
            messages.append({'role': 'user', 'content': message})

            # Pass full messages to LLM for multi-turn support
            response = await get_llm_response_async(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(conversation_id, 'assistant', response)

            return build_response_json(response, provider, database_context,
                                       is_async=True, conversation_id=conversation_id)

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"NiCOBot async chat error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': f'Error communicating with LLM: {str(e)}'}), 500

    app.register_blueprint(nicobot_bp)
