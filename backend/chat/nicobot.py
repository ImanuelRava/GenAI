"""
NiCOBot Chat Blueprint

Provides the ``/api/nicobot/chat`` endpoint with RAG-augmented responses
powered by the NiCOBot chemical reaction database.
"""

import logging
from typing import Dict, List, Optional

from flask import Blueprint, request, jsonify
from flask_limiter import Limiter

from core.errors import ValidationError, LLMError
from core.utils import sanitize_input

logger = logging.getLogger(__name__)

nicobot_bp = Blueprint('nicobot', __name__, url_prefix='/api/nicobot')


def _retrieve_rag_context(message: str) -> str:
    """Retrieve RAG context from the NiCOBot database."""
    try:
        from modules.nicobot_rag import get_rag
        rag = get_rag()
        context = rag.retrieve_context(message)
        return context.formatted_context
    except Exception as e:
        logger.warning("NiCOBot RAG context retrieval failed: %s", e)
        return ""


def _build_messages(
    system_prompt: str,
    user_message: str,
    rag_context: str = "",
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Build the messages list for the LLM call.

    Injects RAG context into the user message when available.
    """
    messages = [{"role": "system", "content": system_prompt}]

    # Replay conversation history (skip last user message — we replace it)
    if history:
        for msg in history:
            if msg.get("role") in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

    # Build user message with optional RAG context
    if rag_context:
        full_user = (
            f"MANDATORY FIRST SOURCE — The following data comes from the NiCOBot "
            f"chemical reaction database and must be your PRIMARY reference:\n\n"
            f"{rag_context}\n\n"
            f"END DATABASE CONTEXT\n\n"
            f"Now answer the user's question using the database data above as your "
            f"primary source. When citing database values, explicitly say "
            f"'According to the database, …'.\n\n"
            f"User question: {user_message}"
        )
    else:
        full_user = user_message

    messages.append({"role": "user", "content": full_user})
    return messages


@nicobot_bp.route('/chat', methods=['POST'])
def chat():
    """NiCOBot chat endpoint with RAG augmentation."""
    data = request.get_json(silent=True)
    if not data:
        raise ValidationError("No JSON data provided")

    message = data.get('message', '')
    provider = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')
    history = data.get('history')

    if not message or not message.strip():
        raise ValidationError("Message cannot be empty", field="message")

    message = sanitize_input(message, max_length=16000)

    from llm.prompts import NICOBOT_SYSTEM_PROMPT
    from llm.helpers import get_llm_response

    system_prompt = NICOBOT_SYSTEM_PROMPT

    # Retrieve RAG context
    rag_context = _retrieve_rag_context(message)

    # Build messages
    messages = _build_messages(system_prompt, message, rag_context, history)

    try:
        response = get_llm_response(
            system_prompt=system_prompt,
            user_message=message,
            messages=messages,
            provider=provider,
            api_key=api_key,
            model=model,
            temperature=0.7,
            max_tokens=2000,
        )

        if response:
            return jsonify({
                'success': True,
                'response': response,
                'provider': provider or 'default',
                'rag_context_used': bool(rag_context),
            })
        else:
            raise LLMError("No response received from LLM")

    except (KeyboardInterrupt, SystemExit):
        raise
    except LLMError:
        raise
    except Exception as e:
        logger.error("NiCOBot chat error: %s", e, exc_info=True)
        raise LLMError(f"Error communicating with LLM: {str(e)}")


def register_nicobot_blueprint(app, limiter: Limiter):
    """Register NiCOBot blueprint with rate limiting."""
    app.register_blueprint(nicobot_bp)
    limiter.limit("20 per minute")(app.view_functions['nicobot.chat'])
    logger.info("[STARTUP] NiCOBot chat blueprint registered at /api/nicobot/chat")