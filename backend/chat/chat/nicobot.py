"""
NiCOBot chat blueprint — sync and async endpoints with conversation history.

RAG context is placed in the USER message (not the system prompt) so the
LLM actually reads and uses it.
"""

import logging
from flask import Blueprint, request, jsonify

from core.errors import ValidationError

from llm import get_llm_response, get_llm_response_async
from llm.prompts import NICOBOT_SYSTEM_PROMPT

from .helpers import (
    extract_chat_params,
    build_response_json,
    get_explanation,
)
from .conversation import conversation_store

logger = logging.getLogger(__name__)

nicobot_bp = Blueprint('nicobot', __name__)

# Short system prompt used when database context is available — keeps
# the system message focused on identity + one clear rule.
_NICOBOT_RAG_SYSTEM_PROMPT = (
    NICOBOT_SYSTEM_PROMPT + "\n\n"
    "IMPORTANT: When the user provides database context in their message, "
    "you MUST answer using the specific compounds, papers, and data shown "
    "in that context. Quote specific DOIs, SMILES strings, and reaction "
    "types from the provided data. Do NOT substitute your own training-data "
    "values."
)


def _retrieve_rag_context(message: str,
                          log_prefix: str = "NiCOBot"):
    """Retrieve RAG context for a user message.

    Returns (formatted_context_string_or_None, rag_stats_dict_or_None).
    Never raises — falls back to (None, None) on any error.
    """
    try:
        from modules.nicobot_rag import get_rag

        rag = get_rag()
        context = rag.retrieve_context(message)

        if context.formatted_context:
            stats = {
                'compounds': len(context.compounds),
                'papers': len(context.papers),
            }
            logger.info(
                f"[{log_prefix}] RAG context retrieved: "
                f"{stats['compounds']} compounds, {stats['papers']} papers"
            )
            return context.formatted_context, stats
        else:
            logger.info(f"[{log_prefix}] RAG: no database match for query")

    except Exception as e:
        logger.warning(
            f"[{log_prefix}] RAG error: {e}. Falling back to base prompt."
        )

    return None, None


def _build_messages(message: str, conversation_id: str,
                    use_rag: bool = True,
                    log_prefix: str = "NiCOBot"):
    """Build the messages list for the LLM call.

    When RAG context is available, the database data is placed in the USER
    message (right next to the question) rather than buried in a long system
    prompt.  LLMs attend far more strongly to user-message content.

    Returns (messages, system_prompt, database_context_or_None).
    """
    database_context = None

    if use_rag:
        database_context, _ = _retrieve_rag_context(message, log_prefix)

    if database_context:
        system_prompt = _NICOBOT_RAG_SYSTEM_PROMPT
        # Prepend database context directly into the user message.
        rag_user_message = (
            f"[The following data was retrieved from the NiCOBot chemical "
            f"database. Use these specific values in your answer.]\n\n"
            f"{database_context}\n\n"
            f"[End of database context]\n\n"
            f"{message}"
        )
    else:
        system_prompt = NICOBOT_SYSTEM_PROMPT
        rag_user_message = message

    # Build full messages list with conversation history
    history = conversation_store.get_history(conversation_id)
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(history)
    messages.append({'role': 'user', 'content': rag_user_message})

    return messages, system_prompt, database_context


def register_nicobot_blueprint(app, limiter):
    """Register NiCOBot routes on the Flask app with rate limiting."""

    @nicobot_bp.route('/api/nicobot/chat', methods=['POST'])
    @limiter.limit("20 per minute")
    def nicobot_chat():
        """NiCOBot chat endpoint with RAG integration and conversation history."""
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(
                f"[NiCOBot] Provider: {provider}, RAG: {use_rag}"
            )

            # Build messages (RAG context goes into the user message)
            messages, system_prompt, database_context = _build_messages(
                message, conversation_id, use_rag=use_rag
            )

            response = get_llm_response(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history (original message, not RAG-wrapped)
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(
                    conversation_id, 'assistant', response
                )

            return build_response_json(
                response, provider, database_context,
                conversation_id=conversation_id
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("NiCOBot chat error", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error communicating with LLM. Check server logs for details.'
            }), 500

    @nicobot_bp.route('/api/nicobot/chat/async', methods=['POST'])
    @limiter.limit("20 per minute")
    async def nicobot_chat_async():
        """Async NiCOBot chat endpoint with RAG integration."""
        try:
            data = request.get_json()
            message, provider, api_key, model = extract_chat_params(data)
            use_rag = data.get('use_rag', True)
            conversation_id = data.get('conversation_id')

            logger.info(
                f"[NiCOBot Async] Provider: {provider}, RAG: {use_rag}"
            )

            # Build messages (RAG context goes into the user message)
            messages, system_prompt, database_context = _build_messages(
                message, conversation_id, use_rag=use_rag
            )

            response = await get_llm_response_async(
                system_prompt, message,
                provider=provider, api_key=api_key, model=model,
                messages=messages,
            )

            # Store conversation history (original message, not RAG-wrapped)
            conversation_store.add_message(conversation_id, 'user', message)
            if response:
                conversation_store.add_message(
                    conversation_id, 'assistant', response
                )

            return build_response_json(
                response, provider, database_context,
                is_async=True, conversation_id=conversation_id
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("NiCOBot async chat error", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error communicating with LLM. Check server logs for details.'
            }), 500

    app.register_blueprint(nicobot_bp)