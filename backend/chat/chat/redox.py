"""
Redox-Active Ligands chat blueprint — sync and async endpoints.

Wire RAL RAG into the chat flow (mirrors the NiCOBot pattern).
"""

import logging
from flask import Blueprint, request, jsonify

from core.errors import ValidationError
from llm import get_llm_response, get_llm_response_async
from llm.prompts import REDOX_SYSTEM_PROMPT

from .helpers import (
    extract_chat_params,
    build_response_json,
)
from .conversation import conversation_store

logger = logging.getLogger(__name__)

redox_bp = Blueprint('redox', __name__)

# System prompt used when database context is available.
# The database is the PRIMARY reference, but the LLM is allowed to
# supplement with domain knowledge for mechanistic reasoning.
_REDOX_RAG_SYSTEM_PROMPT = (
    REDOX_SYSTEM_PROMPT + "\n\n"
    "DATABASE-GROUNDED ANSWERING RULES:\n"
    "1. Use the database context as your PRIMARY reference point. "
    "Quote exact HOMO, LUMO, Gap, and omega values from the database when available. "
    "Cite specific DOIs when discussing reactions from the database.\n"
    "2. You MAY supplement with your domain knowledge to provide: "
    "mechanistic explanations, electronic-structure reasoning, "
    "coordination-chemistry principles, and literature context beyond the database.\n"
    "3. When the database provides quantitative data (yields, electronic parameters), "
    "always present those numbers first, then add your interpretive analysis.\n"
    "4. When comparing ligands, discuss ALL ligand classes mentioned by the user. "
    "Do NOT dismiss a ligand class as having 'no data' unless its section in the "
    "database context is genuinely empty."
)


def _retrieve_rag_context(message: str,
                          log_prefix: str = "RAL-Bot"):
    """Retrieve RAG context for a user message.

    Returns (formatted_context_string_or_None, rag_stats_dict_or_None).
    Never raises — falls back to (None, None) on any error.
    """
    try:
        from modules.ral_rag import get_ral_rag

        rag = get_ral_rag()
        context = rag.retrieve_context(message)

        if context.formatted_context:
            stats = {
                'ligands': len(context.ligands),
                'reactions': len(context.reactions),
                'class': context.detected_class or 'none',
            }
            logger.info(
                f"[{log_prefix}] RAG context retrieved: "
                f"{stats['ligands']} ligands, {stats['reactions']} reactions, "
                f"class={stats['class']}"
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
                    log_prefix: str = "RAL-Bot"):
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
        system_prompt = _REDOX_RAG_SYSTEM_PROMPT
        # Prepend database context directly into the user message — this is
        # the most reliable way to force the LLM to use RAG data.
        rag_user_message = (
            f"[The following data was retrieved from the RAL research database. "
            f"Treat this as your primary reference — quote specific values and DOIs. "
            f"You may supplement with your chemistry knowledge for reasoning.]\n\n"
            f"{database_context}\n\n"
            f"[End of database context]\n\n"
            f"{message}"
        )
    else:
        system_prompt = REDOX_SYSTEM_PROMPT
        rag_user_message = message

    # Build full messages list with conversation history
    history = conversation_store.get_history(conversation_id)
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(history)
    messages.append({'role': 'user', 'content': rag_user_message})

    return messages, system_prompt, database_context


def register_redox_blueprint(app, limiter, backend_dir=None, base_dir=None):
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
                f"[RAL Bot] Provider: {provider}, RAG: {use_rag}"
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
                conversation_id=conversation_id,
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("RAL chat error", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error communicating with LLM. Check server logs for details.'
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
                f"[RAL Bot Async] Provider: {provider}, RAG: {use_rag}"
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
                is_async=True, conversation_id=conversation_id,
            )

        except ValidationError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("RAL async chat error", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Error communicating with LLM. Check server logs for details.'
            }), 500

    app.register_blueprint(redox_bp)