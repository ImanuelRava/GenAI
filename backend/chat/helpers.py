"""
Shared helpers for chat endpoints.
"""

import logging
from typing import Optional

from flask import jsonify

from core.errors import ValidationError
from core.config import config
from core.utils import sanitize_input

from llm.prompts import NICOBOT_SYSTEM_PROMPT, REDOX_SYSTEM_PROMPT, EXPLAIN_SYSTEM_PROMPT, PREDEFINED_EXPLANATIONS
from llm import get_llm_response, get_llm_response_async

from .conversation import conversation_store

logger = logging.getLogger(__name__)

# Lazy import guard for RAG
_rag_available = None
_ral_rag_available = None


def _is_rag_available():
    global _rag_available
    if _rag_available is None:
        try:
            from modules.nicobot_rag import get_rag
            _rag_available = True
        except ImportError:
            _rag_available = False
            logger.warning("NiCOBot RAG service not available.")
    return _rag_available


def _is_ral_rag_available():
    global _ral_rag_available
    if _ral_rag_available is None:
        try:
            from modules.ral_rag import get_ral_rag
            _ral_rag_available = True
        except ImportError:
            _ral_rag_available = False
            logger.warning("RAL RAG service not available.")
    return _ral_rag_available


def extract_chat_params(data: dict) -> tuple:
    """Extract and validate common chat parameters from request JSON."""
    if not data:
        raise ValidationError("No JSON data provided")

    message = sanitize_input(data.get('message', ''), max_length=config.MAX_PROMPT_LENGTH)
    if not message:
        raise ValidationError("Message cannot be empty", field="message")

    return (
        message,
        data.get('provider'),
        data.get('api_key'),
        data.get('model'),
    )


def build_nicobot_system_prompt(message: str, use_rag: bool,
                                  log_prefix: str = "NiCOBot") -> tuple:
    """Build NiCOBot system prompt with optional RAG context.

    Returns (system_prompt, database_context_or_None).
    """
    database_context = None
    if _is_rag_available() and use_rag:
        try:
            from modules.nicobot_rag import get_rag
            rag = get_rag()
            context = rag.retrieve_context(message)
            if context.formatted_context:
                database_context = context.formatted_context
                system_prompt = (
                    f"{NICOBOT_SYSTEM_PROMPT}\n\n"
                    f"The following information has been retrieved from the NiCOBot chemical database. "
                    f"Use this to enhance your response:\n\n"
                    f"{context.formatted_context}\n\n"
                    "When answering, reference specific compounds, papers, or data from the database "
                    "when relevant."
                )
                logger.info(
                    f"[{log_prefix}] RAG context retrieved: "
                    f"{len(context.compounds)} compounds, {len(context.papers)} papers"
                )
                return system_prompt, database_context
        except (ImportError, OSError, ValueError, KeyError, AttributeError) as e:
            # ImportError: nicobot_rag module not available. OSError: data
            # files missing. ValueError/KeyError/AttributeError: unexpected
            # data structure. Fall back to base prompt — chat continues.
            logger.warning(f"[{log_prefix}] RAG error: {e}. Falling back to base prompt.")

    return NICOBOT_SYSTEM_PROMPT, None


def build_redox_system_prompt(message: str, use_rag: bool,
                                log_prefix: str = "RAL-Bot") -> tuple:
    """Build Redox/RAL-Bot system prompt with optional RAG context.

    Returns (system_prompt, database_context_or_None).
    """
    database_context = None
    if _is_ral_rag_available() and use_rag:
        try:
            from modules.ral_rag import get_ral_rag
            rag = get_ral_rag()
            enhanced = rag.build_enhanced_prompt(message, REDOX_SYSTEM_PROMPT)
            # Check if RAG actually added data (enhanced will be longer than base)
            if len(enhanced) > len(REDOX_SYSTEM_PROMPT) + 50:
                database_context = rag.retrieve_context(message).formatted_context
                logger.info(
                    f"[{log_prefix}] RAG context retrieved: "
                    f"database-enhanced prompt generated"
                )
                return enhanced, database_context
        except (ImportError, OSError, ValueError, KeyError, AttributeError) as e:
            logger.warning(f"[{log_prefix}] RAG error: {e}. Falling back to base prompt.")

    return REDOX_SYSTEM_PROMPT, None


def build_response_json(
    response,
    provider,
    database_context=None,
    is_async=False,
    conversation_id=None,
):
    """Build a standardised JSON chat response."""
    if response:
        result = {
            'success': True,
            'response': response,
            'provider': provider or 'default',
        }
        if is_async:
            result['async'] = True
        if database_context:
            result['database_enhanced'] = True
        if conversation_id:
            result['conversation_id'] = conversation_id
        return jsonify(result)

    return jsonify({
        'success': False,
        'error': 'No response received from LLM. Please check your API key and provider settings.'
    }), 500


def get_explanation(node_label, context, is_async=False):
    """Shared logic for knowledge graph node explanation."""
    extra = {}
    if is_async:
        extra['async'] = True

    explanation = PREDEFINED_EXPLANATIONS.get(
        node_label.lower(),
        f'{node_label} is an important concept in transition metal catalysis. '
        f'It plays a crucial role in the catalytic cycle and influences reaction outcomes.'
    )

    return jsonify({
        'success': True,
        'node': node_label,
        'explanation': explanation,
        'source': 'predefined',
        **extra,
    })
