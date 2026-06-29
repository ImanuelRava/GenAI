"""
Public helper functions for LLM interactions.

These are the functions imported and used by route / chat modules.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from .factory import LLMProviderFactory
from .base import BaseLLMProvider
from .prompts import KNOWLEDGE_GRAPH_SYSTEM_PROMPT, EXPLAIN_SYSTEM_PROMPT

from core.utils import sanitize_input

logger = logging.getLogger(__name__)


def get_llm_response(
    system_prompt: str,
    user_message: str,
    provider: str = None,
    api_key: str = None,
    model: str = None,
    messages: List[Dict[str, str]] = None,
    **kwargs,
) -> Optional[str]:
    """Single-turn LLM call (sync).

    If *messages* is provided the full conversation history is sent to the
    provider (multi-turn support).  Otherwise falls back to the simple
    system_prompt + user_message interface.
    """
    system_prompt = sanitize_input(system_prompt, max_length=4000)
    user_message = sanitize_input(user_message, max_length=2000)

    if provider is None:
        provider = LLMProviderFactory.get_default_provider()

    try:
        constructor_kwargs = {}
        if api_key:
            constructor_kwargs['api_key'] = api_key
        if model:
            constructor_kwargs['model'] = model

        llm: BaseLLMProvider = LLMProviderFactory.create(provider, **constructor_kwargs)

        chat_kwargs = {}
        if 'temperature' in kwargs:
            chat_kwargs['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            chat_kwargs['max_tokens'] = kwargs['max_tokens']

        if messages:
            return llm.chat_with_messages(messages, **chat_kwargs)

        return llm.chat(system_prompt, user_message, **chat_kwargs)
    except (ValueError, KeyError, TypeError, ConnectionError) as e:
        # Catch only the expected failure modes from LLM provider calls —
        # network/HTTP errors are already caught inside llm.chat(), so this
        # outer guard is for unexpected bugs in payload building or message
        # extraction. Other exceptions (e.g. AttributeError from a malformed
        # provider instance) should propagate so they're visible.
        logger.error(f"LLM Error: {e}")
        return None


async def get_llm_response_async(
    system_prompt: str,
    user_message: str,
    provider: str = None,
    api_key: str = None,
    model: str = None,
    messages: List[Dict[str, str]] = None,
    **kwargs,
) -> Optional[str]:
    """Single-turn LLM call (async).

    If *messages* is provided the full conversation history is sent to the
    provider (multi-turn support).
    """
    system_prompt = sanitize_input(system_prompt, max_length=4000)
    user_message = sanitize_input(user_message, max_length=2000)

    if provider is None:
        provider = LLMProviderFactory.get_default_provider()

    try:
        constructor_kwargs = {}
        if api_key:
            constructor_kwargs['api_key'] = api_key
        if model:
            constructor_kwargs['model'] = model

        llm: BaseLLMProvider = LLMProviderFactory.create(provider, **constructor_kwargs)

        chat_kwargs = {}
        if 'temperature' in kwargs:
            chat_kwargs['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            chat_kwargs['max_tokens'] = kwargs['max_tokens']

        if messages:
            return await llm.chat_with_messages_async(messages, **chat_kwargs)

        return await llm.chat_async(system_prompt, user_message, **chat_kwargs)
    except (ValueError, KeyError, TypeError, ConnectionError) as e:
        logger.error(f"LLM Async Error: {e}")
        return None


def _parse_kg_json(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from a knowledge-graph LLM response, stripping markdown fences."""
    if not response:
        return None
    try:
        json_str = response.strip()
        if json_str.startswith('```'):
            lines = json_str.split('\n')
            json_str = '\n'.join(lines[1:-1] if lines[-1] == '```' else lines[1:])
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}")
        logger.debug(f"Response was: {response[:500]}...")
    return None


def generate_knowledge_graph(
    topic: str,
    provider: str = None,
    api_key: str = None,
) -> Optional[Dict[str, Any]]:
    topic = sanitize_input(topic, max_length=500)
    user_message = f"Generate a knowledge graph for: {topic}"
    response = get_llm_response(KNOWLEDGE_GRAPH_SYSTEM_PROMPT, user_message,
                                provider=provider, api_key=api_key)
    return _parse_kg_json(response)


async def generate_knowledge_graph_async(
    topic: str,
    provider: str = None,
    api_key: str = None,
) -> Optional[Dict[str, Any]]:
    topic = sanitize_input(topic, max_length=500)
    user_message = f"Generate a knowledge graph for: {topic}"
    response = await get_llm_response_async(
        KNOWLEDGE_GRAPH_SYSTEM_PROMPT, user_message,
        provider=provider, api_key=api_key,
    )
    return _parse_kg_json(response)


def explain_concept(
    concept: str,
    context: str = "",
    provider: str = None,
    api_key: str = None,
) -> Optional[str]:
    concept = sanitize_input(concept, max_length=200)
    context = sanitize_input(context, max_length=500)
    user_message = f"Explain {concept} in the context of transition metal catalysis. Context: {context}"
    return get_llm_response(EXPLAIN_SYSTEM_PROMPT, user_message,
                            provider=provider, api_key=api_key)


async def explain_concept_async(
    concept: str,
    context: str = "",
    provider: str = None,
    api_key: str = None,
) -> Optional[str]:
    concept = sanitize_input(concept, max_length=200)
    context = sanitize_input(context, max_length=500)
    user_message = f"Explain {concept} in the context of transition metal catalysis. Context: {context}"
    return await get_llm_response_async(EXPLAIN_SYSTEM_PROMPT, user_message,
                                       provider=provider, api_key=api_key)
