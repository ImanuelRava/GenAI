"""
ChemExtract LLM provider shims (delegating to ``llm.client.LLMClient``).

This file preserves the original public function signatures used by
``chemextract/extractor.py`` and ``chemextract/standalone.py``:

  - ``call_vision_llm(base64_image, provider, model, api_key, system_prompt, user_message)``
  - ``call_vision_llm_async(...)``
  - ``_call_text_provider(provider, model, api_key, text)``
  - ``_call_text_provider_async(...)``
  - ``_call_gemini_text(text, model, api_key)``
  - ``_call_gemini_text_async(...)``
  - ``_call_anthropic_text(text, model, api_key)``
  - ``_call_anthropic_text_async(...)``
  - ``_retry_on_failure(func, *args, **kwargs)``
  - ``_retry_on_failure_async(func, *args, **kwargs)``

All HTTP work is delegated to ``llm.client.LLMClient``, which is shared
with the chat providers and the ReactionLens pipeline. The retry helpers
delegate to ``llm.client.retry_with_backoff[_async]``.

Pre-consolidation: ~523 LOC with 16 near-identical functions
(8 vision + 8 text, each duplicated for sync/async).
Post-consolidation: ~150 LOC of thin shims.
"""

import logging
from typing import Optional, Dict

from llm.client import (
    LLMClient,
    VISION_CAPABLE_PROVIDERS,
    retry_with_backoff,
    retry_with_backoff_async,
)

from .config import (
    MAX_OUTPUT_TOKENS, MAX_RETRIES, RETRY_DELAY,
    EXTRACTION_TEMPERATURE, EXTRACTION_SEED,
)
from .prompts import SYSTEM_PROMPT_COMPREHENSIVE
from .json_utils import _parse_json_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-exported retry helpers (backwards compat with chemextract/__init__.py)
# ---------------------------------------------------------------------------

def _retry_on_failure(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Retry a sync function on failure with linear back-off.

    Delegates to llm.client.retry_with_backoff to keep retry logic in one place.
    """
    return retry_with_backoff(
        func, *args,
        max_retries=max_retries,
        retry_delay=RETRY_DELAY,
        **kwargs,
    )


async def _retry_on_failure_async(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Async retry wrapper. Delegates to llm.client.retry_with_backoff_async."""
    return await retry_with_backoff_async(
        func, *args,
        max_retries=max_retries,
        retry_delay=RETRY_DELAY,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Vision LLM calls
# ---------------------------------------------------------------------------

def call_vision_llm(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Call a vision-capable LLM and return parsed JSON, or None on failure.

    Returns the parsed JSON dict (not the raw text) to preserve the original
    function's contract — callers in extractor.py expect a dict.
    """
    try:
        if provider not in VISION_CAPABLE_PROVIDERS:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None

        client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ]

        def _do_call():
            text = client.vision(
                messages,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                seed=EXTRACTION_SEED,
            )
            if text is None:
                return None
            return _parse_json_response(text)

        return _retry_on_failure(_do_call)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        # JSON parse errors, unexpected response shapes, or retry-exhausted
        # RuntimeError from _retry_on_failure. VisionNotSupportedError is
        # raised before the try block, so it propagates correctly.
        logger.error(f"[ChemExtract] Vision LLM call failed: {e}")
        return None


async def call_vision_llm_async(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Async vision LLM call. See ``call_vision_llm``."""
    try:
        if provider not in VISION_CAPABLE_PROVIDERS:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None

        client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            },
        ]

        async def _do_call():
            text = await client.vision_async(
                messages,
                temperature=EXTRACTION_TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                response_format={"type": "json_object"},
                seed=EXTRACTION_SEED,
            )
            if text is None:
                return None
            return _parse_json_response(text)

        return await _retry_on_failure_async(_do_call)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error(f"[ChemExtract] Async vision LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Text LLM calls
# ---------------------------------------------------------------------------

def _build_text_messages(text: str):
    """Build the standard ChemExtract text-extraction messages list."""
    user_content = (
        "Extract ALL chemical reactions and compounds from this text. "
        "List EVERY reaction as a separate entry:\n\n" + text
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
        {"role": "user", "content": user_content},
    ]


def _call_text_provider(
    provider: str,
    model: str,
    api_key: str,
    text: str,
) -> Optional[Dict]:
    """Text LLM call for OpenAI-compatible providers (deepseek, openai).

    Delegates to LLMClient.chat with JSON-mode response_format.
    """
    client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    def _do_call():
        result = client.chat(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return _retry_on_failure(_do_call)


async def _call_text_provider_async(
    provider: str,
    model: str,
    api_key: str,
    text: str,
) -> Optional[Dict]:
    """Async text LLM call for OpenAI-compatible providers."""
    client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    async def _do_call():
        result = await client.chat_async(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return await _retry_on_failure_async(_do_call)


def _call_gemini_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Text LLM call for Gemini. Delegates to LLMClient.chat."""
    client = LLMClient(provider='gemini', api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    def _do_call():
        result = client.chat(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},  # LLMClient translates to responseMimeType
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return _retry_on_failure(_do_call)


async def _call_gemini_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Async text LLM call for Gemini."""
    client = LLMClient(provider='gemini', api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    async def _do_call():
        result = await client.chat_async(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return await _retry_on_failure_async(_do_call)


def _call_anthropic_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Text LLM call for Anthropic. Delegates to LLMClient.chat.

    Note: Anthropic doesn't support response_format natively, so JSON mode
    is requested via the prompt (SYSTEM_PROMPT_COMPREHENSIVE instructs
    JSON output) and the response is parsed with _parse_json_response.
    """
    client = LLMClient(provider='anthropic', api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    def _do_call():
        result = client.chat(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return _retry_on_failure(_do_call)


async def _call_anthropic_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Async text LLM call for Anthropic."""
    client = LLMClient(provider='anthropic', api_key=api_key, model=model, timeout=300)
    messages = _build_text_messages(text)

    async def _do_call():
        result = await client.chat_async(
            messages,
            temperature=EXTRACTION_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            seed=EXTRACTION_SEED,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return await _retry_on_failure_async(_do_call)
