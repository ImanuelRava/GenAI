"""
ReactionLens provider shims (delegating to ``llm.client.LLMClient``).

Preserves the original public API used by ``reaction/extraction.py``:

  - ``rl_call_text(text, provider, model, api_key, system_prompt)``
  - ``rl_call_text_async(text, provider, model, api_key, system_prompt)``
  - ``_rl_retry(func, *args, **kwargs)``
  - ``_rl_retry_async(func, *args, **kwargs)``

All HTTP work is delegated to ``llm.client.LLMClient``. The retry helpers
delegate to ``llm.client.retry_with_backoff[_async]``.

Pre-consolidation: ~191 LOC with 8 near-identical per-provider functions
(4 providers × sync/async) plus retry helpers.
Post-consolidation: ~80 LOC of thin shims.
"""

import asyncio
import logging
from typing import Optional, Dict

from llm.client import (
    LLMClient,
    retry_with_backoff,
    retry_with_backoff_async,
)

from .parsing import _parse_json_response
from .prompts import RL_MAX_RETRIES, RL_RETRY_DELAY, RL_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helpers (backwards compat — same names as the original module)
# ---------------------------------------------------------------------------

def _rl_retry(func, *args, max_retries: int = RL_MAX_RETRIES, **kwargs):
    """Synchronous retry with linear back-off.

    Delegates to llm.client.retry_with_backoff to keep retry logic in one place.
    """
    return retry_with_backoff(
        func, *args,
        max_retries=max_retries,
        retry_delay=RL_RETRY_DELAY,
        **kwargs,
    )


async def _rl_retry_async(func, *args, max_retries: int = RL_MAX_RETRIES, **kwargs):
    """Async retry with linear back-off. Delegates to llm.client.retry_with_backoff_async."""
    return await retry_with_backoff_async(
        func, *args,
        max_retries=max_retries,
        retry_delay=RL_RETRY_DELAY,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Unified text-LLM dispatch
# ---------------------------------------------------------------------------

# ReactionLens supports these 4 text providers (originally hardcoded in
# _text_deepseek / _text_openai / _text_gemini / _text_anthropic).
RL_SUPPORTED_PROVIDERS = ('deepseek', 'openai', 'gemini', 'anthropic')


def rl_call_text(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
) -> Optional[Dict]:
    """Send a text-extraction request to the named provider.

    Returns the parsed JSON dict (ReactionLens schema), or None on failure.
    """
    if provider not in RL_SUPPORTED_PROVIDERS:
        logger.error(f"[ReactionLens] Unsupported provider: {provider}")
        return None

    client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    def _do_call():
        result = client.chat(
            messages,
            temperature=0.1,  # ReactionLens uses low temperature for determinism
            max_tokens=RL_MAX_OUTPUT_TOKENS,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return _rl_retry(_do_call)


async def rl_call_text_async(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
) -> Optional[Dict]:
    """Async version of ``rl_call_text``."""
    if provider not in RL_SUPPORTED_PROVIDERS:
        logger.error(f"[ReactionLens] Unsupported provider: {provider}")
        return None

    client = LLMClient(provider=provider, api_key=api_key, model=model, timeout=300)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    async def _do_call():
        result = await client.chat_async(
            messages,
            temperature=0.1,
            max_tokens=RL_MAX_OUTPUT_TOKENS,
        )
        if result is None:
            return None
        return _parse_json_response(result)

    return await _rl_retry_async(_do_call)
