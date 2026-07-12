"""
Concrete LLM Provider implementations (thin shims over ``LLMClient``).

These classes preserve the original public API (constructor signature,
``chat()``, ``chat_async()``, ``chat_with_messages()``,
``chat_with_messages_async()``) so that ``llm/factory.py``, ``llm/helpers.py``,
and all downstream callers continue to work unchanged.

All actual HTTP work is delegated to ``llm.client.LLMClient``, which is
the single source of truth for provider request/response handling.

Historical note: prior to v2.3.0, each provider class contained its own
copy of the HTTP request/response logic, with near-identical boilerplate
duplicated across DeepSeek, OpenAI, Groq, and OpenRouter (all OpenAI-
compatible). Anthropic, Gemini, HuggingFace, and Ollama each had their
own bespoke implementations. This file is now ~80 LOC instead of ~350 LOC,
and the same logic is shared with the chemextract and reaction pipelines.
"""

import logging
from typing import Optional, List, Dict

from .base import BaseLLMProvider
from .client import LLMClient

logger = logging.getLogger(__name__)


class _LLMClientBackedProvider(BaseLLMProvider):
    """Base class for providers that delegate to LLMClient.

    Subclasses just declare the provider name; everything else is handled
    by this base + LLMClient.
    """

    # Subclasses override this.
    _PROVIDER_NAME: str = ''

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        # Stash the constructor args so introspection works (some tests
        # and routes inspect provider.api_key / provider.model directly).
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # Construct the underlying client. LLMClient resolves env-var
        # fallbacks for api_key / model / base_url.
        self._client = LLMClient(
            provider=self._PROVIDER_NAME,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
        # Sync visible attrs with what LLMClient resolved (so callers that
        # inspect provider.api_key see the env-var-resolved value too).
        self.api_key = self._client.api_key
        self.base_url = self._client.base_url
        self.model = self._client.model

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self._client.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
        )

    async def chat_async(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return await self._client.chat_async(
            messages, temperature=temperature, max_tokens=max_tokens,
        )

    # Multi-turn methods inherited from BaseLLMProvider — they call
    # chat()/chat_async() with the extracted system+user, which is fine
    # for the chat use case. If we later want true multi-turn for
    # OpenAI-compatible providers (which support it natively), we can
    # override chat_with_messages here to pass the full messages list
    # straight through to LLMClient.chat().
    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Override to pass the full messages list to LLMClient (native
        multi-turn support for OpenAI-compatible providers)."""
        return self._client.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
        )

    async def chat_with_messages_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        return await self._client.chat_async(
            messages, temperature=temperature, max_tokens=max_tokens,
        )


# ---------------------------------------------------------------------------
# Concrete providers — each is a one-liner declaring the provider name.
# ---------------------------------------------------------------------------

class DeepSeekProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'deepseek'


class OpenAIProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'openai'


class GroqProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'groq'


class OpenRouterProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'openrouter'


class AnthropicProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'anthropic'


class GeminiProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'gemini'


class HuggingFaceProvider(_LLMClientBackedProvider):
    _PROVIDER_NAME = 'huggingface'


class OllamaProvider(_LLMClientBackedProvider):
    """Ollama provider — accepts ``base_url`` instead of ``api_key`` in
    its constructor signature for backwards compat with the original
    OllamaProvider (which didn't take an api_key)."""

    _PROVIDER_NAME = 'ollama'

    def __init__(self, base_url: str = None, model: str = None, api_key: str = None):
        # Ignore api_key (Ollama is local, no auth) but accept the kwarg
        # so callers using the factory's uniform **kwargs don't break.
        super().__init__(api_key=None, model=model, base_url=base_url)
