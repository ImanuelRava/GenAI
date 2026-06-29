"""
Unified LLM Client — single source of truth for all LLM provider calls.

This module replaces three previously-duplicated provider implementations:
  - backend/llm/providers.py            (chat, 8 providers, sync+async)
  - backend/modules/chemextract/llm_providers.py  (vision + text, 4 providers)
  - backend/modules/reaction/providers.py         (text, 4 providers)

All three are now thin shims that delegate to ``LLMClient`` defined here.

Design
------
``LLMClient`` is constructed per-call with (provider, api_key, model) and
exposes two methods, each in sync + async variants:

  - ``chat(messages, ...)``       — text chat. ``messages`` is a list of
    ``{"role": ..., "content": ...}`` dicts. Multi-turn conversations are
    passed natively to providers that support them; for providers that
    only accept a flat prompt (HuggingFace, legacy Gemini), the messages
    are flattened into a single prompt string.

  - ``vision(messages, ...)``     — vision-capable chat. ``messages`` follows
    the OpenAI multimodal format: user-message content may be a list of
    ``{"type": "text"|"image_url", ...}`` parts. The client translates this
    to each provider's native vision format (OpenAI/DeepSeek use image_url,
    Gemini uses inline_data, Anthropic uses image source).

Both methods return the raw text content from the model, or ``None`` on
error. JSON-mode responses are returned as raw text — callers that want
parsed JSON wrap the call with their own JSON parser (the codebase has
three slightly different parsers; keeping them separate is intentional
because each handles slightly different edge cases).

Provider capabilities
---------------------
VISION_CAPABLE_PROVIDERS lists the providers that support vision input.
``LLMClient.vision()`` raises ``ValueError`` for non-vision providers.

The ``response_format`` kwarg is honored for providers that support JSON
mode (OpenAI, DeepSeek, Gemini). For providers that don't (Anthropic,
Groq, Ollama, HuggingFace, OpenRouter), it is silently ignored.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

# Providers that support vision input (image + text in the same message).
# Used by routes/data_extraction.py VISION_CAPABLE_PROVIDERS and by
# ChemExtractAI.vision_providers — kept here as the single source of truth.
VISION_CAPABLE_PROVIDERS: List[str] = ['deepseek', 'openai', 'gemini', 'anthropic']

# Default models per provider (single source of truth — replaces three
# separate default-model dicts that previously lived in providers.py,
# chemextract/extractor.py, and routes/data_extraction.py).
PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    'deepseek': 'deepseek-chat',
    'openai': 'gpt-4o-mini',
    'anthropic': 'claude-3-haiku-20240307',
    'gemini': 'gemini-2.0-flash',
    'groq': 'llama-3.3-70b-versatile',
    'openrouter': 'meta-llama/llama-3-8b-instruct:free',
    'huggingface': 'meta-llama/Llama-3.2-3B-Instruct',
    'ollama': 'llama3',
}

# Default base URLs per provider. Can be overridden via env vars
# (e.g. DEEPSEEK_BASE_URL, OLLAMA_BASE_URL). Providers marked None have
# URLs that are built dynamically per-request (Gemini) or per-model
# (HuggingFace), so they don't have a single base URL.
PROVIDER_BASE_URLS: Dict[str, Optional[str]] = {
    'deepseek': 'https://api.deepseek.com/v1',
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com/v1',
    'gemini': None,  # URL is built per-model in _build_gemini().
    'groq': 'https://api.groq.com/openai/v1',
    'openrouter': 'https://openrouter.ai/api/v1',
    'huggingface': 'https://api-inference.huggingface.co/models',
    'ollama': 'http://localhost:11434',
}

# API "style" — determines how the request payload and response are shaped.
# - 'openai_compat': DeepSeek, OpenAI, Groq, OpenRouter all use the
#   /chat/completions endpoint with identical payload shape.
# - 'anthropic': Anthropic uses /messages with x-api-key auth and a
#   different payload shape (system as a top-level field, no system message).
# - 'gemini': Google's generateContent endpoint with contents/parts.
# - 'huggingface': HF inference API with a single 'inputs' string.
# - 'ollama': Local Ollama /api/chat endpoint.
PROVIDER_API_STYLES: Dict[str, str] = {
    'deepseek': 'openai_compat',
    'openai': 'openai_compat',
    'groq': 'openai_compat',
    'openrouter': 'openai_compat',
    'anthropic': 'anthropic',
    'gemini': 'gemini',
    'huggingface': 'huggingface',
    'ollama': 'ollama',
}

# Env var that holds the API key for each provider.
PROVIDER_ENV_KEY: Dict[str, List[str]] = {
    'deepseek': ['DEEPSEEK_API_KEY'],
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    'gemini': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'openrouter': ['OPENROUTER_API_KEY'],
    'huggingface': ['HF_API_KEY', 'HUGGINGFACE_API_KEY'],
    'ollama': [],  # No API key — local.
}

# Env var that overrides the base URL for each provider.
PROVIDER_BASE_URL_ENV: Dict[str, str] = {
    'deepseek': 'DEEPSEEK_BASE_URL',
    'openai': 'OPENAI_BASE_URL',
    'anthropic': '',  # Fixed URL.
    'gemini': '',     # Fixed URL.
    'groq': '',       # Fixed URL.
    'openrouter': '', # Fixed URL.
    'huggingface': '',# URL is model-specific.
    'ollama': 'OLLAMA_BASE_URL',
}

# Env var that overrides the default model for each provider.
PROVIDER_MODEL_ENV: Dict[str, str] = {
    'deepseek': 'DEEPSEEK_MODEL',
    'openai': 'OPENAI_MODEL',
    'anthropic': 'ANTHROPIC_MODEL',
    'gemini': 'GEMINI_MODEL',
    'groq': 'GROQ_MODEL',
    'openrouter': 'OPENROUTER_MODEL',
    'huggingface': 'HF_MODEL',
    'ollama': 'OLLAMA_MODEL',
}

ALL_PROVIDERS: List[str] = list(PROVIDER_DEFAULT_MODELS.keys())


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Base error for LLMClient."""


class UnsupportedProviderError(LLMClientError):
    """Raised when an unknown provider name is passed."""


class VisionNotSupportedError(LLMClientError):
    """Raised when vision() is called on a non-vision-capable provider."""


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified client for all 8 supported LLM providers.

    Construct one client per call (they're cheap — just config, no
    persistent connections). Sync and async methods share payload-building
    logic via private helpers.

    Example
    -------
    >>> client = LLMClient(provider='deepseek', api_key='sk-...', model='deepseek-chat')
    >>> text = client.chat([
    ...     {"role": "system", "content": "You are a chemist."},
    ...     {"role": "user", "content": "What is Suzuki coupling?"},
    ... ], temperature=0.7, max_tokens=2000)

    >>> # Vision:
    >>> text = client.vision([
    ...     {"role": "system", "content": "Extract reactions from this image."},
    ...     {"role": "user", "content": [
    ...         {"type": "text", "text": "What reactions are shown?"},
    ...         {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ...     ]},
    ... ], response_format={"type": "json_object"})
    """

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        provider_lower = provider.lower()
        if provider_lower not in PROVIDER_DEFAULT_MODELS:
            raise UnsupportedProviderError(
                f"Unknown provider: {provider}. "
                f"Available: {ALL_PROVIDERS}"
            )

        self.provider = provider_lower
        self.api_style = PROVIDER_API_STYLES[provider_lower]
        self.timeout = timeout

        # Resolve API key from arg or env.
        self.api_key = api_key or self._env_first(PROVIDER_ENV_KEY[provider_lower])

        # Resolve model from arg, env, or default.
        self.model = (
            model
            or self._env_first([PROVIDER_MODEL_ENV[provider_lower]])
            or PROVIDER_DEFAULT_MODELS[provider_lower]
        )

        # Resolve base URL.
        # - If caller passed base_url, use it.
        # - Else if provider has an env-override var set (e.g. OLLAMA_BASE_URL), use it.
        # - Else fall back to the provider's default base URL (may be None
        #   for providers like Gemini whose URL is built per-model).
        default_base_url = PROVIDER_BASE_URLS[provider_lower]
        base_url_env_var = PROVIDER_BASE_URL_ENV[provider_lower]
        if base_url:
            self.base_url = base_url
        elif base_url_env_var and os.environ.get(base_url_env_var):
            self.base_url = os.environ.get(base_url_env_var)
        else:
            self.base_url = default_base_url  # may be None for gemini/huggingface

    # ------------------------------------------------------------------
    # Public: text chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Synchronous text chat. Returns raw model output, or None on error."""
        url, headers, payload = self._build_text_request(
            messages, temperature, max_tokens, response_format, seed, extra_headers,
        )
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            return self._parse_text_response(response)
        except requests.exceptions.Timeout:
            logger.error(f"[LLMClient:{self.provider}] request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[LLMClient:{self.provider}] request error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            # JSON decode errors (ValueError), unexpected response shapes
            # (KeyError), or None where a dict was expected (TypeError).
            logger.error(f"[LLMClient:{self.provider}] response parse error: {e}")
            return None

    async def chat_async(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Async text chat. Returns raw model output, or None on error."""
        url, headers, payload = self._build_text_request(
            messages, temperature, max_tokens, response_format, seed, extra_headers,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(
                            f"[LLMClient:{self.provider}] API error: "
                            f"{response.status} - {text[:200]}"
                        )
                        return None
                    data = await response.json()
                    return self._extract_text_from_response(data)
        except asyncio.TimeoutError:
            logger.error(f"[LLMClient:{self.provider}] async request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"[LLMClient:{self.provider}] async client error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            # JSON decode errors (ValueError), unexpected response shapes
            # (KeyError), or None where a dict was expected (TypeError).
            logger.error(f"[LLMClient:{self.provider}] async response parse error: {e}")
            return None

    # ------------------------------------------------------------------
    # Public: vision chat
    # ------------------------------------------------------------------

    def vision(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Synchronous vision chat. ``messages`` follows OpenAI multimodal format.

        Raises ``VisionNotSupportedError`` if the provider doesn't support vision.
        """
        if self.provider not in VISION_CAPABLE_PROVIDERS:
            raise VisionNotSupportedError(
                f"Provider '{self.provider}' does not support vision input. "
                f"Vision-capable providers: {VISION_CAPABLE_PROVIDERS}"
            )
        url, headers, payload = self._build_vision_request(
            messages, temperature, max_tokens, response_format, seed, extra_headers,
        )
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            return self._parse_text_response(response)
        except requests.exceptions.Timeout:
            logger.error(f"[LLMClient:{self.provider}] vision request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[LLMClient:{self.provider}] vision request error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[LLMClient:{self.provider}] vision response parse error: {e}")
            return None

    async def vision_async(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Async vision chat. See ``vision()`` for the messages format."""
        if self.provider not in VISION_CAPABLE_PROVIDERS:
            raise VisionNotSupportedError(
                f"Provider '{self.provider}' does not support vision input. "
                f"Vision-capable providers: {VISION_CAPABLE_PROVIDERS}"
            )
        url, headers, payload = self._build_vision_request(
            messages, temperature, max_tokens, response_format, seed, extra_headers,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(
                            f"[LLMClient:{self.provider}] vision API error: "
                            f"{response.status} - {text[:200]}"
                        )
                        return None
                    data = await response.json()
                    return self._extract_text_from_response(data)
        except asyncio.TimeoutError:
            logger.error(f"[LLMClient:{self.provider}] async vision request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"[LLMClient:{self.provider}] async vision client error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"[LLMClient:{self.provider}] async vision parse error: {e}")
            return None

    # ------------------------------------------------------------------
    # Capability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def supports_vision(provider: str) -> bool:
        """Return True if the given provider supports vision input."""
        return provider.lower() in VISION_CAPABLE_PROVIDERS

    # ------------------------------------------------------------------
    # Internals: request building
    # ------------------------------------------------------------------

    @staticmethod
    def _env_first(env_vars: List[str]) -> Optional[str]:
        """Return the first non-empty env var from the list."""
        for var in env_vars:
            val = os.environ.get(var)
            if val:
                return val
        return None

    def _build_text_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        extra_headers: Optional[Dict[str, str]],
    ) -> tuple:
        """Build (url, headers, payload) for a text chat request."""
        if self.api_style == 'openai_compat':
            return self._build_openai_compat(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
            )
        elif self.api_style == 'anthropic':
            return self._build_anthropic(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
                is_vision=False,
            )
        elif self.api_style == 'gemini':
            return self._build_gemini(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
                is_vision=False,
            )
        elif self.api_style == 'huggingface':
            return self._build_huggingface(
                messages, temperature, max_tokens, extra_headers,
            )
        elif self.api_style == 'ollama':
            return self._build_ollama(
                messages, temperature, max_tokens, extra_headers,
            )
        raise UnsupportedProviderError(f"Unknown API style: {self.api_style}")

    def _build_vision_request(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        extra_headers: Optional[Dict[str, str]],
    ) -> tuple:
        """Build (url, headers, payload) for a vision chat request."""
        if self.api_style == 'openai_compat':
            return self._build_openai_compat(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
            )
        elif self.api_style == 'anthropic':
            return self._build_anthropic(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
                is_vision=True,
            )
        elif self.api_style == 'gemini':
            return self._build_gemini(
                messages, temperature, max_tokens, response_format, seed, extra_headers,
                is_vision=True,
            )
        # HuggingFace and Ollama don't support vision in our abstraction.
        raise VisionNotSupportedError(
            f"Provider '{self.provider}' (API style '{self.api_style}') "
            f"does not support vision input."
        )

    # --- OpenAI-compatible (DeepSeek, OpenAI, Groq, OpenRouter) ---

    def _build_openai_compat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        extra_headers: Optional[Dict[str, str]],
    ) -> tuple:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter requires extra attribution headers.
        if self.provider == 'openrouter':
            headers["HTTP-Referer"] = "https://genai-research.local"
            headers["X-Title"] = "GenAI Research"
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed

        return url, headers, payload

    # --- Anthropic ---

    def _build_anthropic(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        extra_headers: Optional[Dict[str, str]],
        is_vision: bool,
    ) -> tuple:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if is_vision:
            # Anthropic recommends prompt caching beta for vision-heavy workloads.
            headers["anthropic-beta"] = "prompt-caching-2024-07-31"
        if extra_headers:
            headers.update(extra_headers)

        # Anthropic takes the system prompt as a top-level field, not as a
        # message. Extract it from the messages list.
        system_prompt = ""
        conversation_messages: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m["content"] + "\n"
            else:
                # For vision messages, Anthropic expects content as a list
                # of typed parts. The OpenAI format already uses lists for
                # multimodal content, so we pass it through. For plain text,
                # we keep content as a string.
                content = m["content"]
                if isinstance(content, list):
                    # Translate OpenAI image_url parts to Anthropic image source.
                    translated_parts = []
                    for part in content:
                        if part.get("type") == "image_url":
                            image_url = part["image_url"]["url"]
                            # Extract base64 data from data URL.
                            # Format: "data:image/png;base64,<data>"
                            if image_url.startswith("data:"):
                                _, _, b64data = image_url.partition(",")
                                translated_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64data,
                                    },
                                })
                            else:
                                logger.warning(
                                    f"[LLMClient:anthropic] Non-data URL image "
                                    f"not supported, skipping."
                                )
                        elif part.get("type") == "text":
                            translated_parts.append({"type": "text", "text": part["text"]})
                    conversation_messages.append({"role": m["role"], "content": translated_parts})
                else:
                    conversation_messages.append({"role": m["role"], "content": content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt.strip(),
            "messages": conversation_messages,
        }
        # Anthropic doesn't support response_format natively, but we honor
        # seed if provided (Claude 3.5+ supports it).
        if seed is not None:
            payload["seed"] = seed

        return url, headers, payload

    # --- Gemini ---

    def _build_gemini(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]],
        seed: Optional[int],
        extra_headers: Optional[Dict[str, str]],
        is_vision: bool,
    ) -> tuple:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        # Gemini doesn't use Authorization headers — the key is in the URL.
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        # Gemini flattens system + user into a single 'contents' list with
        # parts. For multi-turn, we'd ideally preserve roles, but the
        # original code flattened everything into one prompt — we keep that
        # behavior for backwards compat.
        parts: List[Dict[str, Any]] = []
        system_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n\n"
            else:
                content = m["content"]
                if isinstance(content, list):
                    # Vision message — translate OpenAI parts to Gemini parts.
                    for part in content:
                        if part.get("type") == "text":
                            parts.append({"text": part["text"]})
                        elif part.get("type") == "image_url":
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                _, _, b64data = image_url.partition(",")
                                parts.append({
                                    "inline_data": {
                                        "mime_type": "image/png",
                                        "data": b64data,
                                    }
                                })
                else:
                    parts.append({"text": content})

        # Prepend the system text as the first text part.
        if system_text:
            parts.insert(0, {"text": system_text.strip()})

        generation_config: Dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if seed is not None:
            generation_config["seed"] = seed
        if response_format and response_format.get("type") == "json_object":
            generation_config["responseMimeType"] = "application/json"

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        }
        return url, headers, payload

    # --- HuggingFace ---

    def _build_huggingface(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra_headers: Optional[Dict[str, str]],
    ) -> tuple:
        # HF URL is per-model: <base>/<model>. self.base_url may be None
        # if not explicitly overridden, so fall back to the default.
        base = self.base_url or 'https://api-inference.huggingface.co/models'
        url = f"{base}/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        # HF uses a single prompt string with Llama-3 chat template.
        prompt = self._build_hf_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        return url, headers, payload

    @staticmethod
    def _build_hf_prompt(messages: List[Dict[str, Any]]) -> str:
        """Flatten messages into a Llama-3 chat-template prompt string."""
        parts = ["<|begin_of_text|>"]
        for m in messages:
            if m["role"] == "system":
                parts.append(f"{m['content']}<|eot_id|>")
            elif m["role"] == "user":
                parts.append(
                    f"<|start_header_id|>user<|end_header_id|>\n{m['content']}<|eot_id|>"
                )
            elif m["role"] == "assistant":
                parts.append(
                    f"<|start_header_id|>assistant<|end_header_id|>\n{m['content']}<|eot_id|>"
                )
        parts.append("<|start_header_id|>assistant<|end_header_id|>")
        return "\n".join(parts)

    # --- Ollama ---

    def _build_ollama(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        extra_headers: Optional[Dict[str, str]],
    ) -> tuple:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        return url, headers, payload

    # ------------------------------------------------------------------
    # Internals: response parsing
    # ------------------------------------------------------------------

    def _parse_text_response(self, response: requests.Response) -> Optional[str]:
        """Parse a sync requests.Response into text content."""
        if response.status_code != 200:
            logger.error(
                f"[LLMClient:{self.provider}] API error: "
                f"{response.status_code} - {response.text[:200]}"
            )
            return None
        try:
            data = response.json()
        except ValueError:
            logger.error(f"[LLMClient:{self.provider}] non-JSON response: {response.text[:200]}")
            return None
        return self._extract_text_from_response(data)

    def _extract_text_from_response(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract the text content from a parsed JSON response, dispatching
        by provider API style. Handles both text and vision responses."""
        if self.api_style == 'openai_compat':
            # /chat/completions: choices[0].message.content
            try:
                return data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                logger.error(f"[LLMClient:{self.provider}] unexpected response shape: {str(data)[:200]}")
                return None
        elif self.api_style == 'anthropic':
            # /messages: content[0].text
            try:
                return data["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                logger.error(f"[LLMClient:{self.provider}] unexpected response shape: {str(data)[:200]}")
                return None
        elif self.api_style == 'gemini':
            # generateContent: candidates[0].content.parts[0].text
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                logger.error(f"[LLMClient:{self.provider}] unexpected response shape: {str(data)[:200]}")
                return None
        elif self.api_style == 'huggingface':
            # HF returns either a list with generated_text or a dict.
            if isinstance(data, list) and len(data) > 0:
                return data[0].get('generated_text', '')
            if isinstance(data, dict):
                if data.get("error"):
                    # HF returns 503 with {"error": "..."} when model is loading.
                    logger.warning(f"[LLMClient:huggingface] {data['error']}")
                    return None
                return data.get('generated_text', '')
            return None
        elif self.api_style == 'ollama':
            # /api/chat: message.content
            try:
                return data["message"]["content"]
            except (KeyError, TypeError):
                logger.error(f"[LLMClient:{self.provider}] unexpected response shape: {str(data)[:200]}")
                return None
        return None


# ---------------------------------------------------------------------------
# Retry helpers (shared by chemextract + reaction shims)
# ---------------------------------------------------------------------------

def retry_with_backoff(
    func,
    *args,
    max_retries: int = 3,
    retry_delay: float = 3.0,
    **kwargs,
):
    """Retry a sync callable on failure with linear back-off.

    Used by the chemextract and reaction shims to preserve their original
    retry semantics. The unified LLMClient itself does NOT retry — that's
    the caller's policy decision.

    Catches all exceptions except ``KeyboardInterrupt`` and ``SystemExit``
    (which propagate immediately so Ctrl+C / process shutdown work).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_delay * (attempt + 1)
                logger.warning(
                    f"[LLMClient] call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(f"[LLMClient] call failed after {max_retries + 1} attempts: {exc}")
    if last_exc:
        raise last_exc


async def retry_with_backoff_async(
    func,
    *args,
    max_retries: int = 3,
    retry_delay: float = 3.0,
    **kwargs,
):
    """Async retry with linear back-off. See ``retry_with_backoff``."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_delay * (attempt + 1)
                logger.warning(
                    f"[LLMClient] async call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"[LLMClient] async call failed after {max_retries + 1} attempts: {exc}")
    if last_exc:
        raise last_exc
