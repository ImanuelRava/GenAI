"""
Tests for the unified LLMClient (backend.llm.client).

These tests verify:
  - All 8 providers can be constructed and resolve env-var defaults correctly
  - The provider capability matrix (vision-capable vs text-only)
  - Request payload building for each API style (openai_compat, anthropic,
    gemini, huggingface, ollama) — both text and vision
  - Response parsing for each API style
  - Error handling: unknown provider, vision on non-vision provider
  - Retry helpers (retry_with_backoff, retry_with_backoff_async)
  - The thin shims in llm/providers.py, chemextract/llm_providers.py,
    and reaction/providers.py still expose their original public APIs

No actual HTTP calls are made — all tests use the internal _build_*_request
and _extract_text_from_response methods directly, plus unittest.mock to
simulate responses.
"""

import sys
import os
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# Make backend/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from llm.client import (
    LLMClient,
    VISION_CAPABLE_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_BASE_URLS,
    ALL_PROVIDERS,
    UnsupportedProviderError,
    VisionNotSupportedError,
    retry_with_backoff,
    retry_with_backoff_async,
)


# =============================================================================
# Provider registry tests
# =============================================================================

class TestProviderRegistry:
    """Tests for the provider registry constants."""

    def test_all_providers_listed(self):
        assert set(ALL_PROVIDERS) == {
            'deepseek', 'openai', 'anthropic', 'gemini',
            'groq', 'openrouter', 'huggingface', 'ollama',
        }

    def test_vision_capable_providers(self):
        assert set(VISION_CAPABLE_PROVIDERS) == {
            'deepseek', 'openai', 'gemini', 'anthropic',
        }

    def test_every_provider_has_default_model(self):
        for p in ALL_PROVIDERS:
            assert p in PROVIDER_DEFAULT_MODELS
            assert PROVIDER_DEFAULT_MODELS[p]

    def test_every_provider_has_base_url(self):
        for p in ALL_PROVIDERS:
            assert p in PROVIDER_BASE_URLS
            # Gemini's base URL is None (built per-model), others must be str.
            if p == 'gemini':
                assert PROVIDER_BASE_URLS[p] is None
            else:
                assert PROVIDER_BASE_URLS[p]

    def test_vision_providers_are_subset_of_all_providers(self):
        assert set(VISION_CAPABLE_PROVIDERS).issubset(set(ALL_PROVIDERS))


# =============================================================================
# Construction tests
# =============================================================================

class TestLLMClientConstruction:
    """Tests for LLMClient.__init__ — env-var resolution, model defaults, etc."""

    @pytest.mark.parametrize("provider_name", ALL_PROVIDERS)
    def test_all_providers_construct_with_explicit_args(self, provider_name):
        """Every provider should construct with explicit api_key + model."""
        client = LLMClient(
            provider=provider_name,
            api_key='sk-test-12345',
            model='custom-model',
        )
        assert client.provider == provider_name
        assert client.api_key == 'sk-test-12345'
        assert client.model == 'custom-model'

    @pytest.mark.parametrize("provider_name", ALL_PROVIDERS)
    def test_all_providers_use_default_model_when_none(self, provider_name):
        client = LLMClient(provider=provider_name, api_key='sk-test-12345')
        assert client.model == PROVIDER_DEFAULT_MODELS[provider_name]

    def test_unknown_provider_raises(self):
        with pytest.raises(UnsupportedProviderError) as exc_info:
            LLMClient(provider='unknown-provider')
        assert 'unknown-provider' in str(exc_info.value)

    def test_provider_name_case_insensitive(self):
        """Provider name should be normalized to lowercase."""
        c = LLMClient(provider='DeepSeek', api_key='sk-test')
        assert c.provider == 'deepseek'

    def test_env_var_api_key_resolution(self):
        """When api_key not passed, LLMClient should fall back to env var."""
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'sk-from-env-12345'}):
            c = LLMClient(provider='deepseek')
            assert c.api_key == 'sk-from-env-12345'

    def test_explicit_api_key_takes_precedence_over_env(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-from-env'}):
            c = LLMClient(provider='openai', api_key='sk-explicit')
            assert c.api_key == 'sk-explicit'

    def test_gemini_falls_back_to_google_api_key_env(self):
        """Gemini accepts either GEMINI_API_KEY or GOOGLE_API_KEY."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': 'google-key'}):
            c = LLMClient(provider='gemini')
            assert c.api_key == 'google-key'

    def test_huggingface_falls_back_to_huggingface_api_key_env(self):
        with patch.dict(os.environ, {'HF_API_KEY': '', 'HUGGINGFACE_API_KEY': 'hf-key'}):
            c = LLMClient(provider='huggingface')
            assert c.api_key == 'hf-key'

    def test_env_var_model_resolution(self):
        with patch.dict(os.environ, {'DEEPSEEK_MODEL': 'deepseek-reasoner'}):
            c = LLMClient(provider='deepseek', api_key='sk-test')
            assert c.model == 'deepseek-reasoner'

    def test_env_var_base_url_resolution(self):
        with patch.dict(os.environ, {'OLLAMA_BASE_URL': 'http://my-ollama:11434'}):
            c = LLMClient(provider='ollama')
            assert c.base_url == 'http://my-ollama:11434'

    def test_gemini_base_url_is_none(self):
        """Gemini's base URL is None because it's built per-model."""
        c = LLMClient(provider='gemini', api_key='test')
        assert c.base_url is None

    def test_supports_vision_static_helper(self):
        assert LLMClient.supports_vision('deepseek') is True
        assert LLMClient.supports_vision('openai') is True
        assert LLMClient.supports_vision('gemini') is True
        assert LLMClient.supports_vision('anthropic') is True
        assert LLMClient.supports_vision('groq') is False
        assert LLMClient.supports_vision('openrouter') is False
        assert LLMClient.supports_vision('huggingface') is False
        assert LLMClient.supports_vision('ollama') is False


# =============================================================================
# Text request building tests (per API style)
# =============================================================================

class TestTextRequestBuilding:
    """Verify the (url, headers, payload) tuples built for text chat."""

    @pytest.mark.parametrize("provider_name", ['deepseek', 'openai', 'groq', 'openrouter'])
    def test_openai_compat_providers_build_correct_payload(self, provider_name):
        """DeepSeek, OpenAI, Groq, OpenRouter all share the OpenAI-compatible
        /chat/completions endpoint with identical payload shape."""
        c = LLMClient(provider=provider_name, api_key='sk-test')
        messages = [
            {"role": "system", "content": "You are a chemist."},
            {"role": "user", "content": "Hello"},
        ]
        url, headers, payload = c._build_text_request(
            messages, temperature=0.5, max_tokens=1000,
            response_format={"type": "json_object"}, seed=42, extra_headers=None,
        )
        # URL ends with /chat/completions
        assert url.endswith('/chat/completions')
        # Auth header
        assert headers['Authorization'] == 'Bearer sk-test'
        assert headers['Content-Type'] == 'application/json'
        # Payload shape
        assert payload['model'] == PROVIDER_DEFAULT_MODELS[provider_name]
        assert payload['temperature'] == 0.5
        assert payload['max_tokens'] == 1000
        assert payload['response_format'] == {"type": "json_object"}
        assert payload['seed'] == 42
        assert payload['messages'] == messages

    def test_openrouter_adds_attribution_headers(self):
        """OpenRouter requires HTTP-Referer and X-Title headers."""
        c = LLMClient(provider='openrouter', api_key='sk-test')
        _, headers, _ = c._build_text_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.5, max_tokens=100,
            response_format=None, seed=None, extra_headers=None,
        )
        assert 'HTTP-Referer' in headers
        assert 'X-Title' in headers

    def test_other_openai_compat_providers_do_not_have_attribution_headers(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')
        _, headers, _ = c._build_text_request(
            [{"role": "user", "content": "hi"}],
            temperature=0.5, max_tokens=100,
            response_format=None, seed=None, extra_headers=None,
        )
        assert 'HTTP-Referer' not in headers
        assert 'X-Title' not in headers

    def test_anthropic_extracts_system_to_top_level(self):
        """Anthropic takes system as a top-level field, not as a message."""
        c = LLMClient(provider='anthropic', api_key='sk-ant-test')
        messages = [
            {"role": "system", "content": "You are a chemist."},
            {"role": "user", "content": "Hello"},
        ]
        url, headers, payload = c._build_text_request(
            messages, temperature=0.5, max_tokens=1000,
            response_format=None, seed=None, extra_headers=None,
        )
        assert url == 'https://api.anthropic.com/v1/messages'
        assert headers['x-api-key'] == 'sk-ant-test'
        assert headers['anthropic-version'] == '2023-06-01'
        # System extracted to top-level
        assert payload['system'] == 'You are a chemist.'
        # Only non-system messages remain in messages list
        assert len(payload['messages']) == 1
        assert payload['messages'][0]['role'] == 'user'

    def test_gemini_flattens_messages_to_parts(self):
        """Gemini uses contents/parts format with system text prepended."""
        c = LLMClient(provider='gemini', api_key='test-key')
        messages = [
            {"role": "system", "content": "You are a chemist."},
            {"role": "user", "content": "Hello"},
        ]
        url, headers, payload = c._build_text_request(
            messages, temperature=0.5, max_tokens=1000,
            response_format={"type": "json_object"}, seed=42, extra_headers=None,
        )
        # URL contains the API key and model name
        assert 'gemini-2.0-flash' in url
        assert 'test-key' in url
        # Payload shape
        assert 'contents' in payload
        parts = payload['contents'][0]['parts']
        # System text prepended, then user message
        assert len(parts) == 2
        assert parts[0]['text'] == 'You are a chemist.'
        assert parts[1]['text'] == 'Hello'
        # response_format translates to responseMimeType
        assert payload['generationConfig']['responseMimeType'] == 'application/json'
        assert payload['generationConfig']['seed'] == 42

    def test_huggingface_uses_llama3_prompt_template(self):
        """HuggingFace flattens messages into a Llama-3 chat-template string."""
        c = LLMClient(provider='huggingface', api_key='hf-test')
        messages = [
            {"role": "system", "content": "You are a chemist."},
            {"role": "user", "content": "Hello"},
        ]
        url, headers, payload = c._build_huggingface(messages, 0.5, 1000, None)
        # URL is per-model
        assert url == 'https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct'
        # Prompt contains the chat template markers
        assert '<|begin_of_text|>' in payload['inputs']
        assert '<|start_header_id|>user<|end_header_id|>' in payload['inputs']
        assert 'You are a chemist.' in payload['inputs']
        assert 'Hello' in payload['inputs']

    def test_ollama_uses_native_chat_format(self):
        c = LLMClient(provider='ollama', api_key=None)
        messages = [
            {"role": "system", "content": "You are a chemist."},
            {"role": "user", "content": "Hello"},
        ]
        url, headers, payload = c._build_ollama(messages, 0.5, 1000, None)
        assert url == 'http://localhost:11434/api/chat'
        assert payload['model'] == 'llama3'
        assert payload['stream'] is False
        assert payload['options']['temperature'] == 0.5
        assert payload['options']['num_predict'] == 1000
        assert payload['messages'] == messages


# =============================================================================
# Vision request building tests
# =============================================================================

class TestVisionRequestBuilding:
    """Verify vision request payloads for each vision-capable provider."""

    @pytest.fixture
    def vision_messages(self):
        return [
            {"role": "system", "content": "Extract reactions."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            },
        ]

    def test_openai_vision_passes_image_url_through(self, vision_messages):
        """DeepSeek + OpenAI pass the OpenAI image_url format through unchanged."""
        for provider in ['deepseek', 'openai']:
            c = LLMClient(provider=provider, api_key='sk-test')
            url, headers, payload = c._build_vision_request(
                vision_messages, temperature=0.0, max_tokens=16384,
                response_format={"type": "json_object"}, seed=42, extra_headers=None,
            )
            user_msg = payload['messages'][1]
            assert isinstance(user_msg['content'], list)
            assert user_msg['content'][1]['type'] == 'image_url'
            assert 'data:image/png;base64,abc123' in user_msg['content'][1]['image_url']['url']

    def test_anthropic_vision_translates_image_url_to_source(self, vision_messages):
        """Anthropic uses 'image' + 'source' instead of 'image_url'."""
        c = LLMClient(provider='anthropic', api_key='sk-ant-test')
        url, headers, payload = c._build_vision_request(
            vision_messages, temperature=0.0, max_tokens=16384,
            response_format=None, seed=None, extra_headers=None,
        )
        # anthropic-beta header added for vision
        assert 'anthropic-beta' in headers
        # User message content is a list of typed parts
        user_msg = payload['messages'][0]
        assert isinstance(user_msg['content'], list)
        # Find the image part
        image_parts = [p for p in user_msg['content'] if p.get('type') == 'image']
        assert len(image_parts) == 1
        assert image_parts[0]['source']['type'] == 'base64'
        assert image_parts[0]['source']['media_type'] == 'image/png'
        assert image_parts[0]['source']['data'] == 'abc123'

    def test_gemini_vision_translates_image_url_to_inline_data(self, vision_messages):
        """Gemini uses 'inline_data' instead of 'image_url'."""
        c = LLMClient(provider='gemini', api_key='test-key')
        url, headers, payload = c._build_vision_request(
            vision_messages, temperature=0.0, max_tokens=16384,
            response_format={"type": "json_object"}, seed=42, extra_headers=None,
        )
        parts = payload['contents'][0]['parts']
        # System text + user text + image data = 3 parts
        assert len(parts) == 3
        # Last part should be inline_data
        assert 'inline_data' in parts[2]
        assert parts[2]['inline_data']['mime_type'] == 'image/png'
        assert parts[2]['inline_data']['data'] == 'abc123'

    @pytest.mark.parametrize("non_vision_provider", ['groq', 'openrouter', 'huggingface', 'ollama'])
    def test_vision_on_non_vision_provider_raises(self, non_vision_provider, vision_messages):
        c = LLMClient(provider=non_vision_provider, api_key='sk-test')
        with pytest.raises(VisionNotSupportedError):
            c.vision(vision_messages)

    @pytest.mark.parametrize("non_vision_provider", ['groq', 'openrouter', 'huggingface', 'ollama'])
    @pytest.mark.asyncio
    async def test_vision_async_on_non_vision_provider_raises(self, non_vision_provider, vision_messages):
        c = LLMClient(provider=non_vision_provider, api_key='sk-test')
        with pytest.raises(VisionNotSupportedError):
            await c.vision_async(vision_messages)


# =============================================================================
# Response parsing tests
# =============================================================================

class TestResponseParsing:
    """Verify _extract_text_from_response handles each provider's response shape."""

    def test_openai_compat_response(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')
        data = {"choices": [{"message": {"content": "Hello!"}}]}
        assert c._extract_text_from_response(data) == "Hello!"

    def test_anthropic_response(self):
        c = LLMClient(provider='anthropic', api_key='sk-test')
        data = {"content": [{"text": "Hello!"}]}
        assert c._extract_text_from_response(data) == "Hello!"

    def test_gemini_response(self):
        c = LLMClient(provider='gemini', api_key='test')
        data = {"candidates": [{"content": {"parts": [{"text": "Hello!"}]}}]}
        assert c._extract_text_from_response(data) == "Hello!"

    def test_huggingface_response_list(self):
        c = LLMClient(provider='huggingface', api_key='test')
        data = [{"generated_text": "Hello!"}]
        assert c._extract_text_from_response(data) == "Hello!"

    def test_huggingface_response_dict(self):
        c = LLMClient(provider='huggingface', api_key='test')
        data = {"generated_text": "Hello!"}
        assert c._extract_text_from_response(data) == "Hello!"

    def test_huggingface_response_with_error(self):
        """HF returns {"error": "..."} when model is loading — should return None."""
        c = LLMClient(provider='huggingface', api_key='test')
        data = {"error": "Model is loading"}
        assert c._extract_text_from_response(data) is None

    def test_ollama_response(self):
        c = LLMClient(provider='ollama', api_key=None)
        data = {"message": {"content": "Hello!"}}
        assert c._extract_text_from_response(data) == "Hello!"

    def test_malformed_response_returns_none(self):
        """If the response shape is unexpected, return None (don't raise)."""
        c = LLMClient(provider='deepseek', api_key='sk-test')
        # Missing 'choices' key
        assert c._extract_text_from_response({}) is None
        # Empty choices list
        assert c._extract_text_from_response({"choices": []}) is None


# =============================================================================
# Sync chat() with mocked HTTP tests
# =============================================================================

class TestSyncChatWithMockedHTTP:
    """Test the sync chat() path with requests.post mocked."""

    def test_chat_returns_text_on_200(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        with patch('llm.client.requests.post', return_value=mock_response) as mock_post:
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result == "Hello!"
            # Verify the request was made with the right URL
            call_args = mock_post.call_args
            assert 'deepseek.com' in call_args[0][0] or 'deepseek.com' in str(call_args)

    def test_chat_returns_none_on_error_status(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        with patch('llm.client.requests.post', return_value=mock_response):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None

    def test_chat_returns_none_on_timeout(self):
        import requests as _requests
        c = LLMClient(provider='deepseek', api_key='sk-test')
        with patch('llm.client.requests.post', side_effect=_requests.exceptions.Timeout()):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None

    def test_chat_returns_none_on_connection_error(self):
        """Connection errors (requests.exceptions.ConnectionError is a subclass
        of RequestException) should be caught and return None."""
        import requests as _requests
        c = LLMClient(provider='deepseek', api_key='sk-test')
        with patch('llm.client.requests.post',
                   side_effect=_requests.exceptions.ConnectionError("refused")):
            result = c.chat([{"role": "user", "content": "hi"}])
            assert result is None


# =============================================================================
# Async chat() with mocked HTTP tests
# =============================================================================

class TestAsyncChatWithMockedHTTP:
    """Test the async chat_async() path with aiohttp mocked."""

    @pytest.mark.asyncio
    async def test_chat_async_returns_text_on_200(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')

        # Mock the aiohttp.ClientSession context manager
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Hello async!"}}]
        })
        mock_response.text = AsyncMock(return_value='')

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('llm.client.aiohttp.ClientSession', return_value=mock_session):
            result = await c.chat_async([{"role": "user", "content": "hi"}])
            assert result == "Hello async!"

    @pytest.mark.asyncio
    async def test_chat_async_returns_none_on_error_status(self):
        c = LLMClient(provider='deepseek', api_key='sk-test')

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="error")
        mock_response.json = AsyncMock(return_value={})

        mock_session = MagicMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('llm.client.aiohttp.ClientSession', return_value=mock_session):
            result = await c.chat_async([{"role": "user", "content": "hi"}])
            assert result is None


# =============================================================================
# Retry helper tests
# =============================================================================

class TestRetryHelpers:
    """Test retry_with_backoff and retry_with_backoff_async."""

    def test_retry_succeeds_on_first_attempt(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            return "success"
        result = retry_with_backoff(func, max_retries=3, retry_delay=0.01)
        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        result = retry_with_backoff(func, max_retries=3, retry_delay=0.01)
        assert result == "success"
        assert call_count == 3

    def test_retry_raises_after_max_attempts(self):
        call_count = 0
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")
        with pytest.raises(ValueError):
            retry_with_backoff(func, max_retries=2, retry_delay=0.01)
        # Initial call + 2 retries = 3 calls total
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_succeeds_on_first_attempt(self):
        call_count = 0
        async def func():
            nonlocal call_count
            call_count += 1
            return "success"
        result = await retry_with_backoff_async(func, max_retries=3, retry_delay=0.01)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_async_raises_after_max_attempts(self):
        call_count = 0
        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")
        with pytest.raises(ValueError):
            await retry_with_backoff_async(func, max_retries=2, retry_delay=0.01)
        assert call_count == 3


# =============================================================================
# Shim compatibility tests
# =============================================================================

class TestLLMProvidersShim:
    """Verify llm/providers.py classes still expose the original API."""

    @pytest.mark.parametrize("provider_name, class_name", [
        ('deepseek', 'DeepSeekProvider'),
        ('openai', 'OpenAIProvider'),
        ('groq', 'GroqProvider'),
        ('openrouter', 'OpenRouterProvider'),
        ('anthropic', 'AnthropicProvider'),
        ('gemini', 'GeminiProvider'),
        ('huggingface', 'HuggingFaceProvider'),
        ('ollama', 'OllamaProvider'),
    ])
    def test_provider_class_constructs_and_resolves_model(self, provider_name, class_name):
        from llm.providers import (
            DeepSeekProvider, OpenAIProvider, GroqProvider, OpenRouterProvider,
            AnthropicProvider, GeminiProvider, HuggingFaceProvider, OllamaProvider,
        )
        cls = locals()[class_name]
        instance = cls(api_key='sk-test')
        assert instance.model == PROVIDER_DEFAULT_MODELS[provider_name]
        assert hasattr(instance, 'chat')
        assert hasattr(instance, 'chat_async')
        assert hasattr(instance, 'chat_with_messages')
        assert hasattr(instance, 'chat_with_messages_async')

    def test_provider_chat_delegates_to_llmclient(self):
        """Provider.chat() should produce the same payload as LLMClient.chat()."""
        from llm.providers import DeepSeekProvider
        provider = DeepSeekProvider(api_key='sk-test')

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        with patch('llm.client.requests.post', return_value=mock_response):
            result = provider.chat("system", "user", temperature=0.5, max_tokens=100)
            assert result == "Hello!"

    def test_provider_chat_with_messages_passes_full_messages_list(self):
        """chat_with_messages should send the full multi-turn messages list,
        not just system + last user."""
        from llm.providers import DeepSeekProvider
        provider = DeepSeekProvider(api_key='sk-test')

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        with patch('llm.client.requests.post', return_value=mock_response) as mock_post:
            messages = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "msg1"},
                {"role": "assistant", "content": "reply1"},
                {"role": "user", "content": "msg2"},
            ]
            provider.chat_with_messages(messages)
            # The payload sent to requests.post should contain ALL 4 messages
            sent_payload = mock_post.call_args[1]['json']
            assert len(sent_payload['messages']) == 4

    def test_factory_creates_all_providers(self):
        from llm.factory import LLMProviderFactory
        for provider_name in ALL_PROVIDERS:
            instance = LLMProviderFactory.create(provider_name, api_key='sk-test')
            assert instance is not None


class TestChemextractShim:
    """Verify chemextract/llm_providers.py preserves its public API."""

    def test_all_public_functions_importable(self):
        from modules.chemextract.llm_providers import (
            call_vision_llm, call_vision_llm_async,
            _call_text_provider, _call_text_provider_async,
            _call_gemini_text, _call_gemini_text_async,
            _call_anthropic_text, _call_anthropic_text_async,
            _retry_on_failure, _retry_on_failure_async,
        )
        # All 10 must be callable.
        for fn in [call_vision_llm, call_vision_llm_async,
                   _call_text_provider, _call_text_provider_async,
                   _call_gemini_text, _call_gemini_text_async,
                   _call_anthropic_text, _call_anthropic_text_async,
                   _retry_on_failure, _retry_on_failure_async]:
            assert callable(fn)

    def test_call_vision_llm_rejects_non_vision_provider(self):
        """call_vision_llm should return None for non-vision providers."""
        from modules.chemextract.llm_providers import call_vision_llm
        result = call_vision_llm(
            base64_image='abc',
            provider='groq',  # not vision-capable
            model='llama-3.3-70b-versatile',
            api_key='sk-test',
            system_prompt='sys',
            user_message='msg',
        )
        assert result is None

    def test_call_vision_llm_uses_retry_and_returns_parsed_json(self):
        """call_vision_llm should call LLMClient.vision and parse the JSON response."""
        from modules.chemextract.llm_providers import call_vision_llm

        # Mock LLMClient.vision to return a JSON string.
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"reactions": [], "compounds": []}'}}]
        }
        with patch('llm.client.requests.post', return_value=mock_response):
            result = call_vision_llm(
                base64_image='abc',
                provider='deepseek',
                model='deepseek-chat',
                api_key='sk-test',
                system_prompt='sys',
                user_message='msg',
            )
        # Result should be parsed JSON, not a string.
        assert isinstance(result, dict)
        assert 'reactions' in result


class TestReactionShim:
    """Verify reaction/providers.py preserves its public API."""

    def test_all_public_functions_importable(self):
        from modules.reaction.providers import (
            rl_call_text, rl_call_text_async,
            _rl_retry, _rl_retry_async,
        )
        for fn in [rl_call_text, rl_call_text_async, _rl_retry, _rl_retry_async]:
            assert callable(fn)

    def test_rl_call_text_rejects_unsupported_provider(self):
        """rl_call_text only supports deepseek/openai/gemini/anthropic."""
        from modules.reaction.providers import rl_call_text
        result = rl_call_text(
            text='some text',
            provider='groq',  # not in RL_SUPPORTED_PROVIDERS
            model='llama-3.3-70b-versatile',
            api_key='sk-test',
            system_prompt='sys',
        )
        assert result is None

    def test_rl_call_text_returns_parsed_json_on_success(self):
        from modules.reaction.providers import rl_call_text

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"has_reactions": false}'}}]
        }
        with patch('llm.client.requests.post', return_value=mock_response):
            result = rl_call_text(
                text='no reactions here',
                provider='deepseek',
                model='deepseek-chat',
                api_key='sk-test',
                system_prompt='sys',
            )
        assert isinstance(result, dict)
        assert result.get('has_reactions') is False
