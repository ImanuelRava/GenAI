"""
LLM Provider Factory — creates provider instances by name.
"""

import os
import logging
from typing import List

from .providers import (
    DeepSeekProvider,
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    OpenRouterProvider,
)
from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


class LLMProviderFactory:
    HK_FRIENDLY_PROVIDERS: List[str] = ['ollama', 'openrouter', 'huggingface', 'deepseek']
    ALL_PROVIDERS: List[str] = [
        'ollama', 'openrouter', 'huggingface', 'deepseek',
        'gemini', 'groq', 'openai', 'anthropic',
    ]

    _PROVIDER_MAP = {
        'ollama': OllamaProvider,
        'openrouter': OpenRouterProvider,
        'huggingface': HuggingFaceProvider,
        'hf': HuggingFaceProvider,
        'deepseek': DeepSeekProvider,
        'gemini': GeminiProvider,
        'groq': GroqProvider,
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }

    @staticmethod
    def create(provider: str = 'ollama', **kwargs) -> BaseLLMProvider:
        provider_lower = provider.lower()
        if provider_lower not in LLMProviderFactory._PROVIDER_MAP:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(LLMProviderFactory._PROVIDER_MAP.keys())}"
            )
        return LLMProviderFactory._PROVIDER_MAP[provider_lower](**kwargs)

    @staticmethod
    def get_default_provider() -> str:
        env_checks = [
            ('GROQ_API_KEY', 'groq'),
            ('GEMINI_API_KEY', 'gemini'),
            ('GOOGLE_API_KEY', 'gemini'),
            ('HF_API_KEY', 'huggingface'),
            ('HUGGINGFACE_API_KEY', 'huggingface'),
            ('OLLAMA_BASE_URL', 'ollama'),
            ('OLLAMA_HOST', 'ollama'),
            ('DEEPSEEK_API_KEY', 'deepseek'),
            ('OPENROUTER_API_KEY', 'openrouter'),
            ('OPENAI_API_KEY', 'openai'),
            ('ANTHROPIC_API_KEY', 'anthropic'),
        ]
        for env_var, provider in env_checks:
            if os.environ.get(env_var):
                return provider
        return 'groq'
