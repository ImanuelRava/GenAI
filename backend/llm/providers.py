"""
Concrete LLM Provider implementations.

OpenAI-compatible providers (DeepSeek, OpenAI, Groq, OpenRouter) share
header/payload builders from BaseLLMProvider to reduce duplication.
Provider-specific APIs (Anthropic, Ollama, Gemini, HuggingFace) retain
their own implementations.
"""

import os
import logging
from typing import Optional

import requests
import aiohttp

from .base import BaseLLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI-compatible providers (use shared helpers)
# ---------------------------------------------------------------------------


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY')
        self.base_url = os.environ.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
        self.model = model or os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return self._make_request(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return await self._make_request_async(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        self.model = model or os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return self._make_request(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return await self._make_request_async(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        self.base_url = 'https://api.groq.com/openai/v1'
        self.model = model or os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return self._make_request(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return await self._make_request_async(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('OPENROUTER_API_KEY')
        self.base_url = 'https://openrouter.ai/api/v1'
        self.model = model or os.environ.get('OPENROUTER_MODEL', 'meta-llama/llama-3-8b-instruct:free')

    def _build_openai_headers(self):
        headers = super()._build_openai_headers()
        headers["HTTP-Referer"] = "https://genai-research.local"
        headers["X-Title"] = "GenAI Research"
        return headers

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return self._make_request(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        return await self._make_request_async(
            self._build_openai_headers(),
            self._build_openai_payload(system_prompt, user_message, temperature, max_tokens),
        )


# ---------------------------------------------------------------------------
# Provider-specific API implementations
# ---------------------------------------------------------------------------


class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.base_url = 'https://api.anthropic.com/v1'
        self.model = model or os.environ.get('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            response = requests.post(
                f"{self.base_url}/messages", headers=headers, json=payload, timeout=60,
            )
            if response.status_code == 200:
                return response.json()["content"][0]["text"]
            else:
                logger.error(f"Anthropic API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Anthropic request error: {e}")
            return None

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["content"][0]["text"]
                    else:
                        text = await response.text()
                        logger.error(f"Anthropic API Error: {response.status} - {text}")
                        return None
        except Exception as e:
            logger.error(f"Anthropic async request error: {e}")
            return None


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url=None, model=None):
        self.api_key = None
        self.base_url = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = model or os.environ.get('OLLAMA_MODEL', 'llama3')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=120,
            )
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                logger.error(f"Ollama API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Ollama request error: {e}")
            return None

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["message"]["content"]
                    else:
                        text = await response.text()
                        logger.error(f"Ollama API Error: {response.status} - {text}")
                        return None
        except Exception as e:
            logger.error(f"Ollama async request error: {e}")
            return None


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.base_url = None
        self.model = model or os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        try:
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                logger.error(f"Gemini API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Gemini request error: {e}")
            return None

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_message}"}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        text = await response.text()
                        logger.error(f"Gemini API Error: {response.status} - {text}")
                        return None
        except Exception as e:
            logger.error(f"Gemini async request error: {e}")
            return None


class HuggingFaceProvider(BaseLLMProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.environ.get('HF_API_KEY') or os.environ.get('HUGGINGFACE_API_KEY')
        self.model = model or os.environ.get('HF_MODEL', 'meta-llama/Llama-3.2-3B-Instruct')
        self.base_url = f'https://api-inference.huggingface.co/models/{self.model}'

    def _build_hf_prompt(self, system_prompt, user_message):
        return (
            f"<|begin_of_text|>\n{system_prompt}\n<|eot_id|>\n"
            f"<|start_header_id|>user<|end_header_id|>\n{user_message}\n"
            f"<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"
        )

    def chat(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "inputs": self._build_hf_prompt(system_prompt, user_message),
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature,
                          "return_full_text": False},
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get('generated_text', '')
                return data.get('generated_text', '')
            elif response.status_code == 503:
                logger.warning("Hugging Face: Model is loading, please wait...")
                return None
            else:
                logger.error(f"Hugging Face API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Hugging Face request error: {e}")
            return None

    async def chat_async(self, system_prompt, user_message, temperature=0.7, max_tokens=2000):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "inputs": self._build_hf_prompt(system_prompt, user_message),
            "parameters": {"max_new_tokens": max_tokens, "temperature": temperature,
                          "return_full_text": False},
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url, headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, list) and len(data) > 0:
                            return data[0].get('generated_text', '')
                        return data.get('generated_text', '')
                    elif response.status == 503:
                        logger.warning("Hugging Face: Model is loading...")
                        return None
                    else:
                        text = await response.text()
                        logger.error(f"Hugging Face API Error: {response.status} - {text}")
                        return None
        except Exception as e:
            logger.error(f"Hugging Face async request error: {e}")
            return None
