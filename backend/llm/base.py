"""
Base LLM Provider with shared sync/async HTTP helpers.
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import requests
import aiohttp

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        pass

    @abstractmethod
    async def chat_async(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        pass

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Multi-turn sync chat using a full messages list.

        Default implementation extracts system prompt and last user message,
        then delegates to chat().  Providers with native multi-turn support
        can override this to pass the entire messages list.
        """
        system_prompt = ""
        user_message = ""
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            elif m["role"] == "user":
                user_message = m["content"]
        if not user_message:
            return None
        return self.chat(system_prompt, user_message,
                         temperature=temperature, max_tokens=max_tokens)

    async def chat_with_messages_async(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Optional[str]:
        """Multi-turn async chat using a full messages list.

        Default implementation delegates to chat_with_messages().
        """
        return self.chat_with_messages(messages,
                                       temperature=temperature,
                                       max_tokens=max_tokens)

    def _build_openai_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_openai_payload(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _make_request(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: int = 60,
    ) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.error(f"LLM API Error: {response.status_code} - {response.text}")
                return None
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            # JSON decode errors (ValueError), unexpected response shapes
            # (KeyError), or None where a dict was expected (TypeError).
            logger.error(f"LLM response parse error: {e}")
            return None

    async def _make_request_async(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: int = 60,
    ) -> Optional[str]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    else:
                        text = await response.text()
                        logger.error(f"LLM API Error: {response.status} - {text}")
                        return None
        except asyncio.TimeoutError:
            logger.error("LLM async request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"LLM async request error: {e}")
            return None
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"LLM async response parse error: {e}")
            return None
