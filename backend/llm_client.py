import os
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import requests
import aiohttp

from utils import sanitize_input

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:5000"
    timeout: int = 60
    max_retries: int = 3

    def __post_init__(self):
        self.base_url = os.environ.get('LLM_SERVICE_URL', self.base_url)
        self.timeout = int(os.environ.get('LLM_TIMEOUT', self.timeout))


class LLMClientError(Exception):
    pass


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._session: Optional[requests.Session] = None
        self._async_session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            })
        return self._session

    async def _get_async_session(self) -> aiohttp.ClientSession:
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession(
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self._async_session

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        system_prompt = sanitize_input(system_prompt, max_length=4000)
        user_message = sanitize_input(user_message, max_length=2000)

        url = f"{self.config.base_url}/api/llm/chat"
        payload = {
            'message': user_message,
            'system_prompt': system_prompt,
            'provider': provider,
            'api_key': api_key,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        try:
            session = self._get_session()
            response = session.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('response')
                else:
                    logger.error(f"LLM API error: {data.get('error')}")
                    return None
            else:
                logger.error(f"LLM HTTP error: {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM client error: {e}")
            return None

    async def chat_async(
        self,
        system_prompt: str,
        user_message: str,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Optional[str]:
        system_prompt = sanitize_input(system_prompt, max_length=4000)
        user_message = sanitize_input(user_message, max_length=2000)

        url = f"{self.config.base_url}/api/llm/chat"
        payload = {
            'message': user_message,
            'system_prompt': system_prompt,
            'provider': provider,
            'api_key': api_key,
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        try:
            session = await self._get_async_session()
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        return data.get('response')
                    else:
                        logger.error(f"LLM API error: {data.get('error')}")
                        return None
                else:
                    logger.error(f"LLM HTTP error: {response.status}")
                    return None

        except asyncio.TimeoutError:
            logger.error("LLM async request timed out")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"LLM async connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM async client error: {e}")
            return None

    def generate_knowledge_graph(
        self,
        topic: str,
        provider: str = None,
        api_key: str = None
    ) -> Optional[Dict[str, Any]]:
        topic = sanitize_input(topic, max_length=500)

        url = f"{self.config.base_url}/api/knowledge-graph"
        payload = {
            'topic': topic,
            'use_llm': True,
            'provider': provider,
            'api_key': api_key
        }

        try:
            session = self._get_session()
            response = session.post(
                url,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('graph')
            return None

        except Exception as e:
            logger.error(f"Knowledge graph error: {e}")
            return None

    async def generate_knowledge_graph_async(
        self,
        topic: str,
        provider: str = None,
        api_key: str = None
    ) -> Optional[Dict[str, Any]]:
        topic = sanitize_input(topic, max_length=500)

        url = f"{self.config.base_url}/api/knowledge-graph"
        payload = {
            'topic': topic,
            'use_llm': True,
            'provider': provider,
            'api_key': api_key
        }

        try:
            session = await self._get_async_session()
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('success'):
                        return data.get('graph')
            return None

        except Exception as e:
            logger.error(f"Async knowledge graph error: {e}")
            return None

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    async def close_async(self):
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
            self._async_session = None


_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def get_llm_response(
    system_prompt: str,
    user_message: str,
    provider: str = None,
    api_key: str = None,
    model: str = None,
    **kwargs
) -> Optional[str]:
    client = get_llm_client()
    return client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=kwargs.get('temperature', 0.7),
        max_tokens=kwargs.get('max_tokens', 2000)
    )


async def get_llm_response_async(
    system_prompt: str,
    user_message: str,
    provider: str = None,
    api_key: str = None,
    model: str = None,
    **kwargs
) -> Optional[str]:
    client = get_llm_client()
    return await client.chat_async(
        system_prompt=system_prompt,
        user_message=user_message,
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=kwargs.get('temperature', 0.7),
        max_tokens=kwargs.get('max_tokens', 2000)
    )


def generate_knowledge_graph(
    topic: str,
    provider: str = None,
    api_key: str = None
) -> Optional[Dict[str, Any]]:
    client = get_llm_client()
    return client.generate_knowledge_graph(
        topic=topic,
        provider=provider,
        api_key=api_key
    )


async def generate_knowledge_graph_async(
    topic: str,
    provider: str = None,
    api_key: str = None
) -> Optional[Dict[str, Any]]:
    client = get_llm_client()
    return await client.generate_knowledge_graph_async(
        topic=topic,
        provider=provider,
        api_key=api_key
    )
