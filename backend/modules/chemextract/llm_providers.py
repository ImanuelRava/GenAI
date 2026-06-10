import json
import logging
import time
import asyncio
from typing import Dict, Optional, List

import requests
import aiohttp

from .config import (
    MAX_OUTPUT_TOKENS, MAX_RETRIES, RETRY_DELAY,
    EXTRACTION_TEMPERATURE, EXTRACTION_SEED,
)
from .prompts import SYSTEM_PROMPT_COMPREHENSIVE
from .json_utils import _parse_json_response

logger = logging.getLogger(__name__)


def _retry_on_failure(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Retry a function call on failure with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"[ChemExtract] Call failed (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"[ChemExtract] Call failed after {max_retries+1} attempts: {e}")
                raise


async def _retry_on_failure_async(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Async retry wrapper with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"[ChemExtract] Async call failed (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[ChemExtract] Async call failed after {max_retries+1} attempts: {e}")
                raise


def call_vision_llm(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str
) -> Optional[Dict]:
    """Call a vision-capable LLM via HTTP to extract data from an image."""
    try:
        if provider == 'deepseek':
            return _call_deepseek_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'openai':
            return _call_openai_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'gemini':
            return _call_gemini_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'anthropic':
            return _call_anthropic_vision(base64_image, model, api_key, system_prompt, user_message)
        else:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"[ChemExtract] Vision LLM call failed: {e}")
        return None


async def call_vision_llm_async(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str
) -> Optional[Dict]:
    """Async call to a vision-capable LLM via HTTP."""
    try:
        if provider == 'deepseek':
            return await _call_deepseek_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'openai':
            return await _call_openai_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'gemini':
            return await _call_gemini_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'anthropic':
            return await _call_anthropic_vision_async(base64_image, model, api_key, system_prompt, user_message)
        else:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"[ChemExtract] Async vision LLM call failed: {e}")
        return None


def _call_deepseek_vision(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "seed": EXTRACTION_SEED,
        "response_format": {"type": "json_object"},
    }
    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] DeepSeek API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek vision request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_deepseek_vision_async(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "seed": EXTRACTION_SEED,
        "response_format": {"type": "json_object"},
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek async vision error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


def _call_openai_vision(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
            ]},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "seed": EXTRACTION_SEED,
        "response_format": {"type": "json_object"},
    }
    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] OpenAI API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI vision request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_openai_vision_async(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
            ]},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "seed": EXTRACTION_SEED,
        "response_format": {"type": "json_object"},
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI async vision error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


def _call_gemini_vision(base64_image, model, api_key, system_prompt, user_message):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"text": f"{system_prompt}\n\n{user_message}"},
            {"inline_data": {"mime_type": "image/png", "data": base64_image}}
        ]}],
        "generationConfig": {
            "temperature": EXTRACTION_TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "seed": EXTRACTION_SEED,
            "responseMimeType": "application/json",
        }
    }
    def _do_call():
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['candidates'][0]['content']['parts'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Gemini API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini vision request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_gemini_vision_async(base64_image, model, api_key, system_prompt, user_message):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [
            {"text": f"{system_prompt}\n\n{user_message}"},
            {"inline_data": {"mime_type": "image/png", "data": base64_image}}
        ]}],
        "generationConfig": {
            "temperature": EXTRACTION_TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "seed": EXTRACTION_SEED,
            "responseMimeType": "application/json",
        }
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini async vision error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


def _call_anthropic_vision(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
    }
    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
            {"type": "text", "text": user_message}
        ]}]
    }
    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['content'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Anthropic API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic vision request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_anthropic_vision_async(base64_image, model, api_key, system_prompt, user_message):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "prompt-caching-2024-07-31",
    }
    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "system": system_prompt,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
            {"type": "text", "text": user_message}
        ]}]
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic async vision error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


# ---------------------------------------------------------------------------
# Text LLM calls
# ---------------------------------------------------------------------------

def _build_text_payload(provider, model, text):
    user_content = f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "seed": EXTRACTION_SEED,
        "response_format": {"type": "json_object"},
    }


def _call_text_provider(provider, model, api_key, text):
    urls = {
        'deepseek': "https://api.deepseek.com/v1/chat/completions",
        'openai': "https://api.openai.com/v1/chat/completions",
    }
    url = urls.get(provider, f"https://api.{provider}.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_text_payload(provider, model, text)
    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] {provider} API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] {provider} text request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_text_provider_async(provider, model, api_key, text):
    urls = {
        'deepseek': "https://api.deepseek.com/v1/chat/completions",
        'openai': "https://api.openai.com/v1/chat/completions",
    }
    url = urls.get(provider, f"https://api.{provider}.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = _build_text_payload(provider, model, text)
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] {provider} async text error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


def _call_gemini_text(text, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    user_content = f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\n{user_content}"}]}],
        "generationConfig": {
            "temperature": EXTRACTION_TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "seed": EXTRACTION_SEED,
            "responseMimeType": "application/json",
        }
    }
    def _do_call():
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['candidates'][0]['content']['parts'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Gemini API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini text request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_gemini_text_async(text, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    user_content = f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"
    payload = {
        "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\n{user_content}"}]}],
        "generationConfig": {
            "temperature": EXTRACTION_TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "seed": EXTRACTION_SEED,
            "responseMimeType": "application/json",
        }
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini async text error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)


def _call_anthropic_text(text, model, api_key):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    user_content = f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"
    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [{"role": "user", "content": user_content}]
    }
    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['content'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Anthropic API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic text request error: {e}")
            return None
    return _retry_on_failure(_do_call)


async def _call_anthropic_text_async(text, model, api_key):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    user_content = f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"
    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": EXTRACTION_TEMPERATURE,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [{"role": "user", "content": user_content}]
    }
    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        return _parse_json_response(content)
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic async text error: {e}")
            return None
    return await _retry_on_failure_async(_do_call)
