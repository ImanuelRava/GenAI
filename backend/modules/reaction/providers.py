"""
ReactionLens — Provider-specific text LLM calls and retry helpers.
"""

import asyncio
import logging
from typing import Optional, Dict

import requests
import aiohttp

from .parsing import _parse_json_response
from .prompts import RL_MAX_RETRIES, RL_RETRY_DELAY, RL_MAX_OUTPUT_TOKENS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _rl_retry(func, *args, max_retries: int = RL_MAX_RETRIES, **kwargs):
    """Synchronous retry with linear back-off."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            if attempt < max_retries:
                wait = RL_RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"[ReactionLens] Call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                import time
                time.sleep(wait)
            else:
                logger.error(f"[ReactionLens] Call failed after {max_retries + 1} attempts: {exc}")
                raise


async def _rl_retry_async(func, *args, max_retries: int = RL_MAX_RETRIES, **kwargs):
    """Async retry with linear back-off."""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if attempt < max_retries:
                wait = RL_RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"[ReactionLens] Async call failed (attempt {attempt + 1}/{max_retries + 1}): {exc}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                logger.error(f"[ReactionLens] Async call failed after {max_retries + 1} attempts: {exc}")
                raise


# ---------------------------------------------------------------------------
# Provider-specific text LLM calls
# ---------------------------------------------------------------------------

def _text_deepseek(text, sys_prompt, model, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
         "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    logger.error(f"[ReactionLens] DeepSeek text error: {resp.status_code}")
    return None


async def _text_deepseek_async(text, sys_prompt, model, api_key):
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
         "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _text_openai(text, sys_prompt, model, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
         "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    return None


async def _text_openai_async(text, sys_prompt, model, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": text}],
         "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _text_gemini(text, sys_prompt, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    p = {"contents": [{"parts": [{"text": f"{sys_prompt}\n\n{text}"}]}],
         "generationConfig": {"temperature": 0.1, "maxOutputTokens": RL_MAX_OUTPUT_TOKENS}}
    resp = requests.post(url, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
    return None


async def _text_gemini_async(text, sys_prompt, model, api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    p = {"contents": [{"parts": [{"text": f"{sys_prompt}\n\n{text}"}]}],
         "generationConfig": {"temperature": 0.1, "maxOutputTokens": RL_MAX_OUTPUT_TOKENS}}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["candidates"][0]["content"]["parts"][0]["text"])
    return None


def _text_anthropic(text, sys_prompt, model, api_key):
    url = "https://api.anthropic.com/v1/messages"
    h = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    p = {"model": model, "max_tokens": RL_MAX_OUTPUT_TOKENS, "system": sys_prompt,
         "messages": [{"role": "user", "content": text}]}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["content"][0]["text"])
    return None


async def _text_anthropic_async(text, sys_prompt, model, api_key):
    url = "https://api.anthropic.com/v1/messages"
    h = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    p = {"model": model, "max_tokens": RL_MAX_OUTPUT_TOKENS, "system": sys_prompt,
         "messages": [{"role": "user", "content": text}]}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["content"][0]["text"])
    return None


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def rl_call_text(text, provider, model, api_key, system_prompt):
    def _do():
        if provider == "deepseek":
            return _text_deepseek(text, system_prompt, model, api_key)
        elif provider == "openai":
            return _text_openai(text, system_prompt, model, api_key)
        elif provider == "gemini":
            return _text_gemini(text, system_prompt, model, api_key)
        elif provider == "anthropic":
            return _text_anthropic(text, system_prompt, model, api_key)
        else:
            logger.error(f"[ReactionLens] Unsupported provider: {provider}")
            return None
    return _rl_retry(_do)


async def rl_call_text_async(text, provider, model, api_key, system_prompt):
    async def _do():
        if provider == "deepseek":
            return await _text_deepseek_async(text, system_prompt, model, api_key)
        elif provider == "openai":
            return await _text_openai_async(text, system_prompt, model, api_key)
        elif provider == "gemini":
            return await _text_gemini_async(text, system_prompt, model, api_key)
        elif provider == "anthropic":
            return await _text_anthropic_async(text, system_prompt, model, api_key)
        else:
            logger.error(f"[ReactionLens] Unsupported provider: {provider}")
            return None
    return await _rl_retry_async(_do)
