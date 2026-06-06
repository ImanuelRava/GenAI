"""
ReactionLens - Signal-Driven Adaptive Reaction Extraction

A chemistry extraction module using signal detection and adaptive multi-strategy
extraction from scientific PDF documents.

Components:
1. ReactionSignalDetector - Single-pass VLM signal detection from PDF pages
2. AdaptiveExtractor - Content-adaptive extraction with self-verification
3. ReactionNormalizer - Entity resolution, abbreviation expansion, quantity parsing
4. ReactionLens - Orchestrated extraction pipeline

No external chemical database dependencies. SMILES generation is handled natively
by vision-language models during extraction.
"""

# === Imports ===
import re
import json
import logging
import base64
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import requests
import aiohttp

logger = logging.getLogger(__name__)

# === Optional Dependency Detection ===

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.debug("[ReactionLens] PyMuPDF not available; signal detector PDF support disabled")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.debug("[ReactionLens] OpenCV not available; page segmentation disabled")

# NO PubChem dependency -- SMILES resolution is handled by LLM natively

# === Imports from ChemExtract (best-effort; module works standalone too) ===

try:
    from .chemextract_extractor import (
        call_vision_llm,
        call_vision_llm_async,
        call_text_llm,
        call_text_llm_async,
        _parse_json_response,
        _retry_on_failure,
        _retry_on_failure_async,
        MAX_OUTPUT_TOKENS,
        MAX_RETRIES,
        RETRY_DELAY,
        pdf_to_images,
        extract_text_from_pdf,
        ChemExtractAI,
        format_reaction_schemes,
    )
    HAS_CHEMEXTRACT = True
    logger.debug("[ReactionLens] ChemExtractAI integration available")
except ImportError:
    HAS_CHEMEXTRACT = False
    logger.warning(
        "[ReactionLens] chemextract_extractor not found; "
        "running in standalone mode (LLM wrappers will be used)"
    )


# === Constants ===

RL_MAX_RETRIES: int = 3
RL_RETRY_DELAY: float = 3.0  # seconds
RL_MAX_OUTPUT_TOKENS: int = 16384
RL_DEFAULT_DPI: int = 200
RL_MIN_IMAGE_DIMENSION: int = 50  # pixels - skip tiny embedded images
RL_WHITE_RATIO_THRESHOLD: float = 0.90
RL_MIN_GAP_HEIGHT: int = 30  # pixels for page segmentation
RL_DEFAULT_VISION_PROVIDERS = ("deepseek", "openai", "gemini", "anthropic")


# === Enums & Data Classes ===

class SignalType(str, Enum):
    """Classification label for a detected content signal on a PDF page."""
    REACTION_SCHEME = "reaction_scheme"
    OPTIMIZATION_TABLE = "optimization_table"
    SUBSTRATE_SCOPE = "substrate_scope"
    GENERAL_TABLE = "general_table"
    CATALYTIC_CYCLE = "catalytic_cycle"
    FIGURE = "figure"
    TEXT_ONLY = "text_only"
    UNKNOWN = "unknown"


class ExtractionMode(str, Enum):
    """Extraction strategy for the adaptive extractor."""
    REACTION_SCHEME = "reaction_scheme"
    OPTIMIZATION_TABLE = "optimization_table"
    SUBSTRATE_SCOPE = "substrate_scope"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ContentSignal:
    """A detected content signal from a PDF page.

    Represents one piece of chemistry-relevant content found on a page,
    identified by a single VLM pass rather than sequential classification.

    Attributes:
        page_number: 1-indexed page number in the source PDF.
        signal_index: Zero-indexed position of this signal on its page.
        signal_type: Type classification (reaction_scheme, optimization_table, etc.).
        priority: Extraction priority (higher = more important, 1-5 scale).
        bbox: ``(x0, y0, x1, y1)`` bounding box in PDF points.
        base64_image: Base64-encoded PNG of the signal region (or full page).
        confidence: VLM detection confidence 0.0-1.0.
        caption: Nearby caption text if detected.
        source: How the signal was detected (``"signal_detector"``, ``"full_page"``, etc.).
    """
    page_number: int = 1
    signal_index: int = 0
    signal_type: str = SignalType.UNKNOWN.value
    priority: int = 0
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    base64_image: str = ""
    confidence: float = 0.0
    caption: str = ""
    source: str = "signal_detector"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict."""
        return {
            "page_number": self.page_number,
            "signal_index": self.signal_index,
            "signal_type": self.signal_type,
            "priority": self.priority,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "source": self.source,
            "caption": self.caption,
            "has_image": bool(self.base64_image),
        }


# === Local _parse_json_response fallback ===

def _local_parse_json_response(content: str) -> Optional[Union[Dict, List]]:
    """Parse JSON from LLM response text. Handles truncated, partial, and
    multiple JSON outputs. This is a local fallback when ChemExtract is not available."""
    if not content:
        return None

    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON via brace/bracket matching
    # First try array
    depth = 0
    start = -1
    best_json = None

    for i, char in enumerate(content):
        if char == '[':
            if depth == 0:
                start = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(content[start:i + 1])
                    if best_json is None or len(str(candidate)) > len(str(best_json)):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    # Then try dict
    depth = 0
    start = -1
    for i, char in enumerate(content):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(content[start:i + 1])
                    if best_json is None or len(str(candidate)) > len(str(best_json)):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    if best_json is not None:
        return best_json

    # Fallback: try fixing common LLM JSON issues (truncated output)
    try:
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if open_braces > 0 or open_brackets > 0:
            fixed = content + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try stripping markdown code fences
    try:
        cleaned = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"[ReactionLens] JSON parse failed for content of length {len(content)}")
    return None


# Choose the parse function based on availability
if HAS_CHEMEXTRACT:
    _parse_json_response = _parse_json_response
else:
    _parse_json_response = _local_parse_json_response


# === Retry Helpers (standalone) ===

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


# === Standalone LLM Wrappers (fallback when ChemExtract not importable) ===

def _standalone_call_vision(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Minimal vision LLM call covering the four major providers."""
    def _do():
        if provider == "deepseek":
            return _vision_deepseek(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "openai":
            return _vision_openai(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "gemini":
            return _vision_gemini(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "anthropic":
            return _vision_anthropic(base64_image, model, api_key, system_prompt, user_message)
        else:
            logger.error(f"[ReactionLens] Unsupported provider: {provider}")
            return None
    return _rl_retry(_do)


async def _standalone_call_vision_async(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Async version of the standalone vision call."""
    async def _do():
        if provider == "deepseek":
            return await _vision_deepseek_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "openai":
            return await _vision_openai_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "gemini":
            return await _vision_gemini_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == "anthropic":
            return await _vision_anthropic_async(base64_image, model, api_key, system_prompt, user_message)
        else:
            return None
    return await _rl_retry_async(_do)


def _standalone_call_text(text: str, system_prompt: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Minimal text LLM call."""
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
            return None
    return _rl_retry(_do)


async def _standalone_call_text_async(text: str, system_prompt: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Async version of the standalone text call."""
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
            return None
    return await _rl_retry_async(_do)


# === Provider implementations (standalone) ===

def _vision_deepseek(b64, model, api_key, sys_prompt, usr_msg):
    """DeepSeek vision API call (sync)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": usr_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    logger.error(f"[ReactionLens] DeepSeek vision error: {resp.status_code}")
    return None


async def _vision_deepseek_async(b64, model, api_key, sys_prompt, usr_msg):
    """DeepSeek vision API call (async)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": usr_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _vision_openai(b64, model, api_key, sys_prompt, usr_msg):
    """OpenAI GPT-4 Vision API call (sync)."""
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": usr_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
        ]}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    return None


async def _vision_openai_async(b64, model, api_key, sys_prompt, usr_msg):
    """OpenAI GPT-4 Vision API call (async)."""
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": usr_msg},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}
        ]}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _vision_gemini(b64, model, api_key, sys_prompt, usr_msg):
    """Google Gemini Vision API call (sync)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    p = {"contents": [{"parts": [
        {"text": f"{sys_prompt}\n\n{usr_msg}"},
        {"inline_data": {"mime_type": "image/png", "data": b64}}
    ]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": RL_MAX_OUTPUT_TOKENS}}
    resp = requests.post(url, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
    return None


async def _vision_gemini_async(b64, model, api_key, sys_prompt, usr_msg):
    """Google Gemini Vision API call (async)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    p = {"contents": [{"parts": [
        {"text": f"{sys_prompt}\n\n{usr_msg}"},
        {"inline_data": {"mime_type": "image/png", "data": b64}}
    ]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": RL_MAX_OUTPUT_TOKENS}}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["candidates"][0]["content"]["parts"][0]["text"])
    return None


def _vision_anthropic(b64, model, api_key, sys_prompt, usr_msg):
    """Anthropic Claude Vision API call (sync)."""
    url = "https://api.anthropic.com/v1/messages"
    h = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    p = {"model": model, "max_tokens": RL_MAX_OUTPUT_TOKENS, "system": sys_prompt,
         "messages": [{"role": "user", "content": [
             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
             {"type": "text", "text": usr_msg}
         ]}]}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["content"][0]["text"])
    return None


async def _vision_anthropic_async(b64, model, api_key, sys_prompt, usr_msg):
    """Anthropic Claude Vision API call (async)."""
    url = "https://api.anthropic.com/v1/messages"
    h = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    p = {"model": model, "max_tokens": RL_MAX_OUTPUT_TOKENS, "system": sys_prompt,
         "messages": [{"role": "user", "content": [
             {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
             {"type": "text", "text": usr_msg}
         ]}]}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["content"][0]["text"])
    return None


# === Text LLM provider implementations ===

def _text_deepseek(text, sys_prompt, model, api_key):
    """DeepSeek text API call (sync)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    return None


async def _text_deepseek_async(text, sys_prompt, model, api_key):
    """DeepSeek text API call (async)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _text_openai(text, sys_prompt, model, api_key):
    """OpenAI text API call (sync)."""
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["choices"][0]["message"]["content"])
    return None


async def _text_openai_async(text, sys_prompt, model, api_key):
    """OpenAI text API call (async)."""
    url = "https://api.openai.com/v1/chat/completions"
    h = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    p = {"model": model, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text}],
        "max_tokens": RL_MAX_OUTPUT_TOKENS, "temperature": 0.1}
    async with aiohttp.ClientSession() as s:
        async with s.post(url, headers=h, json=p, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 200:
                data = await resp.json()
                return _parse_json_response(data["choices"][0]["message"]["content"])
    return None


def _text_gemini(text, sys_prompt, model, api_key):
    """Google Gemini text API call (sync)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    p = {"contents": [{"parts": [{"text": f"{sys_prompt}\n\n{text}"}]}],
         "generationConfig": {"temperature": 0.1, "maxOutputTokens": RL_MAX_OUTPUT_TOKENS}}
    resp = requests.post(url, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
    return None


async def _text_gemini_async(text, sys_prompt, model, api_key):
    """Google Gemini text API call (async)."""
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
    """Anthropic Claude text API call (sync)."""
    url = "https://api.anthropic.com/v1/messages"
    h = {"x-api-key": api_key, "Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    p = {"model": model, "max_tokens": RL_MAX_OUTPUT_TOKENS, "system": sys_prompt,
         "messages": [{"role": "user", "content": text}]}
    resp = requests.post(url, headers=h, json=p, timeout=300)
    if resp.status_code == 200:
        return _parse_json_response(resp.json()["content"][0]["text"])
    return None


async def _text_anthropic_async(text, sys_prompt, model, api_key):
    """Anthropic Claude text API call (async)."""
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


# === Unified dispatchers ===

def rl_call_vision(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Dispatch vision call to ChemExtract or standalone implementation."""
    if HAS_CHEMEXTRACT:
        return call_vision_llm(base64_image, provider, model, api_key, system_prompt, user_message)
    return _standalone_call_vision(base64_image, provider, model, api_key, system_prompt, user_message)


async def rl_call_vision_async(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str,
) -> Optional[Dict]:
    """Async dispatch for vision calls."""
    if HAS_CHEMEXTRACT:
        return await call_vision_llm_async(base64_image, provider, model, api_key, system_prompt, user_message)
    return await _standalone_call_vision_async(base64_image, provider, model, api_key, system_prompt, user_message)


def rl_call_text(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
) -> Optional[Dict]:
    """Dispatch text call to ChemExtract or standalone implementation."""
    if HAS_CHEMEXTRACT:
        return _call_text_with_prompt(text, system_prompt, provider, model, api_key)
    return _standalone_call_text(text, system_prompt, provider, model, api_key)


async def rl_call_text_async(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
) -> Optional[Dict]:
    """Async dispatch for text calls."""
    if HAS_CHEMEXTRACT:
        return await _call_text_with_prompt_async(text, system_prompt, provider, model, api_key)
    return await _standalone_call_text_async(text, system_prompt, provider, model, api_key)


def _call_text_with_prompt(text: str, system_prompt: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Wrap ChemExtract's text LLM with a custom system prompt."""
    if provider == "deepseek":
        return _text_deepseek(text, system_prompt, model, api_key)
    elif provider == "openai":
        return _text_openai(text, system_prompt, model, api_key)
    elif provider == "gemini":
        return _text_gemini(text, system_prompt, model, api_key)
    elif provider == "anthropic":
        return _text_anthropic(text, system_prompt, model, api_key)
    return None


async def _call_text_with_prompt_async(text: str, system_prompt: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Async version of _call_text_with_prompt."""
    if provider == "deepseek":
        return await _text_deepseek_async(text, system_prompt, model, api_key)
    elif provider == "openai":
        return await _text_openai_async(text, system_prompt, model, api_key)
    elif provider == "gemini":
        return await _text_gemini_async(text, system_prompt, model, api_key)
    elif provider == "anthropic":
        return await _text_anthropic_async(text, system_prompt, model, api_key)
    return None


# === System Prompts ===

# Signal Detection Prompt (single-pass VLM)
SIGNAL_DETECTION_PROMPT = """You are a chemistry document analyst. Analyze this page image from a scientific paper.

Identify ALL chemistry-relevant content regions on this page. For EACH region, provide:
1. Type classification (reaction_scheme, optimization_table, substrate_scope, catalytic_cycle, general_table, figure, text_only)
2. Approximate location on the page (top/middle/bottom, left/center/right)
3. Priority (1-5, where reaction schemes and optimization tables are highest priority)
4. Brief caption if visible

Return JSON array:
[
  {
    "type": "reaction_scheme",
    "location": "top-center",
    "priority": 5,
    "caption": "Table 1. Optimization of reaction conditions",
    "confidence": 0.95,
    "description": "Reaction scheme with varying catalyst loadings"
  },
  ...
]

If no chemistry content found, return: [{"type": "text_only", "priority": 0, "confidence": 1.0}]
Mark pages as relevant if they contain ANY chemistry-related visual content."""

# Adaptive Extraction Prompt (single adaptive prompt for all signal types)
ADAPTIVE_EXTRACTION_PROMPT = """You are an expert chemistry data extraction system analyzing a {signal_type} from a scientific paper.

Extraction Strategy for {signal_type}:
{strategy_guidance}

CRITICAL RULES:
1. Extract EVERY entry/row visible. Do NOT skip, summarize, or combine.
2. Every chemical must include quantity with UNITS when shown (e.g. "Pd(OAc)2 (5 mol%)", "THF (5 mL)")
3. Mixed solvents: record exactly as shown ("DMF:H2O (4:1)", "DCM/MeCN")
4. R-group substitutions: use "symbol = value" format (e.g. "R = 4-MeOC6H4")
5. Identify standard conditions from the reaction diagram or first footnote
6. Deviations SUBSTITUTE standard conditions unless stated as addition
7. Use "N.R." for missing information; do NOT include internal standards
8. For each compound where you can see the structure, provide a SMILES string

Return JSON with two sections:

"Optimization Runs Dictionary": {entry_key: {
  "Entry": "entry number",
  "Reactants": "reactant indices with quantities",
  "Reactant_SMILES": ["SMILES for each reactant if visible"],
  "Substitutions": "R-group substitutions (symbol = value)",
  "Products": "product identity with yield",
  "Product_SMILES": ["SMILES for each product if visible"],
  "Catalyst": "catalyst (quantity with units)",
  "Catalyst_SMILES": "SMILES if visible",
  "Ligand": "ligand (quantity with units)",
  "Ligand_SMILES": "SMILES if visible",
  "Anode": "anode material",
  "Cathode": "cathode material",
  "Current": "current value",
  "Electrolytes": "non-solvent chemicals (quantity)",
  "Photocatalyst": "photocatalyst (quantity)",
  "Photocatalyst_SMILES": "SMILES if visible",
  "irradiation conditions": "light source details",
  "Chemicals": "non-solvent chemicals with roles (ROLE, quantity)",
  "Solvents": "solvents (quantity with units, mixed solvents as-is)",
  "Duration": "time",
  "Pressure": "pressure",
  "Air/Inert": "atmosphere",
  "Temperature": "temperature",
  "Others": "uncaptured conditions",
  "Yield": "yield value",
  "Yield type": "isolated/NMR/HPLC/GC",
  "Other product info": "ee, er, conversion, dr, selectivity",
  "Footnote": "superscript references"
}}

"Footnotes Dictionary": {"a": "description", "*": "description", ...}

Be EXHAUSTIVE -- extract every single entry."""

STRATEGY_GUIDANCE = {
    "reaction_scheme": """Focus on extracting reaction transformations:
- For each arrow: identify reactants, products, reagents above/below arrow
- Record ALL conditions (catalyst, ligand, solvent, temp, time, atmosphere)
- If R-group table present: extract scaffold SMILES with [*] placeholders and all substituent values
- Record yields and selectivities for each reaction""",

    "optimization_table": """Focus on systematic condition variation:
- Identify the STANDARD CONDITIONS from the reaction diagram or first footnote
- For each entry: note ONLY what DEVIATES from standard conditions
- Entries that don't mention a condition use the standard value
- Pay special attention to catalyst loading, ligand identity, solvent identity, temperature""",

    "substrate_scope": """Focus on substrate diversity:
- Extract the general reaction conditions (shown above/beside the table)
- For each substrate row: identify the specific substrate and product structure
- Record product SMILES when the structure is clearly visible
- Note any conditions that deviate from the general conditions in footnotes""",

    "comprehensive": """Extract ALL chemistry information visible:
- Reaction schemes with full conditions
- Tables with all entries
- Compound names with SMILES when structures are visible
- Catalytic cycles with intermediates
- Any other chemistry data""",
}

# Self-Verification Prompt (replaces footnote-only resolution)
SELF_VERIFICATION_PROMPT = """You are a chemistry data verification system. You will receive:
1. An image from a scientific chemistry paper
2. Previously extracted reaction data (Optimization Runs Dictionary + Footnotes Dictionary)

Your task:
1. CROSS-CHECK each entry against the image for accuracy
2. RESOLVE all footnote references by applying footnote descriptions to the matching entries
3. FIX any missed chemicals, incorrect quantities, or wrong yields
4. ENSURE consistency: if entry 1 uses "Pd(OAc)2 (5 mol%)" and entry 2 says "same catalyst", fill in the value
5. VERIFY SMILES strings are chemically reasonable for the described compound

Return the CORRECTED Optimization Runs Dictionary as JSON.
If no corrections needed, return the dictionary unchanged."""

# SMILES Validation Prompt (replaces PubChem lookup)
SMILES_VALIDATION_PROMPT = """You are a chemistry expert. Verify that the following SMILES strings are chemically reasonable for the described compounds.

For each pair, respond with:
- "valid" if the SMILES correctly represents the compound
- "corrected: [SMILES]" if you can provide a better SMILES
- "invalid" if the SMILES is clearly wrong

Return JSON: {{"compound_name": "valid/corrected_SMILES/invalid", ...}}

Pairs:
{smiles_pairs}"""

# Page Relevance Prompt
PAGE_RELEVANCE_PROMPT = """Analyze this image from a scientific paper. Does it contain chemistry-relevant visual content?

Chemistry-relevant: reaction schemes, optimization tables, substrate scope tables, catalytic cycles, mechanism diagrams, selectivity graphs
NOT relevant: title pages, references, affiliations, acknowledgements without chemistry

Return JSON: {{"is_relevant": true/false, "content_type": "...", "confidence": 0.0-1.0}}
Be generous -- mark as relevant for anything potentially chemistry-related."""


# ================================================================
# 1. ReactionSignalDetector
# ================================================================

class ReactionSignalDetector:
    """Detects chemistry content signals from PDF pages using a single VLM pass.

    Instead of sequentially segmenting and classifying regions, this detector
    analyzes each page in one VLM call and outputs a structured signal map
    with prioritized content regions.

    The key efficiency gain over per-region classification is: one VLM call
    per page instead of N calls per page (where N = number of detected regions).

    Args:
        provider: LLM provider for VLM calls.
        model: Model name for VLM calls.
        api_key: API key for the LLM provider.
        dpi: Rendering DPI for page-to-image conversion (default 200).
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        api_key: str = "",
        dpi: int = RL_DEFAULT_DPI,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.dpi = dpi

        if not HAS_PYMUPDF:
            logger.warning("[ReactionSignalDetector] PyMuPDF not available; "
                           "detector will return empty signals")

    def detect_signals(self, pdf_path: str, max_pages: int = 50) -> List[ContentSignal]:
        """Detect all chemistry signals from a PDF.

        Iterates through each page, renders it as an image, and runs a single
        VLM call to identify all content signals on that page.

        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to process.

        Returns:
            List of :class:`ContentSignal` objects, sorted by priority (descending).
        """
        if not HAS_PYMUPDF:
            logger.error("[ReactionSignalDetector] Cannot detect signals without PyMuPDF")
            return []

        signals: List[ContentSignal] = []
        try:
            doc = fitz.open(pdf_path)
            page_count = min(len(doc), max_pages)
            for page_num in range(page_count):
                page = doc[page_num]
                page_signals = self._detect_page_signals(page, page_num + 1)
                signals.extend(page_signals)
            doc.close()
            logger.info(
                f"[ReactionSignalDetector] Detected {len(signals)} signals "
                f"across {page_count} pages"
            )
        except Exception as e:
            logger.error(f"[ReactionSignalDetector] PDF analysis failed: {e}")

        # Sort by priority descending
        signals.sort(key=lambda s: s.priority, reverse=True)
        return signals

    async def detect_signals_async(self, pdf_path: str, max_pages: int = 50) -> List[ContentSignal]:
        """Async version of signal detection."""
        if not HAS_PYMUPDF:
            return []

        signals: List[ContentSignal] = []
        try:
            doc = fitz.open(pdf_path)
            page_count = min(len(doc), max_pages)
            pages_data = []
            for page_num in range(page_count):
                page = doc[page_num]
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                page_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                pages_data.append((page_num + 1, page, page_b64))
            doc.close()

            async def detect_one(pn, page, pb64):
                return self._detect_page_signals(page, pn, pb64)

            tasks = [detect_one(pn, pg, pb) for pn, pg, pb in pages_data]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"[ReactionSignalDetector] Async detection error: {result}")
                elif isinstance(result, list):
                    signals.extend(result)
        except Exception as e:
            logger.error(f"[ReactionSignalDetector] Async PDF analysis failed: {e}")

        signals.sort(key=lambda s: s.priority, reverse=True)
        return signals

    def _detect_page_signals(self, page, page_number: int, page_b64: str = None) -> List[ContentSignal]:
        """Single VLM pass to detect all signals on a page.

        Renders the page (if not pre-rendered), runs one VLM call to identify
        content regions, then attempts to match detected signals with
        extracted image regions for more accurate cropping.

        Args:
            page: A PyMuPDF page object.
            page_number: 1-indexed page number.
            page_b64: Optional pre-rendered base64 image of the full page.

        Returns:
            List of :class:`ContentSignal` objects found on this page.
        """
        signals: List[ContentSignal] = []

        # Render page if not pre-rendered
        if page_b64 is None:
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            page_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        if not self.api_key:
            # No API key: return full page as unknown signal
            signals.append(ContentSignal(
                page_number=page_number,
                base64_image=page_b64,
                source="full_page",
            ))
            return signals

        # Single VLM call for signal detection
        try:
            result = rl_call_vision(
                page_b64, self.provider, self.model, self.api_key,
                SIGNAL_DETECTION_PROMPT,
                "Identify all chemistry content regions on this page."
            )
        except Exception as e:
            logger.debug(f"[ReactionSignalDetector] VLM call failed for page {page_number}: {e}")
            signals.append(ContentSignal(
                page_number=page_number,
                base64_image=page_b64,
                source="full_page_fallback",
            ))
            return signals

        if result is None or not isinstance(result, list) or len(result) == 0:
            signals.append(ContentSignal(
                page_number=page_number,
                base64_image=page_b64,
                source="full_page_fallback",
            ))
            return signals

        # Extract embedded images and layout blocks for potential cropping
        regions = self._get_page_regions(page, page_number)

        for idx, signal_data in enumerate(result):
            if not isinstance(signal_data, dict):
                continue

            sig_type = signal_data.get("type", "unknown")
            priority = int(signal_data.get("priority", 0))
            confidence = float(signal_data.get("confidence", 0.5))
            caption = signal_data.get("caption", "")

            # Skip text-only signals
            if sig_type == "text_only" and priority == 0:
                continue

            # Try to match with a cropped region by location heuristics
            best_region = self._find_best_region(regions, signal_data)

            signals.append(ContentSignal(
                page_number=page_number,
                signal_index=idx,
                signal_type=sig_type,
                priority=priority,
                bbox=best_region.get("bbox", (0, 0, 0, 0)) if best_region else (0, 0, 0, 0),
                base64_image=best_region.get("image", "") if best_region else page_b64,
                confidence=confidence,
                caption=caption,
                source="signal_detector",
            ))

        # If no chemistry signals found, still return a low-priority full page signal
        if not signals:
            signals.append(ContentSignal(
                page_number=page_number,
                base64_image=page_b64,
                source="full_page",
            ))

        return signals

    def _get_page_regions(self, page, page_number: int) -> List[Dict]:
        """Extract embedded images and layout blocks from page as potential crops.

        Returns a list of dicts with 'bbox' and 'image' (base64) keys.

        Args:
            page: A PyMuPDF page object.
            page_number: 1-indexed page number.

        Returns:
            List of region dicts.
        """
        regions: List[Dict] = []
        if not HAS_PYMUPDF:
            return regions

        try:
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)

            # Extract embedded images
            try:
                image_info = page.get_image_info(xrefs=True)
                for idx, img in enumerate(image_info):
                    bbox = img.get("bbox", (0, 0, 0, 0))
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    pw = w * (self.dpi / 72)
                    ph = h * (self.dpi / 72)

                    if pw < RL_MIN_IMAGE_DIMENSION or ph < RL_MIN_IMAGE_DIMENSION:
                        continue

                    clip = fitz.Rect(bbox)
                    try:
                        pix = page.get_pixmap(matrix=mat, clip=clip)
                        img_data = pix.tobytes("png")
                        b64_img = base64.b64encode(img_data).decode("utf-8")
                        regions.append({"bbox": bbox, "image": b64_img, "source": "embedded"})
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"[ReactionSignalDetector] Embedded image extraction failed: {e}")

            # Detect image-type blocks from layout
            try:
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_IMAGES).get("blocks", [])
                image_blocks = [b for b in blocks if b.get("type") == 1]

                for idx, block in enumerate(image_blocks):
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    # Skip if already covered by an embedded image
                    already_covered = False
                    for existing in regions:
                        eb = existing["bbox"]
                        if self._bbox_overlap_ratio(eb, bbox) > 0.7:
                            already_covered = True
                            break
                    if already_covered:
                        continue

                    clip = fitz.Rect(bbox)
                    try:
                        pix = page.get_pixmap(matrix=mat, clip=clip)
                        img_data = pix.tobytes("png")
                        b64_img = base64.b64encode(img_data).decode("utf-8")
                        regions.append({"bbox": bbox, "image": b64_img, "source": "layout"})
                    except Exception:
                        continue
            except Exception as e:
                logger.debug(f"[ReactionSignalDetector] Layout block extraction failed: {e}")

        except Exception as e:
            logger.warning(f"[ReactionSignalDetector] Region extraction failed: {e}")

        return regions

    def _find_best_region(self, regions: List[Dict], signal_data: Dict) -> Optional[Dict]:
        """Match a signal to the best region by location heuristics.

        Uses the "location" field from the VLM output (e.g. "top-center",
        "bottom-left") to find the region that best overlaps with the
        expected position on the page.

        Args:
            regions: List of region dicts with 'bbox' and 'image' keys.
            signal_data: Signal data dict from VLM, including 'location' field.

        Returns:
            Best matching region dict, or None.
        """
        if not regions:
            return None

        location = signal_data.get("location", "").lower()
        sig_type = signal_data.get("type", "")

        # Score each region based on position match
        best_score = -1
        best_region = None

        for region in regions:
            score = 0
            bbox = region.get("bbox", (0, 0, 0, 0))
            y_center = (bbox[1] + bbox[3]) / 2.0
            x_center = (bbox[0] + bbox[2]) / 2.0

            # Vertical position matching
            if "top" in location:
                score += 2 if y_center < 0.33 else 0
            elif "bottom" in location:
                score += 2 if y_center > 0.67 else 0
            elif "middle" in location:
                score += 2 if 0.33 <= y_center <= 0.67 else 0

            # Horizontal position matching
            if "left" in location:
                score += 1 if x_center < 0.33 else 0
            elif "right" in location:
                score += 1 if x_center > 0.67 else 0
            elif "center" in location:
                score += 1 if 0.33 <= x_center <= 0.67 else 0

            # Prefer regions that are images (not full-page fallback)
            if region.get("source") == "embedded":
                score += 1
            elif region.get("source") == "layout":
                score += 0.5

            # Type-based preference for certain signal types
            if sig_type in ("reaction_scheme", "optimization_table", "substrate_scope"):
                # Prefer larger regions for these types
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                score += min(area / 100000.0, 1.0)

            if score > best_score:
                best_score = score
                best_region = region

        return best_region if best_score > 0 else None

    @staticmethod
    def _bbox_overlap_ratio(bbox_a: Tuple, bbox_b: Tuple) -> float:
        """Compute overlap ratio of two bounding boxes (smaller / larger area)."""
        ax0, ay0, ax1, ay1 = bbox_a
        bx0, by0, bx1, by1 = bbox_b

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        inter_area = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)

        min_area = min(area_a, area_b)
        return inter_area / min_area if min_area > 0 else 0.0

    def segment_page(self, base64_image: str) -> List[str]:
        """Optional: segment compound pages using OpenCV.

        Splits a full-page image into sub-images at white-line gaps.

        Args:
            base64_image: Base64-encoded PNG of a full page.

        Returns:
            List of base64-encoded PNG sub-images.
        """
        if not HAS_CV2:
            return [base64_image]

        try:
            img_bytes = base64.b64decode(base64_image)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return [base64_image]

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

            gap_rows = []
            for y in range(h):
                white_count = cv2.countNonZero(thresh[y:y + 1, :])
                if white_count / w >= RL_WHITE_RATIO_THRESHOLD:
                    gap_rows.append(y)

            if not gap_rows:
                return [base64_image]

            # Group consecutive gap rows
            gap_regions_list: List[Tuple[int, int]] = []
            start = gap_rows[0]
            for i in range(1, len(gap_rows)):
                if gap_rows[i] - gap_rows[i - 1] > 1:
                    gap_regions_list.append((start, gap_rows[i - 1]))
                    start = gap_rows[i]
            gap_regions_list.append((start, gap_rows[-1]))

            split_points = [
                (s + e) // 2 for s, e in gap_regions_list
                if e - s + 1 >= RL_MIN_GAP_HEIGHT
            ]

            if not split_points:
                return [base64_image]

            sub_images: List[str] = []
            prev = 0
            for sp in split_points:
                crop = img[prev:sp, :]
                _, buf = cv2.imencode(".png", crop)
                sub_images.append(base64.b64encode(buf).decode("utf-8"))
                prev = sp
            crop = img[prev:h, :]
            _, buf = cv2.imencode(".png", crop)
            sub_images.append(base64.b64encode(buf).decode("utf-8"))

            logger.debug(f"[ReactionSignalDetector] Segmented into {len(sub_images)} sub-images")
            return sub_images

        except Exception as e:
            logger.warning(f"[ReactionSignalDetector] Segmentation failed: {e}")
            return [base64_image]


# ================================================================
# 2. AdaptiveExtractor
# ================================================================

class AdaptiveExtractor:
    """Extracts reaction data using content-adaptive prompts with self-verification.

    Instead of separate mode-specific extraction passes and a footnote-only
    resolution pass, this extractor:

    1. Selects extraction strategy based on detected signal type
    2. Runs a single adaptive extraction pass with the appropriate strategy
    3. Runs a self-verification pass that checks ALL extracted data against
       the image (not just footnotes)

    This approach reduces the total number of LLM calls while improving
    data quality through comprehensive cross-checking.

    Args:
        provider: LLM provider for VLM and text calls.
        model: Model name.
        api_key: API key for the provider.
        enable_verification: Whether to run the self-verification pass (default True).
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        api_key: str = "",
        enable_verification: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.enable_verification = enable_verification

    def extract(self, signal: ContentSignal, context_text: str = "") -> Dict[str, Any]:
        """Extract reaction data from a content signal.

        Pipeline:
        1. Map signal type to extraction mode
        2. Build adaptive prompt with strategy guidance
        3. Pass 1: VLM extraction
        4. Pass 2: Self-verification (optional)

        Args:
            signal: A :class:`ContentSignal` to extract from.
            context_text: Optional surrounding text context to prepend.

        Returns:
            A dict with:
            - ``reactions``: list of formatted reaction dicts
            - ``raw_optimization_runs``: the (verified) optimization runs dict
            - ``footnotes``: the footnotes dictionary
            - ``extraction_mode``: the mode used
        """
        mode = self._signal_to_mode(signal)
        strategy = STRATEGY_GUIDANCE.get(mode, STRATEGY_GUIDANCE["comprehensive"])

        prompt = ADAPTIVE_EXTRACTION_PROMPT.format(
            signal_type=signal.signal_type,
            strategy_guidance=strategy,
        )

        # Build user message with optional context
        user_parts = []
        if context_text:
            user_parts.append(f"CONTEXT (surrounding text):\n{context_text}\n")
        if signal.caption:
            user_parts.append(f"CAPTION: {signal.caption}\n")
        user_parts.append("Extract ALL reaction entries from this image.")
        user_message = "\n".join(user_parts)

        # Pass 1: Adaptive extraction
        logger.info(f"[AdaptiveExtractor] Pass 1: extraction (mode={mode}, signal_type={signal.signal_type})")

        pass1_result = rl_call_vision(
            signal.base64_image, self.provider, self.model, self.api_key,
            prompt, user_message
        )

        if pass1_result is None:
            logger.warning("[AdaptiveExtractor] Pass 1 returned None")
            return {
                "reactions": [],
                "raw_optimization_runs": {},
                "footnotes": {},
                "extraction_mode": mode,
            }

        # Parse the two dictionaries from Pass 1
        opt_runs, footnotes = self._parse_extraction_result(pass1_result)

        logger.info(
            f"[AdaptiveExtractor] Pass 1 complete: "
            f"{len(opt_runs)} optimization runs, {len(footnotes)} footnotes"
        )

        # Pass 2: Self-verification (replaces footnote-only resolution)
        if self.enable_verification and opt_runs:
            logger.info("[AdaptiveExtractor] Pass 2: self-verification")
            opt_runs = self._self_verify(signal.base64_image, opt_runs, footnotes)

        # Format to ChemExtract reaction format
        reactions = self._format_to_reactions(opt_runs, footnotes)

        return {
            "reactions": reactions,
            "raw_optimization_runs": opt_runs,
            "footnotes": footnotes,
            "extraction_mode": mode,
        }

    async def extract_async(self, signal: ContentSignal, context_text: str = "") -> Dict[str, Any]:
        """Async version of extract."""
        mode = self._signal_to_mode(signal)
        strategy = STRATEGY_GUIDANCE.get(mode, STRATEGY_GUIDANCE["comprehensive"])

        prompt = ADAPTIVE_EXTRACTION_PROMPT.format(
            signal_type=signal.signal_type,
            strategy_guidance=strategy,
        )

        user_parts = []
        if context_text:
            user_parts.append(f"CONTEXT:\n{context_text}\n")
        if signal.caption:
            user_parts.append(f"CAPTION: {signal.caption}\n")
        user_parts.append("Extract ALL reaction entries from this image.")
        user_message = "\n".join(user_parts)

        logger.info(f"[AdaptiveExtractor] Pass 1 async (mode={mode})")

        pass1_result = await rl_call_vision_async(
            signal.base64_image, self.provider, self.model, self.api_key,
            prompt, user_message
        )

        if pass1_result is None:
            return {"reactions": [], "raw_optimization_runs": {}, "footnotes": {}, "extraction_mode": mode}

        opt_runs, footnotes = self._parse_extraction_result(pass1_result)

        if self.enable_verification and opt_runs:
            opt_runs = await self._self_verify_async(signal.base64_image, opt_runs, footnotes)

        reactions = self._format_to_reactions(opt_runs, footnotes)

        return {
            "reactions": reactions,
            "raw_optimization_runs": opt_runs,
            "footnotes": footnotes,
            "extraction_mode": mode,
        }

    def _signal_to_mode(self, signal: ContentSignal) -> str:
        """Map signal type to extraction mode.

        Args:
            signal: A :class:`ContentSignal`.

        Returns:
            Extraction mode string.
        """
        mapping = {
            SignalType.REACTION_SCHEME.value: ExtractionMode.REACTION_SCHEME.value,
            SignalType.OPTIMIZATION_TABLE.value: ExtractionMode.OPTIMIZATION_TABLE.value,
            SignalType.SUBSTRATE_SCOPE.value: ExtractionMode.SUBSTRATE_SCOPE.value,
            SignalType.CATALYTIC_CYCLE.value: ExtractionMode.REACTION_SCHEME.value,
        }
        return mapping.get(signal.signal_type, ExtractionMode.COMPREHENSIVE.value)

    def _self_verify(self, base64_image: str, opt_runs: Dict, footnotes: Dict) -> Dict:
        """Self-verification pass: cross-check extracted data against the image.

        Sends the extracted data back to the VLM alongside the original image
        for a comprehensive accuracy check. This goes beyond simple footnote
        resolution by checking all fields for accuracy and consistency.

        Args:
            base64_image: The original page/region image.
            opt_runs: Optimization runs dictionary from Pass 1.
            footnotes: Footnotes dictionary from Pass 1.

        Returns:
            Corrected optimization runs dictionary.
        """
        combined = json.dumps({
            "Optimization Runs Dictionary": opt_runs,
            "Footnotes Dictionary": footnotes,
        }, indent=2, ensure_ascii=False)

        # Truncate if too long for context window
        if len(combined) > 12000:
            combined = combined[:12000] + "\n... (truncated for context length)"

        try:
            result = rl_call_vision(
                base64_image, self.provider, self.model, self.api_key,
                SELF_VERIFICATION_PROMPT,
                f"Verify and correct this extracted data:\n\n{combined}"
            )

            if result:
                # Try to find the corrected dictionary in the response
                for key, value in result.items():
                    if isinstance(value, dict):
                        kl = key.lower()
                        if "optimization" in kl or "runs" in kl or "corrected" in kl or "updated" in kl:
                            opt_runs = value
                            break
                        # If it's a dict-of-dicts, assume it's the corrected runs
                        sample = list(value.values())[:2]
                        if sample and all(isinstance(v, dict) for v in sample):
                            opt_runs = value
                            break
                logger.info("[AdaptiveExtractor] Self-verification complete")
            else:
                logger.warning("[AdaptiveExtractor] Self-verification returned None")
        except Exception as e:
            logger.warning(f"[AdaptiveExtractor] Self-verification failed: {e}")

        return opt_runs

    async def _self_verify_async(self, base64_image: str, opt_runs: Dict, footnotes: Dict) -> Dict:
        """Async version of self-verification."""
        combined = json.dumps({
            "Optimization Runs Dictionary": opt_runs,
            "Footnotes Dictionary": footnotes,
        }, indent=2, ensure_ascii=False)

        if len(combined) > 12000:
            combined = combined[:12000] + "\n... (truncated for context length)"

        try:
            result = await rl_call_vision_async(
                base64_image, self.provider, self.model, self.api_key,
                SELF_VERIFICATION_PROMPT,
                f"Verify and correct this extracted data:\n\n{combined}"
            )

            if result:
                for key, value in result.items():
                    if isinstance(value, dict):
                        kl = key.lower()
                        if "optimization" in kl or "runs" in kl or "corrected" in kl or "updated" in kl:
                            opt_runs = value
                            break
                        sample = list(value.values())[:2]
                        if sample and all(isinstance(v, dict) for v in sample):
                            opt_runs = value
                            break
        except Exception as e:
            logger.warning(f"[AdaptiveExtractor] Async self-verification failed: {e}")

        return opt_runs

    @staticmethod
    def _parse_extraction_result(result: Union[Dict, List]) -> Tuple[Dict, Dict]:
        """Parse the VLM output into optimization runs and footnotes dicts.

        Handles flexible key naming and various output structures.

        Args:
            result: Raw JSON dict/list from the VLM.

        Returns:
            A tuple ``(optimization_runs, footnotes)``.
        """
        opt_runs: Dict = {}
        footnotes: Dict = {}

        if not isinstance(result, dict):
            return opt_runs, footnotes

        for key, value in result.items():
            if not isinstance(value, dict):
                continue
            kl = key.lower()
            if "optimization" in kl or "runs" in kl:
                opt_runs = value
            elif "footnote" in kl:
                footnotes = value
            elif not opt_runs:
                # Treat any dict-of-dicts as potential optimization runs
                sample = list(value.values())[:3]
                if any(isinstance(v, dict) for v in sample):
                    opt_runs = value

        return opt_runs, footnotes

    @staticmethod
    def _format_to_reactions(
        opt_runs: Dict,
        footnotes: Optional[Dict] = None,
    ) -> List[Dict]:
        """Convert optimization runs dict to ChemExtract-compatible reaction list.

        Each entry in ``opt_runs`` is converted to a reaction dict. Also
        extracts SMILES from the new ``_SMILES`` fields added by the
        adaptive extraction prompt.

        Args:
            opt_runs: Optimization runs dictionary.
            footnotes: Optional footnotes dictionary.

        Returns:
            A list of reaction dicts.
        """
        if not opt_runs:
            return []

        reactions: List[Dict] = []
        for entry_key, entry_data in opt_runs.items():
            if not isinstance(entry_data, dict):
                continue

            entry_label = re.sub(r"[^0-9a-zA-Z]", "", entry_key) or entry_key

            reaction: Dict[str, Any] = {
                "id": f"reactionlens_{entry_label}",
                "entry": entry_label,
                "source": "reactionlens",
            }

            # Reactants
            reactants_raw = entry_data.get("Reactants") or entry_data.get("reactants") or entry_data.get("Substrates") or []
            reaction["reactants"] = _normalize_chemical_list(reactants_raw)

            # Reactant SMILES
            reactant_smiles = entry_data.get("Reactant_SMILES") or entry_data.get("reactant_smiles") or []
            if reactant_smiles and isinstance(reactant_smiles, list):
                for i, smi in enumerate(reactant_smiles):
                    if i < len(reaction["reactants"]) and smi:
                        reaction["reactants"][i]["smiles"] = smi

            # Products
            products_raw = entry_data.get("Products") or entry_data.get("products") or []
            reaction["products"] = _normalize_chemical_list(products_raw)

            # Product SMILES
            product_smiles = entry_data.get("Product_SMILES") or entry_data.get("product_smiles") or []
            if product_smiles and isinstance(product_smiles, list):
                for i, smi in enumerate(product_smiles):
                    if i < len(reaction["products"]) and smi:
                        reaction["products"][i]["smiles"] = smi

            # Catalyst
            catalyst = _clean_field(entry_data.get("Catalyst") or entry_data.get("catalyst"))
            reaction["catalyst"] = catalyst
            catalyst_smiles = entry_data.get("Catalyst_SMILES") or entry_data.get("catalyst_smiles")
            if catalyst_smiles and catalyst != "N.R.":
                reaction["catalyst_smiles"] = str(catalyst_smiles)

            # Ligand
            ligand = _clean_field(entry_data.get("Ligand") or entry_data.get("ligand"))
            reaction["ligand"] = ligand
            ligand_smiles = entry_data.get("Ligand_SMILES") or entry_data.get("ligand_smiles")
            if ligand_smiles and ligand != "N.R.":
                reaction["ligand_smiles"] = str(ligand_smiles)

            # Solvents
            solvents_raw = entry_data.get("Solvents") or entry_data.get("Solvent") or entry_data.get("solvents") or "N.R."
            if isinstance(solvents_raw, str):
                solvents_list = [s.strip() for s in solvents_raw.split(",") if s.strip()]
            else:
                solvents_list = solvents_raw if isinstance(solvents_raw, list) else []
            reaction["solvents"] = [s for s in solvents_list if str(s).upper() not in ("N.R.", "NONE", "")]

            # Conditions
            reaction["conditions"] = {
                "temperature": _clean_field(entry_data.get("Temperature") or entry_data.get("temperature")),
                "time": _clean_field(entry_data.get("Duration") or entry_data.get("Time") or entry_data.get("time")),
                "atmosphere": _clean_field(entry_data.get("Air/Inert") or entry_data.get("Air/Inert")),
                "pressure": _clean_field(entry_data.get("Pressure") or entry_data.get("pressure")),
            }

            # Outcomes
            yield_val = _clean_field(entry_data.get("Yield") or entry_data.get("yield"))
            outcomes: Dict[str, str] = {"yield": yield_val}
            yield_type = _clean_field(entry_data.get("Yield type") or entry_data.get("Yield type"))
            if yield_type and yield_type != "N.R.":
                outcomes["yield_type"] = yield_type
            other_info = _clean_field(entry_data.get("Other product info") or entry_data.get("Other product info"))
            if other_info and other_info != "N.R.":
                outcomes["other_product_info"] = other_info
            reaction["outcomes"] = outcomes

            # Substitutions
            subs = entry_data.get("Substitutions") or entry_data.get("substitutions")
            if subs:
                reaction["substitutions"] = subs if isinstance(subs, str) else json.dumps(subs)

            # Electrochemistry fields
            for ef in ("Anode", "Cathode", "Current", "Electrolytes"):
                val = entry_data.get(ef)
                if val:
                    reaction[ef.lower()] = _clean_field(val)

            # Photocatalysis fields
            pc = entry_data.get("Photocatalyst") or entry_data.get("photocatalyst")
            if pc:
                reaction["photocatalyst"] = _clean_field(pc)
            pc_smiles = entry_data.get("Photocatalyst_SMILES") or entry_data.get("photocatalyst_smiles")
            if pc_smiles:
                reaction["photocatalyst_smiles"] = str(pc_smiles)
            irr = entry_data.get("irradiation conditions") or entry_data.get("irradiation conditions")
            if irr:
                reaction["irradiation_conditions"] = _clean_field(irr)

            # Chemicals / Additives
            chems = entry_data.get("Chemicals") or entry_data.get("chemicals") or entry_data.get("Additives") or []
            reaction["additives"] = _normalize_chemical_list(chems)

            # Others
            others = entry_data.get("Others") or entry_data.get("others")
            if others:
                reaction["others"] = others if isinstance(others, str) else json.dumps(others)

            # Footnotes (resolved text)
            fn_refs = entry_data.get("Footnote") or entry_data.get("footnotes") or ""
            if fn_refs and footnotes:
                resolved = []
                refs = fn_refs if isinstance(fn_refs, list) else [fn_refs]
                for ref in refs:
                    ref_str = str(ref).strip()
                    if ref_str in footnotes:
                        resolved.append(f"{ref_str}: {footnotes[ref_str]}")
                if resolved:
                    reaction["footnotes"] = resolved

            reactions.append(reaction)

        return reactions


# === Shared formatting utilities ===

def _normalize_chemical_list(raw) -> List[Dict[str, str]]:
    """Normalise a raw chemical list to ``[{"name": "...", "quantity": "..."}]``.

    Handles entries like ``"Pd(OAc)2 (5 mol%)"`` by looking for a
    trailing parenthetical group that looks like a quantity.
    """
    if not raw:
        return []
    if isinstance(raw, str):
        raw = [raw]
    result: List[Dict[str, str]] = []
    _qty_re = re.compile(
        r"^(.+?)\s+([\[(]([^\)\]]*?(?:\d|mol[%\s]|equiv|mmol|mL|mg|g|mol|M|mL|h|°C|wt%|v/v)[^\)\]]*?)[\])]?)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    for item in raw:
        if isinstance(item, dict):
            name = item.get("name", item.get("Name", ""))
            qty = item.get("quantity", item.get("Quantity", item.get("loading", item.get("Loading", ""))))
            smiles = item.get("smiles", item.get("SMILES", ""))
            if name:
                entry = {"name": str(name).strip(), "quantity": str(qty).strip() if qty else ""}
                if smiles:
                    entry["smiles"] = str(smiles).strip()
                result.append(entry)
        elif isinstance(item, str) and item.strip():
            stripped = item.strip()
            m = _qty_re.match(stripped)
            if m:
                name_part = m.group(1).strip()
                qty_part = m.group(3).strip()
                result.append({"name": name_part, "quantity": qty_part})
            else:
                result.append({"name": stripped, "quantity": ""})
    return result


def _clean_field(value) -> str:
    """Normalise a field value to a clean string."""
    if value is None:
        return "N.R."
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False) if value else "N.R."
    s = str(value).strip()
    return s if s and s.upper() not in ("NONE", "NULL", "") else "N.R."


# ================================================================
# 3. ReactionNormalizer
# ================================================================

# --- Chemical name dictionary ---

COMMON_CHEMICAL_NAMES: Dict[str, str] = {
    # -- Solvents --
    "DCM": "Dichloromethane", "CH2Cl2": "Dichloromethane",
    "THF": "Tetrahydrofuran", "DMF": "N,N-Dimethylformamide",
    "DMA": "N,N-Dimethylacetamide", "NMP": "N-Methyl-2-pyrrolidone",
    "DMSO": "Dimethyl sulfoxide", "MeCN": "Acetonitrile", "ACN": "Acetonitrile",
    "EtOH": "Ethanol", "MeOH": "Methanol", "iPrOH": "Isopropanol",
    "IPA": "Isopropanol", "tBuOH": "tert-Butanol", "nBuOH": "n-Butanol",
    "toluene": "Toluene", "PhMe": "Toluene", "hexane": "n-Hexane",
    "hexanes": "n-Hexanes", "EtOAc": "Ethyl acetate", "EA": "Ethyl acetate",
    "MTBE": "Methyl tert-butyl ether", "DCE": "1,2-Dichloroethane",
    "CHCl3": "Chloroform", "CPME": "Cyclopentyl methyl ether",
    "dioxane": "1,4-Dioxane", "DME": "1,2-Dimethoxyethane",
    "TFA": "Trifluoroacetic acid", "AcOH": "Acetic acid", "HOAc": "Acetic acid",
    "Py": "Pyridine", "DMAP": "4-Dimethylaminopyridine",
    "HFIP": "Hexafluoroisopropanol", "TFE": "2,2,2-Trifluoroethanol",
    "1,2-DCE": "1,2-Dichloroethane",
    "nBuOAc": "n-Butyl acetate", "iPrOAc": "Isopropyl acetate",
    # -- Bases --
    "K2CO3": "Potassium carbonate", "Cs2CO3": "Cesium carbonate",
    "Na2CO3": "Sodium carbonate", "NaHCO3": "Sodium bicarbonate",
    "K3PO4": "Tripotassium phosphate", "KOtBu": "Potassium tert-butoxide",
    "NaOtBu": "Sodium tert-butoxide", "Et3N": "Triethylamine",
    "TEA": "Triethylamine", "DIPEA": "N,N-Diisopropylethylamine",
    "DABCO": "1,4-Diazabicyclo[2.2.2]octane",
    "DBU": "1,8-Diazabicyclo[5.4.0]undec-7-ene",
    "DBN": "1,5-Diazabicyclo[4.3.0]non-5-ene",
    "TMG": "Tetramethylguanidine", "LDA": "Lithium diisopropylamide",
    "nBuLi": "n-Butyllithium", "NaH": "Sodium hydride", "KH": "Potassium hydride",
    "NaHMDS": "Sodium hexamethyldisilazide", "LiHMDS": "Lithium hexamethyldisilazide",
    "KHMDS": "Potassium hexamethyldisilazide",
    # -- Ligands --
    "PPh3": "Triphenylphosphine", "XPhos": "2-Dicyclohexylphosphino-2',4',6'-triisopropylbiphenyl",
    "SPhos": "Diphenylphosphino(2,6-dimethoxyphenyl)",
    "BINAP": "2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl",
    "R-BINAP": "(R)-2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl",
    "S-BINAP": "(S)-2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl",
    "dppe": "1,2-Bis(diphenylphosphino)ethane",
    "dppp": "1,3-Bis(diphenylphosphino)propane",
    "dppb": "1,4-Bis(diphenylphosphino)butane",
    "dppf": "1,1'-Bis(diphenylphosphino)ferrocene",
    "PCy3": "Tricyclohexylphosphine",
    "JohnPhos": "2-(Di-tert-butylphosphino)biphenyl",
    "BrettPhos": "2-(Dicyclohexylphosphino)-2',4',6'-triisopropylbiphenyl",
    "tBuXPhos": "2-Di-tert-butylphosphino-2',4',6'-triisopropylbiphenyl",
    "DavePhos": "2-Dicyclohexylphosphino-2',6'-dimethoxybiphenyl",
    "MeO-Biphep": "(R)-6,6'-Dimethoxy-2,2'-bis(diphenylphosphino)-1,1'-biphenyl",
    # -- Catalysts --
    "Pd(PPh3)4": "Tetrakis(triphenylphosphine)palladium(0)",
    "Pd(dppf)Cl2": "[1,1'-Bis(diphenylphosphino)ferrocene]palladium(II) dichloride",
    "Pd(OAc)2": "Palladium(II) acetate", "PdCl2": "Palladium(II) chloride",
    "Pd/C": "Palladium on carbon",
    "Pd2(dba)3": "Tris(dibenzylideneacetone)dipalladium(0)",
    "Pd(dba)2": "Bis(dibenzylideneacetone)palladium(0)",
    "Ni(cod)2": "Bis(1,5-cyclooctadiene)nickel(0)",
    "CuI": "Copper(I) iodide", "CuBr": "Copper(I) bromide",
    "CuCl": "Copper(I) chloride", "Cu(OAc)2": "Copper(II) acetate",
    "CuTC": "Copper(I) thiophene-2-carboxylate",
    "FeCl3": "Iron(III) chloride", "FeCl2": "Iron(II) chloride",
    "AlCl3": "Aluminium chloride", "TiCl4": "Titanium(IV) chloride",
    "SnCl4": "Tin(IV) chloride",
    "NiCl2": "Nickel(II) chloride", "CoCl2": "Cobalt(II) chloride",
    "RuCl2(p-cymene)2": "Dichloro(p-cymene)ruthenium(II) dimer",
    # -- Photocatalysts --
    "Ir(ppy)3": "Tris(2-phenylpyridinato-C2,N)iridium(III)",
    "Ir[dF(CF3)ppy]2(dtbbpy)PF6": "Iridium(III) bis[4,4'-di-tert-butyl-2,2'-bipyridine] bis[3,5-difluoro-2-(trifluoromethyl)phenylpyridinato]",
    "Ru(bpy)3Cl2": "Tris(2,2'-bipyridine)ruthenium(II) chloride",
    "Eosin Y": "2',4',5',7'-Tetrabromofluorescein",
    "Acr+Mes Cl-": "9-Mesityl-10-methylacridinium perchlorate",
    "4CzIPN": "2,4,5,6-tetra(9H-carbazol-9-yl)isophthalonitrile",
    "Mes-Acr+Cl-": "9-Mesityl-10-methylacridinium chloride",
    "fac-Ir(ppy)3": "Facial-tris(2-phenylpyridinato)iridium(III)",
    # -- Electrolytes / Additives / Salts --
    "nBu4NBF4": "Tetrabutylammonium tetrafluoroborate",
    "n-Bu4NBF4": "Tetrabutylammonium tetrafluoroborate",
    "Bu4NBF4": "Tetrabutylammonium tetrafluoroborate",
    "TBABF4": "Tetrabutylammonium tetrafluoroborate",
    "nBu4NCl": "Tetrabutylammonium chloride",
    "n-Bu4NCl": "Tetrabutylammonium chloride",
    "Bu4NCl": "Tetrabutylammonium chloride",
    "TBAC": "Tetrabutylammonium chloride",
    "TBACl": "Tetrabutylammonium chloride",
    "nBu4NPF6": "Tetrabutylammonium hexafluorophosphate",
    "n-Bu4NPF6": "Tetrabutylammonium hexafluorophosphate",
    "Bu4NPF6": "Tetrabutylammonium hexafluorophosphate",
    "TBAPF6": "Tetrabutylammonium hexafluorophosphate",
    "nBu4NI": "Tetrabutylammonium iodide",
    "n-Bu4NI": "Tetrabutylammonium iodide",
    "nBu4NClO4": "Tetrabutylammonium perchlorate",
    "TBAClO4": "Tetrabutylammonium perchlorate",
    "TBAB": "Tetrabutylammonium bromide",
    "nBu4NBr": "Tetrabutylammonium bromide",
    "TBAF": "Tetrabutylammonium fluoride",
    "LiClO4": "Lithium perchlorate", "NaBF4": "Sodium tetrafluoroborate",
    "NaBArF": "Sodium tetrakis[3,5-bis(trifluoromethyl)phenyl]borate",
    "KF": "Potassium fluoride", "CsF": "Cesium fluoride",
    "NaI": "Sodium iodide", "KI": "Potassium iodide",
    "NaCl": "Sodium chloride", "KCl": "Potassium chloride",
    "LiBr": "Lithium bromide", "NaBr": "Sodium bromide",
    "KBr": "Potassium bromide", "LiCl": "Lithium chloride",
    # -- Protecting groups / Reagents --
    "TMS": "Trimethylsilyl", "TBS": "tert-Butyldimethylsilyl",
    "TBDPS": "tert-Butyldiphenylsilyl", "TIPS": "Triisopropylsilyl",
    "MsCl": "Methanesulfonyl chloride", "TsCl": "p-Toluenesulfonyl chloride",
    "TfOH": "Trifluoromethanesulfonic acid", "Tf2O": "Trifluoromethanesulfonic anhydride",
    "TfNH2": "Trifluoromethanesulfonamide",
    "TMSN3": "Trimethylsilyl azide", "TMS-CN": "Trimethylsilyl cyanide",
    # -- Oxidants / Reductants --
    "DIBAL-H": "Diisobutylaluminium hydride", "DIBAL": "Diisobutylaluminium hydride",
    "NaBH4": "Sodium borohydride", "LiAlH4": "Lithium aluminium hydride",
    "PCC": "Pyridinium chlorochromate", "PDC": "Pyridinium dichromate",
    "DMP": "Dess-Martin periodinane", "mCPBA": "meta-Chloroperoxybenzoic acid",
    "IBX": "2-Iodoxybenzoic acid", "NBS": "N-Bromosuccinimide",
    "NCS": "N-Chlorosuccinimide", "NIS": "N-Iodosuccinimide",
    "TBHP": "tert-Butyl hydroperoxide", "H2O2": "Hydrogen peroxide",
    "O2": "Oxygen", "CO2": "Carbon dioxide", "NH3": "Ammonia",
    "H2": "Hydrogen", "N2": "Nitrogen", "Ar": "Argon",
    "NEt3": "Triethylamine", "NMe2": "Dimethylamine",
    # -- Coupling reagents --
    "DCC": "N,N'-Dicyclohexylcarbodiimide",
    "EDC": "1-Ethyl-3-(3-dimethylaminopropyl)carbodiimide",
    "HATU": "Hexafluorophosphate azabenzotriazole tetramethyl uronium",
    "DIAD": "Diisopropyl azodicarboxylate", "DEAD": "Diethyl azodicarboxylate",
    "PPTS": "Pyridinium p-toluenesulfonate",
    # -- Metal triflates --
    "Sc(OTf)3": "Scandium(III) triflate", "In(OTf)3": "Indium(III) triflate",
    "Yb(OTf)3": "Ytterbium(III) triflate", "AgOTf": "Silver triflate",
    "AgNO3": "Silver nitrate", "Ag2O": "Silver(I) oxide",
    "Cu(OTf)2": "Copper(II) triflate", "Zn(OTf)2": "Zinc(II) triflate",
    "Fe(OTf)3": "Iron(III) triflate",
    # -- Others --
    "TPAP": "Tetrapropylammonium perruthenate",
    "NMO": "N-Methylmorpholine N-oxide",
    "pyridine": "Pyridine", "quinoline": "Quinoline",
    "TEMPO": "2,2,6,6-Tetramethylpiperidine 1-oxyl",
    "BAIB": "Bis(acetoxy)iodobenzene",
    "PIDA": "Phenyliodine diacetate",
    "Selectfluor": "1-Chloromethyl-4-fluoro-1,4-diazoniabicyclo[2.2.2]octane bis(tetrafluoroborate)",
    "NFSI": "N-Fluorobenzenesulfonimide",
}

ENTITY_RESOLUTION_KEYS: List[str] = [
    "Catalyst", "Ligand", "Solvents", "Chemicals", "Additives", "Electrolytes",
    "Photocatalyst", "Anode", "Cathode",
]


class ReactionNormalizer:
    """Post-processing for extracted chemical data.

    Provides:
    - Abbreviation resolution via dictionary lookup (COMMON_CHEMICAL_NAMES)
    - Chemical-quantity pair splitting
    - Mixed solvent handling
    - Optional LLM-based SMILES validation (NO PubChem dependency)

    Args:
        provider: LLM provider for optional SMILES validation.
        model: Model name for SMILES validation.
        api_key: API key for SMILES validation.
        enable_smiles_validation: Whether to validate SMILES via LLM (default False).
        common_names: Custom abbreviation dictionary override.
        resolution_keys: List of dict keys to process in reaction entries.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: str = "",
        enable_smiles_validation: bool = False,
        common_names: Optional[Dict[str, str]] = None,
        resolution_keys: Optional[List[str]] = None,
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.enable_smiles_validation = enable_smiles_validation
        self.common_names = common_names or COMMON_CHEMICAL_NAMES
        self.resolution_keys = resolution_keys or ENTITY_RESOLUTION_KEYS

        if enable_smiles_validation and not api_key:
            logger.info("[ReactionNormalizer] SMILES validation requested but no API key provided; "
                        "validation will be skipped")

    def normalize_reactions(self, reactions: List[Dict]) -> List[Dict]:
        """Normalize a list of reaction dicts.

        Applies abbreviation resolution, quantity parsing, and mixed
        solvent handling to each reaction entry.

        Args:
            reactions: List of reaction dicts.

        Returns:
            The same list, normalized in-place.
        """
        for reaction in reactions:
            self._normalize_reaction(reaction)
        return reactions

    def _normalize_reaction(self, reaction: Dict) -> Dict:
        """Normalize a single reaction dict."""
        for key in self.resolution_keys:
            try:
                value = reaction.get(key)
                if value:
                    if isinstance(value, str):
                        reaction[key] = self._replace_abbreviations(value)
                    elif isinstance(value, list):
                        reaction[key] = [
                            self._replace_abbreviations(str(v)) if isinstance(v, str) else v
                            for v in value
                        ]
            except Exception as e:
                logger.debug(f"[ReactionNormalizer] Error normalizing key '{key}': {e}")

        # Also normalize catalyst and ligand if present as strings
        for key in ("catalyst", "ligand", "photocatalyst"):
            val = reaction.get(key)
            if isinstance(val, str) and val and val != "N.R.":
                reaction[key] = self._replace_abbreviations(val)

        # Normalize reactants and products
        for list_key in ("reactants", "products", "additives"):
            items = reaction.get(list_key, [])
            for item in items:
                if isinstance(item, dict) and item.get("name"):
                    item["name"] = self._replace_abbreviations(item["name"])

        # Normalize solvents list
        solvents = reaction.get("solvents", [])
        reaction["solvents"] = [
            self._replace_abbreviations(s) if isinstance(s, str) else s
            for s in solvents
        ]

        return reaction

    def resolve_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve chemical entities in a single optimization run entry.

        Walks through the resolution keys, splits chemical-quantity pairs,
        and resolves abbreviations.

        Args:
            entry: A single reaction entry dict (optimization runs format).

        Returns:
            The same dict, mutated in-place with resolved entities.
        """
        for key in self.resolution_keys:
            try:
                value = entry.get(key)
                if value:
                    resolved = self._split_and_resolve(str(value))
                    entry[key] = resolved
            except Exception as e:
                logger.debug(f"[ReactionNormalizer] Error resolving key '{key}': {e}")

        # Consolidate multi-solvent entries
        solvents = entry.get("Solvents") or entry.get("Solvent")
        if isinstance(solvents, list) and len(solvents) > 1:
            names = ":".join(str(s[0]) if isinstance(s, (list, tuple)) else str(s) for s in solvents)
            values_parts = [str(s[1]) if isinstance(s, (list, tuple)) and len(s) > 1 else ""
                            for s in solvents]
            values_str = ":".join(values_parts)
            values_str = values_str if not all(v in ("", "None", "N.R.") for v in values_str) else None
            entry["Solvents"] = [[names, values_str]] if values_str else [[names, None]]

        return entry

    def validate_smiles_batch(self, compounds: List[Dict]) -> List[Dict]:
        """LLM-based SMILES validation (replaces PubChem lookup).

        Sends compound name + SMILES pairs to a text LLM for validation.
        Corrects invalid SMILES and removes clearly wrong ones.

        Args:
            compounds: List of compound dicts with 'name' and 'smiles' keys.

        Returns:
            The same list with SMILES corrected or removed.
        """
        if not self.enable_smiles_validation or not self.api_key or not self.provider:
            return compounds

        # Build pairs for validation
        pairs = []
        for c in compounds:
            if c.get("name") and c.get("smiles"):
                pairs.append(f'  "{c["name"]}": "{c["smiles"]}"')

        if not pairs:
            return compounds

        prompt = SMILES_VALIDATION_PROMPT.format(smiles_pairs="\n".join(pairs))
        user_text = f"Validate these SMILES:\n\n" + "\n".join(pairs)

        try:
            result = rl_call_text(
                user_text, self.provider, self.model or "deepseek-chat",
                self.api_key, prompt
            )
            if result and isinstance(result, dict):
                for c in compounds:
                    name = c.get("name", "")
                    val = result.get(name)
                    if val is None:
                        # Try case-insensitive match
                        for rk, rv in result.items():
                            if rk.lower() == name.lower():
                                val = rv
                                break
                    if val is not None:
                        val = str(val).strip().lower()
                        if val.startswith("corrected:"):
                            c["smiles"] = val.replace("corrected:", "").strip()
                        elif val == "invalid":
                            c["smiles"] = ""
                logger.debug(f"[ReactionNormalizer] SMILES validation complete for {len(pairs)} compounds")
        except Exception as e:
            logger.debug(f"[ReactionNormalizer] SMILES validation failed: {e}")

        return compounds

    def _split_and_resolve(self, value: str) -> List[Tuple[str, Optional[str]]]:
        """Split a chemical string into (name, quantity) pairs and resolve.

        Handles comma-separated lists with parenthetical quantities and
        respects bracket nesting.

        Args:
            value: A string like ``"Pd(OAc)2 (5 mol%), THF (5 mL)"``.

        Returns:
            A list of ``(resolved_name, quantity_or_None)`` tuples.
        """
        components = self._split_respecting_brackets(value)
        result: List[Tuple[str, Optional[str]]] = []

        for component in components:
            component = component.strip()
            if not component:
                continue

            match = re.match(r'(.+?)(\s*(\([^\)]+\)|\[[^\]]+\]))?\s*$', component)
            if match:
                name = match.group(1).strip()
                qty_raw = match.group(2)
                quantity = (qty_raw.strip().replace('(', '').replace(')', '')
                            .replace('[', '').replace(']', '') if qty_raw else None)

                resolved = self._process_mixed_chemicals(name)
                result.append((resolved, quantity))
            else:
                result.append((component, None))

        return result

    def _split_respecting_brackets(self, text: str) -> List[str]:
        """Split text on commas, respecting bracket nesting.

        Args:
            text: Comma-separated chemical string.

        Returns:
            List of individual chemical component strings.
        """
        components: List[str] = []
        current: List[str] = []
        bracket_level = 0

        for char in text:
            if char in "([":
                bracket_level += 1
            elif char in ")]":
                bracket_level -= 1

            if char == ',' and bracket_level == 0:
                components.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            components.append(''.join(current).strip())

        return components

    def _process_mixed_chemicals(self, chemical: str) -> str:
        """Resolve mixed chemical systems (solvents, electrolytes).

        Handles delimiters ``:``, ``/``, en-dash. Individual
        components are resolved via abbreviation lookup.

        Args:
            chemical: A chemical name, possibly with mixed system delimiters.

        Returns:
            The resolved string.
        """
        if ":" in chemical or "/" in chemical or "\u2013" in chemical:
            # Normalize all delimiters to colon
            chemical = re.sub(r"[:/\u2013]", ":", chemical)
            parts = chemical.split(":")
            resolved = [self._resolve_single(p.strip()) for p in parts]
            return ":".join(resolved)
        else:
            return self._resolve_single(chemical)

    def _resolve_single(self, chemical: str) -> str:
        """Resolve a single chemical name via abbreviation lookup.

        Args:
            chemical: A chemical name or abbreviation.

        Returns:
            The resolved name.
        """
        return self._replace_chemical(chemical)

    def _replace_chemical(self, chemical: str) -> str:
        """Look up a chemical abbreviation in the common names dictionary.

        Args:
            chemical: Abbreviation to look up.

        Returns:
            Full name if found, otherwise the original abbreviation.
        """
        try:
            return self.common_names[chemical]
        except KeyError:
            return chemical

    def _replace_abbreviations(self, text: str) -> str:
        """Replace all known abbreviations in a text string.

        Uses word-boundary matching for safety.

        Args:
            text: Text containing potential abbreviations.

        Returns:
            Text with abbreviations expanded.
        """
        if not isinstance(text, str) or not text.strip():
            return text

        result = text
        for abbrev, full_name in self.common_names.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, full_name, result, flags=re.IGNORECASE)

        return result


# ================================================================
# 4. ReactionLens Pipeline
# ================================================================

class ReactionLens:
    """Orchestrated extraction pipeline using signal detection and adaptive extraction.

    Same public API as before, but internally uses the new components:
    - ReactionSignalDetector (single-pass VLM signal detection)
    - AdaptiveExtractor (content-adaptive extraction with self-verification)
    - ReactionNormalizer (no PubChem; LLM-based SMILES validation)

    Pipeline flow:
    1. PDF to Pages (PyMuPDF rendering)
    2. Signal Detection (single VLM pass per page to find content signals)
    3. Filtering (skip low-priority/irrelevant signals)
    4. Adaptive Extraction per signal (with self-verification)
    5. Normalization (abbreviation resolution, quantity parsing)
    6. Deduplication and compound extraction

    Usage::

        lens = ReactionLens(provider="deepseek", api_key="sk-...")
        result = lens.extract_from_pdf("paper.pdf")
        # result["reactions"] -- list of reaction dicts
        # result["compounds"] -- list of compound dicts
        # result["metadata"] -- extraction metadata

    Args:
        provider: LLM provider (``deepseek``, ``openai``, ``gemini``, ``anthropic``).
        model: Model name. If None, uses the provider's default.
        api_key: API key for the LLM provider.
        enable_filtering: Whether to filter irrelevant pages (default True).
        enable_segmentation: Whether to segment compound pages (default True).
        enable_entity_resolution: Whether to run post-processing (default True).
        default_extraction_mode: Default extraction mode (default ``comprehensive``).
        dpi: Page rendering DPI (default 200).
        max_pages: Maximum pages to process (default 50).
    """

    DEFAULT_MODELS = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o",
        "gemini": "gemini-2.0-flash",
        "anthropic": "claude-3-5-sonnet-20241022",
    }

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        api_key: str = "",
        enable_filtering: bool = True,
        enable_segmentation: bool = True,
        enable_entity_resolution: bool = True,
        default_extraction_mode: str = ExtractionMode.COMPREHENSIVE.value,
        dpi: int = RL_DEFAULT_DPI,
        max_pages: int = 50,
    ):
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS.get(provider, "deepseek-chat")
        self.api_key = api_key
        self.enable_filtering = enable_filtering
        self.enable_segmentation = enable_segmentation
        self.enable_entity_resolution = enable_entity_resolution
        self.default_extraction_mode = default_extraction_mode
        self.dpi = dpi
        self.max_pages = max_pages

        # Initialize sub-components with new architecture
        self.detector = ReactionSignalDetector(
            provider=provider,
            model=model or self.model,
            api_key=api_key,
            dpi=dpi,
        )
        self.extractor = AdaptiveExtractor(
            provider=provider,
            model=model or self.model,
            api_key=api_key,
            enable_verification=True,
        )
        self.normalizer = ReactionNormalizer(
            provider=provider,
            model=model or self.model,
            api_key=api_key,
            enable_smiles_validation=enable_entity_resolution,
        )

    def extract_from_pdf(
        self,
        pdf_path: str,
        extraction_mode: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the full ReactionLens pipeline on a PDF (synchronous).

        Args:
            pdf_path: Path to the PDF file.
            extraction_mode: Override the default extraction mode.
            max_pages: Override the max pages setting.

        Returns:
            A dict with:
            - ``reactions``: list of all extracted reaction dicts
            - ``compounds``: list of compound dicts
            - ``metadata``: pipeline execution metadata
        """
        max_pages = max_pages or self.max_pages
        mode = extraction_mode or self.default_extraction_mode

        result: Dict[str, Any] = {
            "reactions": [],
            "compounds": [],
            "metadata": {
                "pipeline": "reactionlens",
                "architecture": "signal-driven-adaptive",
                "provider": self.provider,
                "model": self.model,
                "extraction_mode": mode,
                "total_pages": 0,
                "signals_detected": 0,
                "signals_processed": 0,
                "filtered_signals": 0,
                "pages_processed": [],
                "entity_resolution": self.enable_entity_resolution,
                "filtering": self.enable_filtering,
                "segmentation": self.enable_segmentation,
            },
        }

        if not HAS_PYMUPDF:
            logger.error("[ReactionLens] PyMuPDF not available; cannot process PDF")
            return result

        try:
            # Stage 1: Signal Detection (single-pass VLM per page)
            logger.info("[ReactionLens] Stage 1: Signal detection")
            signals = self.detector.detect_signals(pdf_path, max_pages)
            result["metadata"]["signals_detected"] = len(signals)

            if not signals:
                logger.info("[ReactionLens] No signals detected")
                return result

            # Stage 2: Filter signals
            filtered_signals = self._filter_signals(signals)
            result["metadata"]["filtered_signals"] = len(signals) - len(filtered_signals)
            signals = filtered_signals

            # Stage 3: Extract text for context
            try:
                doc = fitz.open(pdf_path)
                page_texts = {}
                for pn in range(min(len(doc), max_pages)):
                    try:
                        page_texts[pn + 1] = doc[pn].get_text()
                    except Exception:
                        page_texts[pn + 1] = ""
                doc.close()
            except Exception:
                page_texts = {}

            # Stage 4: Adaptive Extraction per signal
            logger.info(f"[ReactionLens] Stage 4: Adaptive extraction on {len(signals)} signals")
            for signal in signals:
                if not signal.base64_image:
                    continue

                page_info = {
                    "page": signal.page_number,
                    "signal_type": signal.signal_type,
                    "priority": signal.priority,
                    "reactions": 0,
                }

                try:
                    context = page_texts.get(signal.page_number, "")[:2000]

                    extraction = self.extractor.extract(signal, context_text=context)

                    reactions = extraction.get("reactions", [])
                    result["reactions"].extend(reactions)
                    result["metadata"]["signals_processed"] += 1
                    page_info["reactions"] = len(reactions)

                except Exception as e:
                    logger.error(
                        f"[ReactionLens] Extraction failed for signal "
                        f"(page {signal.page_number}, type={signal.signal_type}): {e}"
                    )

                result["metadata"]["pages_processed"].append(page_info)

            # Stage 5: Entity Resolution
            if self.enable_entity_resolution and result["reactions"]:
                logger.info("[ReactionLens] Stage 5: Entity resolution")
                result["reactions"] = self.normalizer.normalize_reactions(result["reactions"])

            # Stage 6: Deduplication and compound extraction
            result["reactions"] = self._deduplicate_reactions(result["reactions"])
            result["compounds"] = self._extract_compounds(result["reactions"])

            logger.info(
                f"[ReactionLens] Pipeline complete: {len(result['reactions'])} reactions, "
                f"{len(result['compounds'])} compounds from {len(signals)} signals"
            )

        except Exception as e:
            logger.error(f"[ReactionLens] Pipeline failed: {e}")
            result["metadata"]["error"] = str(e)

        return result

    async def extract_from_pdf_async(
        self,
        pdf_path: str,
        extraction_mode: Optional[str] = None,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the full ReactionLens pipeline on a PDF (asynchronous).

        Processes signals concurrently where possible for faster extraction.

        Args:
            pdf_path: Path to the PDF file.
            extraction_mode: Override the default extraction mode.
            max_pages: Override the max pages setting.

        Returns:
            Same dict format as :meth:`extract_from_pdf`.
        """
        max_pages = max_pages or self.max_pages
        mode = extraction_mode or self.default_extraction_mode

        result: Dict[str, Any] = {
            "reactions": [],
            "compounds": [],
            "metadata": {
                "pipeline": "reactionlens",
                "architecture": "signal-driven-adaptive",
                "provider": self.provider,
                "model": self.model,
                "extraction_mode": mode,
                "total_pages": 0,
                "signals_detected": 0,
                "signals_processed": 0,
                "filtered_signals": 0,
                "async": True,
            },
        }

        if not HAS_PYMUPDF:
            return result

        try:
            # Stage 1: Async signal detection
            signals = await self.detector.detect_signals_async(pdf_path, max_pages)
            result["metadata"]["signals_detected"] = len(signals)

            if not signals:
                return result

            # Stage 2: Filter signals
            filtered_signals = self._filter_signals(signals)
            result["metadata"]["filtered_signals"] = len(signals) - len(filtered_signals)
            signals = filtered_signals

            # Stage 3: Extract page texts
            try:
                doc = fitz.open(pdf_path)
                page_texts = {}
                for pn in range(min(len(doc), max_pages)):
                    try:
                        page_texts[pn + 1] = doc[pn].get_text()
                    except Exception:
                        page_texts[pn + 1] = ""
                doc.close()
            except Exception:
                page_texts = {}

            # Stage 4: Async extraction per signal
            async def process_signal(signal):
                if not signal.base64_image:
                    return []
                try:
                    context = page_texts.get(signal.page_number, "")[:2000]
                    extraction = await self.extractor.extract_async(signal, context_text=context)
                    return extraction.get("reactions", [])
                except Exception as e:
                    logger.error(f"[ReactionLens] Async extraction error (p{signal.page_number}): {e}")
                    return []

            tasks = [process_signal(s) for s in signals]
            all_reactions = await asyncio.gather(*tasks)

            for i, signal_reactions in enumerate(all_reactions):
                result["reactions"].extend(signal_reactions)
                if signal_reactions:
                    result["metadata"]["signals_processed"] += 1

            # Stage 5: Entity resolution
            if self.enable_entity_resolution and result["reactions"]:
                result["reactions"] = self.normalizer.normalize_reactions(result["reactions"])

            # Stage 6: Dedup and compound extraction
            result["reactions"] = self._deduplicate_reactions(result["reactions"])
            result["compounds"] = self._extract_compounds(result["reactions"])

        except Exception as e:
            logger.error(f"[ReactionLens] Async pipeline failed: {e}")
            result["metadata"]["error"] = str(e)

        return result

    def extract_from_images(
        self,
        images: List[Tuple[int, str]],
        extraction_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract reactions from a list of pre-rendered page images.

        Wraps each image as a ContentSignal and runs the adaptive extractor.

        Args:
            images: List of ``(page_number, base64_image)`` tuples.
            extraction_mode: Override the default extraction mode.

        Returns:
            Same dict format as :meth:`extract_from_pdf`.
        """
        mode = extraction_mode or self.default_extraction_mode
        result: Dict[str, Any] = {
            "reactions": [],
            "compounds": [],
            "metadata": {
                "pipeline": "reactionlens",
                "architecture": "signal-driven-adaptive",
                "provider": self.provider,
                "model": self.model,
                "extraction_mode": mode,
                "total_images": len(images),
            },
        }

        for page_number, base64_img in images:
            # Create a signal from the image directly
            signal = ContentSignal(
                page_number=page_number,
                signal_type="unknown",
                priority=3,
                base64_image=base64_img,
                source="pre_rendered",
            )

            extraction = self.extractor.extract(signal)
            result["reactions"].extend(extraction.get("reactions", []))

        if self.enable_entity_resolution and result["reactions"]:
            result["reactions"] = self.normalizer.normalize_reactions(result["reactions"])

        result["reactions"] = self._deduplicate_reactions(result["reactions"])
        result["compounds"] = self._extract_compounds(result["reactions"])
        return result

    def _filter_signals(self, signals: List[ContentSignal]) -> List[ContentSignal]:
        """Filter signals based on priority and relevance.

        Args:
            signals: List of detected signals.

        Returns:
            Filtered list with low-priority and irrelevant signals removed.
        """
        if not self.enable_filtering:
            return signals

        filtered = []
        for signal in signals:
            # Skip text-only signals
            if signal.signal_type == SignalType.TEXT_ONLY.value and signal.priority == 0:
                continue
            # Skip very low confidence signals
            if signal.confidence < 0.3 and signal.priority < 2:
                continue
            # Skip unknown type with low priority
            if signal.signal_type == SignalType.UNKNOWN.value and signal.priority == 0:
                continue
            filtered.append(signal)

        logger.info(f"[ReactionLens] Filtered {len(signals)} -> {len(filtered)} signals")
        return filtered

    @staticmethod
    def _deduplicate_reactions(reactions: List[Dict]) -> List[Dict]:
        """Remove duplicate reactions based on entry label and key fields.

        Args:
            reactions: List of reaction dicts.

        Returns:
            Deduplicated list, preserving order of first occurrence.
        """
        if len(reactions) <= 1:
            return reactions

        seen: Dict[str, Dict] = {}
        for reaction in reactions:
            # Create a dedup key from entry and key fields
            entry = str(reaction.get("entry", ""))
            catalyst = str(reaction.get("catalyst", ""))
            ligand = str(reaction.get("ligand", ""))
            key = f"{entry}|{catalyst}|{ligand}"

            if key not in seen:
                seen[key] = reaction
            else:
                # Prefer the one with more data
                existing = seen[key]
                existing_fields = sum(1 for v in existing.values() if v and v != "N.R." and v != [])
                new_fields = sum(1 for v in reaction.values() if v and v != "N.R." and v != [])
                if new_fields > existing_fields:
                    seen[key] = reaction

        deduped = list(seen.values())
        removed = len(reactions) - len(deduped)
        if removed > 0:
            logger.debug(f"[ReactionLens] Deduplicated: removed {removed} duplicate reactions")
        return deduped

    @staticmethod
    def _extract_compounds(reactions: List[Dict]) -> List[Dict]:
        """Extract unique compound list from reactions, including SMILES.

        Builds a deduplicated list of all compounds mentioned across
        reactions, including SMILES when available.

        Args:
            reactions: List of reaction dicts.

        Returns:
            List of unique compound dicts.
        """
        compounds_map: Dict[str, Dict] = {}

        for reaction in reactions:
            # Extract from reactants
            for r in reaction.get("reactants", []):
                if isinstance(r, dict) and r.get("name"):
                    name = r["name"]
                    if name not in compounds_map:
                        compounds_map[name] = {
                            "name": name,
                            "smiles": r.get("smiles", ""),
                            "role": "reactant",
                        }
                    elif r.get("smiles") and not compounds_map[name].get("smiles"):
                        compounds_map[name]["smiles"] = r["smiles"]

            # Extract from products
            for p in reaction.get("products", []):
                if isinstance(p, dict) and p.get("name"):
                    name = p["name"]
                    if name not in compounds_map:
                        compounds_map[name] = {
                            "name": name,
                            "smiles": p.get("smiles", ""),
                            "role": "product",
                        }
                    elif p.get("smiles") and not compounds_map[name].get("smiles"):
                        compounds_map[name]["smiles"] = p["smiles"]

            # Extract catalyst with SMILES
            catalyst = reaction.get("catalyst", "N.R.")
            catalyst_smiles = reaction.get("catalyst_smiles", "")
            if catalyst and catalyst != "N.R." and catalyst not in compounds_map:
                compounds_map[catalyst] = {
                    "name": catalyst,
                    "smiles": catalyst_smiles or "",
                    "role": "catalyst",
                }

            # Extract ligand with SMILES
            ligand = reaction.get("ligand", "N.R.")
            ligand_smiles = reaction.get("ligand_smiles", "")
            if ligand and ligand != "N.R." and ligand not in compounds_map:
                compounds_map[ligand] = {
                    "name": ligand,
                    "smiles": ligand_smiles or "",
                    "role": "ligand",
                }

            # Extract photocatalyst with SMILES
            pc = reaction.get("photocatalyst", "")
            pc_smiles = reaction.get("photocatalyst_smiles", "")
            if pc and pc not in compounds_map:
                compounds_map[pc] = {
                    "name": pc,
                    "smiles": pc_smiles or "",
                    "role": "photocatalyst",
                }

            # Extract solvents
            for s in reaction.get("solvents", []):
                if isinstance(s, str) and s and s not in compounds_map:
                    compounds_map[s] = {
                        "name": s,
                        "smiles": "",
                        "role": "solvent",
                    }

        compounds = list(compounds_map.values())

        # Clean up empty SMILES
        for c in compounds:
            if not c.get("smiles"):
                c["smiles"] = ""

        return compounds


# ================================================================
# Convenience Functions
# ================================================================

def extract_with_reactionlens(
    pdf_path: str,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function for one-shot extraction.

    Creates a ReactionLens instance and extracts all reactions from a PDF.

    Args:
        pdf_path: Path to the PDF file.
        provider: LLM provider (``deepseek``, ``openai``, ``gemini``, ``anthropic``).
        api_key: API key for the provider.
        model: Optional model name override.
        **kwargs: Additional keyword arguments passed to ReactionLens constructor.

    Returns:
        Same dict format as :meth:`ReactionLens.extract_from_pdf`.

    Example::

        result = extract_with_reactionlens(
            "paper.pdf", "deepseek", "sk-...", model="deepseek-chat"
        )
        for rxn in result["reactions"]:
            print(rxn["entry"], rxn["outcomes"]["yield"])
    """
    lens = ReactionLens(provider=provider, model=model, api_key=api_key, **kwargs)
    return lens.extract_from_pdf(pdf_path)


async def extract_with_reactionlens_async(
    pdf_path: str,
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Async convenience function for one-shot extraction.

    Args:
        pdf_path: Path to the PDF file.
        provider: LLM provider.
        api_key: API key for the provider.
        model: Optional model name override.
        **kwargs: Additional keyword arguments passed to ReactionLens constructor.

    Returns:
        Same dict format as :meth:`ReactionLens.extract_from_pdf_async`.
    """
    lens = ReactionLens(provider=provider, model=model, api_key=api_key, **kwargs)
    return await lens.extract_from_pdf_async(pdf_path)
