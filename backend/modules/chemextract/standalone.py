import logging
from typing import Dict, List, Optional, Any

from .config import TEXT_CHUNK_SIZE, CHUNK_OVERLAP
from .llm_providers import (
    _call_text_provider, _call_text_provider_async,
    _call_gemini_text, _call_gemini_text_async,
    _call_anthropic_text, _call_anthropic_text_async,
)
from .json_utils import _parse_json_response

logger = logging.getLogger(__name__)


def call_text_llm(text, provider, model, api_key):
    try:
        if provider in ('deepseek', 'openai'):
            return _call_text_provider(provider, model, api_key, text)
        elif provider == 'gemini':
            return _call_gemini_text(text, model, api_key)
        elif provider == 'anthropic':
            return _call_anthropic_text(text, model, api_key)
        else:
            return _call_text_provider(provider, model, api_key, text)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        # JSON parse errors, unexpected response shapes, or retry-exhausted
        # RuntimeError from _retry_on_failure. Other exceptions (bugs) propagate.
        logger.error(f"[ChemExtract] Text LLM call failed: {e}")
        return None


async def call_text_llm_async(text, provider, model, api_key):
    try:
        if provider in ('deepseek', 'openai'):
            return await _call_text_provider_async(provider, model, api_key, text)
        elif provider == 'gemini':
            return await _call_gemini_text_async(text, model, api_key)
        elif provider == 'anthropic':
            return await _call_anthropic_text_async(text, model, api_key)
        else:
            return await _call_text_provider_async(provider, model, api_key, text)
    except (ValueError, KeyError, TypeError, RuntimeError) as e:
        logger.error(f"[ChemExtract] Async text LLM call failed: {e}")
        return None


def _split_text_aware(text, chunk_size=TEXT_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text or len(text) <= chunk_size:
        return [text] if text and text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        search_start = max(start, end - int(chunk_size * 0.2))
        break_point = text.rfind('\n\n', search_start, end + overlap)
        if break_point != -1 and break_point > start + chunk_size * 0.5:
            end = break_point + 2
        else:
            break_point = text.rfind('. ', search_start, end + overlap)
            if break_point != -1 and break_point > start + chunk_size * 0.5:
                end = break_point + 1
        chunk = text[start:end].strip()
        if len(chunk) > 200:
            chunks.append(chunk)
        start = max(start, end - overlap)
    return [c for c in chunks if c]


def call_text_llm_chunked(text, provider, model, api_key, chunk_size=TEXT_CHUNK_SIZE):
    if not text or not text.strip():
        return None
    if len(text) <= chunk_size:
        return call_text_llm(text, provider, model, api_key)
    chunks = _split_text_aware(text, chunk_size=chunk_size, overlap=CHUNK_OVERLAP)
    logger.info(f"[ChemExtract] Text ({len(text)} chars) split into {len(chunks)} chunks")
    merged = {"reactions": [], "compounds": []}
    for i, chunk in enumerate(chunks):
        logger.info(f"[ChemExtract] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        chunk_result = call_text_llm(chunk, provider, model, api_key)
        if chunk_result:
            for reaction in chunk_result.get("reactions", []):
                merged["reactions"].append(reaction)
            for compound in chunk_result.get("compounds", []):
                merged["compounds"].append(compound)
            for key in ("scaffold_smiles", "rgroup_table", "experimental_procedures", "characterization_data"):
                if key in chunk_result and key not in merged:
                    merged[key] = chunk_result[key]
    for idx, reaction in enumerate(merged["reactions"], 1):
        reaction["id"] = f"reaction_{idx}"
    logger.info(f"[ChemExtract] Chunked extraction: {len(merged['reactions'])} reactions, {len(merged['compounds'])} compounds")
    return merged if merged["reactions"] or merged["compounds"] else None


async def call_text_llm_chunked_async(text, provider, model, api_key, chunk_size=TEXT_CHUNK_SIZE):
    if not text or not text.strip():
        return None
    if len(text) <= chunk_size:
        return await call_text_llm_async(text, provider, model, api_key)
    chunks = _split_text_aware(text, chunk_size=chunk_size, overlap=CHUNK_OVERLAP)
    logger.info(f"[ChemExtract Async] Text ({len(text)} chars) split into {len(chunks)} chunks")
    merged = {"reactions": [], "compounds": []}
    for i, chunk in enumerate(chunks):
        logger.info(f"[ChemExtract Async] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        chunk_result = await call_text_llm_async(chunk, provider, model, api_key)
        if chunk_result:
            for reaction in chunk_result.get("reactions", []):
                merged["reactions"].append(reaction)
            for compound in chunk_result.get("compounds", []):
                merged["compounds"].append(compound)
            for key in ("scaffold_smiles", "rgroup_table", "experimental_procedures", "characterization_data"):
                if key in chunk_result and key not in merged:
                    merged[key] = chunk_result[key]
    for idx, reaction in enumerate(merged["reactions"], 1):
        reaction["id"] = f"reaction_{idx}"
    logger.info(f"[ChemExtract Async] Chunked: {len(merged['reactions'])} reactions, {len(merged['compounds'])} compounds")
    return merged if merged["reactions"] or merged["compounds"] else None
