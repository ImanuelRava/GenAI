"""
LLM Routes Blueprint
Handles LLM integration endpoints
"""

import os
import json
import logging
from typing import Optional, Dict, Any

from flask import Blueprint, request, jsonify

from core.errors import APIError, ValidationError, LLMError
from core.utils import sanitize_input
from core.config import config, LLM_PROVIDER_CONFIG
from core.cache import cache_llm_response

logger = logging.getLogger(__name__)

llm_bp = Blueprint('llm', __name__, url_prefix='/api/llm')


def is_valid_api_key_format(key_value: str) -> bool:
    if not key_value:
        return False

    key_lower = key_value.lower()
    invalid_patterns = [
        'your_', 'placeholder', 'example', 'xxx', 'test_key',
        'sk-your', 'api_key_here', 'key_here', '_here',
        'replace_', 'insert_', 'change_'
    ]

    for pattern in invalid_patterns:
        if pattern in key_lower:
            return False

    if len(key_value) < 10:
        return False

    return True


@llm_bp.route('/status')
def api_llm_status():
    import requests

    def test_ollama_connection(base_url: str) -> bool:
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # Connection refused, timeout, DNS failure — Ollama not running.
            return False

    providers_with_keys = []

    provider_env_keys = {
        'groq': 'GROQ_API_KEY',
        'gemini': ('GEMINI_API_KEY', 'GOOGLE_API_KEY'),
        'huggingface': ('HF_API_KEY', 'HUGGINGFACE_API_KEY'),
        'deepseek': 'DEEPSEEK_API_KEY',
        'openrouter': 'OPENROUTER_API_KEY',
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY'
    }

    for provider, env_keys in provider_env_keys.items():
        if isinstance(env_keys, tuple):
            key_value = next((os.environ.get(k) for k in env_keys if os.environ.get(k)), None)
        else:
            key_value = os.environ.get(env_keys)

        if is_valid_api_key_format(key_value):
            providers_with_keys.append({
                'provider': provider,
                'name': LLM_PROVIDER_CONFIG.get(provider, {}).get('name', provider),
                'configured': True,
                'free_tier': LLM_PROVIDER_CONFIG.get(provider, {}).get('free_tier', False)
            })

    ollama_url = os.environ.get('OLLAMA_BASE_URL') or os.environ.get('OLLAMA_HOST') or 'http://localhost:11434'
    ollama_available = test_ollama_connection(ollama_url)
    if ollama_available:
        providers_with_keys.append({
            'provider': 'ollama',
            'name': 'Ollama (Local)',
            'configured': True,
            'url': ollama_url,
            'free_tier': True
        })

    default_provider = None
    for provider in ['groq', 'gemini', 'huggingface', 'ollama', 'deepseek', 'openrouter', 'openai', 'anthropic']:
        if any(p['provider'] == provider for p in providers_with_keys):
            default_provider = provider
            break

    verify_keys = os.environ.get('VERIFY_BACKEND_KEYS', 'false').lower() == 'true'
    has_backend_key = bool((verify_keys and providers_with_keys) or ollama_available)

    return jsonify({
        'success': True,
        'has_backend_key': has_backend_key,
        'providers': providers_with_keys,
        'default_provider': default_provider if has_backend_key else None,
        'ollama_available': ollama_available,
        'note': 'Backend API key verification is disabled by default to avoid unnecessary API calls.'
    })


@llm_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if not data:
        raise ValidationError("No JSON data provided")

    message = data.get('message', '')
    provider = data.get('provider')
    api_key = data.get('api_key')
    model = data.get('model')

    if not message:
        raise ValidationError("Message cannot be empty", field="message")

    from llm import get_llm_response
    from llm.prompts import NICOBOT_SYSTEM_PROMPT

    system_prompt = NICOBOT_SYSTEM_PROMPT

    try:
        response = get_llm_response(
            system_prompt,
            message,
            provider=provider,
            api_key=api_key,
            model=model
        )

        if response:
            return jsonify({
                'success': True,
                'response': response,
                'provider': provider or 'default'
            })
        else:
            raise LLMError("No response received from LLM")

    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"LLM chat error: {e}", exc_info=True)
        raise LLMError(f"Error communicating with LLM: {str(e)}")


@llm_bp.route('/providers')
def get_providers():
    providers = []
    for key, info in LLM_PROVIDER_CONFIG.items():
        providers.append({
            'id': key,
            'name': info.get('name', key),
            'url': info.get('url', ''),
            'free_tier': info.get('free_tier', False),
            'default_model': info.get('default_model', '')
        })

    return jsonify({
        'success': True,
        'providers': providers,
        'count': len(providers)
    })
