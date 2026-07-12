"""
LLM Package - Provider abstraction and public helpers
"""

from .factory import LLMProviderFactory
from .helpers import (
    get_llm_response,
    get_llm_response_async,
    generate_knowledge_graph,
    generate_knowledge_graph_async,
    explain_concept,
    explain_concept_async,
)
from .client import (
    LLMClient,
    VISION_CAPABLE_PROVIDERS,
    PROVIDER_DEFAULT_MODELS,
    ALL_PROVIDERS,
    UnsupportedProviderError,
    VisionNotSupportedError,
)
from .prompts import (
    NICOBOT_SYSTEM_PROMPT,
    REDCROSS_SYSTEM_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
    PREDEFINED_EXPLANATIONS,
)

__all__ = [
    'LLMProviderFactory',
    'LLMClient',
    'VISION_CAPABLE_PROVIDERS',
    'PROVIDER_DEFAULT_MODELS',
    'ALL_PROVIDERS',
    'UnsupportedProviderError',
    'VisionNotSupportedError',
    'get_llm_response',
    'get_llm_response_async',
    'generate_knowledge_graph',
    'generate_knowledge_graph_async',
    'explain_concept',
    'explain_concept_async',
    'NICOBOT_SYSTEM_PROMPT',
    'REDCROSS_SYSTEM_PROMPT',
    'EXPLAIN_SYSTEM_PROMPT',
    'PREDEFINED_EXPLANATIONS',
]
