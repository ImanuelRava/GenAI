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
from .prompts import (
    NICOBOT_SYSTEM_PROMPT,
    REDOX_SYSTEM_PROMPT,
    EXPLAIN_SYSTEM_PROMPT,
    PREDEFINED_EXPLANATIONS,
)

__all__ = [
    'LLMProviderFactory',
    'get_llm_response',
    'get_llm_response_async',
    'generate_knowledge_graph',
    'generate_knowledge_graph_async',
    'explain_concept',
    'explain_concept_async',
    'NICOBOT_SYSTEM_PROMPT',
    'REDOX_SYSTEM_PROMPT',
    'EXPLAIN_SYSTEM_PROMPT',
    'PREDEFINED_EXPLANATIONS',
]
