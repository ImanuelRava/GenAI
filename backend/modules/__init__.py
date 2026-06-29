"""
Backend Modules Package for GenAI Research Platform
"""

from .Forward_Reference import build_forward_network
from .Backward_Reference import build_reference_network
from .Cross_Reference import build_cross_reference_network

from .nicobot_database import NiCOBotDatabase, get_database
from .nicobot_rag import NiCOBotRAG, get_rag, enhance_prompt_with_context

__all__ = [
    'build_forward_network',
    'build_reference_network',
    'build_cross_reference_network',
    'NiCOBotDatabase',
    'get_database',
    'NiCOBotRAG',
    'get_rag',
    'enhance_prompt_with_context',
]
