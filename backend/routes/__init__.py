"""
Routes Package for GenAI Research Platform
Contains Flask blueprints for organized API endpoints
"""

from .network import network_bp
from .chemistry import chemistry_bp
from .llm import llm_bp
from .visualization import viz_bp
from .data_extraction import data_extraction_bp

__all__ = ['network_bp', 'chemistry_bp', 'llm_bp', 'viz_bp', 'data_extraction_bp']
