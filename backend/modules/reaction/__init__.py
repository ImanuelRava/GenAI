"""
ReactionLens — Text-Driven Chemical Reaction Detection & Extraction

Split into sub-modules:
- extraction: main pipeline (sync + async)
- providers: provider-specific text LLM calls
- parsing: JSON parsing + paragraph segmentation + PDF text extraction
- prompts: REACTION_DETECTION_PROMPT + constants
"""

from .extraction import (
    extract_with_reactionlens,
    extract_with_reactionlens_async,
    ReactionLens,
)
from .parsing import extract_text_from_pdf

__all__ = [
    'extract_with_reactionlens',
    'extract_with_reactionlens_async',
    'ReactionLens',
    'extract_text_from_pdf',
]
