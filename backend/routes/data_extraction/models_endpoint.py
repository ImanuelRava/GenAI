"""
GET /api/extract/models — list available LLM models and capabilities.

Returns the static catalog of models the frontend can offer in its
extraction UI, along with which providers support vision input and
metadata about the ReactionLens pipeline.
"""

from flask import jsonify

from ._helpers import (
    data_extraction_bp,
    AVAILABLE_MODELS,
    VISION_CAPABLE_PROVIDERS,
    REACTIONLENS_INFO,
)


@data_extraction_bp.route('/extract/models', methods=['GET'])
def get_models():
    """Return the catalog of available extraction models + capabilities."""
    return jsonify({
        "success": True,
        "models": AVAILABLE_MODELS,
        "vision_providers": VISION_CAPABLE_PROVIDERS,
        "reactionlens": REACTIONLENS_INFO,
        "async_support": True,
    })
