"""
Knowledge Graph Chat Blueprint

Provides the ``/api/knowledge-graph`` endpoints for generating and
querying chemistry knowledge graphs via LLM.
"""

import logging
from typing import Dict, Any, Optional

from flask import Blueprint, request, jsonify
from flask_limiter import Limiter

from core.errors import ValidationError, LLMError
from core.utils import sanitize_input

logger = logging.getLogger(__name__)

kg_bp = Blueprint('knowledge_graph', __name__, url_prefix='/api/knowledge-graph')


@kg_bp.route('', methods=['POST'])
def generate():
    """Generate a knowledge graph for a given chemistry topic."""
    data = request.get_json(silent=True)
    if not data:
        raise ValidationError("No JSON data provided")

    topic = data.get('topic', '')
    provider = data.get('provider')
    api_key = data.get('api_key')

    if not topic or not topic.strip():
        raise ValidationError("Topic cannot be empty", field="topic")

    topic = sanitize_input(topic, max_length=500)

    try:
        from llm.helpers import generate_knowledge_graph

        graph = generate_knowledge_graph(
            topic=topic,
            provider=provider,
            api_key=api_key,
        )

        if graph:
            return jsonify({
                'success': True,
                'graph': graph,
                'provider': provider or 'default',
            })
        else:
            # Fallback to mock data when LLM is unavailable
            from llm.knowledge_graph import generate_mock_knowledge_graph
            mock_graph = generate_mock_knowledge_graph(topic)
            return jsonify({
                'success': True,
                'graph': mock_graph,
                'provider': 'mock_fallback',
                'note': 'LLM unavailable — using pre-built graph',
            })

    except (KeyboardInterrupt, SystemExit):
        raise
    except LLMError:
        raise
    except Exception as e:
        logger.error("Knowledge graph generation error: %s", e, exc_info=True)
        raise LLMError(f"Error generating knowledge graph: {str(e)}")


def register_knowledge_graph_blueprint(app, limiter: Limiter):
    """Register Knowledge Graph blueprint with rate limiting."""
    app.register_blueprint(kg_bp)
    limiter.limit("15 per minute")(app.view_functions['knowledge_graph.generate'])
    logger.info("[STARTUP] Knowledge Graph blueprint registered at /api/knowledge-graph")