"""
Knowledge graph generation blueprint — sync and async endpoints.
"""

import logging
from flask import Blueprint, request, jsonify

from core.errors import ValidationError, APIError
from core.utils import sanitize_input
from llm import generate_knowledge_graph, generate_knowledge_graph_async, get_llm_response, get_llm_response_async
from llm.knowledge_graph import generate_mock_knowledge_graph
from llm.prompts import EXPLAIN_SYSTEM_PROMPT

from .helpers import get_explanation

logger = logging.getLogger(__name__)

kg_bp = Blueprint('knowledge_graph', __name__)


def register_knowledge_graph_blueprint(app, limiter):
    """Register Knowledge Graph routes on the Flask app with rate limiting."""

    @kg_bp.route('/api/knowledge-graph', methods=['POST'])
    @limiter.limit("10 per minute")
    def api_knowledge_graph():
        try:
            data = request.get_json()
            if not data:
                raise ValidationError("No JSON data provided")

            topic = sanitize_input(data.get('topic', 'cross-coupling'), max_length=500)
            use_llm = data.get('use_llm', True)
            provider = data.get('provider')
            api_key = data.get('api_key')

            logger.info(f"[KG API] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

            graph_data = None
            llm_used = False

            if use_llm:
                logger.info(f"[KG API] Attempting LLM generation for: {topic}")
                graph_data = generate_knowledge_graph(topic, provider=provider, api_key=api_key)
                if graph_data:
                    llm_used = True
                    logger.info(f"[KG API] LLM generation successful, {len(graph_data.get('nodes', []))} nodes")

            if not graph_data:
                logger.info(f"[KG API] Using mock data for: {topic}")
                graph_data = generate_mock_knowledge_graph(topic)

            return jsonify({
                'success': True,
                'topic': topic,
                'graph': graph_data,
                'llm_used': llm_used,
            })

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("Knowledge Graph Error", exc_info=True)
            raise APIError("Error generating knowledge graph. Check server logs for details.", 500)

    @kg_bp.route('/api/knowledge-graph/async', methods=['POST'])
    @limiter.limit("10 per minute")
    async def api_knowledge_graph_async():
        try:
            data = request.get_json()
            if not data:
                raise ValidationError("No JSON data provided")

            topic = sanitize_input(data.get('topic', 'cross-coupling'), max_length=500)
            use_llm = data.get('use_llm', True)
            provider = data.get('provider')
            api_key = data.get('api_key')

            logger.info(f"[KG API Async] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

            graph_data = None
            llm_used = False

            if use_llm:
                logger.info(f"[KG API Async] Attempting LLM generation for: {topic}")
                graph_data = await generate_knowledge_graph_async(topic, provider=provider, api_key=api_key)
                if graph_data:
                    llm_used = True
                    logger.info(f"[KG API Async] LLM generation successful, {len(graph_data.get('nodes', []))} nodes")

            if not graph_data:
                logger.info(f"[KG API Async] Using mock data for: {topic}")
                graph_data = generate_mock_knowledge_graph(topic)

            return jsonify({
                'success': True,
                'topic': topic,
                'graph': graph_data,
                'llm_used': llm_used,
                'async': True,
            })

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("Knowledge Graph Async Error", exc_info=True)
            raise APIError("Error generating knowledge graph. Check server logs for details.", 500)

    @kg_bp.route('/api/knowledge-graph/explain', methods=['POST'])
    def api_knowledge_graph_explain():
        try:
            data = request.get_json()
            node_label = sanitize_input(data.get('node', ''), max_length=200)
            context = sanitize_input(data.get('context', ''), max_length=500)
            provider = data.get('provider')
            api_key = data.get('api_key')

            if not node_label:
                raise APIError("Node label is required", 400)

            user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"
            llm_response = get_llm_response(EXPLAIN_SYSTEM_PROMPT, user_message,
                                            provider=provider, api_key=api_key)

            if llm_response:
                return jsonify({
                    'success': True,
                    'node': node_label,
                    'explanation': llm_response,
                    'source': 'llm',
                })

            return get_explanation(node_label, context)

        except APIError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("Explanation Error", exc_info=True)
            raise APIError("Error generating explanation. Check server logs for details.", 500)

    @kg_bp.route('/api/knowledge-graph/explain/async', methods=['POST'])
    async def api_knowledge_graph_explain_async():
        try:
            data = request.get_json()
            if not data:
                raise ValidationError("No JSON data provided")

            node_label = sanitize_input(data.get('node', ''), max_length=200)
            context = sanitize_input(data.get('context', ''), max_length=500)
            provider = data.get('provider')
            api_key = data.get('api_key')

            if not node_label:
                raise APIError("Node label is required", 400)

            user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"
            llm_response = await get_llm_response_async(EXPLAIN_SYSTEM_PROMPT, user_message,
                                                           provider=provider, api_key=api_key)

            if llm_response:
                return jsonify({
                    'success': True,
                    'node': node_label,
                    'explanation': llm_response,
                    'source': 'llm',
                    'async': True,
                })

            return get_explanation(node_label, context, is_async=True)

        except APIError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error("Explanation Async Error", exc_info=True)
            raise APIError("Error generating explanation. Check server logs for details.", 500)

    app.register_blueprint(kg_bp)
