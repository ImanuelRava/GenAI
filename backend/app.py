"""
GenAI Research Platform - Main Flask Application
An AI-powered chemistry research platform for transition metal catalysis.
"""

import os
import sys
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dotenv import load_dotenv
env_paths = [
    os.path.join(BASE_DIR, '.env'),
    os.path.join(BACKEND_DIR, '.env'),
    '/home/z/my-project/.env',
    os.path.expanduser('~/.env')
]

for env_path in env_paths:
    if os.path.exists(env_path):
        print(f"[STARTUP] Loading environment from: {env_path}")
        load_dotenv(env_path)
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key = line.split('=', 1)[0]
                    print(f"  {key}=***REDACTED***")
        break

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import config
from errors import APIError, register_error_handlers, success_response
from cache import get_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=config.static_folder, static_url_path='')
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

CORS(app, resources={
    r"/api/*": {
        "origins": config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
    storage_uri="memory://"
)

register_error_handlers(app)

@app.before_request
def before_request():
    g.start_time = time.time()
    g.request_id = request.headers.get('X-Request-ID', '-')
    if not request.path.startswith('/static'):
        logger.info(f"[{request.method}] {request.path} - Started (ID: {g.request_id})")

@app.after_request
def after_request(response):
    if not request.path.startswith('/static'):
        duration = time.time() - g.get('start_time', time.time())
        logger.info(
            f"[{request.method}] {request.path} - "
            f"{response.status_code} ({duration:.3f}s) (ID: {g.get('request_id', '-')})"
        )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/api/health')
def health_check():
    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        rdkit_available = False

    cache_stats = get_cache().stats()

    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': '2.0.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'components': {
            'rdkit': rdkit_available,
            'cache': cache_stats,
            'async_support': True
        }
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        'success': True,
        'service': 'GenAI Research Platform',
        'version': '2.1.0',
        'features': {
            'async_support': True,
            'http_llm_client': True,
            'rag_integration': True,
            'database_integration': True
        },
        'endpoints': {
            'network': '/api/network',
            'molecules': '/api/molecules',
            'reactions': '/api/reactions',
            'gnn': '/api/gnn/*',
            'pca': '/api/pca/*',
            'llm': '/api/llm/*',
            'knowledge_graph': '/api/knowledge-graph',
            'nicobot_chat': '/api/nicobot/chat',
            'database': '/api/database/*'
        }
    })

@app.route('/')
def index():
    return send_from_directory(config.static_folder, 'index.html')

@app.route('/TMC/')
def TMC_index():
    return send_from_directory(os.path.join(config.static_folder, 'TMC'), 'index.html')

@app.route('/TMC/<path:filename>')
def TMC_files(filename: str):
    return send_from_directory(os.path.join(config.static_folder, 'TMC'), filename)

@app.route('/AI/')
def AI_index():
    return send_from_directory(os.path.join(config.static_folder, 'AI'), 'index.html')

@app.route('/AI/<path:filename>')
def AI_files(filename: str):
    return send_from_directory(os.path.join(config.static_folder, 'AI'), filename)

@app.route('/virus/')
def virus_index():
    return send_from_directory(os.path.join(config.static_folder, 'virus'), 'index.html')

@app.route('/virus/<path:filename>')
def virus_files(filename: str):
    return send_from_directory(os.path.join(config.static_folder, 'virus'), filename)

@app.route('/redox-ligands/')
def redox_index():
    return send_from_directory(os.path.join(config.static_folder, 'redox-ligands'), 'index.html')

@app.route('/redox-ligands/<path:filename>')
def redox_files(filename: str):
    return send_from_directory(os.path.join(config.static_folder, 'redox-ligands'), filename)

from routes.network import network_bp
from routes.chemistry import chemistry_bp
from routes.llm import llm_bp
from routes.visualization import viz_bp
from routes.data_extraction import data_extraction_bp

app.register_blueprint(network_bp)
app.register_blueprint(chemistry_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(viz_bp)
app.register_blueprint(data_extraction_bp)

# Register database blueprint if available
try:
    from routes.database import database_bp
    app.register_blueprint(database_bp)
    logger.info("Registered API blueprints: network, chemistry, llm, viz, data_extraction, database")
except ImportError:
    logger.warning("Database blueprint not available. Database API endpoints disabled.")
    logger.info("Registered API blueprints: network, chemistry, llm, viz, data_extraction")

# Use HTTP client for LLM requests instead of direct imports
from llm_client import get_llm_response, generate_knowledge_graph
from utils import sanitize_input
from errors import ValidationError, LLMError

# Import RAG service for database integration
try:
    from modules.nicobot_rag import get_rag, enhance_prompt_with_context
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("NiCOBot RAG service not available. Running without database integration.")

@app.route('/api/nicobot/chat', methods=['POST'])
@limiter.limit("20 per minute")
def nicobot_chat():
    """
    NiCOBot chat endpoint using HTTP-based LLM client with RAG integration.
    Integrates database context for enhanced responses about compounds and papers.
    """
    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No JSON data provided")

        message = data.get('message', '')
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')
        use_rag = data.get('use_rag', True)  # Enable RAG by default

        message = sanitize_input(message, max_length=config.MAX_PROMPT_LENGTH)
        if not message:
            raise ValidationError("Message cannot be empty", field="message")

        logger.info(f"[NiCOBot] Provider: {provider}, Has API Key: {bool(api_key)}, RAG: {use_rag}")

        base_system_prompt = """You are NiCOBot, a specialized AI assistant for Nickel-catalyzed cross-coupling reactions and C-O bond activation chemistry.
Provide accurate, helpful responses about:
- Nickel catalysis mechanisms and applications
- C-O bond activation strategies
- Cross-coupling reactions (Suzuki, Heck, Kumada, etc.)
- Ligand design for transition metal catalysis
- Comparison of Ni vs Pd catalysis

Keep responses concise but informative. Use proper chemical nomenclature."""

        # Enhance prompt with database context if RAG is available
        database_context = None
        if RAG_AVAILABLE and use_rag:
            try:
                rag = get_rag()
                context = rag.retrieve_context(message)
                if context.formatted_context:
                    database_context = context.formatted_context
                    system_prompt = f"""{base_system_prompt}

## Database Context
The following information has been retrieved from the NiCOBot chemical database. Use this to enhance your response:

{context.formatted_context}

When answering, reference specific compounds, papers, or data from the database when relevant. If the user asks about a specific compound or reaction, provide the SMILES notation and any relevant publication references."""
                    logger.info(f"[NiCOBot] RAG context retrieved: {len(context.compounds)} compounds, {len(context.papers)} papers")
                else:
                    system_prompt = base_system_prompt
            except Exception as e:
                logger.warning(f"[NiCOBot] RAG error: {e}. Falling back to base prompt.")
                system_prompt = base_system_prompt
        else:
            system_prompt = base_system_prompt

        # Use HTTP client for LLM response
        response = get_llm_response(
            system_prompt,
            message,
            provider=provider,
            api_key=api_key,
            model=model
        )

        if response:
            result = {
                'success': True,
                'response': response,
                'provider': provider or 'default'
            }
            if database_context:
                result['database_enhanced'] = True
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'No response received from LLM. Please check your API key and provider settings.'
            }), 500

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"NiCOBot chat error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error communicating with LLM: {str(e)}'
        }), 500


@app.route('/api/nicobot/chat/async', methods=['POST'])
@limiter.limit("20 per minute")
async def nicobot_chat_async():
    """
    Async NiCOBot chat endpoint using HTTP-based LLM client with RAG integration.
    """
    from llm_client import get_llm_response_async

    try:
        data = request.get_json()

        if not data:
            raise ValidationError("No JSON data provided")

        message = data.get('message', '')
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')
        use_rag = data.get('use_rag', True)

        message = sanitize_input(message, max_length=config.MAX_PROMPT_LENGTH)
        if not message:
            raise ValidationError("Message cannot be empty", field="message")

        logger.info(f"[NiCOBot Async] Provider: {provider}, Has API Key: {bool(api_key)}, RAG: {use_rag}")

        base_system_prompt = """You are NiCOBot, a specialized AI assistant for Nickel-catalyzed cross-coupling reactions and C-O bond activation chemistry.
Provide accurate, helpful responses about:
- Nickel catalysis mechanisms and applications
- C-O bond activation strategies
- Cross-coupling reactions (Suzuki, Heck, Kumada, etc.)
- Ligand design for transition metal catalysis
- Comparison of Ni vs Pd catalysis

Keep responses concise but informative. Use proper chemical nomenclature."""

        # Enhance prompt with database context if RAG is available
        database_context = None
        if RAG_AVAILABLE and use_rag:
            try:
                rag = get_rag()
                context = rag.retrieve_context(message)
                if context.formatted_context:
                    database_context = context.formatted_context
                    system_prompt = f"""{base_system_prompt}

## Database Context
The following information has been retrieved from the NiCOBot chemical database. Use this to enhance your response:

{context.formatted_context}

When answering, reference specific compounds, papers, or data from the database when relevant."""
                    logger.info(f"[NiCOBot Async] RAG context retrieved")
                else:
                    system_prompt = base_system_prompt
            except Exception as e:
                logger.warning(f"[NiCOBot Async] RAG error: {e}")
                system_prompt = base_system_prompt
        else:
            system_prompt = base_system_prompt

        # Use async HTTP client
        response = await get_llm_response_async(
            system_prompt,
            message,
            provider=provider,
            api_key=api_key,
            model=model
        )

        if response:
            result = {
                'success': True,
                'response': response,
                'provider': provider or 'default',
                'async': True
            }
            if database_context:
                result['database_enhanced'] = True
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': 'No response received from LLM. Please check your API key and provider settings.'
            }), 500

    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        logger.error(f"NiCOBot async chat error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error communicating with LLM: {str(e)}'
        }), 500


@app.route('/api/redox/chat', methods=['POST'])
@limiter.limit("20 per minute")
def redox_chat():
    """
    Redox-Active Ligands chat endpoint using HTTP-based LLM client.
    """
    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        message = data.get('message', '')
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')

        message = sanitize_input(message, max_length=config.MAX_PROMPT_LENGTH)
        if not message:
            raise ValidationError("Message cannot be empty", field="message")

        logger.info(f"[RAL Bot] Provider: {provider}, Has API Key: {bool(api_key)}")

        system_prompt = """You are a specialized AI assistant for Redox-Active Ligands chemistry.
Provide accurate, helpful responses about:
- Redox-active (non-innocent) ligands and their behavior
- Metal-ligand cooperativity and electron reservoir concepts
- Ligand classes: PDI (bis-imino)pyridine, catecholate/o-quinone, dithiolenes
- Nickel and first-row transition metal catalysis
Keep responses concise but informative."""

        # Use HTTP client instead of direct import
        response = get_llm_response(system_prompt, message, provider=provider, api_key=api_key, model=model)

        if response:
            return jsonify({'success': True, 'response': response, 'provider': provider or 'default'})
        else:
            return jsonify({'success': False, 'error': 'No response from LLM'}), 500

    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"RAL chat error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500


@app.route('/api/redox/chat/async', methods=['POST'])
@limiter.limit("20 per minute")
async def redox_chat_async():
    """
    Async Redox-Active Ligands chat endpoint using HTTP-based LLM client.
    """
    from llm_client import get_llm_response_async

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        message = data.get('message', '')
        provider = data.get('provider')
        api_key = data.get('api_key')
        model = data.get('model')

        message = sanitize_input(message, max_length=config.MAX_PROMPT_LENGTH)
        if not message:
            raise ValidationError("Message cannot be empty", field="message")

        logger.info(f"[RAL Bot Async] Provider: {provider}, Has API Key: {bool(api_key)}")

        system_prompt = """You are a specialized AI assistant for Redox-Active Ligands chemistry.
Provide accurate, helpful responses about:
- Redox-active (non-innocent) ligands and their behavior
- Metal-ligand cooperativity and electron reservoir concepts
- Ligand classes: PDI (bis-imino)pyridine, catecholate/o-quinone, dithiolenes
- Nickel and first-row transition metal catalysis
Keep responses concise but informative."""

        # Use async HTTP client
        response = await get_llm_response_async(system_prompt, message, provider=provider, api_key=api_key, model=model)

        if response:
            return jsonify({'success': True, 'response': response, 'provider': provider or 'default', 'async': True})
        else:
            return jsonify({'success': False, 'error': 'No response from LLM'}), 500

    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"RAL async chat error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500


@app.route('/api/knowledge-graph', methods=['POST'])
@limiter.limit("10 per minute")
def api_knowledge_graph():
    """
    Knowledge graph generation endpoint using HTTP-based LLM client.
    """
    try:
        data = request.get_json()
        topic = sanitize_input(data.get('topic', 'cross-coupling'), max_length=500)
        use_llm = data.get('use_llm', True)
        provider = data.get('provider')
        api_key = data.get('api_key')

        logger.info(f"[KG API] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

        graph_data = None
        llm_used = False

        if use_llm:
            logger.info(f"[KG API] Attempting LLM generation for: {topic}")
            # Use HTTP client instead of direct import
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
            'llm_used': llm_used
        })

    except Exception as e:
        logger.error(f"Knowledge Graph Error: {e}", exc_info=True)
        raise APIError(f"Error generating knowledge graph: {str(e)}", 500)


@app.route('/api/knowledge-graph/async', methods=['POST'])
@limiter.limit("10 per minute")
async def api_knowledge_graph_async():
    """
    Async knowledge graph generation endpoint using HTTP-based LLM client.
    """
    from llm_client import generate_knowledge_graph_async

    try:
        data = request.get_json()
        topic = sanitize_input(data.get('topic', 'cross-coupling'), max_length=500)
        use_llm = data.get('use_llm', True)
        provider = data.get('provider')
        api_key = data.get('api_key')

        logger.info(f"[KG API Async] Topic: {topic}, Use LLM: {use_llm}, Provider: {provider}")

        graph_data = None
        llm_used = False

        if use_llm:
            logger.info(f"[KG API Async] Attempting LLM generation for: {topic}")
            # Use async HTTP client
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
            'async': True
        })

    except Exception as e:
        logger.error(f"Knowledge Graph Async Error: {e}", exc_info=True)
        raise APIError(f"Error generating knowledge graph: {str(e)}", 500)


@app.route('/api/knowledge-graph/explain', methods=['POST'])
def api_knowledge_graph_explain():
    """
    Knowledge graph node explanation endpoint using HTTP-based LLM client.
    """
    try:
        data = request.get_json()
        node_label = sanitize_input(data.get('node', ''), max_length=200)
        context = sanitize_input(data.get('context', ''), max_length=500)
        provider = data.get('provider')
        api_key = data.get('api_key')

        if not node_label:
            raise APIError("Node label is required", 400)

        system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

        user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"

        # Use HTTP client instead of direct import
        llm_response = get_llm_response(system_prompt, user_message, provider=provider, api_key=api_key)

        if llm_response:
            return jsonify({
                'success': True,
                'node': node_label,
                'explanation': llm_response,
                'source': 'llm'
            })

        explanations = {
            'oxidative addition': 'Oxidative addition is the first step in cross-coupling. The metal catalyst (M) inserts into the C-X bond of the organic halide. The metal oxidation state increases by 2 (e.g., Pd(0) -> Pd(II)) as it forms two new bonds.',
            'transmetalation': 'Transmetalation is the transfer of an organic group from the nucleophilic reagent (R-M) to the metal center. This step pairs the two organic fragments on the metal before coupling.',
            'reductive elimination': 'Reductive elimination is the final step where the two organic groups couple together and are released as the product. The metal is reduced back to its original oxidation state (e.g., Pd(II) -> Pd(0)).',
            'palladium': 'Palladium is the most widely used catalyst for cross-coupling reactions. Pd(0) complexes are nucleophilic and readily undergo oxidative addition. The 2010 Nobel Prize was awarded for Pd-catalyzed cross-couplings.',
            'nickel': 'Nickel is a cost-effective alternative to palladium. Ni is more electrophilic and can activate stronger bonds like C-Cl and C-O. This makes it valuable for sustainable chemistry using biomass-derived feedstocks.',
            'suzuki': 'Suzuki-Miyaura coupling uses organoboron reagents. Key advantages: non-toxic, air-stable reagents, aqueous compatible. Won the 2010 Nobel Prize (Suzuki).',
            'heck': 'Heck reaction couples aryl halides with alkenes. Unique in that it does not require an organometallic nucleophile. Products are substituted alkenes.',
            'ligand': 'Ligands control the reactivity, selectivity, and stability of metal catalysts. Electron-rich ligands favor oxidative addition, while bulky ligands prevent unwanted side reactions.'
        }

        explanation = explanations.get(
            node_label.lower(),
            f'{node_label} is an important concept in transition metal catalysis. It plays a crucial role in the catalytic cycle and influences reaction outcomes.'
        )

        return jsonify({
            'success': True,
            'node': node_label,
            'explanation': explanation,
            'source': 'predefined'
        })

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Explanation Error: {e}", exc_info=True)
        raise APIError(f"Error generating explanation: {str(e)}", 500)


@app.route('/api/knowledge-graph/explain/async', methods=['POST'])
async def api_knowledge_graph_explain_async():
    """
    Async knowledge graph node explanation endpoint using HTTP-based LLM client.
    """
    from llm_client import get_llm_response_async

    try:
        data = request.get_json()
        node_label = sanitize_input(data.get('node', ''), max_length=200)
        context = sanitize_input(data.get('context', ''), max_length=500)
        provider = data.get('provider')
        api_key = data.get('api_key')

        if not node_label:
            raise APIError("Node label is required", 400)

        system_prompt = """You are an expert chemistry educator specializing in transition metal catalysis.
Provide a clear, concise explanation (2-3 sentences) for the given chemistry concept.
Focus on practical understanding and real-world applications.
Keep the explanation accessible to graduate-level chemistry students."""

        user_message = f"Explain {node_label} in the context of transition metal catalysis. Context: {context}"

        # Use async HTTP client
        llm_response = await get_llm_response_async(system_prompt, user_message, provider=provider, api_key=api_key)

        if llm_response:
            return jsonify({
                'success': True,
                'node': node_label,
                'explanation': llm_response,
                'source': 'llm',
                'async': True
            })

        explanations = {
            'oxidative addition': 'Oxidative addition is the first step in cross-coupling. The metal catalyst (M) inserts into the C-X bond of the organic halide. The metal oxidation state increases by 2 (e.g., Pd(0) -> Pd(II)) as it forms two new bonds.',
            'transmetalation': 'Transmetalation is the transfer of an organic group from the nucleophilic reagent (R-M) to the metal center. This step pairs the two organic fragments on the metal before coupling.',
            'reductive elimination': 'Reductive elimination is the final step where the two organic groups couple together and are released as the product. The metal is reduced back to its original oxidation state (e.g., Pd(II) -> Pd(0)).',
            'palladium': 'Palladium is the most widely used catalyst for cross-coupling reactions. Pd(0) complexes are nucleophilic and readily undergo oxidative addition. The 2010 Nobel Prize was awarded for Pd-catalyzed cross-couplings.',
            'nickel': 'Nickel is a cost-effective alternative to palladium. Ni is more electrophilic and can activate stronger bonds like C-Cl and C-O. This makes it valuable for sustainable chemistry using biomass-derived feedstocks.',
            'suzuki': 'Suzuki-Miyaura coupling uses organoboron reagents. Key advantages: non-toxic, air-stable reagents, aqueous compatible. Won the 2010 Nobel Prize (Suzuki).',
            'heck': 'Heck reaction couples aryl halides with alkenes. Unique in that it does not require an organometallic nucleophile. Products are substituted alkenes.',
            'ligand': 'Ligands control the reactivity, selectivity, and stability of metal catalysts. Electron-rich ligands favor oxidative addition, while bulky ligands prevent unwanted side reactions.'
        }

        explanation = explanations.get(
            node_label.lower(),
            f'{node_label} is an important concept in transition metal catalysis. It plays a crucial role in the catalytic cycle and influences reaction outcomes.'
        )

        return jsonify({
            'success': True,
            'node': node_label,
            'explanation': explanation,
            'source': 'predefined',
            'async': True
        })

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Explanation Async Error: {e}", exc_info=True)
        raise APIError(f"Error generating explanation: {str(e)}", 500)


def generate_mock_knowledge_graph(topic: str) -> dict:
    topic_lower = topic.lower()

    graphs = {
        'suzuki': {
            "nodes": [
                {"id": "suzuki", "label": "Suzuki-Miyaura Coupling", "type": "reaction", "description": "Pd-catalyzed cross-coupling between organoboron and organic halide"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst", "description": "Transition metal catalyst essential for Suzuki reaction"},
                {"id": "boronic", "label": "Organoboron Reagent", "type": "reagent", "description": "R-B(OH)2 nucleophilic partner"},
                {"id": "halide", "label": "Organic Halide", "type": "reagent", "description": "R'-X electrophilic partner"},
                {"id": "oxidative", "label": "Oxidative Addition", "type": "mechanism", "description": "Pd(0) -> Pd(II), inserts into C-X bond"},
                {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism", "description": "Transfer of R group from boron to Pd"},
                {"id": "reductive", "label": "Reductive Elimination", "type": "mechanism", "description": "Pd(II) -> Pd(0), forms C-C bond"},
                {"id": "biaryl", "label": "Biaryl Product", "type": "product", "description": "R-R' coupled product"}
            ],
            "edges": [
                {"source": "suzuki", "target": "palladium", "label": "catalyzed by"},
                {"source": "suzuki", "target": "boronic", "label": "uses"},
                {"source": "suzuki", "target": "halide", "label": "uses"},
                {"source": "palladium", "target": "oxidative", "label": "undergoes"},
                {"source": "oxidative", "target": "transmetalation", "label": "followed by"},
                {"source": "transmetalation", "target": "reductive", "label": "followed by"},
                {"source": "reductive", "target": "biaryl", "label": "produces"}
            ]
        },
        'heck': {
            "nodes": [
                {"id": "heck", "label": "Heck Reaction", "type": "reaction", "description": "Pd-catalyzed coupling of aryl halide with alkene"},
                {"id": "palladium", "label": "Palladium Catalyst", "type": "catalyst", "description": "Pd(0)/Pd(II) catalytic cycle"},
                {"id": "aryl_halide", "label": "Aryl Halide", "type": "reagent", "description": "Ar-X electrophile"},
                {"id": "alkene", "label": "Alkene", "type": "reagent", "description": "C=C nucleophilic partner"},
                {"id": "migratory", "label": "Migratory Insertion", "type": "mechanism", "description": "Alkene inserts into Pd-Ar bond"},
                {"id": "styrene", "label": "Styrene Derivative", "type": "product", "description": "Ar-CH=CH2 type product"}
            ],
            "edges": [
                {"source": "heck", "target": "palladium", "label": "catalyzed by"},
                {"source": "heck", "target": "aryl_halide", "label": "uses"},
                {"source": "heck", "target": "alkene", "label": "uses"},
                {"source": "palladium", "target": "migratory", "label": "undergoes"},
                {"source": "migratory", "target": "styrene", "label": "produces"}
            ]
        }
    }

    default_graph = {
        "nodes": [
            {"id": "cross_coupling", "label": "Cross-Coupling", "type": "reaction", "description": "Metal-catalyzed C-C bond formation"},
            {"id": "oxidative_addition", "label": "Oxidative Addition", "type": "mechanism", "description": "M(0) -> M(II), inserts into C-X bond"},
            {"id": "transmetalation", "label": "Transmetalation", "type": "mechanism", "description": "Exchange of ligands between metals"},
            {"id": "reductive_elimination", "label": "Reductive Elimination", "type": "mechanism", "description": "M(II) -> M(0), forms product"},
            {"id": "palladium", "label": "Palladium", "type": "catalyst", "description": "Most common cross-coupling catalyst"},
            {"id": "nickel", "label": "Nickel", "type": "catalyst", "description": "Cheaper alternative, activates C-Cl/C-O"},
            {"id": "product", "label": "C-C Bond Product", "type": "product", "description": "Coupled organic molecule"}
        ],
        "edges": [
            {"source": "cross_coupling", "target": "oxidative_addition", "label": "step 1"},
            {"source": "oxidative_addition", "target": "transmetalation", "label": "step 2"},
            {"source": "transmetalation", "target": "reductive_elimination", "label": "step 3"},
            {"source": "reductive_elimination", "target": "product", "label": "produces"},
            {"source": "cross_coupling", "target": "palladium", "label": "catalyzed by"},
            {"source": "cross_coupling", "target": "nickel", "label": "catalyzed by"}
        ]
    }

    for key, graph in graphs.items():
        if key in topic_lower:
            return graph

    return default_graph

print(f"[STARTUP] GenAI Research Platform v2.0.0")
print(f"[STARTUP] Backend Dir: {BACKEND_DIR}")
print(f"[STARTUP] Static Folder: {config.static_folder}")
print(f"[STARTUP] Upload Folder: {config.UPLOAD_FOLDER}")
print(f"[STARTUP] CORS Origins: {config.CORS_ORIGINS}")
print(f"[STARTUP] Rate Limits: {config.RATE_LIMIT_DEFAULT}")
print(f"[STARTUP] Async Support: Enabled")
print(f"[STARTUP] LLM Client: HTTP-based")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
