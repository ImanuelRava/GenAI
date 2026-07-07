"""
GenAI Research Platform — Flask Application (v2.2.0)

This is the slim entry-point that wires together all blueprints.
Chat endpoints live in ``chat/``, LLM providers in ``llm/``,
core utilities in ``core/``, and route blueprints in ``routes/``.
"""

import os
import sys
import time
import logging

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BACKEND_DIR)

# Both directories on sys.path so:
#   "from backend.app import app"  (needs BASE_DIR / project root)
#   "from core.config import ..."    (needs BACKEND_DIR)
for _p in (BACKEND_DIR, BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
env_paths = [
    os.path.join(BASE_DIR, '.env'),
    os.path.join(BACKEND_DIR, '.env'),
    os.path.expanduser('~/.env'),
]
for env_path in env_paths:
    if os.path.exists(env_path):
        logging.info(f"[STARTUP] Loading environment from: {env_path}")
        load_dotenv(env_path)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from core.config import config
from core.errors import register_error_handlers
from core.cache import get_cache

app = Flask(__name__, static_folder=config.static_folder, static_url_path='')

logger = logging.getLogger(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_FILE_SIZE

CORS(app, resources={
    r"/api/*": {
        "origins": config.CORS_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }
})

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
    storage_uri="memory://",
)

register_error_handlers(app)

# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------

@app.before_request
def before_request():
    g.start_time = time.time()
    g.request_id = request.headers.get('X-Request-ID', '-')
    if not request.path.startswith('/static'):
        logging.info(f"[{request.method}] {request.path} - Started (ID: {g.request_id})")


@app.after_request
def after_request(response):
    if not request.path.startswith('/static'):
        duration = time.time() - g.get('start_time', time.time())
        logging.info(
            f"[{request.method}] {request.path} - "
            f"{response.status_code} ({duration:.3f}s) (ID: {g.get('request_id', '-')})"
        )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response


# ---------------------------------------------------------------------------
# Health / status
# ---------------------------------------------------------------------------

@app.route('/api/health')
def health_check():
    try:
        from rdkit import Chem
        rdkit_available = True
    except ImportError:
        rdkit_available = False

    return jsonify({
        'success': True,
        'status': 'healthy',
        'version': '2.2.0',
        'components': {
            'rdkit': rdkit_available,
            'cache': get_cache().stats(),
            'async_support': True,
        },
    })


@app.route('/api/status')
def api_status():
    return jsonify({
        'success': True,
        'service': 'GenAI Research Platform',
        'version': '2.2.0',
        'features': {
            'async_support': True,
            'http_llm_client': True,
            'rag_integration': True,
            'database_integration': True,
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
            'redox_chat': '/api/redox/chat',
            'database': '/api/database/*',
            'ral_database': '/api/ral-database/*',
        },
    })


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(config.static_folder, 'index.html')


@app.route('/TMC/')
def TMC_index():
    return send_from_directory(os.path.join(config.static_folder, 'TMC'), 'index.html')


@app.route('/TMC/<path:filename>')
def TMC_files(filename):
    return send_from_directory(os.path.join(config.static_folder, 'TMC'), filename)


@app.route('/AI/')
def AI_index():
    return send_from_directory(os.path.join(config.static_folder, 'AI'), 'index.html')


@app.route('/AI/<path:filename>')
def AI_files(filename):
    return send_from_directory(os.path.join(config.static_folder, 'AI'), filename)


@app.route('/virus/')
def virus_index():
    return send_from_directory(os.path.join(config.static_folder, 'virus'), 'index.html')


@app.route('/virus/<path:filename>')
def virus_files(filename):
    return send_from_directory(os.path.join(config.static_folder, 'virus'), filename)


@app.route('/redox-ligands/')
def redox_index():
    return send_from_directory(os.path.join(config.static_folder, 'redox-ligands'), 'index.html')


@app.route('/redox-ligands/<path:filename>')
def redox_files(filename):
    return send_from_directory(os.path.join(config.static_folder, 'redox-ligands'), filename)


@app.route('/technical-modules/')
def technical_modules_index():
    return send_from_directory(os.path.join(config.static_folder, 'technical-modules'), 'index.html')


@app.route('/technical-modules/<path:filename>')
def technical_modules_files(filename):
    return send_from_directory(os.path.join(config.static_folder, 'technical-modules'), filename)


# ---------------------------------------------------------------------------
# Register route blueprints
# ---------------------------------------------------------------------------

# --- Core route blueprints (always required) ---
from routes.network import network_bp
from routes.chemistry import chemistry_bp
from routes.llm import llm_bp
from routes.data_extraction import data_extraction_bp

app.register_blueprint(network_bp)
app.register_blueprint(chemistry_bp)
app.register_blueprint(llm_bp)
app.register_blueprint(data_extraction_bp)

# --- Optional: visualization (needs numpy) ---
try:
    from routes.visualization import viz_bp
    app.register_blueprint(viz_bp)
    _viz_registered = True
except ImportError:
    _viz_registered = False
    logging.warning("[STARTUP] Visualization blueprint not registered (missing optional deps).")

# --- Optional: database (needs nicobot_data) ---
try:
    from routes.database import database_bp
    app.register_blueprint(database_bp)
    _db_registered = True
except ImportError:
    _db_registered = False

# --- Optional: RAL database (needs ral_data) ---
try:
    from routes.ral_database import ral_database_bp
    app.register_blueprint(ral_database_bp)
    _ral_db_registered = True
except ImportError:
    _ral_db_registered = False
    logging.warning("[STARTUP] RAL database blueprint not registered (missing optional deps).")

# ---------------------------------------------------------------------------
# Register chat blueprints (NiCOBot, Redox, Knowledge Graph)
# ---------------------------------------------------------------------------

from chat.nicobot import register_nicobot_blueprint
from chat.redox import register_redox_blueprint
from chat.knowledge_graph import register_knowledge_graph_blueprint

register_nicobot_blueprint(app, limiter)
register_redox_blueprint(app, limiter)
register_knowledge_graph_blueprint(app, limiter)

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

_bps = ["network", "chemistry", "llm", "data_extraction"]
if _viz_registered:
    _bps.append("viz")
if _db_registered:
    _bps.append("database")
if _ral_db_registered:
    _bps.append("ral_database")
_db_msg = ", ".join(_bps)
logging.info(f"[STARTUP] GenAI Research Platform v2.2.0")
logging.info(f"[STARTUP] Backend Dir: {BACKEND_DIR}")
logging.info(f"[STARTUP] Static Folder: {config.static_folder}")
logging.info(f"[STARTUP] Upload Folder: {config.UPLOAD_FOLDER}")
logging.info(f"[STARTUP] CORS Origins: {config.CORS_ORIGINS}")
logging.info(f"[STARTUP] Rate Limits: {config.RATE_LIMIT_DEFAULT}")
logging.info(f"[STARTUP] Registered API blueprints: {_db_msg}")


if __name__ == '__main__':
    # SECURITY: prefer `python wsgi.py` for local dev — this entry point is kept
    # for backwards compatibility. Debug mode is gated behind FLASK_DEBUG=1 and
    # is forced to 127.0.0.1 to avoid exposing the Werkzeug debugger (RCE risk).
    is_replit = bool(os.environ.get('REPL_ID') or os.environ.get('REPL_SLUG'))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    default_host = '0.0.0.0' if is_replit else '127.0.0.1'
    host = os.environ.get('FLASK_HOST', default_host)
    default_port = os.environ.get('PORT', '5000') if is_replit else '5000'
    port = int(os.environ.get('FLASK_PORT') or default_port)
    if debug and host not in ('127.0.0.1', 'localhost'):
        logging.warning(
            "FLASK_DEBUG=1 with FLASK_HOST=%s — forcing 127.0.0.1 to avoid RCE.", host
        )
        host = '127.0.0.1'
    app.run(debug=debug, host=host, port=port)
