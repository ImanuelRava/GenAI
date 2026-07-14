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

from flask import Flask, request, jsonify, send_from_directory, g, abort
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

_rate_storage = os.environ.get('REDIS_URL', 'memory://')

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT_DEFAULT],
    storage_uri=_rate_storage,
)

if _rate_storage == 'memory://':
    logging.warning(
        "[STARTUP] Rate limiter using in-memory storage. "
        "This is NOT safe for multi-worker deployments (gunicorn -w N). "
        "Set REDIS_URL to enable shared rate-limit state."
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
            'redcross_chat': '/api/redcross/chat',
            'database': '/api/database/*',
            'redcross_database': '/api/redcross-database/*',
        },
    })


# ---------------------------------------------------------------------------
# Static pages — all scoped to ALLOWED subdirectories only.
# ---------------------------------------------------------------------------

_ALLOWED_STATIC_SUBDIRS = config.STATIC_ALLOWED_DIRS


def _safe_serve(subdir: str, filename: str):
    """Serve a static file after validating the subdirectory is allowed."""
    # Reject path-traversal attempts (e.g. '../backend/app.py')
    if '..' in filename or filename.startswith('/'):
        abort(404)
    # Only serve from explicitly whitelisted subdirectories
    if subdir and subdir not in _ALLOWED_STATIC_SUBDIRS:
        abort(404)
    directory = os.path.join(config.static_folder, subdir) if subdir else config.static_folder
    return send_from_directory(directory, filename)


@app.route('/')
def index():
    return _safe_serve('', 'index.html')


@app.route('/TMC/')
def TMC_index():
    return _safe_serve('TMC', 'index.html')


@app.route('/TMC/<path:filename>')
def TMC_files(filename):
    return _safe_serve('TMC', filename)


@app.route('/AI/')
def AI_index():
    return _safe_serve('AI', 'index.html')


@app.route('/AI/<path:filename>')
def AI_files(filename):
    return _safe_serve('AI', filename)


@app.route('/virus/')
def virus_index():
    return _safe_serve('virus', 'index.html')


@app.route('/virus/<path:filename>')
def virus_files(filename):
    return _safe_serve('virus', filename)


@app.route('/reductive-coupling/')
def redcoupling_index():
    return _safe_serve('reductive-coupling', 'index.html')


@app.route('/reductive-coupling/<path:filename>')
def redcoupling_files(filename):
    return _safe_serve('reductive-coupling', filename)


@app.route('/technical-modules/')
def technical_modules_index():
    return _safe_serve('technical-modules', 'index.html')


@app.route('/technical-modules/<path:filename>')
def technical_modules_files(filename):
    return _safe_serve('technical-modules', filename)


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
    logging.warning("[STARTUP] Database blueprint not registered (missing optional deps).")

# --- Optional: RedCross database (needs redcross_data) ---
try:
    from routes.redcross_database import redcross_database_bp
    app.register_blueprint(redcross_database_bp)
    _redcross_db_registered = True
except ImportError:
    _redcross_db_registered = False
    logging.warning("[STARTUP] RedCross database blueprint not registered (missing optional deps).")

# ---------------------------------------------------------------------------
# Register chat blueprints (NiCOBot, RedCross, Knowledge Graph)
# ---------------------------------------------------------------------------

try:
    from chat.nicobot import register_nicobot_blueprint
    register_nicobot_blueprint(app, limiter)
    _nicobot_chat_registered = True
except ImportError:
    _nicobot_chat_registered = False
    logging.warning("[STARTUP] NiCOBot chat blueprint not registered (missing optional deps).")

try:
    from chat.redcross import register_redcross_blueprint
    register_redcross_blueprint(app, limiter)
    _redcross_chat_registered = True
except ImportError:
    _redcross_chat_registered = False
    logging.warning("[STARTUP] RedCross chat blueprint not registered (missing optional deps).")

try:
    from chat.knowledge_graph import register_knowledge_graph_blueprint
    register_knowledge_graph_blueprint(app, limiter)
    _kg_chat_registered = True
except ImportError:
    _kg_chat_registered = False
    logging.warning("[STARTUP] Knowledge Graph chat blueprint not registered (missing optional deps).")

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

_bps = ["network", "chemistry", "llm", "data_extraction"]
if _viz_registered:
    _bps.append("viz")
if _db_registered:
    _bps.append("database")
if _redcross_db_registered:
    _bps.append("redcross_database")
if _nicobot_chat_registered:
    _bps.append("nicobot_chat")
if _redcross_chat_registered:
    _bps.append("redcross_chat")
if _kg_chat_registered:
    _bps.append("knowledge_graph")
_db_msg = ", ".join(_bps)
logging.info(f"[STARTUP] GenAI Research Platform v2.2.0")
logging.info(f"[STARTUP] Backend Dir: {BACKEND_DIR}")
logging.info(f"[STARTUP] Static Folder: {config.static_folder}")
logging.info(f"[STARTUP] Upload Folder: {config.UPLOAD_FOLDER}")
logging.info(f"[STARTUP] CORS Origins: {config.CORS_ORIGINS}")
logging.info(f"[STARTUP] Rate Limits: {config.RATE_LIMIT_DEFAULT}")
logging.info(f"[STARTUP] Registered API blueprints: {_db_msg}")


if __name__ == '__main__':
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
