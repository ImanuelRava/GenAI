"""
WSGI entry point for GenAI Research Platform.

Usage:
    python wsgi.py                  # Development server on port 5000
    gunicorn wsgi:application        # Production WSGI server
    waitress-serve wsgi:application  # Alternative WSGI server
"""

import sys
import os
import logging

# ---------------------------------------------------------------------------
# Path setup — must happen BEFORE any backend imports
# ---------------------------------------------------------------------------

PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(PROJECT_HOME, 'backend')

# Ensure both dirs are on sys.path so that
#   "from backend.app import app"  (needs PROJECT_HOME)
#   "from core.config import ..."  (needs BACKEND_PATH)
# both resolve correctly.
for p in (PROJECT_HOME, BACKEND_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Logging — configure early so startup messages are visible
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('wsgi')

# ---------------------------------------------------------------------------
# .env loading (uses python-dotenv if available, falls back to manual)
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env files from well-known locations."""
    candidates = [
        os.path.join(PROJECT_HOME, '.env'),
        os.path.join(BACKEND_PATH, '.env'),
        os.path.expanduser('~/.env'),
    ]
    for env_path in candidates:
        if os.path.exists(env_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(env_path, override=False)
                logger.info(f"Loaded .env from: {env_path}")
            except ImportError:
                _manual_env_load(env_path)
            return
    logger.debug("No .env file found in standard locations.")


def _manual_env_load(env_path):
    """Minimal .env parser used when python-dotenv is not installed."""
    logger.info(f"Loading .env (manual) from: {env_path}")
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '#' in line:
                line = line.split('#', 1)[0].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()

# ---------------------------------------------------------------------------
# Import the Flask application
# ---------------------------------------------------------------------------

try:
    from backend.app import app as application
except Exception as exc:
    logger.critical("Failed to import the Flask application: %s", exc, exc_info=True)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # SECURITY: never bind debug mode to 0.0.0.0 by default.
    # - FLASK_DEBUG=1 enables the Werkzeug debugger (LOCALHOST ONLY — RCE if exposed).
    # - FLASK_HOST defaults to 127.0.0.1; override only behind a reverse proxy in prod.
    # For production, run via gunicorn/waitress against this module: `gunicorn wsgi:application`

    # Detect Replit — Replit sets REPL_ID and REPL_SLUG env vars, and provides
    # a PORT env var (usually 8080 or 5000) that the web preview proxies to.
    # On Replit, we must bind to 0.0.0.0 so the web preview can reach us.
    is_replit = bool(os.environ.get('REPL_ID') or os.environ.get('REPL_SLUG'))

    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    # On Replit, default to 0.0.0.0 (required for web preview). Elsewhere,
    # default to 127.0.0.1 (secure — no network exposure).
    default_host = '0.0.0.0' if is_replit else '127.0.0.1'
    host = os.environ.get('FLASK_HOST', default_host)
    # On Replit, prefer the PORT env var if set. Otherwise use FLASK_PORT or 5000.
    default_port = os.environ.get('PORT', '5000') if is_replit else '5000'
    port = int(os.environ.get('FLASK_PORT') or default_port)

    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning(
            "FLASK_DEBUG=1 with FLASK_HOST=%s — Werkzeug debugger would be exposed "
            "to the network (RCE risk). Forcing host=127.0.0.1.", host
        )
        host = '127.0.0.1'

    if is_replit:
        logger.info("Replit environment detected (REPL_ID=%s). Binding to %s:%d.",
                    os.environ.get('REPL_ID', '?')[:8], host, port)
    logger.info("Starting development server on http://%s:%d (debug=%s)", host, port, debug)
    application.run(debug=debug, host=host, port=port)
