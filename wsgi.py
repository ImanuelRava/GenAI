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
    logger.info("Starting development server on http://0.0.0.0:5000")
    application.run(debug=True, host='0.0.0.0', port=5000)
