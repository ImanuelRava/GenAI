import sys
import os
import logging

# ---------------------------------------------------------------------------
# Path setup — must happen BEFORE any backend imports
# ---------------------------------------------------------------------------

PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
BACKEND_PATH = os.path.join(PROJECT_HOME, 'backend')

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

def _resolve_host_port():
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'

    # --- Port ---
    platform_port = os.environ.get('PORT', '')
    explicit_port = os.environ.get('FLASK_PORT', '')
    port = int(explicit_port or platform_port or '5000')

    # --- Host ---
    is_hosted = bool(platform_port or explicit_port)
    host = os.environ.get('FLASK_HOST', '0.0.0.0' if is_hosted else '127.0.0.1')

    # Security guard
    if debug and host not in ('127.0.0.1', 'localhost'):
        logger.warning(
            "FLASK_DEBUG=1 with FLASK_HOST=%s — Werkzeug debugger would be exposed "
            "to the network (RCE risk). Forcing host=127.0.0.1.", host
        )
        host = '127.0.0.1'

    return host, port, debug


if __name__ == '__main__':
    host, port, debug = _resolve_host_port()
    logger.info("Starting development server on http://%s:%d (debug=%s)", host, port, debug)
    application.run(debug=debug, host=host, port=port)