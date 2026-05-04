import sys
import os
import logging

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_HOME = '/home/yourusername/genai-research'
BACKEND_PATH = os.path.join(PROJECT_HOME, 'backend')

# ============================================================
# PATH SETUP
# ============================================================
print(f"[WSGI] Project Home: {PROJECT_HOME}")
print(f"[WSGI] Backend Path: {BACKEND_PATH}")

if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)

if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

os.environ['PROJECT_HOME'] = PROJECT_HOME

# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================
env_file = os.path.join(BACKEND_PATH, '.env')
if os.path.exists(env_file):
    print(f"[WSGI] Loading environment from: {env_file}")
    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key and value:
                        os.environ[key] = value
                        print(f"  {key}=***REDACTED***")
    except Exception as e:
        print(f"[WSGI] Error loading .env: {e}")
else:
    print(f"[WSGI] Warning: No .env file found at: {env_file}")
    print(f"[WSGI] Make sure to create one with your configuration!")

# ============================================================
# FLASK APPLICATION
# ============================================================
try:
    from backend.app import app as application
    print("[WSGI] Flask application loaded successfully!")
except ImportError as e:
    print(f"[WSGI] ERROR: Failed to import Flask app: {e}")
    print("[WSGI] Check that all dependencies are installed in your virtualenv")
    raise

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"WSGI started - Project: {PROJECT_HOME}")
logger.info(f"Python version: {sys.version}")

# ============================================================
# FOR LOCAL TESTING (Optional)
# ============================================================
if __name__ == '__main__':
    print("[WSGI] Running in development mode...")
    application.run(debug=True, host='0.0.0.0', port=5000)
