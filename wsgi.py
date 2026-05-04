import sys
import os
import logging

if __name__ == '__main__':
    PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
else:
    PROJECT_HOME = os.environ.get('PROJECT_HOME', '/home/genai-research')

BACKEND_PATH = os.path.join(PROJECT_HOME, 'backend')

print(f"[WSGI] Project Home: {PROJECT_HOME}")
print(f"[WSGI] Backend Path: {BACKEND_PATH}")

if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)

if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

env_file = os.path.join(BACKEND_PATH, '.env')
if os.path.exists(env_file):
    print(f"[WSGI] Loading environment from: {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key and value and key not in os.environ:
                    # Never log actual values for security
                    os.environ[key] = value
                    print(f"  {key}=***REDACTED***")
else:
    print(f"[WSGI] No .env file found at: {env_file}")

# Import the Flask application
from backend.app import app as application

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"WSGI starting - Project home: {PROJECT_HOME}")
logger.info(f"Python version: {sys.version}")

# For local development
if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)
