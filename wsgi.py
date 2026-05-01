"""
WSGI Configuration for PythonAnywhere Deployment

This file is the entry point for the Flask application on PythonAnywhere.
It sets up the correct paths and imports the Flask app.
"""

import sys
import os

# ============================================================
# CONFIGURATION - Edit these paths for your PythonAnywhere account
# ============================================================

# Use environment variable for username, with fallback for development
# Set PYTHONANYWHERE_USERNAME in your PythonAnywhere dashboard or .env file
USERNAME = os.environ.get('PYTHONANYWHERE_USERNAME', 'hbsu')

# Project home directory
PROJECT_HOME = os.environ.get('PROJECT_HOME', f'/home/{USERNAME}/genai-research')

# Backend directory path
BACKEND_PATH = os.path.join(PROJECT_HOME, 'backend')

# ============================================================
# PATH SETUP - Do not modify unless you know what you're doing
# ============================================================

# Add project home to Python path
if PROJECT_HOME not in sys.path:
    sys.path.insert(0, PROJECT_HOME)

# Add backend directory to Python path
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

# ============================================================
# ENVIRONMENT VARIABLES - Set your API keys here or in PythonAnywhere dashboard
# ============================================================

# Uncomment and set these if you need LLM features
# os.environ['LLM_API_KEY'] = 'your-api-key-here'
# os.environ['LLM_PROVIDER'] = 'openai'
# os.environ['LLM_MODEL'] = 'gpt-4'

# Or load from .env file if it exists
env_file = os.path.join(BACKEND_PATH, '.env')
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key and value and key not in os.environ:
                    os.environ[key] = value

# ============================================================
# FLASK APP IMPORT - This is what PythonAnywhere looks for
# ============================================================

# Import the Flask application
from backend.app import app as application

# ============================================================
# OPTIONAL: Add middleware or logging
# ============================================================

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log startup
logging.info(f"WSGI starting - Project home: {PROJECT_HOME}")
logging.info(f"Python version: {sys.version}")

# ============================================================
# ALTERNATIVE: If you want to serve directly from this file
# ============================================================

# You can also run this file directly for local testing:
if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=5000)
