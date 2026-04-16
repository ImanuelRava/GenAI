"""
WSGI Configuration for PythonAnywhere Deployment
ChemAI Research - Flask Application

IMPORTANT: Update the paths below with your PythonAnywhere username!
"""

import sys
import os

# ============================================
# CONFIGURATION - UPDATE THESE PATHS!
# ============================================
# Replace 'yourusername' with your actual PythonAnywhere username
PROJECT_PATH = '/home/yourusername/chemai-research/upload'

# Add project path to Python path
if PROJECT_PATH not in sys.path:
    sys.path.insert(0, PROJECT_PATH)

# ============================================
# SET UP VIRTUAL ENVIRONMENT (Optional)
# ============================================
# Uncomment and adjust if you created a virtual environment
# activate_this = '/home/yourusername/.virtualenvs/chemai/bin/activate_this.py'
# with open(activate_this) as file_:
#     exec(file_.read(), dict(__file__=activate_this))

# ============================================
# IMPORT FLASK APPLICATION
# ============================================
from app import app as application

# ============================================
# LOGGING
# ============================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# ============================================
# FOR LOCAL TESTING
# ============================================
if __name__ == '__main__':
    application.run(debug=True, port=5000)
