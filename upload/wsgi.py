"""
WSGI configuration for PythonAnywhere deployment
ChemAI Research - Flask Backend

This file is used by PythonAnywhere to serve your Flask application.

Setup Instructions for PythonAnywhere:
1. Go to the 'Web' tab in your PythonAnywhere dashboard
2. Create a new web app with manual configuration
3. Set the following paths:
   - Source code: /home/yourusername/chemai-research/upload
   - Working directory: /home/yourusername/chemai-research/upload
   - WSGI configuration file: point to this wsgi.py file

4. In your WSGI configuration file on PythonAnywhere, paste:
   from wsgi import app as application

5. Create a virtual environment and install dependencies:
   pip install flask flask-cors networkx rdkit requests openpyxl

6. Add the following to your WSGI file BEFORE the import:
   import sys
   import os
   path = '/home/yourusername/chemai-research/upload'
   if path not in sys.path:
       sys.path.insert(0, path)
"""

import sys
import os

# Add the upload directory to Python path
# This ensures all modules can be imported correctly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# For PythonAnywhere, you may need to adjust the username in the path below
# Replace 'yourusername' with your actual PythonAnywhere username
PA_PATH = '/home/yourusername/chemai-research/upload'
if os.path.exists(PA_PATH) and PA_PATH not in sys.path:
    sys.path.insert(0, PA_PATH)

# Import the Flask application from app.py
from app import app as application

# If you want to serve both apps, you can use a dispatcher like:
# from werkzeug.middleware.dispatcher import DispatcherMiddleware
# application = DispatcherMiddleware(citation_app, {
#     '/quiz': quiz_app
# })

if __name__ == '__main__':
    application.run()
