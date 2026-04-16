import sys
import os

path = '/home/hbsu/GenAI/upload'
if path not in sys.path:
    sys.path.insert(0, path)

from app import app as application