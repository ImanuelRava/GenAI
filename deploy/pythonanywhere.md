# PythonAnywhere Deployment Guide

## Prerequisites
- Paid plan (Beginner $5/mo minimum — free tier blocks outbound HTTP)

## Steps

### 1. Upload code
Use the **Upload files** page or pull from GitHub via Bash console:
```bash
git clone https://github.com/YOUR_USER/GenAI.git
```

### 2. Create a virtualenv
```bash
mkvirtualenv genai --python=3.12
```

### 3. Install dependencies
```bash
pip install -r GenAI/requirements.txt
```

> **Important**: `rdkit-pypi` and `PyMuPDF` are pre-built wheels — if installation
> fails, PythonAnywhere may not support them. Check with `pip install rdkit-pypi`
> first before proceeding.

### 4. Upload data files
If using the NiCOBot database, upload the CSV files to `GenAI/backend/nicobot_data/`
via the Upload files page (the Git LFS pointers won't work — you need the real files).

### 5. Set environment variables
In the **Web** tab, add a **WSGI configuration file** (see below) and set
environment variables in the **Variables** section:
```
SECRET_KEY=<random-string>
FLASK_DEBUG=0
```

### 6. WSGI config file
Create `/var/www/your-user_pythonanywhere_com_wsgi.py`:
```python
import os
import sys

# Point to your project
project_home = '/home/YOUR_USERNAME/GenAI'
if project_home not in sys.path:
    sys.path.insert(0, project_home)
sys.path.insert(0, os.path.join(project_home, 'backend'))

# Load env
from dotenv import load_dotenv
load_dotenv(os.path.join(project_home, 'backend', '.env'), override=False)

# Import Flask app
from app import app as application
```

### 7. Reload
Click **Reload** in the Web tab.

## Limitations on PythonAnywhere
- No `pdf2image` (no `poppler` system package)
- No Redis (use in-memory cache — already the default)
- No async server (WSGI only — your app works fine)
- 512 MB RAM on Beginner plan
- 100 CPU seconds/day on free tier (irrelevant on paid)