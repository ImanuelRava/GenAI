# GenAI Research - PythonAnywhere Deployment Guide

## Project Structure

```
your-project/
├── index.html                    # Main landing page
├── nicobot/                      # TMC (Transition Metal Catalysis) section
│   ├── index.html
│   ├── introduction.html
│   ├── citation-tool.html
│   ├── quiz.html
│   ├── tmc-embed.html
│   ├── data-extraction.html
│   └── lecture-*.html
├── redox-ligands/                # Redox-Active Ligands section
│   └── index.html
├── AI/                           # AI section
│   └── index.html
├── virus/                        # Virology section
│   └── index.html
└── backend/                      # Flask Backend
    ├── app.py                    # Main Flask application
    ├── requirements.txt          # Python dependencies
    └── modules/                  # Citation network modules
        ├── __init__.py
        ├── DOI.py
        ├── Forward_Reference.py
        ├── Local_Reference.py
        └── Cross_Reference.py
```

## PythonAnywhere Setup Steps

### 1. Upload Files
Upload all files maintaining the folder structure:
- All HTML files and folders (nicobot/, AI/, virus/, redox-ligands/) to your project root
- Backend folder with all Python files

### 2. Create Virtual Environment
```bash
# In PythonAnywhere console
mkvirtualenv --python=python3.10 myenv
pip install -r backend/requirements.txt
```

### 3. Configure Web App
1. Go to **Web** tab in PythonAnywhere
2. Create a new web app
3. Select "Manual configuration" (not Django/Flask template)
4. Choose Python 3.10

### 4. Configure WSGI File
Edit the WSGI file to point to your Flask app:
```python
import sys
import os

# Add your project directory to the path
project_home = '/home/yourusername/your-project'
if project_home not in sys.path:
    sys.path.insert(0, project_home)
    sys.path.insert(0, os.path.join(project_home, 'backend'))

# Activate virtual environment
activate_this = '/home/yourusername/.virtualenvs/myenv/bin/activate_this.py'
exec(open(activate_this).read(), dict(__file__=activate_this))

# Import and return the Flask app
from app import app as application
```

### 5. Set Working Directory
In the Web tab, set:
- **Working directory**: `/home/yourusername/your-project/backend`
- **Virtual environment**: `/home/yourusername/.virtualenvs/myenv`

### 6. Static Files Configuration
For static files in the Web tab:
- **URL**: `/` → **Directory**: `/home/yourusername/your-project/`

Or configure individual static file mappings:
- `/nicobot/` → `/home/yourusername/your-project/nicobot/`
- `/AI/` → `/home/yourusername/your-project/AI/`
- `/virus/` → `/home/yourusername/your-project/virus/`
- `/redox-ligands/` → `/home/yourusername/your-project/redox-ligands/`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main landing page |
| `/nicobot/` | GET | TMC section |
| `/api/network` | POST | Citation network analysis |
| `/api/molecules` | GET | Get molecule data for quiz |

## Testing Locally

```bash
cd backend
python app.py
# Visit http://localhost:5000
```

## Troubleshooting

1. **Import errors**: Ensure virtual environment is activated and all packages installed
2. **Static files not loading**: Check static folder path in app.py
3. **CORS errors**: flask-cors is already configured in app.py
4. **RDKit issues**: RDKit requires specific installation - may need conda on PythonAnywhere

## RDKit on PythonAnywhere

RDKit may require special installation:
```bash
# Option 1: pip (if available)
pip install rdkit

# Option 2: conda (if you have conda)
conda install -c rdkit rdkit
```

If RDKit is not available, the quiz molecule generation will not work, but other features will function normally.
