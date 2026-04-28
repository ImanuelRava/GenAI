# GenAI Research - PythonAnywhere Deployment Guide

## 🇭🇰 For Hong Kong Users

### Local Development (Your Own Machine)
Use **Ollama** - 100% FREE, runs locally, no restrictions!

```bash
# 1. Install from https://ollama.ai
# 2. Download model
ollama pull llama3

# 3. Run Flask
cd backend && python app.py
```

### Deployed Website (PythonAnywhere)
You need a **cloud API** that works in Hong Kong. Choose one:

---

## Option 1: DeepSeek (RECOMMENDED for HK)

**Chinese company - works in Hong Kong!**

1. Go to https://platform.deepseek.com
2. Sign up → Get API key
3. In PythonAnywhere Web tab → Environment variables:
   ```
   DEEPSEEK_API_KEY = sk-your-key-here
   ```
4. Click **Reload**

**Pricing:** ~$0.14/million tokens (very cheap!)

---

## Option 2: OpenRouter (Has FREE models)

**Aggregates multiple LLMs, works globally**

1. Go to https://openrouter.ai
2. Sign up → Get API key (free credits included)
3. In PythonAnywhere:
   ```
   OPENROUTER_API_KEY = sk-or-your-key
   ```
4. Click **Reload**

**Free models available:** `meta-llama/llama-3-8b-instruct:free`

---

## Option 3: Hugging Face (FREE tier)

1. Go to https://huggingface.co/settings/tokens
2. Create token (Read access)
3. In PythonAnywhere:
   ```
   HF_API_KEY = hf_your_token
   ```

---

## Quick Reference

| Environment Variable | Provider | Works in HK? | Cost |
|---------------------|----------|--------------|------|
| `OLLAMA_BASE_URL` | Ollama (local) | ✅ YES | FREE |
| `DEEPSEEK_API_KEY` | DeepSeek | ✅ YES | Cheap |
| `OPENROUTER_API_KEY` | OpenRouter | ✅ YES | Free tier |
| `HF_API_KEY` | Hugging Face | ✅ YES | Free tier |
| `GROQ_API_KEY` | Groq | ❌ NO | - |
| `GEMINI_API_KEY` | Google Gemini | ❌ NO | - |

---

## Project Structure

```
your-project/
├── index.html                    # Main landing page
├── TMC/                          # Transition Metal Catalysis section
│   ├── index.html
│   ├── knowledge-graph.html      # AI-powered knowledge graph
│   ├── citation-tool.html
│   ├── quiz.html
│   └── lecture-*.html
├── redox-ligands/                # Redox-Active Ligands section
│   └── index.html
├── AI/                           # AI section
│   └── index.html
├── virus/                        # Virology section
│   └── index.html
└── backend/                      # Flask Backend
    ├── app.py                    # Main Flask application
    ├── llm_providers.py          # LLM providers (FREE & paid)
    ├── requirements.txt          # Python dependencies
    └── modules/                  # Citation network modules
```

---

## FREE LLM Options for Knowledge Graph

The Knowledge Graph feature supports multiple **FREE** LLM providers. Choose one:

### Option 1: Groq (RECOMMENDED - Free & Fast)
1. Go to https://console.groq.com
2. Sign up (no credit card required)
3. Create an API key
4. Set environment variable: `GROQ_API_KEY=your_key`

**Models available:** `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`

### Option 2: Google Gemini (Free Tier)
1. Go to https://ai.google.dev
2. Sign in with Google account
3. Get free API key
4. Set environment variable: `GEMINI_API_KEY=your_key`

**Models available:** `gemini-2.0-flash`, `gemini-1.5-flash`

### Option 3: Hugging Face (Free)
1. Go to https://huggingface.co/settings/tokens
2. Create a free account
3. Generate an API token
4. Set environment variable: `HF_API_KEY=your_key`

### Option 4: Ollama (100% Free - Local)
1. Install Ollama: https://ollama.ai
2. Run: `ollama pull llama3`
3. Set environment variable: `OLLAMA_BASE_URL=http://localhost:11434`

**Note:** Requires local machine with sufficient RAM.

---

## PythonAnywhere Setup Steps

### 1. Upload Files
Upload all files maintaining the folder structure:
- All HTML files and folders to your project root
- Backend folder with all Python files

### 2. Create Virtual Environment
```bash
# In PythonAnywhere console
mkvirtualenv --python=python3.10 myenv
pip install -r backend/requirements.txt
```

### 3. Set Environment Variables
In PythonAnywhere Web tab, add environment variables:

**For Groq (Free - Recommended):**
```
GROQ_API_KEY = your_groq_api_key_here
```

**For Gemini (Free):**
```
GEMINI_API_KEY = your_gemini_api_key_here
```

### 4. Configure Web App
1. Go to **Web** tab in PythonAnywhere
2. Create a new web app
3. Select "Manual configuration"
4. Choose Python 3.10

### 5. Configure WSGI File
Edit the WSGI file:
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

### 6. Static Files Configuration
In the Web tab:
- **URL**: `/` → **Directory**: `/home/yourusername/your-project/`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main landing page |
| `/TMC/` | GET | TMC section |
| `/api/knowledge-graph` | POST | Generate knowledge graph (uses LLM) |
| `/api/knowledge-graph/explain` | POST | Explain a concept (uses LLM) |
| `/api/network` | POST | Citation network analysis |
| `/api/molecules` | GET | Get molecule data for quiz |

---

## Environment Variables Reference

| Variable | Provider | Get From |
|----------|----------|----------|
| `GROQ_API_KEY` | Groq (FREE) | https://console.groq.com |
| `GEMINI_API_KEY` | Gemini (FREE) | https://ai.google.dev |
| `HF_API_KEY` | Hugging Face (FREE) | https://huggingface.co/settings/tokens |
| `OLLAMA_BASE_URL` | Ollama (FREE local) | http://localhost:11434 |
| `DEEPSEEK_API_KEY` | DeepSeek (paid) | https://platform.deepseek.com |
| `OPENAI_API_KEY` | OpenAI (paid) | https://platform.openai.com |

---

## Testing Locally

```bash
cd backend

# Set your API key (choose one)
export GROQ_API_KEY=your_key        # Free
export GEMINI_API_KEY=your_key      # Free

python app.py
# Visit http://localhost:5000
```

---

## Troubleshooting

1. **LLM not responding**: Check your API key is set correctly
2. **Static files not loading**: Check static folder path in app.py
3. **CORS errors**: flask-cors is already configured in app.py
4. **RDKit issues**: Use `pip install rdkit-pypi` on PythonAnywhere

---

## Quick Start with Free LLM

```bash
# 1. Get free Groq API key from https://console.groq.com

# 2. Set environment variable
export GROQ_API_KEY=gsk_xxxxx

# 3. Run the app
cd backend
pip install flask flask-cors requests
python app.py

# 4. Open http://localhost:5000/TMC/knowledge-graph.html
```

That's it! Your Knowledge Graph is now powered by a free LLM.
