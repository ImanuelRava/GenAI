# GenAI Research Platform

A Flask-based research platform for **transition-metal chemistry** (TMC),
**redox-active ligands**, **AI/ML education**, and **virology**. The backend
exposes citation-network analysis, NiCOBot chat (with RAG over a curated
reaction database), ChemExtract PDF extraction, GNN/PCA visualizations, and
knowledge-graph generation. Eight LLM providers are supported behind a
unified abstraction (DeepSeek, OpenAI, Anthropic, Gemini, Groq, HuggingFace,
OpenRouter, Ollama).

**Version:** 2.2.0
**License:** (add your LICENSE file)
**Python:** 3.10+ (developed on 3.13)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deploy on Replit](#deploy-on-replit) ← fastest way to run it
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Running](#running)
6. [Testing](#testing)
7. [Data Files & Git LFS](#data-files--git-lfs)
8. [API Endpoints](#api-endpoints)
9. [Troubleshooting](#troubleshooting)

---

## Deploy on Replit

This project includes Replit-specific config files (`.replit`, `replit.nix`, `main.py`) so you can deploy in ~5 minutes.

### Steps

1. **Create a new Replit** — go to [replit.com](https://replit.com), click "+ Create Repl", choose the **Python** template, and name it `GenAI`.

2. **Upload the code** — drag-and-drop the entire project zip into the Replit file tree (or use Shell → `git clone` if your repo is on GitHub).

3. **Install dependencies** — Replit will auto-detect `requirements.txt` and prompt you to install. Click **Install**. This installs:
   - Flask + extensions (web framework)
   - pypdf, pdfplumber, PyMuPDF (PDF processing)
   - pandas, numpy, networkx (data + graph analysis)
   - rdkit-pypi (chemistry — pre-built wheel, no system deps needed)
   - aiohttp (async LLM calls)

4. **Set your LLM API key** (optional but recommended):
   - Go to **Tools → Secrets** in the Replit sidebar
   - Add a secret named `DEEPSEEK_API_KEY` (or `GROQ_API_KEY`, `GEMINI_API_KEY`, etc.) with your key value
   - The app reads env vars automatically — no code changes needed
   - You can also set keys per-request via the web UI (BYO-key model)

5. **Click Run** — the green Run button starts `python wsgi.py`. The app auto-detects Replit (via the `REPL_ID` env var) and binds to `0.0.0.0:$PORT` so the web preview works.

6. **Open the web view** — Replit's web preview opens automatically. If not, click the **Open in new tab** icon next to the preview pane.

### What works on Replit

| Feature | Status | Notes |
|---|---|---|
| All HTML pages (TMC, AI, Redox, Virus) | ✅ Works | Served directly from Flask |
| LLM chat (NiCOBot, Redox, Knowledge Graph) | ✅ Works | BYO-key via Secrets or web UI |
| Citation network analysis | ✅ Works | Outbound HTTP to Crossref/OpenAlex allowed |
| PDF extraction (ChemExtract, ReactionLens, vision) | ✅ Works | All PDF libs install fine |
| GNN/PCA visualizations | ✅ Works | Pure numpy, no system deps |
| Molecule/reaction rendering (RDKit) | ✅ Works | Uses `rdkit-pypi` pre-built wheel |
| NiCOBot database search | ⚠️ Partial | Needs LFS data (see below) |

### Optional: Restore the NiCOBot LFS data

The `backend/nicobot_data/` CSVs are Git-LFS pointer stubs (~135 bytes each). Without them, NiCOBot chat works but **without the curated reaction database** (RAG context will be empty). To restore:

```bash
# In Replit Shell, from your original git repo:
git clone https://github.com/your-username/GenAI.git /tmp/genai-source
cd /tmp/genai-source
git lfs install
git lfs pull

# Copy the real CSV files over the stubs:
cp backend/nicobot_data/*.csv /path/to/your/replit/GenAI/backend/nicobot_data/
```

Or, if you have the CSVs locally, drag them into `backend/nicobot_data/` in the Replit file tree, overwriting the stubs.

See [`backend/nicobot_data/DATA_LFS.md`](backend/nicobot_data/DATA_LFS.md) for full recovery instructions.

### Replit config files

| File | Purpose |
|---|---|
| `.replit` | Tells Replit how to run the app (`python wsgi.py`), port mapping, Python version |
| `replit.nix` | System-level deps (libxml2, openssl, fonts) — applied automatically by Replit |
| `main.py` | Shim so Replit's default runner (`python main.py`) also works |

### Troubleshooting on Replit

**"Site can't be reached" in the web preview** — The app is binding to 127.0.0.1 instead of 0.0.0.0. Check that `wsgi.py` logs show `Replit environment detected`. If not, set `FLASK_HOST=0.0.0.0` in Secrets.

**`ModuleNotFoundError: No module named 'rdkit'`** — The `rdkit-pypi` package didn't install. Run `pip install rdkit-pypi` in Shell. If that fails, the `replit.nix` file installs system-level rdkit as a fallback — click **Install packages** when Replit prompts after editing `replit.nix`.

**Port mismatch** — Replit's web preview proxies to the port in your `.replit` file (5000 by default). If the app starts on a different port, either change `.replit`'s `localPort` or set `FLASK_PORT` in Secrets.

**Memory limit on free tier** — The free tier has ~512 MB RAM. If you restore the LFS CSVs (~75 MB), loading them into pandas may spike memory to ~500 MB. If the app crashes with OOM, upgrade to Replit Core or skip the LFS restore.

---

## Quick Start

```bash
# 1. Clone and create a virtual env
git clone <your-repo-url> GenAI
cd GenAI
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment (BYO API key model — no server-side keys required)
cp backend/.env.template backend/.env
# Edit backend/.env to add at least one LLM provider key (DeepSeek, Groq,
# Gemini, OpenRouter, etc.) — or leave blank and provide keys via the web UI.

# 4. Pull the large data files (CSVs are Git-LFS tracked)
git lfs install                       # one-time per machine
git lfs pull                          # downloads ~75 MB of nicobot_data

# 5. Run the dev server
python wsgi.py                        # binds 127.0.0.1:5000
# Open http://127.0.0.1:5000 in your browser.
```

---

## Project Structure

```
GenAI/
├── wsgi.py                       # WSGI entry point (dev + prod)
├── requirements.txt              # Python deps
├── .gitignore                    # excludes .env, __pycache__, uploads, etc.
├── .gitattributes                # Git LFS tracking for large data files
├── index.html                    # Landing page (links to all 4 domains)
├── backend/
│   ├── app.py                    # Flask app factory + blueprint wiring
│   ├── .env.template             # Copy to .env and fill in API keys
│   ├── core/                     # Framework utilities (config, cache, errors)
│   ├── llm/                      # LLM provider abstraction (8 providers)
│   ├── chat/                     # NiCOBot, Redox, Knowledge-Graph blueprints
│   ├── routes/                   # HTTP blueprints (network, chemistry, ...)
│   ├── modules/
│   │   ├── DOI.py                # Crossref/OpenAlex citation lookup
│   │   ├── Forward_Reference.py  # Forward citation graph
│   │   ├── Backward_Reference.py # Backward citation graph
│   │   ├── Cross_Reference.py    # Excel-based cross-citation
│   │   ├── nicobot_database.py   # Compound / paper / reaction indexer
│   │   ├── nicobot_rag.py        # Keyword RAG over nicobot_data/
│   │   ├── gnn_viz.py            # Static GNN visualization data
│   │   ├── pca_viz.py            # Hand-rolled PCA visualization data
│   │   ├── chemextract/          # PDF → chemical entities pipeline
│   │   └── reaction/             # ReactionLens text-screening pipeline
│   └── nicobot_data/             # Curated reaction database (LFS-tracked)
├── AI/                           # ML/AI educational HTML pages
├── TMC/                          # Transition-metal chemistry HTML pages
├── redox-ligands/                # Redox-active ligand HTML pages + pipeline
├── virus/                        # Virology HTML pages
├── logic_flow/                   # Standalone flowchart HTML pages
├── css/                          # Shared header.css
└── tests/                        # pytest suite
```

---

## Configuration

All configuration is via environment variables (loaded from `backend/.env`).

### Required (none)

The platform uses a **BYO-key model** — users supply their own LLM API keys
via the web UI when invoking LLM-powered features. The backend ships without
any server-side keys.

### Recommended

| Variable | Default | Purpose |
|---|---|---|
| `SECRET_KEY` | (random) | Flask session secret. Set to a fixed value in prod for sticky sessions. |
| `CORS_ORIGINS` | `http://localhost:5000,...` | Comma-separated allowed origins. |
| `FLASK_DEBUG` | `0` | Set to `1` to enable Werkzeug debugger (localhost-only). |
| `FLASK_HOST` | `127.0.0.1` | Bind host. Override to `0.0.0.0` only behind a reverse proxy. |
| `FLASK_PORT` | `5000` | Bind port. |

### LLM Providers (optional — for backend defaults)

Users can supply any of these via the web UI; set them in `.env` only if you
want a server-wide default.

| Variable | Provider | Free tier? |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek | No |
| `OPENAI_API_KEY` | OpenAI | No |
| `ANTHROPIC_API_KEY` | Anthropic | No |
| `GEMINI_API_KEY` | Google Gemini | Yes |
| `GROQ_API_KEY` | Groq | Yes |
| `HF_API_KEY` | Hugging Face | Yes |
| `OPENROUTER_API_KEY` | OpenRouter | Yes (some models) |
| `OLLAMA_BASE_URL` | Ollama (local) | Yes (self-hosted) |

### Caching (optional)

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_URL` | (unset) | If set, cache backend switches from in-memory LRU to Redis. |
| `CACHE_TTL` | `3600` | Cache TTL in seconds. |

---

## Running

### Development

```bash
python wsgi.py                      # binds 127.0.0.1:5000 (no debugger)
FLASK_DEBUG=1 python wsgi.py        # enables Werkzeug debugger (localhost only)
```

The Werkzeug debugger allows arbitrary code execution — when `FLASK_DEBUG=1`,
`wsgi.py` automatically forces the bind to `127.0.0.1` even if `FLASK_HOST`
is set to `0.0.0.0`, to prevent remote exploitation.

### Production

Use a real WSGI server. Debug mode is impossible to enable via `wsgi.py`.

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 wsgi:application
```

Or with waitress (Windows-friendly):

```bash
pip install waitress
waitress-serve --port=5000 wsgi:application
```

---

## Testing

```bash
# Install test deps
pip install pytest pytest-asyncio pytest-cov

# Run the full suite
pytest

# Run with coverage
pytest --cov=backend --cov-report=term-missing

# Run a single test file
pytest tests/test_conversation_store.py -v
```

Test files:

| File | Coverage |
|---|---|
| `tests/test_app.py` | Health/status, molecules, reactions, GNN, PCA, LLM status, KG, error handling |
| `tests/test_conversation_store.py` | ConversationStore add/get/trim, TTL eviction, thread-safety |
| `tests/test_nicobot_database.py` | Cross-coupling info, inference helpers, fixture-loaded DB |
| `tests/test_pca_viz.py` | All route data_types now reach distinct branches |
| `tests/test_utils.py` | sanitize_input, sanitize_filename, validate_doi, validate_api_key |

---

## Data Files & Git LFS

The `backend/nicobot_data/` directory contains the curated NiCOBot database.
The 12 CSV files (and any `.xlsx` over 50 MB) are tracked by Git LFS to keep
the repository lightweight.

**If you see files like this:**

```
version https://git-lfs.github.com/spec/v1
oid sha256:1e33cb012da319875753691f7a974a36a63e1aea1c77a678567c4fab2d00c85d
size 36929841
```

…then the LFS objects haven't been pulled. Run:

```bash
git lfs install
git lfs pull
```

See [`backend/nicobot_data/DATA_LFS.md`](backend/nicobot_data/DATA_LFS.md) for
full recovery instructions and the list of affected files.

---

## API Endpoints

All endpoints live under `/api/*`. Key families:

| Family | Endpoints |
|---|---|
| **Health / Status** | `GET /api/health`, `GET /api/status` |
| **Chemistry** | `GET /api/molecules`, `GET /api/reactions`, `GET /api/reaction/<key>` |
| **Citation network** | `POST /api/network` (PDF or Excel upload) |
| **Visualization** | `GET /api/gnn/*`, `GET /api/pca/*` |
| **LLM** | `GET /api/llm/status`, `POST /api/llm/chat`, `GET /api/llm/providers` |
| **Chat** | `POST /api/nicobot/chat[/async]`, `POST /api/redox/chat[/async]` |
| **Knowledge graph** | `POST /api/knowledge-graph[/async][/explain]` |
| **Database** | `GET /api/database/{status,search/compounds,search/papers,...}` |
| **Data extraction** | `POST /api/extract[/pdf/vision/chemextract/reactionlens][...]` |

Each chat / LLM endpoint accepts `provider`, `api_key`, `model`, and the
user's message in the JSON body.

---

## Troubleshooting

### "NiCOBot database returns empty results"

The CSV data files are Git-LFS pointer stubs. Run `git lfs pull`. See
[`backend/nicobot_data/DATA_LFS.md`](backend/nicobot_data/DATA_LFS.md).

### "Werkzeug debugger is exposed to the network"

You set `FLASK_DEBUG=1` and `FLASK_HOST=0.0.0.0`. The fixed `wsgi.py` forces
`127.0.0.1` in this case. To bind publicly with debug off, just remove
`FLASK_DEBUG`.

### "SSL: CERTIFICATE_VERIFY_FAILED" in the redox-ligand pipeline

The pipeline now verifies TLS by default. If you're behind a corporate
TLS-intercepting proxy, set `LIGAND_INSECURE_TLS=1` in your environment as a
last resort (this disables verification and is insecure).

### "ModuleNotFoundError: No module named 'rdkit'"

Some optional features depend on `rdkit`, which isn't always pip-installable
on all platforms. The platform gracefully degrades — `/api/health` will
report `rdkit: false`, and chemistry endpoints will return 503.

### "Flask-Limiter: too many requests"

Default rate limit is `200 per day; 50 per hour` globally, with stricter
limits on chat endpoints (`20 per minute`). Override via
`Config.RATE_LIMIT_DEFAULT` in `backend/core/config.py`.

---

## Contributing

1. Run `pytest` before pushing — CI will reject PRs that break tests.
2. Follow the existing layering: `core/` ← `llm/` ← `chat/` + `routes/` ←
   `modules/`. Don't import `routes/` from `modules/`, etc.
3. Don't commit `.env` files (the `.gitignore` enforces this) — use
   `backend/.env.template` as the canonical template.
4. Don't commit `__pycache__/` or `.pyc` files (also gitignored).
5. Large data files (`*.csv`, `*.xlsx`, `*.json` over ~1 MB in
   `backend/nicobot_data/`) must be Git-LFS-tracked — see `.gitattributes`.
