"""
Shared helpers, constants, and the Blueprint for the data_extraction package.

All endpoint modules in this package register their routes on the shared
``data_extraction_bp`` blueprint defined here. The blueprint is then
re-exported from ``routes/data_extraction/__init__.py`` so that
``app.register_blueprint(data_extraction_bp)`` in ``backend/app.py``
continues to work unchanged.

Pre-split history: this content lived at the top of the monolithic
``routes/data_extraction.py`` (887 LOC). It has been extracted here so
that each endpoint module can be small (~50-150 LOC) and focused on a
single extraction pipeline.
"""

import os
import logging
import tempfile
from typing import Any, Dict, List

from flask import Blueprint, request
from werkzeug.utils import secure_filename

from core.errors import ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blueprint — shared by all endpoint modules in this package.
# url_prefix='/api' matches the original so all routes stay at /api/extract/...
# ---------------------------------------------------------------------------

data_extraction_bp = Blueprint('data_extraction', __name__, url_prefix='/api')

# ---------------------------------------------------------------------------
# Provider / model registry
# ---------------------------------------------------------------------------
# NOTE: These constants are also defined in llm.client (PROVIDER_DEFAULT_MODELS,
# VISION_CAPABLE_PROVIDERS) — the unified LLM client. They're re-declared here
# because (a) data_extraction.py predates the LLMClient consolidation and
# (b) this module adds 'zhipu' (not in the unified client) and uses different
# default models for some providers (e.g. 'gpt-4o-mini' vs 'gpt-4o-mini').
# A future cleanup should consolidate these by importing from llm.client.

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    'deepseek': 'deepseek-chat',
    'openai': 'gpt-4o-mini',
    'gemini': 'gemini-2.0-flash',
    'groq': 'llama-3.3-70b-versatile',
    'ollama': 'llama3',
    'anthropic': 'claude-3-5-sonnet-20241022',
    'zhipu': 'glm-4-flash',
}

VISION_CAPABLE_PROVIDERS: List[str] = ['deepseek', 'openai', 'gemini', 'anthropic']

AVAILABLE_MODELS: List[Dict[str, str]] = [
    {"id": "deepseek-chat", "name": "DeepSeek Chat", "provider": "DeepSeek",
     "description": "Fast and efficient, vision capable"},
    {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "provider": "DeepSeek",
     "description": "Enhanced reasoning"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI",
     "description": "Vision capable"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI",
     "description": "Fast and cheap"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Google",
     "description": "Vision capable"},
    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet",
     "provider": "Anthropic", "description": "Vision capable"},
    {"id": "glm-4-flash", "name": "GLM-4-Flash", "provider": "Zhipu AI",
     "description": "Fast and efficient"},
]

REACTIONLENS_INFO: Dict[str, Any] = {
    "id": "reactionlens",
    "name": "ReactionLens",
    "provider": "Built-in",
    "description": (
        "Text-driven chemical reaction detection: extracts text from paper, "
        "screens paragraphs for reactions, outputs ChemExtract-compatible data"
    ),
    "capabilities": [
        "text_extraction",
        "paragraph_screening",
        "reaction_detection",
        "condition_parsing",
        "chemextract_compatible_output",
    ],
    "supported_formats": ["pdf"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_model_for_provider(provider: str, model: str = None) -> str:
    """Resolve which model to use for a provider.

    If the caller specified an explicit model, use it. Otherwise fall back
    to the provider's default model from PROVIDER_DEFAULT_MODELS, and
    finally to 'deepseek-chat' if the provider is unknown.
    """
    if model:
        return model
    return PROVIDER_DEFAULT_MODELS.get(provider, 'deepseek-chat')


def validate_pdf_upload() -> tuple:
    """Validate a PDF upload from the current Flask request.

    Returns:
        (tmp_path, filename) — the path to a saved temp file and the
        secure original filename.

    Raises:
        ValidationError: if no file was uploaded, the file is empty, or
            the filename doesn't end in .pdf.
    """
    if 'file' not in request.files:
        raise ValidationError("No file provided")

    file = request.files['file']

    if file.filename == '':
        raise ValidationError("No file selected")

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.pdf'):
        raise ValidationError("Only PDF files are supported")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    file.save(tmp.name)
    tmp.close()

    return tmp.name, filename


def cleanup_temp_file(file_path: str) -> None:
    """Best-effort delete of a temp file. Logs a warning on failure.

    Safe to call with None or a non-existent path.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError as e:
        # Only OSError is expected from os.remove (permission denied, file
        # in use, etc.). Other exceptions indicate a bug and should propagate.
        logger.warning(f"Failed to cleanup temp file: {e}")


def merge_extraction_results(all_results: list) -> dict:
    """Merge extraction results from multiple pages/sources.

    Handles two output formats emitted by the vision LLM prompts:
      1. **Flat format** (SYSTEM_PROMPT_VISION): top-level keys
         ``reactants``, ``products``, ``catalysts``, ``ligands``,
         ``solvents``, ``conditions``, ``yields``, ``mechanisms``,
         ``reactionType``, ``selectivity``, ``image_description``.
      2. **Structured format** (SYSTEM_PROMPT_FIGURE_ANALYSIS):
         ``reaction_schemes`` (list of {reactants, products, ...}),
         ``compounds`` (list of {name, smiles, ...}),
         ``table_data`` (list of rows).

    Structured-format entries are also flattened into the flat-format
    keys for backwards compatibility (the frontend reads both shapes).
    Each merged entry from the structured format is tagged with
    ``_page`` and ``_source`` so the UI can trace provenance.
    """
    merged: Dict[str, Any] = {
        # Flat format keys
        "reactants": [],
        "products": [],
        "catalysts": [],
        "ligands": [],
        "solvents": [],
        "conditions": {},
        "yields": [],
        "mechanisms": [],
        "reactionType": None,
        "selectivity": None,
        "image_descriptions": [],
        "pages_with_data": [],
        # Structured format keys
        "reaction_schemes": [],
        "compounds": [],
        "table_data": [],
    }

    for page_result in all_results:
        page_num = page_result["page"]
        source = page_result.get("source", "unknown")
        data = page_result["data"]

        if not data:
            continue

        merged["pages_with_data"].append(page_num)

        # ── Flat format: reactants, products, etc. ──
        for key in ["reactants", "products", "catalysts", "ligands", "solvents", "mechanisms"]:
            if key in data and data[key]:
                for item in data[key]:
                    if item and item not in merged[key]:
                        merged[key].append(item)

        if "yields" in data and data["yields"]:
            for y in data["yields"]:
                if y not in merged["yields"]:
                    merged["yields"].append(y)

        if "conditions" in data and data["conditions"]:
            for cond_key, cond_val in data["conditions"].items():
                if cond_val and not merged["conditions"].get(cond_key):
                    merged["conditions"][cond_key] = cond_val

        if not merged["reactionType"] and data.get("reactionType"):
            merged["reactionType"] = data["reactionType"]
        if not merged["selectivity"] and data.get("selectivity"):
            merged["selectivity"] = data["selectivity"]

        if data.get("image_description"):
            merged["image_descriptions"].append({
                "page": page_num,
                "description": data["image_description"],
            })
        elif data.get("description"):
            merged["image_descriptions"].append({
                "page": page_num,
                "description": data["description"],
                "source": source,
            })

        # ── Structured format: reaction_schemes, compounds, table_data ──
        if data.get("reaction_schemes"):
            for scheme in data["reaction_schemes"]:
                scheme["_page"] = page_num
                scheme["_source"] = source
                merged["reaction_schemes"].append(scheme)
                # Also flatten into the flat format for backward compatibility.
                for entity in scheme.get("reactants", []):
                    name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                    if name and name not in merged["reactants"]:
                        merged["reactants"].append(name)
                for entity in scheme.get("products", []):
                    name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                    if name and name not in merged["products"]:
                        merged["products"].append(name)

        if data.get("compounds"):
            for comp in data["compounds"]:
                name = comp.get("name", "")
                if name:
                    comp_entry = dict(comp)
                    comp_entry["_page"] = page_num
                    comp_entry["_source"] = source
                    merged["compounds"].append(comp_entry)

        if data.get("table_data"):
            for row in data["table_data"]:
                row_entry = dict(row) if isinstance(row, dict) else {"values": row}
                row_entry["_page"] = page_num
                row_entry["_source"] = source
                merged["table_data"].append(row_entry)

    return merged
