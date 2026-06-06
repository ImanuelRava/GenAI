"""
ChemExtract AI - Advanced PDF Extractor for Chemical Reaction Data

This module provides comprehensive chemical information extraction from PDFs,
using AI-powered multimodal analysis including:
- Text extraction with chemical entity recognition
- Image analysis for reaction schemes, structures, and tables
- Molecular formula and SMILES notation extraction
- Reaction condition parsing

Uses LLM Vision models via HTTP API calls for intelligent extraction.
"""

import os
import re
import json
import logging
import base64
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple

import requests
import aiohttp

logger = logging.getLogger(__name__)

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("[ChemExtract] PyMuPDF not available")

try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# ── Constants ───────────────────────────────────────────────────────────────

TEXT_CHUNK_SIZE = 12000
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds


def _retry_on_failure(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Retry a function call on failure with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"[ChemExtract] Call failed (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"[ChemExtract] Call failed after {max_retries+1} attempts: {e}")
                raise


async def _retry_on_failure_async(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """Async retry wrapper with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                wait = RETRY_DELAY * (attempt + 1)
                logger.warning(f"[ChemExtract] Async call failed (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...")
                await asyncio.sleep(wait)
            else:
                logger.error(f"[ChemExtract] Async call failed after {max_retries+1} attempts: {e}")
                raise


SYSTEM_PROMPT_CHEMICAL_ENTITIES = """You are an expert chemist specializing in extracting chemical entities from scientific literature. 

Extract ALL chemical entities from the given text, including:
1. Chemical compounds (by name, formula, or description)
2. Reagents and starting materials
3. Products and intermediates
4. Catalysts and catalyst systems
5. Ligands (phosphines, NHCs, bipyridines, etc.)
6. Solvents
7. Additives and bases/acids

For each entity, provide:
- name: The common or IUPAC name
- smiles: SMILES notation if determinable
- formula: Molecular formula if determinable
- role: reactant, product, catalyst, ligand, solvent, reagent, additive

Return a JSON object:
{
  "entities": [
    {
      "name": "compound name",
      "smiles": "SMILES string or null",
      "formula": "molecular formula or null",
      "role": "reactant|product|catalyst|ligand|solvent|reagent|additive"
    }
  ]
}"""

SYSTEM_PROMPT_REACTION_SCHEME = """You are an expert chemist analyzing reaction scheme images from scientific papers.

Analyze the image carefully and extract:
1. All molecular structures visible (as SMILES if possible, or describe them)
2. Reaction arrows and their direction
3. Reagents and conditions written above/below arrows
4. Yields (percentages or amounts)
5. Temperature, time, pressure conditions
6. Catalyst and ligand structures
7. Any stereochemistry indicators

**CRITICAL: R-Group / Substituent Tables**
Many papers show a general scaffold (e.g., compound 1a) with placeholder substituents
(R1, R2, Y, Ar, etc.) and then list the specific values of those substituents in a
table of entries (1, 2, 3, ...). When you see this pattern you MUST:

  a) Provide the SMILES of the CORE SCAFFOLD (with the placeholder atoms as wildcard *
     atoms, e.g. [R1], [R2], [*] etc.) in the "scaffold_smiles" field.
  b) List every substituent group you find in a "rgroup_table" object.
  c) For EACH table entry provide the FULL SMILES of the complete molecule by
     substituting the actual group into the scaffold.  Use RDKit-style notation:
     replace [*] or [*:1] with the actual SMILES fragment.

Return a JSON object:
{
  "reaction_schemes": [
    {
      "entry": 1,
      "reactants": [{"name": "...", "smiles": "...", "structure_description": "..."}],
      "products": [{"name": "...", "smiles": "...", "structure_description": "..."}],
      "reagents": ["list of reagents"],
      "conditions": {
        "temperature": "e.g., 80\u00b0C",
        "time": "e.g., 12h",
        "solvent": "solvent name",
        "atmosphere": "N2/Ar/air",
        "other": "other conditions"
      },
      "yield": "yield value",
      "catalyst": "catalyst info",
      "ligand": "ligand info"
    }
  ],
  "scaffold_smiles": "SMILES of the general scaffold with [*] placeholders (if applicable)",
  "rgroup_table": {
    "scaffold_label": "1a",
    "rgroups": {
      "R1": {"1a": "SMILES", "1b": "SMILES", "1c": "SMILES"},
      "Y":  {"1a": "Cl", "1b": "SPh", "1c": "Cl"}
    },
    "partner_rgroups": {
      "R2": {"2a": "SMILES", "2b": "SMILES"}
    }
  },
  "description": "overall description of what is shown",
  "notes": "any additional observations"
}

IMPORTANT RULES:
- Extract EVERY reaction scheme visible in the image.
- Do not stop after finding the first reaction - list ALL of them.
- Include every row from R-group tables."""

SYSTEM_PROMPT_TABLE_EXTRACTION = """You are an expert chemist extracting data from chemistry tables in scientific papers.

Extract table data including:
1. Entry/row numbers
2. Substrate/product variations
3. Reaction conditions for each entry
4. Yields and selectivities
5. Any footnotes or special conditions

Return a JSON object:
{
  "table_type": "optimization|substrate_scope|condition_screening|other",
  "columns": ["column names"],
  "data": [
    {
      "entry": 1,
      "values": {"column_name": "value", ...}
    }
  ],
  "footnotes": ["footnote texts"],
  "general_conditions": "general reaction conditions if stated"
}"""

SYSTEM_PROMPT_VISION = """You are an expert chemist specializing in analyzing scientific literature and extracting chemical reaction data from images. 

Analyze the provided image(s) from a scientific paper and extract ALL chemical information you can see, including:

1. **Reaction Schemes**: Chemical structures, reaction arrows, reagents, conditions shown on arrows
2. **Tables**: Any data tables with yields, conditions, compound information
3. **Figures**: Graphs showing yield vs conditions, selectivity data, etc.
4. **Chemical Structures**: Named compounds, SMILES-like notations, molecular formulas
5. **Text in Images**: Any visible text including compound names, conditions, notes

**CRITICAL: R-Group / Substituent Tables**
Many papers show a general scaffold with placeholder substituents (R1, R2, Ar, Y, etc.)
and list the specific substituent values in a separate table of entries.  When you see this
pattern you MUST:

  a) Provide the SMILES of the CORE SCAFFOLD with placeholder atoms as [*] or [*:1] etc.
  b) Record every substituent group found in an "rgroup_table" object.
  c) For EACH entry, assemble the FULL SMILES by substituting actual groups into
     the scaffold (replace [*] with the real fragment SMILES).

Return a JSON object with this structure:
{
  "reactants": ["list of reactant names/structures you can identify"],
  "products": ["list of product names/structures you can identify"],
  "catalysts": ["list of catalysts"],
  "ligands": ["list of ligands"],
  "solvents": ["list of solvents mentioned"],
  "conditions": {
    "temperature": "temperature if shown",
    "time": "reaction time if shown",
    "pressure": "pressure if applicable",
    "atmosphere": "N2, Ar, air, etc."
  },
  "yields": [
    {"product": "product name", "yield": "yield value"}
  ],
  "selectivity": "selectivity information (ee, de, etc.)",
  "reactionType": "type of reaction shown",
  "mechanisms": ["any mechanistic information"],
  "scaffold_smiles": "SMILES of the general scaffold with [*] placeholders (if applicable)",
  "rgroup_table": {
    "scaffold_label": "label of the scaffold compound (e.g. 1a)",
    "rgroups": {
      "R1": {"1a": "SMILES", "1b": "SMILES"},
      "R2": {"2a": "SMILES", "2b": "SMILES"}
    }
  },
  "image_description": "brief description of what is shown",
  "additional_observations": "any other relevant chemical information seen"
}

Be thorough and extract ALL visible chemical information. If you see reaction schemes, describe the complete transformation. If you see tables, extract all relevant data. Pay special attention to scaffold structures and their R-group substituent tables.

IMPORTANT RULES:
- Extract EVERY reaction visible on this page. Do not stop after 1-2 reactions.
- Each row in a table = one separate reaction.
- Include ALL reagents, conditions, and yields for each reaction."""

SYSTEM_PROMPT_COMPREHENSIVE = """You are ChemExtract AI, an expert chemistry data extraction system. Analyze the provided scientific document content and extract ALL chemical information comprehensively.

EXTRACT AND STRUCTURE:

1. **Chemical Entities**
   - All compounds mentioned (reactants, products, intermediates)
   - Catalysts and catalytic systems
   - Ligands (organophosphines, NHCs, nitrogen ligands, etc.)
   - Solvents, reagents, additives, bases, acids
   - Include SMILES and molecular formulas when determinable

2. **Reaction Information**
   - Reaction type (coupling, addition, oxidation, etc.)
   - Transformation description
   - Mechanistic insights if mentioned

3. **Reaction Conditions**
   - Temperature (value and unit)
   - Time (duration)
   - Pressure (if applicable)
   - Atmosphere (N2, Ar, air, etc.)
   - Concentration
   - Scale

4. **Outcomes**
   - Isolated yields
   - Conversions
   - Selectivities (ee, de, regioselectivity)
   - Turnover numbers/frequencies (TON/TOF)

5. **Molecular Structures**
   - SMILES strings
   - InChI if determinable
   - Molecular formulas
   - Structural features (stereocenters, functional groups)

6. **Spectral Data** (if present)
   - NMR shifts
   - MS data
   - IR peaks

7. **R-Group / Substituent Tables (CRITICAL)**
   Many papers define a general scaffold with placeholder groups (R1, R2, Ar, Y, etc.)
   and provide the specific values in a table of numbered entries.  When you see this
   pattern you MUST:

   a) Provide the SMILES of the CORE SCAFFOLD with placeholder atoms as [*] or [*:1].
   b) Record every substituent group in an "rgroup_table" object.
   c) For EACH table entry, assemble the FULL SMILES by substituting the actual
      group SMILES into the scaffold placeholder.

Return structured JSON:
{
  "reactions": [
    {
      "id": "reaction_1",
      "type": "reaction type",
      "entry": 1,
      "reactants": [{"name": "...", "smiles": "...", "formula": "..."}],
      "products": [{"name": "...", "smiles": "...", "formula": "..."}],
      "catalysts": [{"name": "...", "loading": "..."}],
      "ligands": [{"name": "..."}],
      "solvents": ["..."],
      "conditions": {
        "temperature": "...",
        "time": "...",
        "atmosphere": "...",
        "pressure": "...",
        "concentration": "..."
      },
      "outcomes": {
        "yield": "...",
        "conversion": "...",
        "selectivity": {...}
      }
    }
  ],
  "scaffold_smiles": "SMILES of general scaffold with [*] placeholders (if applicable)",
  "rgroup_table": {
    "scaffold_label": "label (e.g. 1a)",
    "rgroups": {
      "R1": {"1a": "SMILES", "1b": "SMILES", "1c": "SMILES"},
      "Y":  {"1a": "Cl", "1b": "SPh"}
    },
    "partner_rgroups": {
      "R2": {"2a": "SMILES", "2b": "SMILES"}
    }
  },
  "compounds": [
    {
      "name": "...",
      "smiles": "...",
      "formula": "...",
      "mw": "...",
      "role": "..."
    }
  ],
  "experimental_procedures": ["..."],
  "characterization_data": {...}
}

IMPORTANT EXTRACTION RULES:
- Extract EVERY individual reaction. Do NOT combine, summarize, or skip any reactions.
- Each row in a substrate scope table = one separate reaction entry.
- Number reactions sequentially: "id": "reaction_1", "reaction_2", etc.
- If you find a table with 20 entries, you MUST produce 20 reaction entries.
- Include the entry/row number from tables in the "entry" field.
- If text is truncated, extract as many reactions as possible from the visible portion."""

def pdf_to_images(file_path: str, dpi: int = 150, max_pages: int = 50) -> List[Tuple[int, str]]:
    """
    Convert PDF pages to base64-encoded images using PyMuPDF.
    
    Returns list of tuples: [(page_number, base64_image), ...]
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required for PDF to image conversion. Install with: pip install PyMuPDF")

    images = []

    try:
        doc = fitz.open(file_path)
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            base64_image = base64.b64encode(img_data).decode('utf-8')
            images.append((page_num + 1, base64_image))

        logger.info(f"[ChemExtract] Converted {len(images)} pages using PyMuPDF")
        return images
    except Exception as e:
        logger.error(f"[ChemExtract] PyMuPDF failed: {e}")
        raise Exception(f"Failed to convert PDF to images: {e}")


def extract_text_from_pdf(file_path: str) -> Tuple[str, Dict]:
    """
    Extract text content from PDF.
    
    Returns (text, metadata) tuple.
    """
    text = ""
    metadata = {"pages": 0, "method": None}

    if HAS_PYPDF:
        try:
            reader = pypdf.PdfReader(file_path)
            metadata["pages"] = len(reader.pages)
            metadata["method"] = "pypdf"

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"

            if text.strip():
                return text, metadata
        except Exception as e:
            logger.warning(f"[ChemExtract] pypdf failed: {e}")

    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                metadata["method"] = "pdfplumber"

                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"

            return text, metadata
        except Exception as e:
            logger.error(f"[ChemExtract] pdfplumber failed: {e}")

    if HAS_PYMUPDF:
        try:
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)
            metadata["method"] = "pymupdf"

            for page in doc:
                text += page.get_text() + "\n\n"

            return text, metadata
        except Exception as e:
            logger.error(f"[ChemExtract] PyMuPDF text extraction failed: {e}")

    raise ImportError("No PDF text extraction library available. Install pypdf, pdfplumber, or PyMuPDF.")



def call_vision_llm(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str
) -> Optional[Dict]:
    """Call a vision-capable LLM via HTTP to extract data from an image."""

    try:
        if provider == 'deepseek':
            return _call_deepseek_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'openai':
            return _call_openai_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'gemini':
            return _call_gemini_vision(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'anthropic':
            return _call_anthropic_vision(base64_image, model, api_key, system_prompt, user_message)
        else:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"[ChemExtract] Vision LLM call failed: {e}")
        return None


async def call_vision_llm_async(
    base64_image: str,
    provider: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_message: str
) -> Optional[Dict]:
    """Async call to a vision-capable LLM via HTTP."""

    try:
        if provider == 'deepseek':
            return await _call_deepseek_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'openai':
            return await _call_openai_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'gemini':
            return await _call_gemini_vision_async(base64_image, model, api_key, system_prompt, user_message)
        elif provider == 'anthropic':
            return await _call_anthropic_vision_async(base64_image, model, api_key, system_prompt, user_message)
        else:
            logger.error(f"[ChemExtract] Unsupported vision provider: {provider}")
            return None
    except Exception as e:
        logger.error(f"[ChemExtract] Async vision LLM call failed: {e}")
        return None


def _parse_json_response(content: str) -> Optional[Dict]:
    """Parse JSON from LLM response. Handles truncated, partial, and multiple JSON outputs."""
    if not content:
        return None

    # Try direct parse first
    content = content.strip()
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON via brace matching (find outermost balanced braces)
    depth = 0
    start = -1
    best_json = None
    
    for i, char in enumerate(content):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(content[start:i+1])
                    # Prefer longer/outermost JSON
                    if best_json is None or len(candidate) > len(best_json):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    if best_json:
        return best_json

    # Fallback: try fixing common LLM JSON issues
    try:
        # Try adding closing brackets if truncated
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if open_braces > 0 or open_brackets > 0:
            fixed = content + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"[ChemExtract] JSON parse failed for content of length {len(content)}")
    return None



def _call_deepseek_vision(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call DeepSeek Vision API via HTTP."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] DeepSeek API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek vision request error: {e}")
            return None

    return _retry_on_failure(_do_call)


def _call_openai_vision(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call OpenAI GPT-4 Vision API via HTTP."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] OpenAI API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI vision request error: {e}")
            return None

    return _retry_on_failure(_do_call)


def _call_gemini_vision(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call Google Gemini Vision API via HTTP."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_message}"},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384
        }
    }

    def _do_call():
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['candidates'][0]['content']['parts'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Gemini API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini vision request error: {e}")
            return None

    return _retry_on_failure(_do_call)


def _call_anthropic_vision(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call Anthropic Claude Vision API via HTTP."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": user_message}
                ]
            }
        ]
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['content'][0]['text']
                return _parse_json_response(content)
            logger.error(f"[ChemExtract] Anthropic API error: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic vision request error: {e}")
            return None

    return _retry_on_failure(_do_call)



async def _call_deepseek_vision_async(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call DeepSeek Vision API via HTTP (async)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    logger.error(f"[ChemExtract] DeepSeek API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek async vision request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


async def _call_openai_vision_async(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call OpenAI GPT-4 Vision API via HTTP (async)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                ]
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    logger.error(f"[ChemExtract] OpenAI API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI async vision request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


async def _call_gemini_vision_async(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call Google Gemini Vision API via HTTP (async)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{system_prompt}\n\n{user_message}"},
                    {"inline_data": {"mime_type": "image/png", "data": base64_image}}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384
        }
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return _parse_json_response(content)
                    logger.error(f"[ChemExtract] Gemini API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini async vision request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


async def _call_anthropic_vision_async(base64_image: str, model: str, api_key: str, system_prompt: str, user_message: str) -> Optional[Dict]:
    """Call Anthropic Claude Vision API via HTTP (async)."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}},
                    {"type": "text", "text": user_message}
                ]
            }
        ]
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        return _parse_json_response(content)
                    logger.error(f"[ChemExtract] Anthropic API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic async vision request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)



def call_text_llm(
    text: str,
    provider: str,
    model: str,
    api_key: str
) -> Optional[Dict]:
    """Call LLM for text analysis via HTTP."""
    try:
        if provider == 'deepseek':
            return _call_deepseek_text(text, model, api_key)
        elif provider == 'openai':
            return _call_openai_text(text, model, api_key)
        elif provider == 'gemini':
            return _call_gemini_text(text, model, api_key)
        elif provider == 'anthropic':
            return _call_anthropic_text(text, model, api_key)
        else:
            return _call_generic_text(text, provider, model, api_key)
    except Exception as e:
        logger.error(f"[ChemExtract] Text LLM call failed: {e}")
        return None


async def call_text_llm_async(
    text: str,
    provider: str,
    model: str,
    api_key: str
) -> Optional[Dict]:
    """Async call LLM for text analysis via HTTP."""
    try:
        if provider == 'deepseek':
            return await _call_deepseek_text_async(text, model, api_key)
        elif provider == 'openai':
            return await _call_openai_text_async(text, model, api_key)
        elif provider == 'gemini':
            return await _call_gemini_text_async(text, model, api_key)
        elif provider == 'anthropic':
            return await _call_anthropic_text_async(text, model, api_key)
        else:
            return await _call_generic_text_async(text, provider, model, api_key)
    except Exception as e:
        logger.error(f"[ChemExtract] Async text LLM call failed: {e}")
        return None


def call_text_llm_chunked(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    chunk_size: int = TEXT_CHUNK_SIZE
) -> Optional[Dict]:
    """Call LLM for text analysis, splitting into chunks and merging results.

    For texts longer than chunk_size, splits the text into overlapping chunks,
    calls the LLM on each chunk, and merges the results (reactions, compounds, etc.)
    into a single combined dictionary.
    """
    if not text or not text.strip():
        return None

    if len(text) <= chunk_size:
        return call_text_llm(text, provider, model, api_key)

    # Split text into chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    logger.info(f"[ChemExtract] Text ({len(text)} chars) split into {len(chunks)} chunks of ~{chunk_size} chars")

    merged: Dict[str, Any] = {
        "reactions": [],
        "compounds": [],
    }

    for i, chunk in enumerate(chunks):
        logger.info(f"[ChemExtract] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        chunk_result = call_text_llm(chunk, provider, model, api_key)
        if chunk_result:
            # Merge reactions
            for reaction in chunk_result.get("reactions", []):
                merged["reactions"].append(reaction)

            # Merge compounds
            for compound in chunk_result.get("compounds", []):
                merged["compounds"].append(compound)

            # Merge other top-level keys if not already present
            for key in ("scaffold_smiles", "rgroup_table", "experimental_procedures",
                        "characterization_data"):
                if key in chunk_result and key not in merged:
                    merged[key] = chunk_result[key]

    # Renumber reactions sequentially
    for idx, reaction in enumerate(merged["reactions"], 1):
        reaction["id"] = f"reaction_{idx}"

    logger.info(f"[ChemExtract] Chunked extraction complete: {len(merged['reactions'])} reactions, {len(merged['compounds'])} compounds")
    return merged if merged["reactions"] or merged["compounds"] else None


async def call_text_llm_chunked_async(
    text: str,
    provider: str,
    model: str,
    api_key: str,
    chunk_size: int = TEXT_CHUNK_SIZE
) -> Optional[Dict]:
    """Async call LLM for text analysis, splitting into chunks and merging results."""
    if not text or not text.strip():
        return None

    if len(text) <= chunk_size:
        return await call_text_llm_async(text, provider, model, api_key)

    # Split text into chunks
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    logger.info(f"[ChemExtract Async] Text ({len(text)} chars) split into {len(chunks)} chunks of ~{chunk_size} chars")

    merged: Dict[str, Any] = {
        "reactions": [],
        "compounds": [],
    }

    for i, chunk in enumerate(chunks):
        logger.info(f"[ChemExtract Async] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        chunk_result = await call_text_llm_async(chunk, provider, model, api_key)
        if chunk_result:
            for reaction in chunk_result.get("reactions", []):
                merged["reactions"].append(reaction)

            for compound in chunk_result.get("compounds", []):
                merged["compounds"].append(compound)

            for key in ("scaffold_smiles", "rgroup_table", "experimental_procedures",
                        "characterization_data"):
                if key in chunk_result and key not in merged:
                    merged[key] = chunk_result[key]

    # Renumber reactions sequentially
    for idx, reaction in enumerate(merged["reactions"], 1):
        reaction["id"] = f"reaction_{idx}"

    logger.info(f"[ChemExtract Async] Chunked extraction complete: {len(merged['reactions'])} reactions, {len(merged['compounds'])} compounds")
    return merged if merged["reactions"] or merged["compounds"] else None


def _call_deepseek_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call DeepSeek for text analysis via HTTP."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            else:
                logger.error(f"[ChemExtract] DeepSeek API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek text request error: {e}")
            return None

    return _retry_on_failure(_do_call)


async def _call_deepseek_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call DeepSeek for text analysis via HTTP (async)."""
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    else:
                        return None
        except Exception as e:
            logger.error(f"[ChemExtract] DeepSeek async text request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


def _call_openai_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call OpenAI for text analysis via HTTP."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            else:
                logger.error(f"[ChemExtract] OpenAI API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI text request error: {e}")
            return None

    return _retry_on_failure(_do_call)


async def _call_openai_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call OpenAI for text analysis via HTTP (async)."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    else:
                        return None
        except Exception as e:
            logger.error(f"[ChemExtract] OpenAI async text request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


def _call_gemini_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Gemini for text analysis via HTTP."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\nExtract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384
        }
    }

    def _do_call():
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['candidates'][0]['content']['parts'][0]['text']
                return _parse_json_response(content)
            else:
                logger.error(f"[ChemExtract] Gemini API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini text request error: {e}")
            return None

    return _retry_on_failure(_do_call)


async def _call_gemini_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Gemini for text analysis via HTTP (async)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\nExtract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 16384
        }
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['candidates'][0]['content']['parts'][0]['text']
                        return _parse_json_response(content)
                    else:
                        return None
        except Exception as e:
            logger.error(f"[ChemExtract] Gemini async text request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


def _call_anthropic_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Anthropic for text analysis via HTTP."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ]
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['content'][0]['text']
                return _parse_json_response(content)
            else:
                logger.error(f"[ChemExtract] Anthropic API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic text request error: {e}")
            return None

    return _retry_on_failure(_do_call)


async def _call_anthropic_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Anthropic for text analysis via HTTP (async)."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ]
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['content'][0]['text']
                        return _parse_json_response(content)
                    else:
                        return None
        except Exception as e:
            logger.error(f"[ChemExtract] Anthropic async text request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)


def _call_generic_text(text: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Generic OpenAI-compatible API call via HTTP."""
    url = f"https://api.{provider}.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    def _do_call():
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                return _parse_json_response(content)
            else:
                logger.error(f"[ChemExtract] Generic API error: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"[ChemExtract] Generic text request error: {e}")
            return None

    return _retry_on_failure(_do_call)


async def _call_generic_text_async(text: str, provider: str, model: str, api_key: str) -> Optional[Dict]:
    """Generic OpenAI-compatible API call via HTTP (async)."""
    url = f"https://api.{provider}.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_COMPREHENSIVE},
            {"role": "user", "content": f"Extract ALL chemical reactions and compounds from this text. List EVERY reaction as a separate entry:\n\n{text}"}
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.1
    }

    async def _do_call():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return _parse_json_response(content)
                    else:
                        return None
        except Exception as e:
            logger.error(f"[ChemExtract] Generic async text request error: {e}")
            return None

    return await _retry_on_failure_async(_do_call)



def _is_pseudo_smiles(s: str) -> bool:
    """Check if a string looks like pseudo-SMILES (R-group/Ar placeholder notation)
    rather than real SMILES.

    Examples of pseudo-SMILES to reject:
      - "ArCH2Cl"
      - "RC(O)Cl"
      - "ArCH2NH2"
      - "RC(O)CH2Ar"

    Real SMILES to accept:
      - "CC(=O)O"
      - "CC(=O)OCC"
      - "ClCCl"
      - "c1ccccc1"
    """
    if not s or len(s) < 2:
        return True
    # Must not contain spaces
    if re.search(r'\s', s):
        return True
    # Single-letter element placeholders outside of brackets: R, Ar, X, Y, Z
    # Check for unbracketed R or Ar used as substituent prefixes
    # e.g. "ArCH2Cl" — Ar is not a real element in this context
    unbracketed = re.findall(r'(?<!\w)(Ar|R|X|Y|Z)(?=[A-Z0-9\[\(]|$)', s)
    if unbracketed:
        return True
    # "R" followed by element-like pattern without bracket: RCl, RBr, RCH2, etc.
    if re.search(r'(?<!\w)R(?=[A-Z][a-z]|[A-Z]\d)', s) and not re.search(r'\[R', s):
        return True
    # "Ar" followed by organic subset chars without bracket
    if re.search(r'(?<!\w)Ar(?=[A-Z]|[a-z])', s) and not re.search(r'\[Ar', s):
        return True
    return False


def _extract_smiles(entity) -> Optional[str]:
    """
    Extract a SMILES string or fallback name from a reaction entity (reactant/product).

    Args:
        entity: Either a dict with 'smiles'/'name' keys, or a raw SMILES string.

    Returns:
        SMILES string if available and valid, compound name as fallback, or None.
    """
    if isinstance(entity, dict):
        smiles = entity.get("smiles")
        if smiles and smiles.strip() and smiles.upper() != "NONE":
            smiles = smiles.strip()
            if not _is_pseudo_smiles(smiles):
                return smiles
        name = entity.get("name")
        if name and name.strip():
            return name.strip()
        return None
    elif isinstance(entity, str):
        cleaned = entity.strip()
        if cleaned and cleaned.upper() != "NONE":
            return cleaned
        return None
    return None


def format_reaction_schemes(
    extraction_result: Dict[str, Any],
    include_metadata: bool = True,
    fallback_to_name: bool = False,
    skip_no_smiles: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert extracted reaction data to SMILES reaction scheme format.

    Takes the output from ChemExtractAI.extract_from_pdf() and formats each
    reaction as a compact SMILES-based reaction scheme string:

        SMILES.SMILES>>SMILES.SMILES

    where molecules on the left of '>>' are reactants and molecules on the right
    are products. Multiple molecules are joined with '.' (dot, meaning separate
    species, no bond between them).

    Handles both text-extracted reactions (from SYSTEM_PROMPT_COMPREHENSIVE)
    and vision-extracted reactions (from SYSTEM_PROMPT_REACTION_SCHEME / VISION).

    Args:
        extraction_result: The dict returned by ChemExtractAI.extract_from_pdf().
            Expected top-level key: "reactions" (list of reaction dicts).
        include_metadata: If True, each entry carries conditions, yield, catalyst,
            ligand, reaction type, and source alongside the scheme string.
        fallback_to_name: If True and a reactant/product lacks a SMILES field,
            its ``name`` is used as a placeholder in the scheme string.
        skip_no_smiles: If True, reactions where ANY reactant or product has no
            SMILES (and fallback_to_name is False) are silently dropped.

    Returns:
        A list of dicts, each containing at minimum:
        - ``scheme``  : str  - e.g. "CCF.CCF>>CCCC"
        - ``reactants_smiles`` : List[str] - individual reactant SMILES
        - ``products_smiles``  : List[str] - individual product SMILES

        When *include_metadata* is True, additional keys are populated:
        - ``reaction_id``, ``type``, ``conditions``, ``yield``, ``catalyst``,  
          ``ligand``, ``reagents``, ``source``, ``page``

    Example::

        >>> result = extractor.extract_from_pdf("paper.pdf")
        >>> schemes = format_reaction_schemes(result)
        >>> for s in schemes:
        ...     print(s["scheme"])
        CCF.CCF>>CCCC
        CC(=O)O.[OH-]>>CC(=O)[O-]
    """
    formatted: List[Dict[str, Any]] = []
    reactions = extraction_result.get("reactions", [])

    for reaction in reactions:
        reactants_smiles: List[str] = []
        products_smiles: List[str] = []
        has_any_smiles = False

        # ── Collect reactant SMILES ──────────────────────────────────────
        for r in reaction.get("reactants", []):
            smiles = _extract_smiles(r)
            if smiles:
                if isinstance(r, dict) and r.get("smiles"):
                    has_any_smiles = True
                reactants_smiles.append(smiles)

        # ── Collect product SMILES ───────────────────────────────────────
        for p in reaction.get("products", []):
            smiles = _extract_smiles(p)
            if smiles:
                if isinstance(p, dict) and p.get("smiles"):
                    has_any_smiles = True
                products_smiles.append(smiles)

        # ── Skip empty or invalid reactions ────────────────────────────────
        if not reactants_smiles or not products_smiles:
            continue

        if skip_no_smiles and not has_any_smiles and not fallback_to_name:
            continue

        # ── Build the scheme string ───────────────────────────────────────
        reactant_str = ".".join(reactants_smiles)
        product_str = ".".join(products_smiles)
        scheme = f"{reactant_str}>>{product_str}"

        entry: Dict[str, Any] = {
            "scheme": scheme,
            "reactants_smiles": reactants_smiles,
            "products_smiles": products_smiles,
        }

        if include_metadata:
            # Yield may be at top-level (vision) or inside outcomes (text)
            yield_val = reaction.get("yield")
            if yield_val is None:
                outcomes = reaction.get("outcomes")
                if isinstance(outcomes, dict):
                    yield_val = outcomes.get("yield")

            entry["reaction_id"] = reaction.get("id", "")
            entry["type"] = reaction.get("type", "unknown")
            entry["conditions"] = reaction.get("conditions", {})
            entry["yield"] = yield_val
            entry["catalyst"] = reaction.get("catalyst")
            entry["ligand"] = reaction.get("ligand")
            entry["source"] = reaction.get("source", "")
            entry["page"] = reaction.get("page")

            reagents = reaction.get("reagents")
            if reagents:
                entry["reagents"] = reagents

        formatted.append(entry)

    return formatted


def format_reaction_schemes_simple(extraction_result: Dict[str, Any]) -> List[str]:
    """
    Convenience wrapper that returns only the bare scheme strings.

    Example::

        >>> schemes = format_reaction_schemes_simple(result)
        >>> schemes
        ['CCF.CCF>>CCCC', 'CC(=O)O.[OH-]>>CC(=O)[O-]']
    """
    return [entry["scheme"] for entry in format_reaction_schemes(
        extraction_result, include_metadata=False
    )]


# ── R-Group Assembly Utilities ─────────────────────────────────────────────

_RGROUP_PLACEHOLDER_RE = re.compile(r'\[(\*|(\*:\d+)|(R\d+)|(R\w+))\]', re.IGNORECASE)


def assemble_rgroup_smiles(
    scaffold_smiles: str,
    rgroups: Dict[str, str]
) -> Optional[str]:
    """
    Replace placeholder atoms in a scaffold SMILES with actual substituent SMILES.

    The scaffold SMILES should contain placeholder atoms written as ``[*]``,
    ``[*:1]``, ``[*:2]``, etc. (standard RDKit / Daylight wildcard notation).
    Legacy placeholders ``[R1]``, ``[R2]`` are also accepted and mapped
    automatically (``[R1]`` -> ``[*:1]``, ``[R2]`` -> ``[*:2]``, etc.).

    Args:
        scaffold_smiles: SMILES string with wildcard / placeholder atoms.
            Example: ``C(C(=O)[*:1])CC[*:2]`` or ``C(C(=O)[R1])CC[R2]``.
        rgroups: Mapping of placeholder label to substituent SMILES fragment.
            The key must match the placeholder tag (e.g. ``"1"``, ``"2"``
            for numbered atoms, or ``"R1"``, ``"R2"`` for named ones).

    Returns:
        The assembled SMILES string, or *None* if substitution fails.

    Example::

        >>> assemble_rgroup_smiles(
        ...     "C(C(=O)[*:1])CC[*:2]",
        ...     {"1": "CCO", "2": "CCCCCC"}
        ... )
        'C(C(=O)CCO)CCCCCCC'
    """
    if not scaffold_smiles or not rgroups:
        return None

    smiles = scaffold_smiles.strip()

    # Normalise legacy [Rn] -> [*:n]
    def _normalize(m):
        tag = m.group(1)
        if tag.startswith('R') and len(tag) > 1 and tag[1:].isdigit():
            return f'[*:{tag[1:]}]'
        if tag.startswith('*:'):
            return m.group(0)  # already [*:n]
        if tag == '*':
            return '[*:1]'  # bare [*] -> assume first group
        return m.group(0)

    smiles = _RGROUP_PLACEHOLDER_RE.sub(_normalize, smiles)

    # Build replacement map: key = [*:n]  value = SMILES fragment
    replacements: Dict[str, str] = {}
    for key, frag in rgroups.items():
        if not frag or frag.strip().upper() in ('NONE', 'H', ''):
            # H placeholder -> just remove the wildcard
            tag = key.lstrip('R')
            replacements[f'[*:{tag}]'] = ''
            continue
        tag = key.lstrip('R')
        replacements[f'[*:{tag}]'] = frag.strip()

    if not replacements:
        return smiles

    # Apply replacements (sorted longest-key first to avoid partial matches)
    for pattern, frag in sorted(replacements.items(), key=lambda x: -len(x[0])):
        smiles = smiles.replace(pattern, frag)

    # Clean up any remaining bare [*] (unmatched) -> remove
    smiles = smiles.replace('[*]', '')

    return smiles if smiles.strip() else None


def assemble_rgroup_reactions(
    extraction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect R-group table data inside an extraction result and rebuild every
    reaction so that reactant / product SMILES reflect the **full, assembled**
    molecule (core scaffold + specific substituents) instead of the bare
    scaffold with placeholder atoms.

    The function looks for an ``rgroup_table`` key at the top level of
    *extraction_result* (populated by the enhanced vision / text prompts).
    If found, it iterates through all reactions, identifies the scaffold
    compound by label / name, assembles complete SMILES via
    :func:`assemble_rgroup_smiles`, and patches the reaction entries in-place.

    The original scaffold-level data is preserved in a new top-level key
    ``_original_reactions`` for auditing.

    Args:
        extraction_result: The dict returned by
        :meth:`ChemExtractAI.extract_from_pdf`.

    Returns:
        The same dict, mutated in-place, with each reaction now carrying
        assembled SMILES.  A new key ``assembled_from_rgroup`` is set to
        ``True`` on each patched reaction.

    Example::

        >>> result = extractor.extract_from_pdf("paper.pdf")
        >>> # result contains rgroup_table with scaffold + substituents
        >>> assemble_rgroup_reactions(result)
        >>> schemes = format_reaction_schemes(result)
        >>> schemes[0]["scheme"]
        'CCOC(=O)CCC(=O)CCCCCCC.CCCCCCCCCCCC>>...'
    """
    rgroup_table = extraction_result.get("rgroup_table")
    if not rgroup_table:
        return extraction_result

    scaffold_smiles = extraction_result.get("scaffold_smiles")
    if not scaffold_smiles:
        return extraction_result

    rgroups = rgroup_table.get("rgroups", {})
    partner_rgroups = rgroup_table.get("partner_rgroups", {})
    if not rgroups and not partner_rgroups:
        return extraction_result

    import copy
    extraction_result["_original_reactions"] = copy.deepcopy(
        extraction_result.get("reactions", [])
    )

    scaffold_label = rgroup_table.get("scaffold_label", "")

    for reaction in extraction_result.get("reactions", []):
        entry_id = str(reaction.get("entry", ""))

        # Also check reactant names for labels like "1a", "1b", etc.
        if not entry_id:
            for r in reaction.get("reactants", []):
                name = (r.get("name") if isinstance(r, dict) else str(r))
                if name and name.strip():
                    for rg_dict in rgroups.values():
                        if name.strip() in rg_dict:
                            entry_id = name.strip()
                            break
                if entry_id:
                    break

        if not entry_id:
            continue

        # Collect the R-group values for this entry from the table
        entry_rgroups: Dict[str, str] = {}
        for rname, variants in rgroups.items():
            if entry_id in variants:
                entry_rgroups[rname] = variants[entry_id]

        # Also look for partner (Reactant 2) R-groups by scanning reactant names
        for r in reaction.get("reactants", []):
            partner_name = (r.get("name") if isinstance(r, dict) else str(r))
            if partner_name and partner_name.strip():
                for prname, variants in partner_rgroups.items():
                    if partner_name.strip() in variants:
                        entry_rgroups[prname] = variants[partner_name.strip()]

        if not entry_rgroups:
            continue

        # Assemble the scaffold SMILES with this entry's substituents
        assembled = assemble_rgroup_smiles(scaffold_smiles, entry_rgroups)
        if not assembled:
            continue

        # Patch reactant SMILES
        patched = False
        for r in reaction.get("reactants", []):
            if isinstance(r, dict):
                old_smiles = r.get("smiles", "")
                old_name = r.get("name", "")
                is_scaffold_entry = False
                for rg_dict in rgroups.values():
                    if old_name.strip() in rg_dict if old_name else False:
                        is_scaffold_entry = True
                        break
                if (not old_smiles
                        or old_smiles == scaffold_smiles
                        or (scaffold_label and old_name == scaffold_label)
                        or is_scaffold_entry):
                    r["smiles"] = assembled
                    r["assembled"] = True
                    patched = True

        if not reaction.get("reactants"):
            reaction["reactants"] = [{
                "name": f"{scaffold_label or 'scaffold'} ({entry_id})",
                "smiles": assembled,
                "assembled": True,
            }]
            patched = True

        # Try to assemble product SMILES if a product scaffold is present
        product_scaffold = extraction_result.get("product_scaffold_smiles")
        if product_scaffold:
            assembled_product = assemble_rgroup_smiles(product_scaffold, entry_rgroups)
            if assembled_product:
                for p in reaction.get("products", []):
                    if isinstance(p, dict) and (not p.get("smiles") or p.get("smiles") == product_scaffold):
                        p["smiles"] = assembled_product
                        p["assembled"] = True

        if patched:
            reaction["assembled_from_rgroup"] = True

    extraction_result["rgroup_assembled"] = True
    return extraction_result


class ChemExtractAI:
    """
    ChemExtract AI - Advanced Chemistry Data Extraction System
    
    Provides comprehensive extraction of chemical information from PDFs:
    - Text analysis for chemical entities and reactions
    - Image analysis for reaction schemes and structures
    - Table extraction for optimization data
    - SMILES and molecular formula recognition
    """

    def __init__(self, llm_provider: str = 'deepseek', api_key: str = None, model: str = None):
        """
        Initialize the extractor.
        
        Args:
            llm_provider: LLM provider ('deepseek', 'openai', 'gemini', 'anthropic')
            api_key: API key for the LLM provider
            model: Specific model to use (optional, uses default if not specified)
        """
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model

        self.vision_providers = ['deepseek', 'openai', 'gemini', 'anthropic']

        self.default_models = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4o',
            'gemini': 'gemini-2.0-flash',
            'anthropic': 'claude-3-5-sonnet-20241022'
        }

        if not self.model:
            self.model = self.default_models.get(llm_provider)

    def extract_from_pdf(
        self,
        pdf_path: str,
        extract_images: bool = True,
        extract_text: bool = True,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive extraction from a PDF (sync).
        
        Strategy:
        1. Extract text and analyze with LLM (primary source)
        2. Extract images and analyze with Vision LLM (supplementary)
        3. Merge results intelligently, avoiding duplicates
        """
        result = {
            "reactions": [],
            "compounds": [],
            "figures": [],
            "tables": [],
            "text_content": "",
            "metadata": {
                "pages_processed": 0,
                "extraction_method": "chemextract_ai",
                "provider": self.llm_provider,
                "model": self.model,
                "text_extracted": False,
                "images_extracted": False
            }
        }

        text_data = None
        if extract_text:
            try:
                text, text_meta = extract_text_from_pdf(pdf_path)
                result["text_content"] = text
                result["metadata"]["pages"] = text_meta.get("pages", 0)
                result["metadata"]["text_method"] = text_meta.get("method")

                if text.strip():
                    logger.info(f"[ChemExtract] Analyzing text ({len(text)} chars)...")
                    text_data = call_text_llm_chunked(text, self.llm_provider, self.model, self.api_key)
                    if text_data:
                        result["metadata"]["text_extracted"] = True
                        result["reactions"] = text_data.get("reactions", [])
                        result["compounds"] = text_data.get("compounds", [])
                        if "experimental_procedures" in text_data:
                            result["experimental_procedures"] = text_data["experimental_procedures"]
                        if "characterization_data" in text_data:
                            result["characterization_data"] = text_data["characterization_data"]
            except Exception as e:
                logger.error(f"[ChemExtract] Text extraction failed: {e}")

        if extract_images and self.llm_provider in self.vision_providers:
            try:
                page_images = pdf_to_images(pdf_path, dpi=150, max_pages=max_pages)
                result["metadata"]["pages_processed"] = len(page_images)

                logger.info(f"[ChemExtract] Analyzing {len(page_images)} pages with vision...")

                for page_num, base64_img in page_images:
                    vision_data = self._extract_from_image(
                        base64_img,
                        extraction_type="comprehensive",
                        page_number=page_num
                    )

                    if vision_data:
                        result["metadata"]["images_extracted"] = True

                        self._merge_vision_results(result, vision_data, page_num)

            except Exception as e:
                logger.error(f"[ChemExtract] Image extraction failed: {e}")

        result["compounds"] = self._deduplicate_compounds(result.get("compounds", []))
        result["reactions"] = self._deduplicate_reactions(result.get("reactions", []))

        return result

    def _merge_vision_results(self, result: Dict, vision_data: Dict, page_num: int):
        """
        Intelligently merge vision extraction results with existing text results.
        Vision data supplements text data but doesn't override it.
        """
        reaction_schemes = vision_data.get("reaction_schemes", [])

        # If vision returned flat format (reactants/products lists instead of reaction_schemes), convert
        if not reaction_schemes and (vision_data.get("reactants") or vision_data.get("products")):
            flat_reactants = vision_data.get("reactants", [])
            flat_products = vision_data.get("products", [])
            if isinstance(flat_reactants, list) and isinstance(flat_products, list):
                # Create a single reaction from flat vision data
                reaction_schemes = [{
                    "entry": 1,
                    "reactants": [{"name": r, "smiles": None} if isinstance(r, str) else r for r in flat_reactants],
                    "products": [{"name": p, "smiles": None} if isinstance(p, str) else p for p in flat_products],
                    "reagents": vision_data.get("reagents", []),
                    "conditions": vision_data.get("conditions", {}),
                    "yield": None,
                    "catalyst": vision_data.get("catalysts", [None])[0] if vision_data.get("catalysts") else None,
                    "ligand": vision_data.get("ligands", [None])[0] if vision_data.get("ligands") else None,
                }]

        for scheme in reaction_schemes:
            reaction = {
                "id": f"vision_page{page_num}_{len(result['reactions'])+1}",
                "source": "vision",
                "page": page_num,
                "type": scheme.get("reactionType", "unknown"),
                "reactants": scheme.get("reactants", []),
                "products": scheme.get("products", []),
                "reagents": scheme.get("reagents", []),
                "catalyst": scheme.get("catalyst"),
                "ligand": scheme.get("ligand"),
                "conditions": scheme.get("conditions", {}),
                "yield": scheme.get("yield"),
                "notes": scheme.get("notes", "")
            }
            result["reactions"].append(reaction)

        if vision_data.get("table_data"):
            result["tables"].append({
                "page": page_num,
                "source": "vision",
                "data": vision_data.get("table_data", []),
                "columns": vision_data.get("table_columns", [])
            })

        existing_names = {c.get("name", "").lower() for c in result.get("compounds", []) if c.get("name")}

        for scheme in reaction_schemes:
            for reactant in scheme.get("reactants", []):
                name = reactant.get("name", "") if isinstance(reactant, dict) else str(reactant)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({
                        "name": name,
                        "smiles": reactant.get("smiles") if isinstance(reactant, dict) else None,
                        "role": "reactant",
                        "source": "vision"
                    })
                    existing_names.add(name.lower())

            for product in scheme.get("products", []):
                name = product.get("name", "") if isinstance(product, dict) else str(product)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({
                        "name": name,
                        "smiles": product.get("smiles") if isinstance(product, dict) else None,
                        "role": "product",
                        "source": "vision"
                    })
                    existing_names.add(name.lower())

        result["figures"].append({
            "page": page_num,
            "type": "vision_analysis",
            "description": vision_data.get("description", ""),
            "notes": vision_data.get("notes", "")
        })

    def _deduplicate_compounds(self, compounds: List) -> List:
        """Remove duplicate compounds based on name or SMILES."""
        seen = set()
        unique = []
        for compound in compounds:
            key = (compound.get("name", "") or compound.get("smiles", "") or "").lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(compound)
        return unique

    def _deduplicate_reactions(self, reactions: List) -> List:
        """Remove exact duplicate reactions, but keep distinct reactions even if they share reactants."""
        seen = set()
        unique = []
        for reaction in reactions:
            entry = reaction.get("entry", "")
            source = reaction.get("source", "")
            reactants_str = str(sorted([str(r) for r in reaction.get("reactants", [])]))
            products_str = str(sorted([str(p) for p in reaction.get("products", [])]))
            
            # Use entry number as primary key if available (from tables)
            if entry:
                key = f"entry_{entry}_{source}"
            else:
                key = f"{reactants_str}|{products_str}|{source}"

            if key not in seen:
                seen.add(key)
                unique.append(reaction)
        return unique

    def format_reaction_schemes(
        self,
        extraction_result: Dict[str, Any],
        include_metadata: bool = True,
        fallback_to_name: bool = True,
        skip_no_smiles: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Instance wrapper around the standalone format_reaction_schemes().

        Converts previously extracted reaction data (stored in *extraction_result*)
        to compact SMILES reaction scheme strings of the form
        ``SMILES.SMILES>>SMILES.SMILES``.

        See :func:`format_reaction_schemes` for full parameter documentation.
        """
        return format_reaction_schemes(
            extraction_result,
            include_metadata=include_metadata,
            fallback_to_name=fallback_to_name,
            skip_no_smiles=skip_no_smiles,
        )

    async def extract_from_pdf_async(
        self,
        pdf_path: str,
        extract_images: bool = True,
        extract_text: bool = True,
        max_pages: int = 50
    ) -> Dict[str, Any]:
        """
        Perform comprehensive extraction from a PDF (async).
        Processes pages concurrently for faster extraction.
        
        Strategy:
        1. Extract text and analyze with LLM (primary source)
        2. Extract images and analyze with Vision LLM (supplementary)
        3. Merge results intelligently, avoiding duplicates
        """
        result = {
            "reactions": [],
            "compounds": [],
            "figures": [],
            "tables": [],
            "text_content": "",
            "metadata": {
                "pages_processed": 0,
                "extraction_method": "chemextract_ai",
                "provider": self.llm_provider,
                "model": self.model,
                "async": True,
                "text_extracted": False,
                "images_extracted": False
            }
        }

        if extract_text:
            try:
                text, text_meta = extract_text_from_pdf(pdf_path)
                result["text_content"] = text
                result["metadata"]["pages"] = text_meta.get("pages", 0)

                if text.strip():
                    logger.info(f"[ChemExtract Async] Analyzing text ({len(text)} chars)...")
                    text_data = await call_text_llm_chunked_async(text, self.llm_provider, self.model, self.api_key)
                    if text_data:
                        result["metadata"]["text_extracted"] = True
                        result["reactions"] = text_data.get("reactions", [])
                        result["compounds"] = text_data.get("compounds", [])
                        if "experimental_procedures" in text_data:
                            result["experimental_procedures"] = text_data["experimental_procedures"]
                        if "characterization_data" in text_data:
                            result["characterization_data"] = text_data["characterization_data"]
            except Exception as e:
                logger.error(f"[ChemExtract] Async text extraction failed: {e}")

        if extract_images and self.llm_provider in self.vision_providers:
            try:
                page_images = pdf_to_images(pdf_path, dpi=150, max_pages=max_pages)
                result["metadata"]["pages_processed"] = len(page_images)

                logger.info(f"[ChemExtract Async] Analyzing {len(page_images)} pages with vision...")

                async def process_page(page_num: int, base64_img: str):
                    vision_data = await self._extract_from_image_async(
                        base64_img,
                        extraction_type="comprehensive",
                        page_number=page_num
                    )
                    return (page_num, vision_data)

                tasks = [process_page(page_num, base64_img) for page_num, base64_img in page_images]
                all_results = await asyncio.gather(*tasks)

                for page_num, vision_data in all_results:
                    if vision_data:
                        result["metadata"]["images_extracted"] = True
                        self._merge_vision_results(result, vision_data, page_num)

            except Exception as e:
                logger.error(f"[ChemExtract] Async image extraction failed: {e}")

        result["compounds"] = self._deduplicate_compounds(result.get("compounds", []))
        result["reactions"] = self._deduplicate_reactions(result.get("reactions", []))

        return result

    def _extract_from_image(
        self,
        base64_image: str,
        extraction_type: str = "reactions",
        page_number: int = 1
    ) -> Optional[Dict]:
        """Extract data from an image using vision LLM (sync)."""

        if extraction_type == "comprehensive":
            system_prompt = SYSTEM_PROMPT_VISION
            user_message = f"""Analyze page {page_number} of this scientific document and extract ALL chemical information:
1. Reaction schemes and molecular structures
2. Tables with yield/condition data
3. Compound names and structures

Return a comprehensive JSON with all extracted data."""
        elif extraction_type == "reactions":
            system_prompt = SYSTEM_PROMPT_REACTION_SCHEME
            user_message = f"Analyze page {page_number} of this scientific document. Extract all reaction schemes, molecular structures, and chemical transformations visible."
        elif extraction_type == "tables":
            system_prompt = SYSTEM_PROMPT_TABLE_EXTRACTION
            user_message = f"Analyze page {page_number}. Extract any chemistry-related tables (optimization tables, substrate scope, condition screening)."
        else:
            system_prompt = SYSTEM_PROMPT_VISION
            user_message = f"Extract all chemical information from page {page_number}."

        return call_vision_llm(
            base64_image,
            self.llm_provider,
            self.model,
            self.api_key,
            system_prompt,
            user_message
        )

    async def _extract_from_image_async(
        self,
        base64_image: str,
        extraction_type: str = "reactions",
        page_number: int = 1
    ) -> Optional[Dict]:
        """Extract data from an image using vision LLM (async)."""

        if extraction_type == "comprehensive":
            system_prompt = SYSTEM_PROMPT_VISION
            user_message = f"""Analyze page {page_number} of this scientific document and extract ALL chemical information:
1. Reaction schemes and molecular structures
2. Tables with yield/condition data
3. Compound names and structures

Return a comprehensive JSON with all extracted data."""
        elif extraction_type == "reactions":
            system_prompt = SYSTEM_PROMPT_REACTION_SCHEME
            user_message = f"Analyze page {page_number} of this scientific document. Extract all reaction schemes, molecular structures, and chemical transformations visible."
        elif extraction_type == "tables":
            system_prompt = SYSTEM_PROMPT_TABLE_EXTRACTION
            user_message = f"Analyze page {page_number}. Extract any chemistry-related tables (optimization tables, substrate scope, condition screening)."
        else:
            system_prompt = SYSTEM_PROMPT_VISION
            user_message = f"Extract all chemical information from page {page_number}."

        return await call_vision_llm_async(
            base64_image,
            self.llm_provider,
            self.model,
            self.api_key,
            system_prompt,
            user_message
        )



def extract_chemical_data_from_pdf(
    pdf_path: str,
    llm_provider: str = 'deepseek',
    api_key: str = None,
    model: str = None,
    max_pages: int = 50,
    extract_images: bool = True,
    extract_text: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to extract chemical data from a PDF (sync).
    """
    extractor = ChemExtractAI(
        llm_provider=llm_provider,
        api_key=api_key,
        model=model
    )

    return extractor.extract_from_pdf(
        pdf_path,
        extract_images=extract_images,
        extract_text=extract_text,
        max_pages=max_pages
    )


async def extract_chemical_data_from_pdf_async(
    pdf_path: str,
    llm_provider: str = 'deepseek',
    api_key: str = None,
    model: str = None,
    max_pages: int = 50,
    extract_images: bool = True,
    extract_text: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to extract chemical data from a PDF (async).
    """
    extractor = ChemExtractAI(
        llm_provider=llm_provider,
        api_key=api_key,
        model=model
    )

    return await extractor.extract_from_pdf_async(
        pdf_path,
        extract_images=extract_images,
        extract_text=extract_text,
        max_pages=max_pages
    )



if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("ChemExtract AI - Chemical Data Extraction Tool")
        print("Usage: python chemextract_extractor.py <pdf_path> [provider] [api_key]")
        print("\nProviders: deepseek, openai, gemini, anthropic")
        sys.exit(1)

    pdf_path = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else 'deepseek'
    api_key = sys.argv[3] if len(sys.argv) > 3 else os.environ.get(f'{provider.upper()}_API_KEY')

    if not api_key:
        print(f"Error: No API key provided. Set {provider.upper()}_API_KEY environment variable or pass as argument.")
        sys.exit(1)

    print(f"Extracting from: {pdf_path}")
    print(f"Provider: {provider}")

    result = extract_chemical_data_from_pdf(
        pdf_path,
        llm_provider=provider,
        api_key=api_key
    )

    print("\n=== Extraction Results ===")
    print(f"Reactions found: {len(result.get('reactions', []))}")
    print(f"Compounds found: {len(result.get('compounds', []))}")
    print(f"Tables found: {len(result.get('tables', []))}")
    print(f"\nFull results saved to extraction_result.json")

    with open("extraction_result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
