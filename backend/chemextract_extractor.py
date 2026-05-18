import os
import re
import json
import logging
import base64
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

Return a JSON object:
{
  "reaction_schemes": [
    {
      "reactants": [{"name": "...", "smiles": "...", "structure_description": "..."}],
      "products": [{"name": "...", "smiles": "...", "structure_description": "..."}],
      "reagents": ["list of reagents"],
      "conditions": {
        "temperature": "e.g., 80°C",
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
  "description": "overall description of what's shown",
  "notes": "any additional observations"
}"""

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
  "image_description": "brief description of what is shown in the image",
  "additional_observations": "any other relevant chemical information seen"
}

Be thorough and extract ALL visible chemical information. If you see reaction schemes, describe the complete transformation. If you see tables, extract all relevant data."""


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

Return structured JSON:
{
  "reactions": [
    {
      "id": "reaction_1",
      "type": "reaction type",
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
}"""



def pdf_to_images(file_path: str, dpi: int = 150, max_pages: int = 10) -> List[Tuple[int, str]]:
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
    """Parse JSON from LLM response."""
    if not content:
        return None

    try:
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"[ChemExtract] JSON parse error: {e}")

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
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return _parse_json_response(content)
        logger.error(f"[ChemExtract] DeepSeek API error: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] DeepSeek vision request error: {e}")
        return None


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
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return _parse_json_response(content)
        logger.error(f"[ChemExtract] OpenAI API error: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] OpenAI vision request error: {e}")
        return None


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
            "maxOutputTokens": 4000
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            return _parse_json_response(content)
        logger.error(f"[ChemExtract] Gemini API error: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] Gemini vision request error: {e}")
        return None


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
        "max_tokens": 4000,
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

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['content'][0]['text']
            return _parse_json_response(content)
        logger.error(f"[ChemExtract] Anthropic API error: {response.status_code}")
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] Anthropic vision request error: {e}")
        return None



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
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return _parse_json_response(content)
                logger.error(f"[ChemExtract] DeepSeek API error: {response.status}")
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] DeepSeek async vision request error: {e}")
        return None


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
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return _parse_json_response(content)
                logger.error(f"[ChemExtract] OpenAI API error: {response.status}")
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] OpenAI async vision request error: {e}")
        return None


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
            "maxOutputTokens": 4000
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['candidates'][0]['content']['parts'][0]['text']
                    return _parse_json_response(content)
                logger.error(f"[ChemExtract] Gemini API error: {response.status}")
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] Gemini async vision request error: {e}")
        return None


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
        "max_tokens": 4000,
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

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['content'][0]['text']
                    return _parse_json_response(content)
                logger.error(f"[ChemExtract] Anthropic API error: {response.status}")
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] Anthropic async vision request error: {e}")
        return None



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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return _parse_json_response(content)
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] DeepSeek text request error: {e}")
        return None


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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return _parse_json_response(content)
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] DeepSeek async text request error: {e}")
        return None


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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return _parse_json_response(content)
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] OpenAI text request error: {e}")
        return None


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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return _parse_json_response(content)
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] OpenAI async text request error: {e}")
        return None


def _call_gemini_text(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Gemini for text analysis via HTTP."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\nExtract all chemical information from this text:\n\n{text[:8000]}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4000
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['candidates'][0]['content']['parts'][0]['text']
            return _parse_json_response(content)
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] Gemini text request error: {e}")
        return None


async def _call_gemini_text_async(text: str, model: str, api_key: str) -> Optional[Dict]:
    """Call Gemini for text analysis via HTTP (async)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": f"{SYSTEM_PROMPT_COMPREHENSIVE}\n\nExtract all chemical information from this text:\n\n{text[:8000]}"}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4000
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['candidates'][0]['content']['parts'][0]['text']
                    return _parse_json_response(content)
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] Gemini async text request error: {e}")
        return None


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
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['content'][0]['text']
            return _parse_json_response(content)
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] Anthropic text request error: {e}")
        return None


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
        "max_tokens": 4000,
        "system": SYSTEM_PROMPT_COMPREHENSIVE,
        "messages": [
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['content'][0]['text']
                    return _parse_json_response(content)
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] Anthropic async text request error: {e}")
        return None


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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            return _parse_json_response(content)
        return None
    except Exception as e:
        logger.error(f"[ChemExtract] Generic text request error: {e}")
        return None


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
            {"role": "user", "content": f"Extract all chemical information from this text:\n\n{text[:8000]}"}
        ],
        "max_tokens": 4000,
        "temperature": 0.1
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    return _parse_json_response(content)
                return None
    except Exception as e:
        logger.error(f"[ChemExtract] Generic async text request error: {e}")
        return None



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
        max_pages: int = 10
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
                    text_data = call_text_llm(text, self.llm_provider, self.model, self.api_key)
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
        """Remove duplicate reactions based on reactants/products similarity."""
        seen = set()
        unique = []
        for reaction in reactions:
            reactants_str = str(sorted([str(r) for r in reaction.get("reactants", [])]))
            products_str = str(sorted([str(p) for p in reaction.get("products", [])]))
            fingerprint = f"{reactants_str}|{products_str}"

            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(reaction)
        return unique

    async def extract_from_pdf_async(
        self,
        pdf_path: str,
        extract_images: bool = True,
        extract_text: bool = True,
        max_pages: int = 10
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
                    text_data = await call_text_llm_async(text, self.llm_provider, self.model, self.api_key)
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
    max_pages: int = 10,
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
    max_pages: int = 10,
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
