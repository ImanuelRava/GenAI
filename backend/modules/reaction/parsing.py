"""
ReactionLens — JSON parsing, paragraph segmentation, and PDF text extraction.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports
# ---------------------------------------------------------------------------

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.debug("[ReactionLens] PyMuPDF not available; PDF text extraction disabled")

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

# ---------------------------------------------------------------------------
# Try to import chemextract's parser
# ---------------------------------------------------------------------------

try:
    from chemextract.json_utils import _parse_json_response as _ce_parse_json_response
    _USE_CHEMEXTRACT_PARSER = True
except ImportError:
    _USE_CHEMEXTRACT_PARSER = False

# ---------------------------------------------------------------------------
# JSON Parsing
# ---------------------------------------------------------------------------

def _local_parse_json_response(content: str) -> Optional[Union[Dict, List]]:
    """Parse JSON from LLM response text.

    Handles truncated, partial, markdown-fenced, and multiple JSON outputs.
    """
    if not content:
        return None

    content = content.strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Find the largest valid JSON object
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
                    candidate = json.loads(content[start:i + 1])
                    if best_json is None or len(str(candidate)) > len(str(best_json)):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    # Also try arrays
    depth = 0
    start = -1
    for i, char in enumerate(content):
        if char == '[':
            if depth == 0:
                start = i
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    candidate = json.loads(content[start:i + 1])
                    if best_json is None or len(str(candidate)) > len(str(best_json)):
                        best_json = candidate
                except (json.JSONDecodeError, ValueError):
                    pass

    if best_json is not None:
        return best_json

    # Try fixing unbalanced braces
    try:
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        if open_braces > 0 or open_brackets > 0:
            fixed = content + '}' * max(0, open_braces) + ']' * max(0, open_brackets)
            return json.loads(fixed)
    except (json.JSONDecodeError, ValueError):
        pass

    logger.warning(f"[ReactionLens] JSON parse failed for content of length {len(content)}")
    return None


# Use chemextract's parser when available, otherwise local fallback
if _USE_CHEMEXTRACT_PARSER:
    _parse_json_response = _ce_parse_json_response
else:
    _parse_json_response = _local_parse_json_response


# ---------------------------------------------------------------------------
# Paragraph Segmentation
# ---------------------------------------------------------------------------

def segment_into_paragraphs(text: str, min_length: int = 80) -> List[Dict[str, Any]]:
    """Split extracted text into paragraphs suitable for reaction screening."""
    from .prompts import RL_MIN_PARAGRAPH_LENGTH
    if min_length == 80:
        min_length = RL_MIN_PARAGRAPH_LENGTH

    if not text or not text.strip():
        return []

    # First try splitting by double newlines
    raw_paragraphs = re.split(r'\n\s*\n', text)

    # If we only get very few paragraphs, also split by single newlines
    if len(raw_paragraphs) < 5:
        raw_paragraphs = re.split(r'\n', text)

    paragraphs = []
    char_offset = 0

    for idx, para in enumerate(raw_paragraphs):
        cleaned = para.strip()
        if len(cleaned) < min_length:
            char_offset += len(para) + 2
            continue

        actual_start = text.find(cleaned, char_offset)
        if actual_start == -1:
            actual_start = char_offset

        paragraphs.append({
            "index": len(paragraphs),
            "text": cleaned,
            "char_start": actual_start,
            "char_end": actual_start + len(cleaned),
        })
        char_offset = actual_start + len(cleaned)

    logger.info(
        f"[ReactionLens] Segmented text into {len(paragraphs)} paragraphs "
        f"(from {len(raw_paragraphs)} raw segments, min_length={min_length})"
    )
    return paragraphs


# ---------------------------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> Tuple[str, Dict]:
    """Extract text content from a PDF file.

    Tries multiple backends (chemextract, pypdf, pdfplumber, PyMuPDF) in order.

    Returns:
        Tuple of (extracted_text, metadata_dict).
    """
    text = ""
    metadata = {"pages": 0, "method": None}

    # Try chemextract first
    try:
        from chemextract.pdf_processor import extract_text_from_pdf as _ce_extract
        return _ce_extract(file_path)
    except Exception as e:
        logger.debug(f"[ReactionLens] ChemExtract text extraction not available: {e}")

    # Fallback implementations
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
            logger.warning(f"[ReactionLens] pypdf failed: {e}")

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
            logger.error(f"[ReactionLens] pdfplumber failed: {e}")

    if HAS_PYMUPDF:
        try:
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)
            metadata["method"] = "pymupdf"
            for page in doc:
                text += page.get_text() + "\n\n"
            return text, metadata
        except Exception as e:
            logger.error(f"[ReactionLens] PyMuPDF text extraction failed: {e}")

    raise ImportError("No PDF text extraction library available. Install pypdf, pdfplumber, or PyMuPDF.")
