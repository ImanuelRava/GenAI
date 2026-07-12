"""
ChemExtract PDF processing - text and image extraction from PDF files.

Provides a two-tier vision pipeline:
  Tier 1: extract_figures_from_pdf() — extracts embedded raster images (photos, charts,
          molecular structure images) with size/quality filtering.
  Tier 2: render_scheme_pages()     — renders full pages that likely contain vector-drawn
          reaction schemes (detected via caption keywords and drawing-annotation density).

Text extraction uses a cascading fallback: pypdf -> pdfplumber -> PyMuPDF.
"""

import base64
import io
import logging
import re
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("[ChemExtract] PyMuPDF not available")

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

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
# Configuration constants for figure extraction
# ---------------------------------------------------------------------------
MIN_FIGURE_WIDTH = 100          # pixels — skip tiny icons / decorative elements
MIN_FIGURE_HEIGHT = 100
MIN_FIGURE_AREA = 15000         # width * height — skip very small images
MAX_EMBEDDED_IMAGES = 200       # safety cap to avoid runaway extraction
SCHEME_PAGE_KEYWORDS = re.compile(
    r'\b(scheme|schem|fig\.|figure|chart|diagram|graph|mechanism|pathway)\b',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def pdf_to_images(file_path: str, dpi: int = 150, max_pages: int = 50) -> List[Tuple[int, str]]:
    """Convert PDF pages to base64-encoded images using PyMuPDF.

    .. deprecated::
        Use :func:`extract_figures_from_pdf` + :func:`render_scheme_pages` instead.
        This renders every page which is wasteful for chemistry PDFs.
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
    except (OSError, RuntimeError, ValueError) as e:
        # PyMuPDF raises OSError on corrupt/encrypted PDFs, RuntimeError on
        # rendering failures, ValueError on invalid page indices.
        logger.error(f"[ChemExtract] PyMuPDF failed: {e}")
        raise RuntimeError(f"Failed to convert PDF to images: {e}") from e


def extract_images_from_pdf(file_path: str, dpi: int = 300) -> List[Tuple[int, bytes]]:
    """Extract embedded images from a PDF (raw bytes, no filtering).

    .. deprecated::
        Use :func:`extract_figures_from_pdf` instead, which adds size filtering
        and returns base64-encoded images with metadata.
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    images = []
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image:
                        images.append((page_num * 1000 + img_idx, base_image["image"]))
                except (KeyError, ValueError, OSError):
                    # Single-image extraction failures (corrupt xref, missing
                    # 'image' key, IO error) should not abort the whole loop.
                    pass
        logger.info(f"[ChemExtract] Extracted {len(images)} embedded images from PDF")
        return images
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"[ChemExtract] Image extraction failed: {e}")
        raise RuntimeError(f"Failed to extract images from PDF: {e}") from e


# ---------------------------------------------------------------------------
# Tier 1 — Embedded figure extraction with quality filtering
# ---------------------------------------------------------------------------

def _image_dimensions(raw_bytes: bytes) -> Optional[Tuple[int, int]]:
    """Return (width, height) of an image from raw bytes, or None on failure."""
    if not HAS_PILLOW:
        return None
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return img.size
    except (OSError, ValueError) as e:
        # PIL raises UnidentifiedImageError (subclass of OSError) on corrupt
        # images, ValueError on truncated data.
        logger.debug(f"[ChemExtract] Could not read image dimensions: {e}")
        return None


def _is_chemical_figure(raw_bytes: bytes, width: int, height: int,
                        mime_type: str = "") -> bool:
    """Heuristic check: is this embedded image likely a chemical figure?

    Filters OUT:
      - Tiny icons / decorative elements (< MIN dimensions)
      - Very large full-page raster scans (likely scanned text pages)
      - Common non-chemical image patterns (tiny 1x1 tracking pixels)
    """
    area = width * height
    if width < MIN_FIGURE_WIDTH or height < MIN_FIGURE_HEIGHT:
        return False
    if area < MIN_FIGURE_AREA:
        return False
    # Skip 1x1 tracking pixels (common in PDFs)
    if width <= 2 or height <= 2:
        return False
    return True


def extract_figures_from_pdf(
    file_path: str,
    min_width: int = MIN_FIGURE_WIDTH,
    min_height: int = MIN_FIGURE_HEIGHT,
    max_images: int = MAX_EMBEDDED_IMAGES,
) -> List[Dict]:
    """Extract embedded raster figures from a PDF with quality filtering.

    Returns a list of dicts, each containing:
      - page (int): 1-based page number
      - index (int): image index on that page
      - width (int): pixel width
      - height (int): pixel height
      - mime_type (str): e.g. "image/png", "image/jpeg"
      - base64 (str): base64-encoded image data
      - size_bytes (int): original byte size

    Embedded images in chemistry papers typically include:
      - Molecular structure diagrams (as PNG/JPEG inlays)
      - Reaction scheme screenshots (from ChemDraw, etc.)
      - Spectral data charts (NMR, IR, MS)
      - Graphs (yield vs. conditions, selectivity plots)
      - Microscope / crystal structure images
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    figures = []
    seen_xrefs = set()  # deduplicate images referenced on multiple pages
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = page.get_images(full=True)
            for img_idx, img_info in enumerate(page_images):
                if len(figures) >= max_images:
                    break
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image:
                        continue
                    raw = base_image["image"]
                    mime = base_image.get("mime_type", "image/png")
                    dims = _image_dimensions(raw)
                    if dims:
                        w, h = dims
                    else:
                        # Fallback: use width/height from PDF image dict if available
                        w = img_info[2] if len(img_info) > 2 else 0
                        h = img_info[3] if len(img_info) > 3 else 0
                    if not _is_chemical_figure(raw, w, h, mime):
                        continue
                    b64 = base64.b64encode(raw).decode('utf-8')
                    figures.append({
                        "page": page_num + 1,
                        "index": img_idx,
                        "width": w,
                        "height": h,
                        "mime_type": mime,
                        "base64": b64,
                        "size_bytes": len(raw),
                    })
                except (KeyError, ValueError, OSError, RuntimeError) as e:
                    # Single-figure extraction failures (corrupt xref, missing
                    # 'image' key, PIL parse error, IO error) should not abort
                    # the whole loop — skip the figure and continue.
                    logger.debug(f"[ChemExtract] Skipped image xref={xref} on page {page_num+1}: {e}")
            if len(figures) >= max_images:
                break
        logger.info(
            f"[ChemExtract] Extracted {len(figures)} embedded figures "
            f"(filtered from {len(seen_xrefs)} total images across {len(doc)} pages)"
        )
        return figures
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"[ChemExtract] Figure extraction failed: {e}")
        raise RuntimeError(f"Failed to extract figures from PDF: {e}") from e


# ---------------------------------------------------------------------------
# Tier 2 — Scheme page detection & rendering
# ---------------------------------------------------------------------------

def detect_scheme_pages(file_path: str, max_pages: int = 50) -> List[int]:
    """Detect pages likely containing reaction schemes or figures.

    Uses two heuristics:
      1. Caption keywords: text containing "Scheme", "Fig.", "Figure", etc.
      2. Drawing-annotation density: pages with many vector drawing operations
         relative to text blocks (suggests reaction schemes drawn with vector paths).

    Returns a sorted list of 1-based page numbers.
    """
    if not HAS_PYMUPDF:
        return []

    scheme_pages = set()
    try:
        doc = fitz.open(file_path)
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            page_text = page.get_text()
            # Heuristic 1: caption keyword match
            if SCHEME_PAGE_KEYWORDS.search(page_text):
                scheme_pages.add(page_num + 1)
                continue
            # Heuristic 2: high drawing-annotation density
            # Pages with many drawing paths but little text often contain
            # reaction schemes drawn as vector graphics
            drawings = page.get_drawings()
            text_blocks = page.get_text("blocks")
            n_drawings = len(drawings)
            n_text_blocks = len([b for b in text_blocks if b[6] == 0])  # type 0 = text block
            # If many drawings and relatively few text blocks, likely a scheme page
            if n_drawings > 10 and n_text_blocks < 5:
                scheme_pages.add(page_num + 1)
    except (OSError, RuntimeError) as e:
        # PyMuPDF raises OSError on corrupt PDFs, RuntimeError on rendering
        # failures. We log and return whatever pages were detected so far.
        logger.error(f"[ChemExtract] Scheme page detection failed: {e}")

    result = sorted(scheme_pages)
    logger.info(f"[ChemExtract] Detected {len(result)} scheme/figure pages: {result}")
    return result


def render_scheme_pages(
    file_path: str,
    page_numbers: Optional[List[int]] = None,
    dpi: int = 200,
    max_pages: int = 50,
) -> List[Dict]:
    """Render specific pages (or auto-detected scheme pages) as base64 images.

    If *page_numbers* is None, auto-detects scheme pages via
    :func:`detect_scheme_pages`.

    Returns a list of dicts:
      - page (int): 1-based page number
      - source (str): "scheme_page"
      - dpi (int): rendering resolution
      - base64 (str): base64-encoded PNG
    """
    if not HAS_PYMUPDF:
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")

    if page_numbers is None:
        page_numbers = detect_scheme_pages(file_path, max_pages)

    rendered = []
    try:
        doc = fitz.open(file_path)
        for pg in page_numbers:
            idx = pg - 1  # convert to 0-based
            if idx < 0 or idx >= len(doc):
                continue
            page = doc[idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            rendered.append({
                "page": pg,
                "source": "scheme_page",
                "dpi": dpi,
                "base64": base64.b64encode(img_data).decode('utf-8'),
            })
        logger.info(f"[ChemExtract] Rendered {len(rendered)} scheme pages at {dpi} DPI")
        return rendered
    except (OSError, RuntimeError, ValueError) as e:
        logger.error(f"[ChemExtract] Scheme page rendering failed: {e}")
        raise RuntimeError(f"Failed to render scheme pages: {e}") from e


# ---------------------------------------------------------------------------
# Combined figure extraction pipeline
# ---------------------------------------------------------------------------

def extract_all_visual_content(
    file_path: str,
    dpi: int = 200,
    max_pages: int = 50,
    include_embedded: bool = True,
    include_scheme_pages: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Run the full two-tier visual extraction pipeline.

    Returns:
        (embedded_figures, scheme_page_images)

    - embedded_figures: list from :func:`extract_figures_from_pdf`
    - scheme_page_images: list from :func:`render_scheme_pages`
    """
    embedded = []
    scheme_pages = []

    if include_embedded:
        try:
            embedded = extract_figures_from_pdf(file_path, max_images=MAX_EMBEDDED_IMAGES)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            # ImportError: PyMuPDF not installed. OSError/RuntimeError/
            # ValueError: PDF parse or rendering failure. Continue with empty
            # embedded list — scheme-page rendering may still succeed.
            logger.warning(f"[ChemExtract] Embedded figure extraction failed, continuing: {e}")

    if include_scheme_pages:
        try:
            scheme_pages = render_scheme_pages(file_path, dpi=dpi, max_pages=max_pages)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"[ChemExtract] Scheme page rendering failed, continuing: {e}")

    return embedded, scheme_pages


# ---------------------------------------------------------------------------
# Text extraction (cascading fallback)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> Tuple[str, Dict]:
    """Extract text content from PDF. Tries pypdf -> pdfplumber -> PyMuPDF."""
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
        except (OSError, ValueError, RuntimeError) as e:
            # pypdf raises OSError on corrupt/encrypted PDFs, ValueError on
            # parse failures. Continue to the next extractor in the cascade.
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
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"[ChemExtract] pdfplumber failed: {e}")

    if HAS_PYMUPDF:
        try:
            doc = fitz.open(file_path)
            metadata["pages"] = len(doc)
            metadata["method"] = "pymupdf"
            for page in doc:
                text += page.get_text() + "\n\n"
            return text, metadata
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"[ChemExtract] PyMuPDF text extraction failed: {e}")

    raise ImportError("No PDF text extraction library available. Install pypdf, pdfplumber, or PyMuPDF.")
