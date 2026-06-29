"""
POST /api/extract/pdf — extract raw text from an uploaded PDF.

This endpoint does NOT call an LLM. It just runs the cascading PDF text
extractor (chemextract → pypdf → pdfplumber → PyMuPDF) and returns the
raw text + page metadata. The frontend uses this for the "Extract Text"
button and as a pre-step before feeding the text into /api/extract.

If the PDF has no extractable text (e.g. it's a scanned image), the
response includes a hint to try /api/extract/pdf/vision instead.
"""

import logging

from flask import jsonify

from ._helpers import data_extraction_bp, validate_pdf_upload, cleanup_temp_file

logger = logging.getLogger(__name__)


@data_extraction_bp.route('/extract/pdf', methods=['POST'])
def extract_text_from_pdf():
    """Extract raw text from an uploaded PDF (no LLM call)."""
    from modules.chemextract.pdf_processor import extract_text_from_pdf
    from core.errors import ValidationError

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        logger.info(f"[PDF Text] Extracting text from: {filename}")

        text, metadata = extract_text_from_pdf(tmp_path)

        if not text or len(text.strip()) < 50:
            return jsonify({
                "success": False,
                "error": (
                    "No extractable text found in PDF. The PDF may be "
                    "image-based (scanned). Try using vision extraction."
                ),
                "text_length": len(text.strip()) if text else 0,
                "metadata": metadata,
            })

        logger.info(
            f"[PDF Text] Extracted {len(text)} chars from "
            f"{metadata.get('pages', 'unknown')} pages"
        )

        return jsonify({
            "success": True,
            "text": text,
            "metadata": metadata,
            "text_length": len(text),
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)
