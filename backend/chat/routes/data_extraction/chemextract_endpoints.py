"""
POST /api/extract/pdf/chemextract         — sync ChemExtract AI pipeline
POST /api/extract/pdf/chemextract/async   — async ChemExtract AI pipeline

ChemExtract AI is the most comprehensive extraction pipeline: it runs
both vision (Tier 1 embedded figures + Tier 2 scheme pages) AND text
extraction (chunked, with chunk-aware splitting), then merges and
post-processes the results.

Accepts a multipart form upload with:
  - file:            the PDF file
  - provider:        LLM provider name
  - api_key:         user's LLM API key
  - model:           (optional) model override
  - max_pages:       (optional) cap on pages, default 50
  - extract_images:  (optional) 'true'/'false', default 'true'
  - extract_text:    (optional) 'true'/'false', default 'true'

Returns the full extraction result dict plus a summary of counts.
"""

import logging

from flask import request, jsonify

from core.errors import ValidationError

from ._helpers import (
    data_extraction_bp,
    validate_pdf_upload,
    cleanup_temp_file,
    get_model_for_provider,
)

logger = logging.getLogger(__name__)


def _parse_chemextract_form():
    """Extract (provider, api_key, model, max_pages, extract_images, extract_text)
    from the multipart form.
    """
    provider = request.form.get('provider', 'deepseek')
    api_key = request.form.get('api_key')
    model = get_model_for_provider(provider, request.form.get('model'))
    max_pages = int(request.form.get('max_pages', 50))
    extract_images = request.form.get('extract_images', 'true').lower() == 'true'
    extract_text = request.form.get('extract_text', 'true').lower() == 'true'
    return provider, api_key, model, max_pages, extract_images, extract_text


def _build_chemextract_summary(result: dict) -> dict:
    """Build the count summary returned alongside the extraction result."""
    return {
        "reactions_found": len(result.get("reactions", [])),
        "compounds_found": len(result.get("compounds", [])),
        "tables_found": len(result.get("tables", [])),
        "figures_found": len(result.get("figures", [])),
    }


def _build_chemextract_extraction_details(meta: dict) -> dict:
    """Build the extraction_details block returned in the sync response."""
    return {
        "vision_mode": meta.get("vision_mode", "N/A"),
        "embedded_figures_found": meta.get("embedded_figures_found", 0),
        "scheme_pages_found": meta.get("scheme_pages_found", 0),
        "text_extracted": meta.get("text_extracted", False),
        "images_extracted": meta.get("images_extracted", False),
        "pages_processed": meta.get("pages_processed", 0),
    }


# ---------------------------------------------------------------------------
# Sync endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/chemextract', methods=['POST'])
def extract_pdf_chemextract():
    """Synchronous ChemExtract AI pipeline (vision + text)."""
    from modules.chemextract.extractor import extract_chemical_data_from_pdf

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages, extract_images, extract_text = (
            _parse_chemextract_form()
        )

        logger.info(f"[ChemExtract AI] Starting extraction for {filename}")
        logger.info(
            f"[ChemExtract AI] Provider: {provider}, Model: {model}, "
            f"Max pages: {max_pages}"
        )

        result = extract_chemical_data_from_pdf(
            tmp_path,
            llm_provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            extract_images=extract_images,
            extract_text=extract_text,
        )

        summary = _build_chemextract_summary(result)
        meta = result.get("metadata", {})

        return jsonify({
            "success": True,
            "data": result,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
            "extraction_details": _build_chemextract_extraction_details(meta),
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"ChemExtract AI error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


# ---------------------------------------------------------------------------
# Async endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/chemextract/async', methods=['POST'])
async def extract_pdf_chemextract_async():
    """Asynchronous ChemExtract AI pipeline."""
    from modules.chemextract.extractor import extract_chemical_data_from_pdf_async

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages, extract_images, extract_text = (
            _parse_chemextract_form()
        )

        logger.info(f"[ChemExtract AI Async] Starting extraction for {filename}")
        logger.info(
            f"[ChemExtract AI Async] Provider: {provider}, Model: {model}"
        )

        result = await extract_chemical_data_from_pdf_async(
            tmp_path,
            llm_provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            extract_images=extract_images,
            extract_text=extract_text,
        )

        summary = _build_chemextract_summary(result)

        return jsonify({
            "success": True,
            "data": result,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
            "async": True,
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"ChemExtract AI async error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)
