"""
POST /api/extract/pdf/reactionlens         — sync ReactionLens extraction
POST /api/extract/pdf/reactionlens/async   — async ReactionLens extraction
POST /api/extract/pdf/reactionlens/text    — sync ReactionLens + raw text dump

ReactionLens is a text-only extraction pipeline: it extracts text from
the PDF, segments it into paragraphs, and screens each paragraph with
an LLM to detect reaction data. The output is ChemExtract-compatible
so the frontend can render it with the same components.

Accepts a multipart form upload with:
  - file:                  the PDF file
  - provider:              LLM provider name (deepseek/openai/gemini/anthropic)
  - api_key:               user's LLM API key
  - model:                 (optional) model override
  - max_pages:             (optional) cap on pages, default 50
  - min_paragraph_length:  (optional) drop paragraphs shorter than this, default 80

The /text variant additionally returns the full extracted text alongside
the structured reaction data, for UIs that want to show both.
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


def _parse_reactionlens_form():
    """Extract (provider, api_key, model, max_pages, min_paragraph_length)
    from the multipart form.
    """
    provider = request.form.get('provider', 'deepseek')
    api_key = request.form.get('api_key')
    model = get_model_for_provider(provider, request.form.get('model'))
    max_pages = int(request.form.get('max_pages', 50))
    min_paragraph_length = int(request.form.get('min_paragraph_length', 80))
    return provider, api_key, model, max_pages, min_paragraph_length


def _build_reactionlens_summary(stats: dict, extra: dict = None) -> dict:
    """Build the count summary returned alongside the ReactionLens result.

    Args:
        stats: the extraction_stats dict from the ReactionLens result
        extra: optional extra fields to merge in (e.g. text_length)
    """
    summary = {
        "reactions_found": stats.get("total_reactions", 0),
        "compounds_found": stats.get("total_compounds", 0),
        "paragraphs_scanned": stats.get("total_paragraphs", 0),
        "paragraphs_with_reactions": stats.get("paragraphs_with_reactions", 0),
    }
    if extra:
        summary.update(extra)
    return summary


# ---------------------------------------------------------------------------
# Sync endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/reactionlens', methods=['POST'])
def extract_pdf_reactionlens():
    """Synchronous ReactionLens text-screening extraction."""
    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages, min_paragraph_length = (
            _parse_reactionlens_form()
        )

        logger.info(f"[ReactionLens] Starting extraction for {filename}")
        logger.info(
            f"[ReactionLens] Provider: {provider}, Model: {model}, "
            f"Max pages: {max_pages}"
        )
        logger.info(f"[ReactionLens] Min paragraph length: {min_paragraph_length}")

        from modules.reaction.extraction import extract_with_reactionlens

        result = extract_with_reactionlens(
            tmp_path,
            provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            min_paragraph_length=min_paragraph_length,
        )

        stats = result.get("extraction_stats", {})
        summary = _build_reactionlens_summary(stats)

        return jsonify({
            "success": True,
            "data": result,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"ReactionLens module not available: {str(e)}",
        }), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"ReactionLens extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


# ---------------------------------------------------------------------------
# Async endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/reactionlens/async', methods=['POST'])
async def extract_pdf_reactionlens_async():
    """Asynchronous ReactionLens text-screening extraction."""
    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages, min_paragraph_length = (
            _parse_reactionlens_form()
        )

        logger.info(f"[ReactionLens Async] Starting extraction for {filename}")
        logger.info(
            f"[ReactionLens Async] Provider: {provider}, Model: {model}, "
            f"Max pages: {max_pages}"
        )

        from modules.reaction.extraction import extract_with_reactionlens_async

        result = await extract_with_reactionlens_async(
            tmp_path,
            provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            min_paragraph_length=min_paragraph_length,
        )

        stats = result.get("extraction_stats", {})
        summary = _build_reactionlens_summary(stats)

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
        return jsonify({
            "success": False,
            "error": f"ReactionLens module not available: {str(e)}",
        }), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"ReactionLens async extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


# ---------------------------------------------------------------------------
# ReactionLens + raw text dump endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/reactionlens/text', methods=['POST'])
def extract_pdf_reactionlens_with_text():
    """Sync ReactionLens extraction that also returns the raw PDF text.

    Useful for UIs that want to show the source text alongside the
    detected reactions (e.g. highlighting which paragraphs contained
    reaction data).
    """
    from modules.reaction.parsing import extract_text_from_pdf as rl_extract_text

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages, min_paragraph_length = (
            _parse_reactionlens_form()
        )

        logger.info(f"[ReactionLens+Text] Starting extraction for {filename}")
        logger.info(
            f"[ReactionLens+Text] Provider: {provider}, Model: {model}, "
            f"Max pages: {max_pages}"
        )

        text_content, pdf_metadata = rl_extract_text(tmp_path)

        from modules.reaction.extraction import extract_with_reactionlens

        result = extract_with_reactionlens(
            tmp_path,
            provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            min_paragraph_length=min_paragraph_length,
        )

        stats = result.get("extraction_stats", {})
        summary = _build_reactionlens_summary(stats, extra={
            "text_length": len(text_content) if text_content else 0,
        })

        return jsonify({
            "success": True,
            "data": result,
            "text_content": text_content,
            "text_metadata": pdf_metadata,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"ReactionLens module not available: {str(e)}",
        }), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"ReactionLens+Text extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)
