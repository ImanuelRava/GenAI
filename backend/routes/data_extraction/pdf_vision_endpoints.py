"""
POST /api/extract/pdf/vision        — sync vision extraction
POST /api/extract/pdf/vision/async  — async vision extraction

Both endpoints accept a multipart form upload with:
  - file:        the PDF file
  - provider:    one of VISION_CAPABLE_PROVIDERS (deepseek/openai/gemini/anthropic)
  - api_key:     the user's LLM provider API key
  - model:       (optional) model override
  - max_pages:   (optional) cap on pages to process, default 50

Pipeline (two-tier, with full-page fallback):
  1. Extract embedded raster figures (Tier 1) — analyze each with the
     SYSTEM_PROMPT_FIGURE_ANALYSIS prompt.
  2. Detect & render scheme/figure pages (Tier 2) — analyze each with the
     SYSTEM_PROMPT_VISION prompt.
  3. If neither tier produced results, fall back to rendering every page
     at low DPI and analyzing each as a full-page image.

The async variant uses asyncio.gather to analyze all figures + scheme
pages concurrently, which is dramatically faster on multi-page PDFs.

Results from all analyzed images are merged via merge_extraction_results()
(which handles both the flat and structured LLM output formats).
"""

import asyncio
import logging

from flask import request, jsonify

from core.errors import ValidationError

from ._helpers import (
    data_extraction_bp,
    validate_pdf_upload,
    cleanup_temp_file,
    get_model_for_provider,
    VISION_CAPABLE_PROVIDERS,
    merge_extraction_results,
)

logger = logging.getLogger(__name__)


def _parse_vision_form():
    """Extract (provider, api_key, model, max_pages) from the multipart form.

    Raises ValidationError if the provider is not vision-capable.
    """
    provider = request.form.get('provider', 'openai')
    api_key = request.form.get('api_key')
    model = get_model_for_provider(provider, request.form.get('model'))
    max_pages = int(request.form.get('max_pages', 50))

    if provider not in VISION_CAPABLE_PROVIDERS:
        raise ValidationError(
            f"Provider '{provider}' does not support vision. "
            f"Use one of: {VISION_CAPABLE_PROVIDERS}"
        )

    return provider, api_key, model, max_pages


# ---------------------------------------------------------------------------
# Sync endpoint
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/vision', methods=['POST'])
def extract_pdf_with_vision():
    """Synchronous PDF vision extraction (two-tier + fallback)."""
    from modules.chemextract.pdf_processor import (
        extract_text_from_pdf,
        extract_all_visual_content,
        pdf_to_images,
    )
    from modules.chemextract.llm_providers import call_vision_llm
    from modules.chemextract.prompts import (
        SYSTEM_PROMPT_VISION,
        SYSTEM_PROMPT_FIGURE_ANALYSIS,
    )

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages = _parse_vision_form()

        logger.info(
            f"[PDF Vision] Processing: {filename}, Provider: {provider}, "
            f"Model: {model}"
        )
        logger.info(
            "[PDF Vision] Using figure extraction pipeline "
            "(embedded figures + scheme pages)"
        )

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)
        embedded_figures, scheme_pages = extract_all_visual_content(
            tmp_path, dpi=200, max_pages=max_pages,
        )
        pdf_metadata['embedded_figures_found'] = len(embedded_figures)
        pdf_metadata['scheme_pages_found'] = len(scheme_pages)
        pdf_metadata['vision_mode'] = 'figure_extraction'

        all_results = []

        # Tier 1: Analyze embedded figures with focused prompt.
        for fig in embedded_figures:
            logger.info(
                f"[PDF Vision] Analyzing embedded figure on page {fig['page']}"
            )
            user_msg = (
                f"Analyze this extracted figure from page {fig['page']} of a "
                f"chemistry paper. Extract all chemical reaction data visible "
                f"in this figure."
            )
            result = call_vision_llm(
                fig["base64"], provider, model, api_key,
                SYSTEM_PROMPT_FIGURE_ANALYSIS, user_msg,
            )
            if result:
                all_results.append({
                    "page": fig["page"],
                    "data": result,
                    "source": "embedded_figure",
                })

        # Tier 2: Analyze scheme pages with comprehensive prompt.
        for sp in scheme_pages:
            logger.info(f"[PDF Vision] Analyzing scheme page {sp['page']}")
            result = call_vision_llm(
                sp["base64"], provider, model, api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this page (page {sp['page']}) and extract all "
                f"chemical reaction data:",
            )
            if result:
                all_results.append({
                    "page": sp["page"],
                    "data": result,
                    "source": "scheme_page",
                })

        # Fallback: render every page if no figures/schemes were detected.
        if not all_results:
            logger.info(
                "[PDF Vision] No figures/schemes detected, falling back to "
                "full-page rendering"
            )
            page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)
            for page_num, base64_image in page_images:
                result = call_vision_llm(
                    base64_image, provider, model, api_key,
                    SYSTEM_PROMPT_VISION,
                    f"Analyze this image (page {page_num}) and extract all "
                    f"chemical reaction data:",
                )
                if result:
                    all_results.append({
                        "page": page_num,
                        "data": result,
                        "source": "full_page",
                    })

        merged = merge_extraction_results(all_results)
        merged['text_content'] = text_content[:5000] if text_content else ""

        return jsonify({
            "success": True,
            "data": merged,
            "metadata": pdf_metadata,
            "embedded_figures": len(embedded_figures),
            "scheme_pages": len(scheme_pages),
            "total_vision_items": len(all_results),
            "model_used": model,
            "provider_used": provider,
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"PDF vision extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


# ---------------------------------------------------------------------------
# Async endpoint — concurrent figure + scheme page analysis
# ---------------------------------------------------------------------------

@data_extraction_bp.route('/extract/pdf/vision/async', methods=['POST'])
async def extract_pdf_with_vision_async():
    """Asynchronous PDF vision extraction with concurrent analysis."""
    from modules.chemextract.pdf_processor import (
        extract_text_from_pdf,
        extract_all_visual_content,
        pdf_to_images,
    )
    from modules.chemextract.llm_providers import call_vision_llm_async
    from modules.chemextract.prompts import (
        SYSTEM_PROMPT_VISION,
        SYSTEM_PROMPT_FIGURE_ANALYSIS,
    )

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()
        provider, api_key, model, max_pages = _parse_vision_form()

        logger.info(f"[PDF Vision Async] Processing: {filename}")
        logger.info("[PDF Vision Async] Using figure extraction pipeline")

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)
        embedded_figures, scheme_pages = extract_all_visual_content(
            tmp_path, dpi=200, max_pages=max_pages,
        )
        pdf_metadata['embedded_figures_found'] = len(embedded_figures)
        pdf_metadata['scheme_pages_found'] = len(scheme_pages)
        pdf_metadata['vision_mode'] = 'figure_extraction'

        # Build concurrent analysis tasks for figures + scheme pages.
        async def analyze_fig(fig):
            return await call_vision_llm_async(
                fig["base64"], provider, model, api_key,
                SYSTEM_PROMPT_FIGURE_ANALYSIS,
                f"Analyze this extracted figure from page {fig['page']} of a "
                f"chemistry paper. Extract all chemical reaction data visible "
                f"in this figure.",
            )

        async def analyze_sp(sp):
            return await call_vision_llm_async(
                sp["base64"], provider, model, api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this page (page {sp['page']}) and extract all "
                f"chemical reaction data:",
            )

        tasks = [analyze_fig(fig) for fig in embedded_figures]
        tasks += [analyze_sp(sp) for sp in scheme_pages]
        results = await asyncio.gather(*tasks)

        all_results = []
        # Map results back to their source figures / scheme pages.
        for fig, sp_idx in zip(embedded_figures, range(len(embedded_figures))):
            if results[sp_idx]:
                all_results.append({
                    "page": fig["page"],
                    "data": results[sp_idx],
                    "source": "embedded_figure",
                })

        for sp, sp_idx in zip(scheme_pages, range(len(scheme_pages))):
            result_idx = len(embedded_figures) + sp_idx
            if results[result_idx]:
                all_results.append({
                    "page": sp["page"],
                    "data": results[result_idx],
                    "source": "scheme_page",
                })

        # Fallback: render every page concurrently if nothing was found.
        if not all_results:
            logger.info(
                "[PDF Vision Async] No figures detected, falling back to "
                "full-page rendering"
            )
            page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)

            async def process_page(page_num, base64_image):
                return await call_vision_llm_async(
                    base64_image, provider, model, api_key,
                    SYSTEM_PROMPT_VISION,
                    f"Analyze this image (page {page_num}) and extract all "
                    f"chemical reaction data:",
                )

            tasks = [process_page(pn, bi) for pn, bi in page_images]
            results = await asyncio.gather(*tasks)
            for i, result in enumerate(results):
                if result:
                    all_results.append({
                        "page": page_images[i][0],
                        "data": result,
                        "source": "full_page",
                    })

        merged = merge_extraction_results(all_results)
        merged['text_content'] = text_content[:5000] if text_content else ""

        return jsonify({
            "success": True,
            "data": merged,
            "metadata": pdf_metadata,
            "embedded_figures": len(embedded_figures),
            "scheme_pages": len(scheme_pages),
            "total_vision_items": len(all_results),
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
        logger.error(f"PDF vision async extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)
