import os
import logging
import tempfile
import json
from typing import Dict, Any

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from core.errors import ValidationError, APIError
from core.utils import sanitize_input

logger = logging.getLogger(__name__)

data_extraction_bp = Blueprint('data_extraction', __name__, url_prefix='/api')

PROVIDER_DEFAULT_MODELS = {
    'deepseek': 'deepseek-chat',
    'openai': 'gpt-4o-mini',
    'gemini': 'gemini-2.0-flash',
    'groq': 'llama-3.3-70b-versatile',
    'ollama': 'llama3',
    'anthropic': 'claude-3-5-sonnet-20241022',
    'zhipu': 'glm-4-flash',
}

VISION_CAPABLE_PROVIDERS = ['deepseek', 'openai', 'gemini', 'anthropic']

AVAILABLE_MODELS = [
    {"id": "deepseek-chat", "name": "DeepSeek Chat", "provider": "DeepSeek", "description": "Fast and efficient, vision capable"},
    {"id": "deepseek-reasoner", "name": "DeepSeek Reasoner", "provider": "DeepSeek", "description": "Enhanced reasoning"},
    {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "description": "Vision capable"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI", "description": "Fast and cheap"},
    {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "Google", "description": "Vision capable"},
    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "provider": "Anthropic", "description": "Vision capable"},
    {"id": "glm-4-flash", "name": "GLM-4-Flash", "provider": "Zhipu AI", "description": "Fast and efficient"},
]

REACTIONLENS_INFO = {
    "id": "reactionlens",
    "name": "ReactionLens",
    "provider": "Built-in",
    "description": "Text-driven chemical reaction detection: extracts text from paper, screens paragraphs for reactions, outputs ChemExtract-compatible data",
    "capabilities": [
        "text_extraction",
        "paragraph_screening",
        "reaction_detection",
        "condition_parsing",
        "chemextract_compatible_output"
    ],
    "supported_formats": ["pdf"]
}


def get_model_for_provider(provider: str, model: str = None) -> str:
    if model:
        return model
    return PROVIDER_DEFAULT_MODELS.get(provider, 'deepseek-chat')


def validate_pdf_upload() -> tuple:
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


def cleanup_temp_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file: {e}")


@data_extraction_bp.route('/extract/models', methods=['GET'])
def get_models():
    return jsonify({
        "success": True,
        "models": AVAILABLE_MODELS,
        "vision_providers": VISION_CAPABLE_PROVIDERS,
        "reactionlens": REACTIONLENS_INFO,
        "async_support": True
    })


@data_extraction_bp.route('/extract', methods=['POST'])
def extract_from_text():
    from modules.chemextract.standalone import call_text_llm

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        text = data.get('text', '')
        provider = data.get('provider', 'deepseek')
        api_key = data.get('api_key')
        model = get_model_for_provider(provider, data.get('model'))

        max_length = 15000
        text = sanitize_input(text, max_length=max_length)
        if not text:
            raise ValidationError("Text cannot be empty", field="text")

        logger.info(f"[Data Extraction] Provider: {provider}, Model: {model}, Text length: {len(text)}")

        result = call_text_llm(text, provider, model, api_key)

        if result:
            total_items = sum(len(result.get(k, [])) for k in ['reactants', 'products', 'catalysts', 'ligands', 'solvents', 'mechanisms'])
            if total_items == 0 and not result.get('conditions') and not result.get('yields'):
                return jsonify({
                    "success": True,
                    "data": result,
                    "warning": "No chemical reaction data was found in the provided text.",
                    "model_used": model,
                    "provider_used": provider
                })

            return jsonify({
                "success": True,
                "data": result,
                "model_used": model,
                "provider_used": provider
            })
        else:
            return jsonify({
                "success": False,
                "error": "No response from LLM. Please check your API key and provider settings."
            }), 500

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Data extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@data_extraction_bp.route('/extract/async', methods=['POST'])
async def extract_from_text_async():
    from modules.chemextract.standalone import call_text_llm_async

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        text = data.get('text', '')
        provider = data.get('provider', 'deepseek')
        api_key = data.get('api_key')
        model = get_model_for_provider(provider, data.get('model'))

        max_length = 15000
        text = sanitize_input(text, max_length=max_length)
        if not text:
            raise ValidationError("Text cannot be empty", field="text")

        logger.info(f"[Data Extraction Async] Provider: {provider}, Model: {model}, Text length: {len(text)}")

        result = await call_text_llm_async(text, provider, model, api_key)

        if result:
            return jsonify({
                "success": True,
                "data": result,
                "model_used": model,
                "provider_used": provider,
                "async": True
            })
        else:
            return jsonify({
                "success": False,
                "error": "No response from LLM."
            }), 500

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"Data extraction async error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@data_extraction_bp.route('/extract/pdf', methods=['POST'])
def extract_text_from_pdf():
    from modules.chemextract.pdf_processor import extract_text_from_pdf

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        logger.info(f"[PDF Text] Extracting text from: {filename}")

        text, metadata = extract_text_from_pdf(tmp_path)

        if not text or len(text.strip()) < 50:
            return jsonify({
                "success": False,
                "error": "No extractable text found in PDF. The PDF may be image-based (scanned). Try using vision extraction.",
                "text_length": len(text.strip()) if text else 0,
                "metadata": metadata
            })

        logger.info(f"[PDF Text] Extracted {len(text)} chars from {metadata.get('pages', 'unknown')} pages")

        return jsonify({
            "success": True,
            "text": text,
            "metadata": metadata,
            "text_length": len(text)
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"PDF text extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/vision', methods=['POST'])
def extract_pdf_with_vision():
    from modules.chemextract.pdf_processor import (
        extract_text_from_pdf, extract_figures_from_pdf,
        render_scheme_pages, extract_all_visual_content,
    )
    from modules.chemextract.llm_providers import call_vision_llm
    from modules.chemextract.prompts import SYSTEM_PROMPT_VISION, SYSTEM_PROMPT_FIGURE_ANALYSIS

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'openai')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))

        if provider not in VISION_CAPABLE_PROVIDERS:
            return jsonify({
                "success": False,
                "error": f"Provider '{provider}' does not support vision. Use: {VISION_CAPABLE_PROVIDERS}"
            }), 400

        logger.info(f"[PDF Vision] Processing: {filename}, Provider: {provider}, Model: {model}")
        logger.info("[PDF Vision] Using figure extraction pipeline (embedded figures + scheme pages)")

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)

        # Two-tier vision extraction
        embedded_figures, scheme_pages = extract_all_visual_content(
            tmp_path, dpi=200, max_pages=max_pages,
        )
        pdf_metadata['embedded_figures_found'] = len(embedded_figures)
        pdf_metadata['scheme_pages_found'] = len(scheme_pages)
        pdf_metadata['vision_mode'] = 'figure_extraction'

        all_results = []

        # Tier 1: Analyze embedded figures with focused prompt
        for fig in embedded_figures:
            logger.info(f"[PDF Vision] Analyzing embedded figure on page {fig['page']}")
            user_msg = (
                f"Analyze this extracted figure from page {fig['page']} of a chemistry paper. "
                f"Extract all chemical reaction data visible in this figure."
            )
            result = call_vision_llm(
                fig["base64"], provider, model, api_key,
                SYSTEM_PROMPT_FIGURE_ANALYSIS, user_msg,
            )
            if result:
                all_results.append({"page": fig["page"], "data": result, "source": "embedded_figure"})

        # Tier 2: Analyze scheme pages with comprehensive prompt
        for sp in scheme_pages:
            logger.info(f"[PDF Vision] Analyzing scheme page {sp['page']}")
            result = call_vision_llm(
                sp["base64"], provider, model, api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this page (page {sp['page']}) and extract all chemical reaction data:"
            )
            if result:
                all_results.append({"page": sp["page"], "data": result, "source": "scheme_page"})

        # Fallback: if no figures/schemes found, use full-page rendering
        if not all_results:
            logger.info("[PDF Vision] No figures/schemes detected, falling back to full-page rendering")
            from modules.chemextract.pdf_processor import pdf_to_images
            page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)
            for page_num, base64_image in page_images:
                result = call_vision_llm(
                    base64_image, provider, model, api_key,
                    SYSTEM_PROMPT_VISION,
                    f"Analyze this image (page {page_num}) and extract all chemical reaction data:"
                )
                if result:
                    all_results.append({"page": page_num, "data": result, "source": "full_page"})

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
            "provider_used": provider
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"PDF vision extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/vision/async', methods=['POST'])
async def extract_pdf_with_vision_async():
    from modules.chemextract.pdf_processor import (
        extract_text_from_pdf, extract_all_visual_content, pdf_to_images,
    )
    from modules.chemextract.llm_providers import call_vision_llm_async
    from modules.chemextract.prompts import SYSTEM_PROMPT_VISION, SYSTEM_PROMPT_FIGURE_ANALYSIS
    import asyncio

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'openai')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))

        if provider not in VISION_CAPABLE_PROVIDERS:
            return jsonify({
                "success": False,
                "error": f"Provider '{provider}' does not support vision."
            }), 400

        logger.info(f"[PDF Vision Async] Processing: {filename}")
        logger.info("[PDF Vision Async] Using figure extraction pipeline")

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)
        embedded_figures, scheme_pages = extract_all_visual_content(
            tmp_path, dpi=200, max_pages=max_pages,
        )
        pdf_metadata['embedded_figures_found'] = len(embedded_figures)
        pdf_metadata['scheme_pages_found'] = len(scheme_pages)
        pdf_metadata['vision_mode'] = 'figure_extraction'

        all_results = []

        # Concurrent analysis of embedded figures
        async def analyze_fig(fig):
            return await call_vision_llm_async(
                fig["base64"], provider, model, api_key,
                SYSTEM_PROMPT_FIGURE_ANALYSIS,
                f"Analyze this extracted figure from page {fig['page']} of a chemistry paper. "
                f"Extract all chemical reaction data visible in this figure."
            )

        # Concurrent analysis of scheme pages
        async def analyze_sp(sp):
            return await call_vision_llm_async(
                sp["base64"], provider, model, api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this page (page {sp['page']}) and extract all chemical reaction data:"
            )

        tasks = [analyze_fig(fig) for fig in embedded_figures]
        tasks += [analyze_sp(sp) for sp in scheme_pages]
        results = await asyncio.gather(*tasks)

        for fig, sp_idx in zip(embedded_figures, range(len(embedded_figures))):
            if results[sp_idx]:
                all_results.append({"page": fig["page"], "data": results[sp_idx], "source": "embedded_figure"})

        for sp, sp_idx in zip(scheme_pages, range(len(scheme_pages))):
            result_idx = len(embedded_figures) + sp_idx
            if results[result_idx]:
                all_results.append({"page": sp["page"], "data": results[result_idx], "source": "scheme_page"})

        # Fallback
        if not all_results:
            logger.info("[PDF Vision Async] No figures detected, falling back to full-page rendering")
            page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)
            async def process_page(page_num, base64_image):
                return await call_vision_llm_async(
                    base64_image, provider, model, api_key,
                    SYSTEM_PROMPT_VISION,
                    f"Analyze this image (page {page_num}) and extract all chemical reaction data:"
                )
            tasks = [process_page(pn, bi) for pn, bi in page_images]
            results = await asyncio.gather(*tasks)
            for i, result in enumerate(results):
                if result:
                    all_results.append({"page": page_images[i][0], "data": result, "source": "full_page"})

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
            "async": True
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"PDF vision async extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/chemextract', methods=['POST'])
def extract_pdf_chemextract():
    from modules.chemextract.extractor import extract_chemical_data_from_pdf

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'

        logger.info(f"[ChemExtract AI] Starting extraction for {filename}")
        logger.info(f"[ChemExtract AI] Provider: {provider}, Model: {model}, Max pages: {max_pages}")

        result = extract_chemical_data_from_pdf(
            tmp_path,
            llm_provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            extract_images=extract_images,
            extract_text=extract_text,
        )

        summary = {
            "reactions_found": len(result.get("reactions", [])),
            "compounds_found": len(result.get("compounds", [])),
            "tables_found": len(result.get("tables", [])),
            "figures_found": len(result.get("figures", []))
        }

        meta = result.get("metadata", {})
        return jsonify({
            "success": True,
            "data": result,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
            "extraction_details": {
                "vision_mode": meta.get("vision_mode", "N/A"),
                "embedded_figures_found": meta.get("embedded_figures_found", 0),
                "scheme_pages_found": meta.get("scheme_pages_found", 0),
                "text_extracted": meta.get("text_extracted", False),
                "images_extracted": meta.get("images_extracted", False),
                "pages_processed": meta.get("pages_processed", 0),
            }
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"ChemExtract AI error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/chemextract/async', methods=['POST'])
async def extract_pdf_chemextract_async():
    from modules.chemextract.extractor import extract_chemical_data_from_pdf_async

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'

        logger.info(f"[ChemExtract AI Async] Starting extraction for {filename}")
        logger.info(f"[ChemExtract AI Async] Provider: {provider}, Model: {model}")

        result = await extract_chemical_data_from_pdf_async(
            tmp_path,
            llm_provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            extract_images=extract_images,
            extract_text=extract_text,
        )

        summary = {
            "reactions_found": len(result.get("reactions", [])),
            "compounds_found": len(result.get("compounds", [])),
            "tables_found": len(result.get("tables", [])),
            "figures_found": len(result.get("figures", []))
        }

        return jsonify({
            "success": True,
            "data": result,
            "summary": summary,
            "filename": filename,
            "model_used": model,
            "provider_used": provider,
            "async": True
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.error(f"ChemExtract AI async error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/reactionlens', methods=['POST'])
def extract_pdf_reactionlens():
    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        min_paragraph_length = int(request.form.get('min_paragraph_length', 80))

        logger.info(f"[ReactionLens] Starting extraction for {filename}")
        logger.info(f"[ReactionLens] Provider: {provider}, Model: {model}, Max pages: {max_pages}")
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
        summary = {
            "reactions_found": stats.get("total_reactions", 0),
            "compounds_found": stats.get("total_compounds", 0),
            "paragraphs_scanned": stats.get("total_paragraphs", 0),
            "paragraphs_with_reactions": stats.get("paragraphs_with_reactions", 0),
        }

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
        return jsonify({"success": False, "error": f"ReactionLens module not available: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"ReactionLens extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/reactionlens/async', methods=['POST'])
async def extract_pdf_reactionlens_async():
    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        min_paragraph_length = int(request.form.get('min_paragraph_length', 80))

        logger.info(f"[ReactionLens Async] Starting extraction for {filename}")
        logger.info(f"[ReactionLens Async] Provider: {provider}, Model: {model}, Max pages: {max_pages}")

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
        summary = {
            "reactions_found": stats.get("total_reactions", 0),
            "compounds_found": stats.get("total_compounds", 0),
            "paragraphs_scanned": stats.get("total_paragraphs", 0),
            "paragraphs_with_reactions": stats.get("paragraphs_with_reactions", 0),
        }

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
        return jsonify({"success": False, "error": f"ReactionLens module not available: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"ReactionLens async extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/pdf/reactionlens/text', methods=['POST'])
def extract_pdf_reactionlens_with_text():
    from modules.reaction.parsing import extract_text_from_pdf as rl_extract_text

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        min_paragraph_length = int(request.form.get('min_paragraph_length', 80))

        logger.info(f"[ReactionLens+Text] Starting extraction for {filename}")
        logger.info(f"[ReactionLens+Text] Provider: {provider}, Model: {model}, Max pages: {max_pages}")

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
        summary = {
            "reactions_found": stats.get("total_reactions", 0),
            "compounds_found": stats.get("total_compounds", 0),
            "paragraphs_scanned": stats.get("total_paragraphs", 0),
            "paragraphs_with_reactions": stats.get("paragraphs_with_reactions", 0),
            "text_length": len(text_content) if text_content else 0,
        }

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
        return jsonify({"success": False, "error": f"ReactionLens module not available: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"ReactionLens+Text extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


@data_extraction_bp.route('/extract/format/schemes', methods=['POST'])
def format_reaction_schemes_endpoint():
    from modules.chemextract.reaction_formatter import format_reaction_schemes

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        extraction_result = data
        output_format = data.get('format', 'smiles')

        if not data.get('reactions') and not data.get('compounds'):
            raise ValidationError("No reaction or compound data provided for formatting")

        logger.info(f"[Format Schemes] Formatting extraction result as {output_format}")

        result = format_reaction_schemes(extraction_result)

        return jsonify({
            "success": True,
            "data": result,
            "format": output_format,
            "reactions_formatted": len(result.get("schemes", []))
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({"success": False, "error": f"Format module not available: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Format reaction schemes error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def merge_extraction_results(all_results: list) -> dict:
    """Merge extraction results from multiple pages/sources.

    Handles two output formats:
      1. Flat format (SYSTEM_PROMPT_VISION): reactants, products, catalysts, etc.
      2. Structured format (SYSTEM_PROMPT_FIGURE_ANALYSIS): reaction_schemes,
         compounds, table_data, etc.
    """
    merged = {
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
                "description": data["image_description"]
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
                # Also flatten into the flat format for backward compatibility
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
