import os
import logging
import tempfile
from typing import Dict, Any

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from errors import ValidationError, APIError
from utils import sanitize_input

logger = logging.getLogger(__name__)

data_extraction_bp = Blueprint('data_extraction', __name__)

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



def get_model_for_provider(provider: str, model: str = None) -> str:
    """Get the appropriate model for a provider."""
    if model:
        return model
    return PROVIDER_DEFAULT_MODELS.get(provider, 'deepseek-chat')


def validate_pdf_upload() -> tuple:
    """Validate PDF file upload and return temp file path."""
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
    """Remove temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file: {e}")



@data_extraction_bp.route('/api/extract/models', methods=['GET'])
def get_models():
    """Get available LLM models for extraction."""
    return jsonify({
        "success": True,
        "models": AVAILABLE_MODELS,
        "vision_providers": VISION_CAPABLE_PROVIDERS,
        "async_support": True
    })


@data_extraction_bp.route('/api/extract', methods=['POST'])
def extract_from_text():
    """
    Extract chemical data from text using LLM (sync).
    
    Request body:
    {
        "text": "Scientific text to analyze...",
        "provider": "deepseek",
        "api_key": "your-api-key",
        "model": "deepseek-chat"  // optional
    }
    """
    from modules.chemextract_extractor import call_text_llm, SYSTEM_PROMPT_COMPREHENSIVE

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


@data_extraction_bp.route('/api/extract/async', methods=['POST'])
async def extract_from_text_async():
    """
    Extract chemical data from text using LLM (async).
    
    Request body:
    {
        "text": "Scientific text to analyze...",
        "provider": "deepseek",
        "api_key": "your-api-key",
        "model": "deepseek-chat"  // optional
    }
    """
    from modules.chemextract_extractor import call_text_llm_async

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


@data_extraction_bp.route('/api/extract/pdf', methods=['POST'])
def extract_text_from_pdf():
    """
    Extract text content from a PDF file.
    
    This endpoint only extracts text, does not analyze with LLM.
    Use /api/extract/pdf/vision for full extraction.
    """
    from modules.chemextract_extractor import extract_text_from_pdf

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


@data_extraction_bp.route('/api/extract/pdf/vision', methods=['POST'])
def extract_pdf_with_vision():
    """
    Extract chemical data from PDF using vision-capable LLM (sync).
    
    Form data:
    - file: PDF file
    - provider: LLM provider (deepseek, openai, gemini, anthropic)
    - api_key: Your API key
    - model: Model to use (optional)
    - max_pages: Maximum pages to process (default: 5)
    """
    from modules.chemextract_extractor import (
        ChemExtractAI,
        extract_text_from_pdf,
        pdf_to_images,
        call_vision_llm,
        SYSTEM_PROMPT_VISION
    )

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'openai')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 5))

        if provider not in VISION_CAPABLE_PROVIDERS:
            return jsonify({
                "success": False,
                "error": f"Provider '{provider}' does not support vision. Use: {VISION_CAPABLE_PROVIDERS}"
            }), 400

        logger.info(f"[PDF Vision] Processing: {filename}, Provider: {provider}, Model: {model}")

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)

        page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)
        pdf_metadata['pages_analyzed'] = len(page_images)

        all_results = []
        for page_num, base64_image in page_images:
            logger.info(f"[PDF Vision] Analyzing page {page_num}")

            result = call_vision_llm(
                base64_image,
                provider,
                model,
                api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this image (page {page_num}) and extract all chemical reaction data:"
            )

            if result:
                all_results.append({"page": page_num, "data": result})

        merged = merge_extraction_results(all_results)
        merged['text_content'] = text_content[:5000] if text_content else ""

        return jsonify({
            "success": True,
            "data": merged,
            "metadata": pdf_metadata,
            "pages_analyzed": len(page_images),
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


@data_extraction_bp.route('/api/extract/pdf/vision/async', methods=['POST'])
async def extract_pdf_with_vision_async():
    """
    Extract chemical data from PDF using vision-capable LLM (async).
    
    Processes all pages concurrently for faster extraction.
    """
    from modules.chemextract_extractor import (
        ChemExtractAI,
        extract_text_from_pdf,
        pdf_to_images,
        call_vision_llm_async,
        SYSTEM_PROMPT_VISION
    )
    import asyncio

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'openai')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 5))

        if provider not in VISION_CAPABLE_PROVIDERS:
            return jsonify({
                "success": False,
                "error": f"Provider '{provider}' does not support vision."
            }), 400

        logger.info(f"[PDF Vision Async] Processing: {filename}")

        text_content, pdf_metadata = extract_text_from_pdf(tmp_path)
        page_images = pdf_to_images(tmp_path, dpi=150, max_pages=max_pages)
        pdf_metadata['pages_analyzed'] = len(page_images)

        async def process_page(page_num: int, base64_image: str):
            return await call_vision_llm_async(
                base64_image,
                provider,
                model,
                api_key,
                SYSTEM_PROMPT_VISION,
                f"Analyze this image (page {page_num}) and extract all chemical reaction data:"
            )

        tasks = [process_page(page_num, base64_image) for page_num, base64_image in page_images]
        results = await asyncio.gather(*tasks)

        all_results = []
        for i, result in enumerate(results):
            if result:
                all_results.append({"page": page_images[i][0], "data": result})

        merged = merge_extraction_results(all_results)
        merged['text_content'] = text_content[:5000] if text_content else ""

        return jsonify({
            "success": True,
            "data": merged,
            "metadata": pdf_metadata,
            "pages_analyzed": len(page_images),
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


@data_extraction_bp.route('/api/extract/pdf/chemextract', methods=['POST'])
def extract_pdf_chemextract():
    """
    Comprehensive chemical data extraction from PDF using ChemExtract AI.
    
    This is the main extraction endpoint that combines:
    - Text analysis for chemical entities
    - Vision analysis for reaction schemes and tables
    - Comprehensive output with reactions, compounds, tables
    
    Form data:
    - file: PDF file
    - provider: LLM provider (default: deepseek)
    - api_key: Your API key
    - model: Model to use (optional)
    - max_pages: Maximum pages to process (default: 10)
    - extract_images: Enable vision analysis (default: true)
    - extract_text: Enable text analysis (default: true)
    """
    from modules.chemextract_extractor import extract_chemical_data_from_pdf

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 10))
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
            extract_text=extract_text
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
            "provider_used": provider
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


@data_extraction_bp.route('/api/extract/pdf/chemextract/async', methods=['POST'])
async def extract_pdf_chemextract_async():
    """
    Async comprehensive chemical data extraction from PDF using ChemExtract AI.
    
    Processes pages concurrently for faster extraction.
    """
    from modules.chemextract_extractor import extract_chemical_data_from_pdf_async

    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 10))
        extract_images = request.form.get('extract_images', 'true').lower() == 'true'
        extract_text = request.form.get('extract_text', 'true').lower() == 'true'

        logger.info(f"[ChemExtract AI Async] Starting extraction for {filename}")

        result = await extract_chemical_data_from_pdf_async(
            tmp_path,
            llm_provider=provider,
            api_key=api_key,
            model=model,
            max_pages=max_pages,
            extract_images=extract_images,
            extract_text=extract_text
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



def merge_extraction_results(all_results: list) -> dict:
    """Merge extraction results from multiple pages."""
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
        "pages_with_data": []
    }

    for page_result in all_results:
        page_num = page_result["page"]
        data = page_result["data"]

        if not data:
            continue

        merged["pages_with_data"].append(page_num)

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

    return merged
