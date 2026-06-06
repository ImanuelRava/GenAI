import os
import logging
import tempfile

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from errors import ValidationError

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
    - max_pages: Maximum pages to process (default: 50)
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


# ─── ReactionLens Endpoints ─────────────────────────────────────────────────

@data_extraction_bp.route('/api/extract/pdf/reactionlens', methods=['POST'])
def extract_pdf_reactionlens():
    """
    Extract chemical reactions from PDF using ReactionLens (sync).
    
    ReactionLens provides specialized chemical reaction extraction with
    optional filtering, segmentation, and entity resolution.
    
    Form data:
    - file: PDF file
    - provider: LLM provider (default: deepseek)
    - api_key: Your API key
    - model: Model to use (optional)
    - max_pages: Maximum pages to process (default: 50)
    - enable_filtering: Enable smart filtering of reactions (default: true)
    - enable_segmentation: Enable reaction segmentation (default: true)
    - enable_entity_resolution: Enable chemical entity resolution (default: true)
    """
    tmp_path = None
    try:
        tmp_path, filename = validate_pdf_upload()

        provider = request.form.get('provider', 'deepseek')
        api_key = request.form.get('api_key')
        model = get_model_for_provider(provider, request.form.get('model'))
        max_pages = int(request.form.get('max_pages', 50))
        enable_filtering = request.form.get('enable_filtering', 'true').lower() == 'true'
        enable_segmentation = request.form.get('enable_segmentation', 'true').lower() == 'true'
        enable_entity_resolution = request.form.get('enable_entity_resolution', 'true').lower() == 'true'

        logger.info(f"[ReactionLens] Starting extraction for {filename}")
        logger.info(f"[ReactionLens] Provider: {provider}, Model: {model}, Max pages: {max_pages}")
        logger.info(f"[ReactionLens] Filtering: {enable_filtering}, Segmentation: {enable_segmentation}, Entity resolution: {enable_entity_resolution}")

        # Try the convenience function first, fall back to class-based approach
        try:
            from modules.reaction_lens import extract_with_reactionlens
            result = extract_with_reactionlens(
                tmp_path,
                provider=provider,
                api_key=api_key,
                model=model,
                max_pages=max_pages,
                enable_filtering=enable_filtering,
                enable_segmentation=enable_segmentation,
                enable_entity_resolution=enable_entity_resolution
            )
        except (ImportError, AttributeError):
            from modules.reaction_lens import ReactionLens
            lens = ReactionLens(
                provider=provider,
                api_key=api_key,
                model=model,
                max_pages=max_pages,
                enable_filtering=enable_filtering,
                enable_segmentation=enable_segmentation,
                enable_entity_resolution=enable_entity_resolution
            )
            result = lens.extract_from_pdf(tmp_path)

        summary = {
            "reactions_found": len(result.get("reactions", [])),
            "filtering_applied": enable_filtering,
            "segmentation_applied": enable_segmentation,
            "entity_resolution_applied": enable_entity_resolution
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
        return jsonify({"success": False, "error": f"ReactionLens module not available: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"ReactionLens extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if tmp_path:
            cleanup_temp_file(tmp_path)


# ─── Format Endpoints ─────────────────────────────────────────────────────────

@data_extraction_bp.route('/api/extract/format/schemes', methods=['POST'])
def format_reaction_schemes_endpoint():
    """
    Format extraction result data into SMILES reaction schemes.
    
    Request body (JSON):
    {
        "reactions": [...],       // Reaction data from extraction results
        "compounds": [...],       // Optional compound data
        "format": "smiles"        // Output format (default: "smiles")
    }
    
    Returns formatted SMILES reaction schemes from extraction results.
    """
    from modules.chemextract_extractor import format_reaction_schemes

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        reactions = data.get('reactions', [])
        compounds = data.get('compounds', [])
        output_format = data.get('format', 'smiles')

        if not reactions and not compounds:
            raise ValidationError("No reaction or compound data provided for formatting")

        logger.info(f"[Format Schemes] Formatting {len(reactions)} reactions, {len(compounds)} compounds as {output_format}")

        result = format_reaction_schemes(
            reactions=reactions,
            compounds=compounds,
            output_format=output_format
        )

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


# ─── Helper Functions ─────────────────────────────────────────────────────────

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
