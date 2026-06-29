"""
POST /api/extract         — sync text extraction
POST /api/extract/async   — async text extraction

Both endpoints accept a JSON body with:
  - text:       the text to extract chemical entities from
  - provider:   'deepseek' | 'openai' | 'gemini' | 'anthropic' | ...
  - api_key:    the user's LLM provider API key
  - model:      (optional) model override

The text is sanitized via core.utils.sanitize_input with the larger
config.MAX_EXTRACTION_TEXT_LENGTH limit (15000 chars by default),
then dispatched to modules.chemextract.standalone.call_text_llm[_async].
"""

import logging

from flask import request, jsonify

from core.config import config
from core.errors import ValidationError
from core.utils import sanitize_input

from ._helpers import data_extraction_bp, get_model_for_provider

logger = logging.getLogger(__name__)


def _parse_text_request(data):
    """Extract and validate the common (text, provider, api_key, model)
    fields from a JSON request body.

    Returns (text, provider, api_key, model). Raises ValidationError if
    the body is missing or text is empty after sanitization.
    """
    if not data:
        raise ValidationError("No JSON data provided")

    text = data.get('text', '')
    provider = data.get('provider', 'deepseek')
    api_key = data.get('api_key')
    model = get_model_for_provider(provider, data.get('model'))

    text = sanitize_input(text, max_length=config.MAX_EXTRACTION_TEXT_LENGTH)
    if not text:
        raise ValidationError("Text cannot be empty", field="text")

    return text, provider, api_key, model


@data_extraction_bp.route('/extract', methods=['POST'])
def extract_from_text():
    """Synchronous text extraction."""
    from modules.chemextract.standalone import call_text_llm

    try:
        text, provider, api_key, model = _parse_text_request(request.get_json())
        logger.info(
            f"[Data Extraction] Provider: {provider}, Model: {model}, "
            f"Text length: {len(text)}"
        )

        result = call_text_llm(text, provider, model, api_key)

        if result:
            total_items = sum(
                len(result.get(k, []))
                for k in ['reactants', 'products', 'catalysts', 'ligands', 'solvents', 'mechanisms']
            )
            if total_items == 0 and not result.get('conditions') and not result.get('yields'):
                return jsonify({
                    "success": True,
                    "data": result,
                    "warning": "No chemical reaction data was found in the provided text.",
                    "model_used": model,
                    "provider_used": provider,
                })

            return jsonify({
                "success": True,
                "data": result,
                "model_used": model,
                "provider_used": provider,
            })
        else:
            return jsonify({
                "success": False,
                "error": "No response from LLM. Please check your API key and provider settings."
            }), 500

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Data extraction error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@data_extraction_bp.route('/extract/async', methods=['POST'])
async def extract_from_text_async():
    """Asynchronous text extraction."""
    from modules.chemextract.standalone import call_text_llm_async

    try:
        text, provider, api_key, model = _parse_text_request(request.get_json())
        logger.info(
            f"[Data Extraction Async] Provider: {provider}, Model: {model}, "
            f"Text length: {len(text)}"
        )

        result = await call_text_llm_async(text, provider, model, api_key)

        if result:
            return jsonify({
                "success": True,
                "data": result,
                "model_used": model,
                "provider_used": provider,
                "async": True,
            })
        else:
            return jsonify({
                "success": False,
                "error": "No response from LLM."
            }), 500

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Data extraction async error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
