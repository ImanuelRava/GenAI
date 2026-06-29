"""
POST /api/extract/format/schemes — format an extraction result as SMILES schemes.

Accepts a JSON body containing a prior extraction result (with
``reactions`` and/or ``compounds`` keys) and returns formatted
reaction schemes in SMILES notation (``reactants>>products``).

This endpoint does NOT call any LLM — it's a pure post-processing
utility that the frontend calls after one of the /extract/pdf/* endpoints
to get a renderable reaction-scheme list.
"""

import logging

from flask import request, jsonify

from core.errors import ValidationError

from ._helpers import data_extraction_bp

logger = logging.getLogger(__name__)


@data_extraction_bp.route('/extract/format/schemes', methods=['POST'])
def format_reaction_schemes_endpoint():
    """Format an extraction result dict into SMILES reaction schemes."""
    from modules.chemextract.reaction_formatter import format_reaction_schemes

    try:
        data = request.get_json()
        if not data:
            raise ValidationError("No JSON data provided")

        extraction_result = data
        output_format = data.get('format', 'smiles')

        if not data.get('reactions') and not data.get('compounds'):
            raise ValidationError(
                "No reaction or compound data provided for formatting"
            )

        logger.info(
            f"[Format Schemes] Formatting extraction result as {output_format}"
        )

        result = format_reaction_schemes(extraction_result)

        return jsonify({
            "success": True,
            "data": result,
            "format": output_format,
            "reactions_formatted": len(result.get("schemes", [])),
        })

    except ValidationError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"Format module not available: {str(e)}",
        }), 500
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(f"Format reaction schemes error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
