"""
Data Extraction blueprint package.

Replaces the monolithic ``routes/data_extraction.py`` (887 LOC) with a
package of focused endpoint modules. Each endpoint module registers its
routes on the shared ``data_extraction_bp`` blueprint defined in
``_helpers.py``, so callers that do::

    from routes.data_extraction import data_extraction_bp
    app.register_blueprint(data_extraction_bp)

continue to work unchanged.

Module layout
-------------
  _helpers.py                  — Blueprint, constants, shared helpers
                                 (validate_pdf_upload, merge_extraction_results, etc.)
  models_endpoint.py           — GET  /extract/models
  text_endpoints.py            — POST /extract, /extract/async
  pdf_text_endpoint.py         — POST /extract/pdf
  pdf_vision_endpoints.py      — POST /extract/pdf/vision, /extract/pdf/vision/async
  chemextract_endpoints.py     — POST /extract/pdf/chemextract, /extract/pdf/chemextract/async
  reactionlens_endpoints.py    — POST /extract/pdf/reactionlens[/async][/text]
  format_schemes_endpoint.py   — POST /extract/format/schemes

Importing this package triggers all endpoint modules to register their
routes on the blueprint (the imports below are intentionally side-effectful).
"""

# The blueprint is defined in _helpers and re-exported here for the
# conventional `from routes.data_extraction import data_extraction_bp` import.
from ._helpers import data_extraction_bp

# Importing the endpoint modules registers their routes on the blueprint
# as a side effect. The imports are intentionally not assigned to names.
from . import (  # noqa: F401  (side-effect imports)
    models_endpoint,
    text_endpoints,
    pdf_text_endpoint,
    pdf_vision_endpoints,
    chemextract_endpoints,
    reactionlens_endpoints,
    format_schemes_endpoint,
)

__all__ = ['data_extraction_bp']
