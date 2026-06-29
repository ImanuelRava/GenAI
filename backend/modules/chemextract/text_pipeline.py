"""
Text pipeline mixin for ChemExtractAI.

Extracts raw text from the PDF using the cascading extractor in
``pdf_processor.extract_text_from_pdf`` (chemextract → pypdf → pdfplumber
→ PyMuPDF), then dispatches it to ``standalone.call_text_llm_chunked[_async]``
which splits long text into paragraph-aware chunks and merges the
per-chunk LLM results.

The text pipeline runs AFTER the vision pipeline (see ``vision_pipeline.py``).
If the vision pipeline already populated ``result["reactions"]``, the text
pipeline OVERWRITES them — this is the original behavior, preserved for
backwards compatibility. The rationale is that text extraction is more
reliable than vision for reaction data when the PDF has extractable text.

This module is a mixin: it expects the host class (``ChemExtractAI``) to
define ``self.llm_provider``, ``self.api_key``, ``self.model``.
"""

import logging

from .pdf_processor import extract_text_from_pdf

logger = logging.getLogger(__name__)


class TextPipelineMixin:
    """Text extraction + chunked LLM analysis methods, mixed into ChemExtractAI."""

    def _run_text_pipeline(self, result: dict, pdf_path: str):
        """Synchronous text extraction + chunked LLM analysis.

        Updates ``result`` in place:
          - result["text_content"]
          - result["metadata"]["pages"], ["text_method"], ["text_extracted"]
          - result["reactions"], result["compounds"] (overwritten if text data found)
          - result["experimental_procedures"], ["characterization_data"]
          - result["scaffold_smiles"], ["rgroup_table"]  (if present in LLM output)
        """
        try:
            text, text_meta = extract_text_from_pdf(pdf_path)
            result["text_content"] = text
            result["metadata"]["pages"] = text_meta.get("pages", 0)
            result["metadata"]["text_method"] = text_meta.get("method")

            if text.strip():
                logger.info(
                    f"[ChemExtract] Analyzing text ({len(text)} chars)..."
                )
                # Lazy import to avoid circular dependency: standalone.py
                # imports from chemextract.llm_providers, which imports
                # from llm.client, which is fine — but keeping the import
                # local means we only pay the cost when text extraction
                # actually runs.
                from .standalone import call_text_llm_chunked

                text_data = call_text_llm_chunked(
                    text, self.llm_provider, self.model, self.api_key,
                )
                if text_data:
                    self._apply_text_data_to_result(result, text_data)
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
            # ImportError: no PDF text extraction library. OSError: corrupt
            # PDF. ValueError/RuntimeError: parse failures. KeyError/TypeError:
            # unexpected LLM response shape. Log and continue — the vision
            # pipeline may have already populated result.
            logger.error(f"[ChemExtract] Text extraction failed: {e}")

    async def _run_text_pipeline_async(self, result: dict, pdf_path: str):
        """Async text extraction + chunked LLM analysis."""
        try:
            text, text_meta = extract_text_from_pdf(pdf_path)
            result["text_content"] = text
            result["metadata"]["pages"] = text_meta.get("pages", 0)
            result["metadata"]["text_method"] = text_meta.get("method")

            if text.strip():
                logger.info(
                    f"[ChemExtract Async] Analyzing text ({len(text)} chars)..."
                )
                from .standalone import call_text_llm_chunked_async

                text_data = await call_text_llm_chunked_async(
                    text, self.llm_provider, self.model, self.api_key,
                )
                if text_data:
                    self._apply_text_data_to_result(result, text_data)
        except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
            logger.error(f"[ChemExtract Async] Text extraction failed: {e}")

    # ------------------------------------------------------------------
    # Shared result-application helper (used by both sync + async paths)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_text_data_to_result(result: dict, text_data: dict):
        """Copy the chunked text-LLM output into the top-level result dict.

        Note: this OVERWRITES result["reactions"] and result["compounds"]
        with the text-pipeline output. The original code had this behavior
        (text pipeline runs after vision and replaces its results) — we
        preserve it for backwards compat. The post-process step that
        follows will deduplicate any compound entries that overlap.
        """
        result["metadata"]["text_extracted"] = True
        result["reactions"] = text_data.get("reactions", [])
        result["compounds"] = text_data.get("compounds", [])

        # Optional keys copied only if the LLM emitted them.
        for opt_key in ("experimental_procedures", "characterization_data"):
            if opt_key in text_data:
                result[opt_key] = text_data[opt_key]
        for opt_key in ("scaffold_smiles", "rgroup_table"):
            if opt_key in text_data:
                result[opt_key] = text_data[opt_key]
