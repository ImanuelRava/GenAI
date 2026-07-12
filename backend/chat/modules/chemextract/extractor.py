"""
ChemExtract AI — main extraction orchestrator (slim).

Pipeline (vision-first):
  1. Extract embedded figures from PDF (Tier 1: raster images with size filtering)
  2. Detect & render scheme/figure pages (Tier 2: vector-drawn reaction schemes)
  3. Analyze all visual content with vision LLM (using focused figure prompts)
  4. Extract text from PDF
  5. Analyze text with text LLM (chunked)
  6. Merge vision + text results, post-process, deduplicate

Pre-decomposition history: this file was 695 LOC with a single ChemExtractAI
class containing ~20 methods. It has been split into three focused modules:

  - ``vision_pipeline.py``  — VisionPipelineMixin (Tier 1 + Tier 2 + fallback)
  - ``text_pipeline.py``    — TextPipelineMixin (PDF text + chunked LLM)
  - ``post_process.py``     — PostProcessMixin (merge + dedup + normalize)
                               + module-level pure functions for unit testing

ChemExtractAI inherits from all three mixins, so its public API
(``extract_from_pdf``, ``extract_from_pdf_async``) is unchanged. The
top-level convenience wrappers ``extract_chemical_data_from_pdf[_async]``
also keep their original signatures.
"""

import logging

from .cache import _get_cached_result, _save_cached_result
from .vision_pipeline import VisionPipelineMixin
from .text_pipeline import TextPipelineMixin
from .post_process import PostProcessMixin

logger = logging.getLogger(__name__)


class ChemExtractAI(VisionPipelineMixin, TextPipelineMixin, PostProcessMixin):
    """Top-level orchestrator for the ChemExtract AI extraction pipeline.

    Inherits the vision, text, and post-process methods from three mixins.
    This class itself only defines:
      - __init__ (provider/key/model config + default model resolution)
      - extract_from_pdf / extract_from_pdf_async (the two public entry points)
      - _make_empty_result (helper used by both entry points)

    All actual extraction / merging / cleanup work is delegated to the
    mixin methods.
    """

    # Single source of truth for which providers support vision input.
    # (Also defined in llm.client.VISION_CAPABLE_PROVIDERS, but kept here
    #  to preserve the original public attribute on ChemExtractAI instances.)
    VISION_PROVIDERS = ['deepseek', 'openai', 'gemini', 'anthropic']

    # Default model per vision-capable provider. Used when the caller
    # doesn't pass an explicit model. Note: these defaults differ slightly
    # from llm.client.PROVIDER_DEFAULT_MODELS (e.g. 'gpt-4o' here vs
    # 'gpt-4o-mini' there) — preserved for backwards compatibility.
    DEFAULT_MODELS = {
        'deepseek': 'deepseek-chat',
        'openai': 'gpt-4o',
        'gemini': 'gemini-2.0-flash',
        'anthropic': 'claude-3-5-sonnet-20241022',
    }

    def __init__(self, llm_provider='deepseek', api_key=None, model=None):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        # Expose the capability lists as instance attributes for backwards
        # compat (some tests / callers inspect chemextract.vision_providers).
        self.vision_providers = list(self.VISION_PROVIDERS)
        self.default_models = dict(self.DEFAULT_MODELS)
        if not self.model:
            self.model = self.default_models.get(llm_provider)

    # ------------------------------------------------------------------
    # Public sync entry point
    # ------------------------------------------------------------------

    def extract_from_pdf(
        self, pdf_path, extract_images=True, extract_text=True, max_pages=50,
    ):
        """Run the full ChemExtract AI pipeline on a PDF.

        Pipeline order:
          1. Check cache — if we've extracted this PDF before with the same
             model + provider, return the cached result immediately.
          2. Vision extraction (if extract_images and provider supports vision)
             — populates result['reactions'], ['compounds'], ['figures'], ['tables'].
          3. Text extraction (if extract_text) — OVERWRITES the vision-extracted
             reactions + compounds if text data is found (text is more reliable
             for reaction data when the PDF has extractable text).
          4. Post-process — clean pseudo-SMILES, deduplicate, normalize, assemble
             R-group reactions.
          5. Save to cache for future calls.

        Args:
            pdf_path: absolute path to the PDF file on disk.
            extract_images: if True (default), run the vision pipeline.
            extract_text: if True (default), run the text pipeline.
            max_pages: cap on the number of pages to process (default 50).

        Returns:
            The result dict with keys: reactions, compounds, figures, tables,
            text_content, metadata.
        """
        cached = _get_cached_result(pdf_path, self.model or "", self.llm_provider)
        if cached is not None:
            return cached

        result = self._make_empty_result()

        # STEP 1: Vision extraction (runs FIRST).
        if extract_images and self.llm_provider in self.vision_providers:
            self._run_vision_pipeline(result, pdf_path, max_pages)

        # STEP 2: Text extraction (may overwrite vision results).
        if extract_text:
            self._run_text_pipeline(result, pdf_path)

        # STEP 3: Post-processing.
        self._post_process(result)

        _save_cached_result(result, pdf_path, self.model or "", self.llm_provider)
        return result

    # ------------------------------------------------------------------
    # Public async entry point
    # ------------------------------------------------------------------

    async def extract_from_pdf_async(
        self, pdf_path, extract_images=True, extract_text=True, max_pages=50,
    ):
        """Async version of ``extract_from_pdf``.

        The vision pipeline uses ``asyncio.gather`` to analyze all figures
        + scheme pages concurrently. The text pipeline awaits the chunked
        text LLM call. Post-processing is synchronous (CPU-bound, fast).
        """
        cached = _get_cached_result(pdf_path, self.model or "", self.llm_provider)
        if cached is not None:
            return cached

        result = self._make_empty_result()

        if extract_images and self.llm_provider in self.vision_providers:
            await self._run_vision_pipeline_async(result, pdf_path, max_pages)

        if extract_text:
            await self._run_text_pipeline_async(result, pdf_path)

        self._post_process(result)

        _save_cached_result(result, pdf_path, self.model or "", self.llm_provider)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_empty_result(self) -> dict:
        """Build the empty result dict that the pipeline will populate.

        Centralized here so the sync and async entry points produce
        identical initial shapes — important for cache consistency.
        """
        return {
            "reactions": [],
            "compounds": [],
            "figures": [],
            "tables": [],
            "text_content": "",
            "metadata": {
                "pages_processed": 0,
                "extraction_method": "chemextract_ai",
                "provider": self.llm_provider,
                "model": self.model,
                "text_extracted": False,
                "images_extracted": False,
                "embedded_figures_found": 0,
                "scheme_pages_found": 0,
                "vision_mode": "figure_extraction",
            },
        }


# ---------------------------------------------------------------------------
# Convenience wrappers — preserved verbatim for backwards compatibility.
# Callers (routes/data_extraction/chemextract_endpoints.py, chemextract/__init__.py)
# import these directly.
# ---------------------------------------------------------------------------

def extract_chemical_data_from_pdf(
    pdf_path, llm_provider='deepseek', api_key=None, model=None,
    max_pages=50, extract_images=True, extract_text=True,
):
    """Construct a ChemExtractAI instance and run the sync pipeline.

    Equivalent to::
        extractor = ChemExtractAI(llm_provider=llm_provider, api_key=api_key, model=model)
        return extractor.extract_from_pdf(pdf_path, ...)
    """
    extractor = ChemExtractAI(llm_provider=llm_provider, api_key=api_key, model=model)
    return extractor.extract_from_pdf(
        pdf_path,
        extract_images=extract_images,
        extract_text=extract_text,
        max_pages=max_pages,
    )


async def extract_chemical_data_from_pdf_async(
    pdf_path, llm_provider='deepseek', api_key=None, model=None,
    max_pages=50, extract_images=True, extract_text=True,
):
    """Construct a ChemExtractAI instance and run the async pipeline."""
    extractor = ChemExtractAI(llm_provider=llm_provider, api_key=api_key, model=model)
    return await extractor.extract_from_pdf_async(
        pdf_path,
        extract_images=extract_images,
        extract_text=extract_text,
        max_pages=max_pages,
    )
