"""
Vision pipeline mixin for ChemExtractAI.

Two-tier vision extraction with full-page fallback:

  Tier 1 — Extract embedded raster figures from the PDF (filtered by size
           and quality). Each figure is analyzed with the focused
           ``SYSTEM_PROMPT_FIGURE_ANALYSIS`` prompt that includes a
           heuristic figure-type hint (spectrum / scheme / table / etc.).

  Tier 2 — Detect pages that contain vector-drawn reaction schemes
           (caption-keyword + drawing-density heuristic) and render them
           at 200 DPI. Each rendered page is analyzed with the
           comprehensive ``SYSTEM_PROMPT_VISION`` prompt.

  Fallback — If neither tier finds any visual content (e.g. scanned PDFs
             or all-vector papers that scheme detection missed), render
             every page at 150 DPI and analyze each as a full-page image.

The async variant uses ``asyncio.gather`` to analyze all figures + scheme
pages concurrently — dramatically faster on multi-page PDFs.

This module is a mixin: it expects the host class (``ChemExtractAI``) to
define ``self.llm_provider``, ``self.api_key``, ``self.model``,
``self.vision_providers``, and the result-merging methods
``_merge_figure_result`` / ``_merge_vision_results`` (which live in
``post_process.py``).
"""

import asyncio
import logging
from typing import Dict, Optional

from .pdf_processor import (
    pdf_to_images,
    extract_figures_from_pdf,
    render_scheme_pages,
)
from .llm_providers import call_vision_llm, call_vision_llm_async
from .prompts import (
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_FIGURE_ANALYSIS,
    SYSTEM_PROMPT_REACTION_SCHEME,
)

logger = logging.getLogger(__name__)


class VisionPipelineMixin:
    """Vision extraction pipeline methods, mixed into ``ChemExtractAI``.

    All methods access ``self.llm_provider``, ``self.api_key``,
    ``self.model``, and the result-merging helpers from
    ``PostProcessMixin``. The mixin itself defines no state.
    """

    # ------------------------------------------------------------------
    # Top-level pipeline entry points
    # ------------------------------------------------------------------

    def _run_vision_pipeline(self, result: dict, pdf_path: str, max_pages: int):
        """Synchronous two-tier vision extraction: embedded figures + scheme pages.

        Updates ``result`` in place:
          - result["metadata"]["embedded_figures_found"]
          - result["metadata"]["scheme_pages_found"]
          - result["metadata"]["pages_processed"]
          - result["metadata"]["images_extracted"]  (set True if any image yields data)
          - result["reactions"], result["compounds"], result["figures"], result["tables"]
            are populated via _merge_figure_result / _merge_vision_results.
        """
        try:
            # Tier 1: Extract embedded figures (filtered by size).
            embedded_figures = extract_figures_from_pdf(pdf_path)
            result["metadata"]["embedded_figures_found"] = len(embedded_figures)
            logger.info(
                f"[ChemExtract] Found {len(embedded_figures)} embedded figures"
            )

            # Tier 2: Detect & render scheme pages (vector-drawn reaction schemes).
            scheme_pages = render_scheme_pages(pdf_path, dpi=200, max_pages=max_pages)
            result["metadata"]["scheme_pages_found"] = len(scheme_pages)
            logger.info(
                f"[ChemExtract] Found {len(scheme_pages)} scheme pages to render"
            )

            total_visuals = len(embedded_figures) + len(scheme_pages)
            if total_visuals == 0:
                logger.info(
                    "[ChemExtract] No visual content found, falling back to "
                    "full-page rendering"
                )
                self._fallback_full_page_vision(result, pdf_path, max_pages)
                return

            # Analyze embedded figures with focused figure prompt.
            for fig in embedded_figures:
                vision_data = self._analyze_embedded_figure(fig)
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_figure_result(
                        result, vision_data, fig.get("page", 0), source="embedded"
                    )

            # Analyze scheme pages with comprehensive vision prompt.
            for sp in scheme_pages:
                vision_data = self._analyze_scheme_page(sp)
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_vision_results(result, vision_data, sp["page"])

            result["metadata"]["pages_processed"] = total_visuals
            logger.info(
                f"[ChemExtract] Vision complete: {len(embedded_figures)} figures + "
                f"{len(scheme_pages)} scheme pages analyzed"
            )

        except (ImportError, OSError, RuntimeError, ValueError) as e:
            # ImportError: PyMuPDF not installed. OSError/RuntimeError/
            # ValueError: PDF parse or rendering failure. Log and continue —
            # the text pipeline may still extract useful data.
            logger.error(f"[ChemExtract] Vision pipeline failed: {e}")

    async def _run_vision_pipeline_async(self, result: dict, pdf_path: str, max_pages: int):
        """Async two-tier vision extraction with concurrent figure + scheme analysis."""
        try:
            embedded_figures = extract_figures_from_pdf(pdf_path)
            result["metadata"]["embedded_figures_found"] = len(embedded_figures)

            scheme_pages = render_scheme_pages(pdf_path, dpi=200, max_pages=max_pages)
            result["metadata"]["scheme_pages_found"] = len(scheme_pages)

            total_visuals = len(embedded_figures) + len(scheme_pages)
            if total_visuals == 0:
                logger.info(
                    "[ChemExtract] No visual content found, falling back to "
                    "full-page rendering"
                )
                self._fallback_full_page_vision(result, pdf_path, max_pages)
                return

            # Concurrent analysis of all embedded figures.
            async def analyze_fig(fig):
                return await self._analyze_embedded_figure_async(fig)

            fig_tasks = [analyze_fig(fig) for fig in embedded_figures]
            fig_results = await asyncio.gather(*fig_tasks)

            for fig, vision_data in zip(embedded_figures, fig_results):
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_figure_result(
                        result, vision_data, fig.get("page", 0), source="embedded"
                    )

            # Concurrent analysis of all scheme pages.
            async def analyze_sp(sp):
                return await self._analyze_scheme_page_async(sp)

            sp_tasks = [analyze_sp(sp) for sp in scheme_pages]
            sp_results = await asyncio.gather(*sp_tasks)

            for sp, vision_data in zip(scheme_pages, sp_results):
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_vision_results(result, vision_data, sp["page"])

            result["metadata"]["pages_processed"] = total_visuals
            logger.info(
                f"[ChemExtract] Async vision complete: analyzed {total_visuals} items"
            )

        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.error(f"[ChemExtract] Async vision pipeline failed: {e}")

    # ------------------------------------------------------------------
    # Full-page fallback
    # ------------------------------------------------------------------

    def _fallback_full_page_vision(self, result: dict, pdf_path: str, max_pages: int):
        """Render every page as an image and analyze each.

        Used when neither Tier 1 (embedded figures) nor Tier 2 (scheme
        pages) finds any visual content. Handles edge cases like scanned
        PDFs or papers with all-vector content that scheme-page detection
        missed.
        """
        logger.info("[ChemExtract] Using full-page rendering fallback")
        page_images = pdf_to_images(pdf_path, dpi=150, max_pages=max_pages)
        result["metadata"]["pages_processed"] = len(page_images)
        for page_num, base64_img in page_images:
            vision_data = self._extract_from_image(
                base64_img, extraction_type="comprehensive", page_number=page_num,
            )
            if vision_data:
                result["metadata"]["images_extracted"] = True
                self._merge_vision_results(result, vision_data, page_num)

    # ------------------------------------------------------------------
    # Individual vision analysis methods
    # ------------------------------------------------------------------

    def _analyze_embedded_figure(self, fig: dict) -> Optional[Dict]:
        """Analyze a single embedded figure with the focused figure prompt.

        The prompt includes a heuristic figure-type hint (spectrum, scheme,
        table, etc.) to help the LLM interpret the image correctly.
        """
        b64 = fig.get("base64", "")
        if not b64:
            return None
        page = fig.get("page", 0)
        idx = fig.get("index", 0)
        fig_type_hint = self._guess_figure_type(fig)
        user_msg = (
            f"Analyze this extracted figure from page {page} (image #{idx + 1}) "
            f"of a chemistry research paper. "
            f"This appears to be a: {fig_type_hint}. "
            f"Extract all chemical reaction data visible in this figure."
        )
        return call_vision_llm(
            b64, self.llm_provider, self.model, self.api_key,
            SYSTEM_PROMPT_FIGURE_ANALYSIS, user_msg,
        )

    async def _analyze_embedded_figure_async(self, fig: dict) -> Optional[Dict]:
        """Async analysis of a single embedded figure."""
        b64 = fig.get("base64", "")
        if not b64:
            return None
        page = fig.get("page", 0)
        idx = fig.get("index", 0)
        fig_type_hint = self._guess_figure_type(fig)
        user_msg = (
            f"Analyze this extracted figure from page {page} (image #{idx + 1}) "
            f"of a chemistry research paper. "
            f"This appears to be a: {fig_type_hint}. "
            f"Extract all chemical reaction data visible in this figure."
        )
        return await call_vision_llm_async(
            b64, self.llm_provider, self.model, self.api_key,
            SYSTEM_PROMPT_FIGURE_ANALYSIS, user_msg,
        )

    def _analyze_scheme_page(self, sp: dict) -> Optional[Dict]:
        """Analyze a rendered scheme page with the comprehensive vision prompt."""
        b64 = sp.get("base64", "")
        if not b64:
            return None
        page = sp.get("page", 0)
        user_msg = (
            f"Analyze this page (page {page}) from a chemistry research paper. "
            f"This page was identified as containing reaction schemes or figures. "
            f"Extract ALL chemical reaction data visible on this page."
        )
        return call_vision_llm(
            b64, self.llm_provider, self.model, self.api_key,
            SYSTEM_PROMPT_VISION, user_msg,
        )

    async def _analyze_scheme_page_async(self, sp: dict) -> Optional[Dict]:
        """Async analysis of a rendered scheme page."""
        b64 = sp.get("base64", "")
        if not b64:
            return None
        page = sp.get("page", 0)
        user_msg = (
            f"Analyze this page (page {page}) from a chemistry research paper. "
            f"This page was identified as containing reaction schemes or figures. "
            f"Extract ALL chemical reaction data visible on this page."
        )
        return await call_vision_llm_async(
            b64, self.llm_provider, self.model, self.api_key,
            SYSTEM_PROMPT_VISION, user_msg,
        )

    # ------------------------------------------------------------------
    # Legacy image analysis (for fallback full-page mode)
    # ------------------------------------------------------------------

    def _extract_from_image(
        self, base64_image, extraction_type="comprehensive", page_number=0,
    ):
        """Analyze a single page image with the comprehensive or reaction-scheme prompt.

        Used by _fallback_full_page_vision. The ``extraction_type`` arg
        selects between SYSTEM_PROMPT_VISION ('comprehensive') and
        SYSTEM_PROMPT_REACTION_SCHEME (anything else).
        """
        system_prompt = (
            SYSTEM_PROMPT_VISION if extraction_type == "comprehensive"
            else SYSTEM_PROMPT_REACTION_SCHEME
        )
        user_message = (
            f"Analyze this image (page {page_number}) and extract all "
            f"chemical reaction data:"
        )
        return call_vision_llm(
            base64_image, self.llm_provider, self.model, self.api_key,
            system_prompt, user_message,
        )

    async def _extract_from_image_async(
        self, base64_image, extraction_type="comprehensive", page_number=0,
    ):
        """Async version of _extract_from_image."""
        system_prompt = (
            SYSTEM_PROMPT_VISION if extraction_type == "comprehensive"
            else SYSTEM_PROMPT_REACTION_SCHEME
        )
        user_message = (
            f"Analyze this image (page {page_number}) and extract all "
            f"chemical reaction data:"
        )
        return await call_vision_llm_async(
            base64_image, self.llm_provider, self.model, self.api_key,
            system_prompt, user_message,
        )

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_figure_type(fig: dict) -> str:
        """Heuristic guess of figure type from width/height metadata.

        Helps the vision LLM interpret the image by hinting whether it's
        likely a spectrum (very wide/narrow), a full-page scheme (large),
        a molecular structure icon (small), a square graph, or a generic
        rectangular figure.
        """
        w = fig.get("width", 0)
        h = fig.get("height", 0)
        aspect = w / max(h, 1)

        if aspect > 3.0 or aspect < 0.33:
            return "wide/narrow figure (possibly a spectrum, chromatogram, or horizontal scheme)"
        if w > 1500 or h > 1500:
            return "large figure (possibly a full-width scheme or data table)"
        if w < 300 and h < 300:
            return "small image (possibly a molecular structure icon)"
        if 0.8 < aspect < 1.2:
            return "square-ish figure (possibly a molecular structure, crystal structure, or graph)"
        return "rectangular figure (possibly a reaction scheme, chart, or table)"
