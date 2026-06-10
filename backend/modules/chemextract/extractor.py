"""
ChemExtract AI — main extraction orchestrator.

Pipeline (vision-first):
  1. Extract embedded figures from PDF (Tier 1: raster images with size filtering)
  2. Detect & render scheme/figure pages (Tier 2: vector-drawn reaction schemes)
  3. Analyze all visual content with vision LLM (using focused figure prompts)
  4. Extract text from PDF
  5. Analyze text with text LLM (chunked)
  6. Merge vision + text results, post-process, deduplicate
"""

import logging
from typing import Dict, Any, List, Optional

from .config import TEXT_CHUNK_SIZE
from .pdf_processor import (
    extract_text_from_pdf,
    pdf_to_images,
    extract_figures_from_pdf,
    render_scheme_pages,
    detect_scheme_pages,
    extract_all_visual_content,
)
from .llm_providers import call_vision_llm, call_vision_llm_async
from .prompts import (
    SYSTEM_PROMPT_VISION,
    SYSTEM_PROMPT_FIGURE_ANALYSIS,
    SYSTEM_PROMPT_REACTION_SCHEME,
)
from .smiles_utils import _is_pseudo_smiles, assemble_rgroup_reactions
from .cache import _get_cached_result, _save_cached_result

logger = logging.getLogger(__name__)


class ChemExtractAI:
    def __init__(self, llm_provider='deepseek', api_key=None, model=None):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.model = model
        self.vision_providers = ['deepseek', 'openai', 'gemini', 'anthropic']
        self.default_models = {
            'deepseek': 'deepseek-chat',
            'openai': 'gpt-4o',
            'gemini': 'gemini-2.0-flash',
            'anthropic': 'claude-3-5-sonnet-20241022'
        }
        if not self.model:
            self.model = self.default_models.get(llm_provider)

    # ------------------------------------------------------------------
    # Sync pipeline
    # ------------------------------------------------------------------

    def extract_from_pdf(self, pdf_path, extract_images=True, extract_text=True,
                        max_pages=50):
        cached = _get_cached_result(pdf_path, self.model or "", self.llm_provider)
        if cached is not None:
            return cached

        result = {
            "reactions": [], "compounds": [], "figures": [], "tables": [],
            "text_content": "",
            "metadata": {
                "pages_processed": 0, "extraction_method": "chemextract_ai",
                "provider": self.llm_provider, "model": self.model,
                "text_extracted": False, "images_extracted": False,
                "embedded_figures_found": 0, "scheme_pages_found": 0,
                "vision_mode": "figure_extraction",
            }
        }

        # ── STEP 1: Vision extraction (runs FIRST) ──────────────────
        if extract_images and self.llm_provider in self.vision_providers:
            self._run_vision_pipeline(result, pdf_path, max_pages)

        # ── STEP 2: Text extraction ───────────────────────────────────
        if extract_text:
            self._run_text_pipeline(result, pdf_path)

        # ── STEP 3: Post-processing ─────────────────────────────────
        self._post_process(result)

        _save_cached_result(result, pdf_path, self.model or "", self.llm_provider)
        return result

    # ------------------------------------------------------------------
    # Async pipeline
    # ------------------------------------------------------------------

    async def extract_from_pdf_async(self, pdf_path, extract_images=True, extract_text=True,
                                      max_pages=50):
        import asyncio

        cached = _get_cached_result(pdf_path, self.model or "", self.llm_provider)
        if cached is not None:
            return cached

        result = {
            "reactions": [], "compounds": [], "figures": [], "tables": [],
            "text_content": "",
            "metadata": {
                "pages_processed": 0, "extraction_method": "chemextract_ai",
                "provider": self.llm_provider, "model": self.model,
                "text_extracted": False, "images_extracted": False,
                "embedded_figures_found": 0, "scheme_pages_found": 0,
                "vision_mode": "figure_extraction",
            }
        }

        # ── STEP 1: Vision extraction (runs FIRST) ──────────────────
        if extract_images and self.llm_provider in self.vision_providers:
            await self._run_vision_pipeline_async(result, pdf_path, max_pages)

        # ── STEP 2: Text extraction ───────────────────────────────────
        if extract_text:
            await self._run_text_pipeline_async(result, pdf_path)

        # ── STEP 3: Post-processing ─────────────────────────────────
        self._post_process(result)

        _save_cached_result(result, pdf_path, self.model or "", self.llm_provider)
        return result

    # ------------------------------------------------------------------
    # Vision pipeline helpers
    # ------------------------------------------------------------------

    def _run_vision_pipeline(self, result: dict, pdf_path: str, max_pages: int):
        """Two-tier vision extraction: embedded figures + scheme pages."""
        try:
            # Tier 1: Extract embedded figures (filtered by size)
            embedded_figures = extract_figures_from_pdf(pdf_path)
            result["metadata"]["embedded_figures_found"] = len(embedded_figures)
            logger.info(f"[ChemExtract] Found {len(embedded_figures)} embedded figures")

            # Tier 2: Detect & render scheme pages (vector-drawn reaction schemes)
            scheme_pages = render_scheme_pages(pdf_path, dpi=200, max_pages=max_pages)
            result["metadata"]["scheme_pages_found"] = len(scheme_pages)
            logger.info(f"[ChemExtract] Found {len(scheme_pages)} scheme pages to render")

            total_visuals = len(embedded_figures) + len(scheme_pages)
            if total_visuals == 0:
                logger.info("[ChemExtract] No visual content found, falling back to full-page rendering")
                self._fallback_full_page_vision(result, pdf_path, max_pages)
                return

            # Analyze embedded figures with focused figure prompt
            for fig in embedded_figures:
                vision_data = self._analyze_embedded_figure(fig)
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_figure_result(result, vision_data, fig.get("page", 0), source="embedded")

            # Analyze scheme pages with comprehensive vision prompt
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

        except Exception as e:
            logger.error(f"[ChemExtract] Vision pipeline failed: {e}")

    async def _run_vision_pipeline_async(self, result: dict, pdf_path: str, max_pages: int):
        """Async two-tier vision extraction."""
        import asyncio
        try:
            embedded_figures = extract_figures_from_pdf(pdf_path)
            result["metadata"]["embedded_figures_found"] = len(embedded_figures)

            scheme_pages = render_scheme_pages(pdf_path, dpi=200, max_pages=max_pages)
            result["metadata"]["scheme_pages_found"] = len(scheme_pages)

            total_visuals = len(embedded_figures) + len(scheme_pages)
            if total_visuals == 0:
                logger.info("[ChemExtract] No visual content found, falling back to full-page rendering")
                self._fallback_full_page_vision(result, pdf_path, max_pages)
                return

            # Concurrent analysis of all embedded figures
            async def analyze_fig(fig):
                return await self._analyze_embedded_figure_async(fig)

            fig_tasks = [analyze_fig(fig) for fig in embedded_figures]
            fig_results = await asyncio.gather(*fig_tasks)

            for fig, vision_data in zip(embedded_figures, fig_results):
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_figure_result(result, vision_data, fig.get("page", 0), source="embedded")

            # Concurrent analysis of all scheme pages
            async def analyze_sp(sp):
                return await self._analyze_scheme_page_async(sp)

            sp_tasks = [analyze_sp(sp) for sp in scheme_pages]
            sp_results = await asyncio.gather(*sp_tasks)

            for sp, vision_data in zip(scheme_pages, sp_results):
                if vision_data:
                    result["metadata"]["images_extracted"] = True
                    self._merge_vision_results(result, vision_data, sp["page"])

            result["metadata"]["pages_processed"] = total_visuals
            logger.info(f"[ChemExtract] Async vision complete: analyzed {total_visuals} items")

        except Exception as e:
            logger.error(f"[ChemExtract] Async vision pipeline failed: {e}")

    def _fallback_full_page_vision(self, result: dict, pdf_path: str, max_pages: int):
        """Fallback: render all pages as images when no figures/schemes detected.

        This handles edge cases like scanned PDFs or papers with all-vector content
        that scheme-page detection missed.
        """
        logger.info("[ChemExtract] Using full-page rendering fallback")
        page_images = pdf_to_images(pdf_path, dpi=150, max_pages=max_pages)
        result["metadata"]["pages_processed"] = len(page_images)
        for page_num, base64_img in page_images:
            vision_data = self._extract_from_image(
                base64_img, extraction_type="comprehensive", page_number=page_num
            )
            if vision_data:
                result["metadata"]["images_extracted"] = True
                self._merge_vision_results(result, vision_data, page_num)

    # ------------------------------------------------------------------
    # Individual vision analysis methods
    # ------------------------------------------------------------------

    def _analyze_embedded_figure(self, fig: dict) -> Optional[Dict]:
        """Analyze a single embedded figure with the focused figure prompt."""
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

    @staticmethod
    def _guess_figure_type(fig: dict) -> str:
        """Heuristic guess of figure type from metadata."""
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

    # ------------------------------------------------------------------
    # Legacy image analysis (for fallback full-page mode)
    # ------------------------------------------------------------------

    def _extract_from_image(self, base64_image, extraction_type="comprehensive", page_number=0):
        system_prompt = SYSTEM_PROMPT_VISION if extraction_type == "comprehensive" else SYSTEM_PROMPT_REACTION_SCHEME
        user_message = f"Analyze this image (page {page_number}) and extract all chemical reaction data:"
        return call_vision_llm(base64_image, self.llm_provider, self.model, self.api_key, system_prompt, user_message)

    async def _extract_from_image_async(self, base64_image, extraction_type="comprehensive", page_number=0):
        system_prompt = SYSTEM_PROMPT_VISION if extraction_type == "comprehensive" else SYSTEM_PROMPT_REACTION_SCHEME
        user_message = f"Analyze this image (page {page_number}) and extract all chemical reaction data:"
        return await call_vision_llm_async(base64_image, self.llm_provider, self.model, self.api_key, system_prompt, user_message)

    # ------------------------------------------------------------------
    # Text pipeline helpers
    # ------------------------------------------------------------------

    def _run_text_pipeline(self, result: dict, pdf_path: str):
        """Extract text from PDF and analyze with text LLM."""
        try:
            text, text_meta = extract_text_from_pdf(pdf_path)
            result["text_content"] = text
            result["metadata"]["pages"] = text_meta.get("pages", 0)
            result["metadata"]["text_method"] = text_meta.get("method")
            if text.strip():
                logger.info(f"[ChemExtract] Analyzing text ({len(text)} chars)...")
                from .standalone import call_text_llm_chunked
                text_data = call_text_llm_chunked(text, self.llm_provider, self.model, self.api_key)
                if text_data:
                    result["metadata"]["text_extracted"] = True
                    result["reactions"] = text_data.get("reactions", [])
                    result["compounds"] = text_data.get("compounds", [])
                    if "experimental_procedures" in text_data:
                        result["experimental_procedures"] = text_data["experimental_procedures"]
                    if "characterization_data" in text_data:
                        result["characterization_data"] = text_data["characterization_data"]
                    for key in ("scaffold_smiles", "rgroup_table"):
                        if key in text_data:
                            result[key] = text_data[key]
        except Exception as e:
            logger.error(f"[ChemExtract] Text extraction failed: {e}")

    async def _run_text_pipeline_async(self, result: dict, pdf_path: str):
        """Async text extraction and analysis."""
        try:
            text, text_meta = extract_text_from_pdf(pdf_path)
            result["text_content"] = text
            result["metadata"]["pages"] = text_meta.get("pages", 0)
            result["metadata"]["text_method"] = text_meta.get("method")
            if text.strip():
                logger.info(f"[ChemExtract Async] Analyzing text ({len(text)} chars)...")
                from .standalone import call_text_llm_chunked_async
                text_data = await call_text_llm_chunked_async(text, self.llm_provider, self.model, self.api_key)
                if text_data:
                    result["metadata"]["text_extracted"] = True
                    result["reactions"] = text_data.get("reactions", [])
                    result["compounds"] = text_data.get("compounds", [])
                    if "experimental_procedures" in text_data:
                        result["experimental_procedures"] = text_data["experimental_procedures"]
                    for key in ("scaffold_smiles", "rgroup_table"):
                        if key in text_data:
                            result[key] = text_data[key]
        except Exception as e:
            logger.error(f"[ChemExtract Async] Text extraction failed: {e}")

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    def _merge_figure_result(self, result: dict, vision_data: dict, page_num: int, source: str = "embedded"):
        """Merge results from an individual embedded figure analysis.

        Uses SYSTEM_PROMPT_FIGURE_ANALYSIS output format which has a
        top-level 'reaction_schemes' key (not flat reactants/products).
        """
        reaction_schemes = vision_data.get("reaction_schemes", [])
        if not reaction_schemes and (vision_data.get("reactants") or vision_data.get("products")):
            # Fallback: flat format (same as _merge_vision_results)
            self._merge_vision_results(result, vision_data, page_num)
            return

        for scheme in reaction_schemes:
            reaction = {
                "id": f"{source}_page{page_num}_{len(result['reactions'])+1}",
                "source": source, "page": page_num,
                "type": scheme.get("reactionType", "unknown"),
                "reactants": scheme.get("reactants", []),
                "products": scheme.get("products", []),
                "reagents": scheme.get("reagents", []),
                "catalyst": scheme.get("catalyst"),
                "ligand": scheme.get("ligand"),
                "conditions": scheme.get("conditions", {}),
                "yield": scheme.get("yield"),
                "notes": scheme.get("notes", ""),
                "entry": scheme.get("entry"),
                "entry_id": scheme.get("entry_id"),
                "rgroup_values": scheme.get("rgroup_values"),
                "assembled_smiles": scheme.get("assembled_smiles"),
            }
            result["reactions"].append(reaction)

        # Handle table_data from figure analysis
        table_data = vision_data.get("table_data", [])
        if table_data:
            result["tables"].append({
                "page": page_num, "source": source,
                "data": table_data,
                "columns": vision_data.get("table_columns", []),
            })

        # Extract compounds
        existing_names = {c.get("name", "").lower() for c in result.get("compounds", []) if c.get("name")}
        for scheme in reaction_schemes:
            for role_key in ("reactants", "products"):
                for entity in scheme.get(role_key, []):
                    name = entity.get("name", "") if isinstance(entity, dict) else str(entity)
                    if name and name.lower() not in existing_names:
                        result["compounds"].append({
                            "name": name,
                            "smiles": entity.get("smiles") if isinstance(entity, dict) else None,
                            "role": "reactant" if role_key == "reactants" else "product",
                            "source": source,
                        })
                        existing_names.add(name.lower())

        # Also merge compounds from top-level "compounds" in figure output
        for comp in vision_data.get("compounds", []):
            name = comp.get("name", "")
            if name and name.lower() not in existing_names:
                result["compounds"].append({
                    "name": name,
                    "smiles": comp.get("smiles"),
                    "formula": comp.get("formula"),
                    "role": comp.get("role", "unknown"),
                    "source": source,
                })
                existing_names.add(name.lower())

        # Record figure metadata
        fig_type = vision_data.get("figure_type", "unknown")
        description = vision_data.get("description", "")
        result["figures"].append({
            "page": page_num, "type": fig_type,
            "source": source, "description": description,
            "notes": vision_data.get("notes", ""),
        })

        # Pull scaffold/rgroup data from figure analysis
        for key in ("scaffold_smiles", "rgroup_table", "rgroup_attachment_map"):
            if key in vision_data and key not in result:
                result[key] = vision_data[key]

    def _merge_vision_results(self, result, vision_data, page_num):
        """Merge results from full-page or scheme-page vision analysis (legacy format)."""
        reaction_schemes = vision_data.get("reaction_schemes", [])
        if not reaction_schemes and (vision_data.get("reactants") or vision_data.get("products")):
            flat_reactants = vision_data.get("reactants", [])
            flat_products = vision_data.get("products", [])
            if isinstance(flat_reactants, list) and isinstance(flat_products, list):
                reaction_schemes = [{
                    "entry": 1,
                    "reactants": [{"name": r, "smiles": None} if isinstance(r, str) else r for r in flat_reactants],
                    "products": [{"name": p, "smiles": None} if isinstance(p, str) else p for p in flat_products],
                    "reagents": vision_data.get("reagents", []),
                    "conditions": vision_data.get("conditions", {}),
                    "yield": None,
                    "catalyst": vision_data.get("catalysts", [None])[0] if vision_data.get("catalysts") else None,
                    "ligand": vision_data.get("ligands", [None])[0] if vision_data.get("ligands") else None,
                }]
        for scheme in reaction_schemes:
            reaction = {
                "id": f"vision_page{page_num}_{len(result['reactions'])+1}",
                "source": "vision", "page": page_num,
                "type": scheme.get("reactionType", "unknown"),
                "reactants": scheme.get("reactants", []),
                "products": scheme.get("products", []),
                "reagents": scheme.get("reagents", []),
                "catalyst": scheme.get("catalyst"),
                "ligand": scheme.get("ligand"),
                "conditions": scheme.get("conditions", {}),
                "yield": scheme.get("yield"),
                "notes": scheme.get("notes", "")
            }
            result["reactions"].append(reaction)
        if vision_data.get("table_data"):
            result["tables"].append({"page": page_num, "source": "vision", "data": vision_data.get("table_data", []), "columns": vision_data.get("table_columns", [])})
        existing_names = {c.get("name", "").lower() for c in result.get("compounds", []) if c.get("name")}
        for scheme in reaction_schemes:
            for reactant in scheme.get("reactants", []):
                name = reactant.get("name", "") if isinstance(reactant, dict) else str(reactant)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({"name": name, "smiles": reactant.get("smiles") if isinstance(reactant, dict) else None, "role": "reactant", "source": "vision"})
                    existing_names.add(name.lower())
            for product in scheme.get("products", []):
                name = product.get("name", "") if isinstance(product, dict) else str(product)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({"name": name, "smiles": product.get("smiles") if isinstance(product, dict) else None, "role": "product", "source": "vision"})
                    existing_names.add(name.lower())
        result["figures"].append({"page": page_num, "type": "vision_analysis", "description": vision_data.get("description", ""), "notes": vision_data.get("notes", "")})

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process(self, result: dict):
        """Clean, deduplicate, and normalize extraction results."""
        result["reactions"] = self._clean_reaction_smiles(result.get("reactions", []))
        result["compounds"] = self._clean_compound_smiles(result.get("compounds", []))
        result["compounds"] = self._deduplicate_compounds(result.get("compounds", []))
        result["reactions"] = self._deduplicate_reactions(result.get("reactions", []))
        result["reactions"] = self._normalize_reactions(result["reactions"])
        result["compounds"] = self._normalize_compounds(result["compounds"])
        assemble_rgroup_reactions(result)

    def _clean_reaction_smiles(self, reactions):
        cleaned = []
        pseudo_count = 0
        for reaction in reactions:
            if not isinstance(reaction, dict):
                cleaned.append(reaction)
                continue
            new_reaction = dict(reaction)
            for role_key in ("reactants", "products"):
                entities = new_reaction.get(role_key, [])
                if isinstance(entities, list):
                    new_entities = []
                    for entity in entities:
                        if isinstance(entity, dict):
                            smiles = entity.get("smiles", "")
                            if smiles and _is_pseudo_smiles(smiles):
                                pseudo_count += 1
                                entity["general_form"] = smiles
                                entity["smiles"] = None
                        new_entities.append(entity)
                    new_reaction[role_key] = new_entities
            for role_key in ("catalysts", "ligands"):
                entities = new_reaction.get(role_key, [])
                if isinstance(entities, list):
                    new_entities = []
                    for entity in entities:
                        if isinstance(entity, dict):
                            smiles = entity.get("smiles", "")
                            if smiles and _is_pseudo_smiles(smiles):
                                pseudo_count += 1
                                entity["smiles"] = None
                        new_entities.append(entity)
                    new_reaction[role_key] = new_entities
            cleaned.append(new_reaction)
        if pseudo_count > 0:
            logger.info(f"[ChemExtract] Cleaned {pseudo_count} pseudo-SMILES from {len(reactions)} reactions")
        return cleaned

    def _clean_compound_smiles(self, compounds):
        cleaned = []
        pseudo_count = 0
        for compound in compounds:
            if not isinstance(compound, dict):
                cleaned.append(compound)
                continue
            new_compound = dict(compound)
            smiles = new_compound.get("smiles", "")
            if smiles and _is_pseudo_smiles(smiles):
                pseudo_count += 1
                new_compound["general_form"] = smiles
                new_compound["smiles"] = None
            cleaned.append(new_compound)
        if pseudo_count > 0:
            logger.info(f"[ChemExtract] Cleaned {pseudo_count} pseudo-SMILES from {len(compounds)} compounds")
        return cleaned

    def _deduplicate_compounds(self, compounds):
        seen = set()
        unique = []
        for compound in compounds:
            key = (compound.get("name", "") or compound.get("smiles", "") or "").lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(compound)
        return unique

    def _deduplicate_reactions(self, reactions):
        def _name_set(reaction, key):
            entities = reaction.get(key, [])
            if not isinstance(entities, list):
                return set()
            return {(e.get("name", "") or "").lower().strip() for e in entities if isinstance(e, dict) and e.get("name")}
        unique = []
        for reaction in reactions:
            r_names = _name_set(reaction, "reactants")
            p_names = _name_set(reaction, "products")
            if not r_names and not p_names:
                unique.append(reaction)
                continue
            is_dup = False
            for existing in unique:
                e_r = _name_set(existing, "reactants")
                e_p = _name_set(existing, "products")
                if not r_names or not p_names or not e_r or not e_p:
                    continue
                r_overlap = len(r_names & e_r) / max(len(r_names | e_r), 1)
                p_overlap = len(p_names & e_p) / max(len(p_names | e_p), 1)
                if r_overlap > 0.8 and p_overlap > 0.8:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(reaction)
        logger.info(f"[ChemExtract] Deduplicated {len(reactions)} reactions -> {len(unique)}")
        return unique

    def _normalize_reactions(self, reactions):
        normalized = []
        for rxn in reactions:
            if not isinstance(rxn, dict):
                normalized.append(rxn)
                continue
            nr = dict(rxn)
            nr.setdefault("id", "")
            nr.setdefault("type", "unknown")
            nr.setdefault("conditions", {})
            nr.setdefault("reactants", [])
            nr.setdefault("products", [])
            yield_val = nr.get("yield")
            if yield_val is None:
                outcomes = nr.get("outcomes")
                if isinstance(outcomes, dict) and "yield" in outcomes:
                    nr["yield"] = outcomes["yield"]
            if nr.get("entry") and not nr.get("entry_id"):
                nr["entry_id"] = str(nr["entry"])
            if isinstance(nr.get("conditions"), dict):
                nr["conditions"] = dict(sorted(nr["conditions"].items()))
            normalized.append(nr)
        return normalized

    def _normalize_compounds(self, compounds):
        normalized = []
        for comp in compounds:
            if not isinstance(comp, dict):
                normalized.append(comp)
                continue
            nc = dict(comp)
            nc.setdefault("name", "")
            nc.setdefault("smiles", None)
            nc.setdefault("role", "unknown")
            nc.setdefault("formula", None)
            if nc.get("smiles") and isinstance(nc["smiles"], str):
                nc["smiles"] = nc["smiles"].strip()
            normalized.append(nc)
        return normalized


# ------------------------------------------------------------------
# Convenience wrappers
# ------------------------------------------------------------------

def extract_chemical_data_from_pdf(pdf_path, llm_provider='deepseek', api_key=None,
                                     model=None, max_pages=50, extract_images=True,
                                     extract_text=True):
    extractor = ChemExtractAI(llm_provider=llm_provider, api_key=api_key, model=model)
    return extractor.extract_from_pdf(pdf_path, extract_images=extract_images, extract_text=extract_text,
                                     max_pages=max_pages)


async def extract_chemical_data_from_pdf_async(pdf_path, llm_provider='deepseek', api_key=None,
                                                model=None, max_pages=50, extract_images=True,
                                                extract_text=True):
    extractor = ChemExtractAI(llm_provider=llm_provider, api_key=api_key, model=model)
    return await extractor.extract_from_pdf_async(pdf_path, extract_images=extract_images,
                                                  extract_text=extract_text, max_pages=max_pages)
