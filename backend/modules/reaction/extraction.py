"""
ReactionLens — Main extraction pipeline (sync + async) and ReactionLens class.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from .parsing import extract_text_from_pdf, segment_into_paragraphs
from .providers import rl_call_text, rl_call_text_async
from .prompts import RL_MIN_PARAGRAPH_LENGTH, REACTION_DETECTION_PROMPT

logger = logging.getLogger(__name__)


def _build_empty_result(metadata):
    return {
        "reactions": [],
        "compounds": [],
        "paragraphs_with_reactions": [],
        "text_metadata": metadata,
        "extraction_stats": {
            "total_paragraphs": 0,
            "paragraphs_with_reactions": 0,
            "total_reactions": 0,
            "total_compounds": 0,
        },
    }


def _merge_results(all_results):
    """Merge extraction results from multiple pages/paragraphs."""
    merged = {
        "reactants": [], "products": [], "catalysts": [], "ligands": [],
        "solvents": [], "conditions": {}, "yields": [], "mechanisms": [],
        "reactionType": None, "selectivity": None,
        "image_descriptions": [], "pages_with_data": [],
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
            merged["image_descriptions"].append({"page": page_num, "description": data["image_description"]})
    return merged


def _finalize_extraction(all_reactions, all_compounds, paragraphs_with_reactions,
                        scaffold_smiles, rgroup_table, experimental_procedures,
                        paragraphs, metadata):
    """Renumber reactions, deduplicate compounds, build final result dict."""
    for idx, rxn in enumerate(all_reactions, 1):
        rxn["id"] = f"reaction_{idx}"
        if "entry" not in rxn:
            rxn["entry"] = idx

    seen_names = set()
    unique_compounds = []
    for comp in all_compounds:
        name = comp.get("name", "")
        if name and name not in seen_names:
            seen_names.add(name)
            unique_compounds.append(comp)

    return {
        "reactions": all_reactions,
        "compounds": unique_compounds,
        "scaffold_smiles": scaffold_smiles,
        "rgroup_table": rgroup_table,
        "experimental_procedures": experimental_procedures,
        "paragraphs_with_reactions": paragraphs_with_reactions,
        "text_metadata": metadata,
        "extraction_stats": {
            "total_paragraphs": len(paragraphs),
            "paragraphs_with_reactions": len(paragraphs_with_reactions),
            "total_reactions": len(all_reactions),
            "total_compounds": len(unique_compounds),
        },
    }


def extract_with_reactionlens(
    pdf_path: str,
    provider: str = "deepseek",
    api_key: str = "",
    model: str = "deepseek-chat",
    max_pages: int = 50,
    min_paragraph_length: int = RL_MIN_PARAGRAPH_LENGTH,
    **kwargs,
) -> Dict[str, Any]:
    """Extract chemical reactions from a PDF using ReactionLens (sync)."""
    logger.info(f"[ReactionLens] Starting text-based extraction from {pdf_path}")
    logger.info(f"[ReactionLens] Provider: {provider}, Model: {model}")

    text, metadata = extract_text_from_pdf(pdf_path)
    if not text or len(text.strip()) < 50:
        logger.warning("[ReactionLens] Insufficient text extracted from PDF")
        return _build_empty_result(metadata)

    logger.info(f"[ReactionLens] Extracted {len(text)} chars from {metadata.get('pages', '?')} pages")

    paragraphs = segment_into_paragraphs(text, min_length=min_paragraph_length)
    if not paragraphs:
        logger.warning("[ReactionLens] No paragraphs found after segmentation")
        return _build_empty_result(metadata)

    all_reactions, all_compounds, paragraphs_with_reactions = [], [], []
    scaffold_smiles, rgroup_table, experimental_procedures = None, None, []

    for i, para in enumerate(paragraphs):
        logger.info(f"[ReactionLens] Screening paragraph {i + 1}/{len(paragraphs)} ({len(para['text'])} chars)...")
        try:
            result = rl_call_text(para["text"], provider, model, api_key, REACTION_DETECTION_PROMPT)
        except Exception as e:
            logger.warning(f"[ReactionLens] LLM call failed for paragraph {i + 1}: {e}")
            continue
        if result is None or not result.get("has_reactions", False):
            continue

        paragraphs_with_reactions.append({
            "paragraph_index": para["index"], "char_start": para["char_start"],
            "char_end": para["char_end"],
            "text_preview": para["text"][:300] + ("..." if len(para["text"]) > 300 else ""),
            "reactions_found": len(result.get("reactions", [])),
            "compounds_found": len(result.get("compounds", [])),
        })
        for rxn in result.get("reactions", []):
            rxn["source"] = "reactionlens_text"
            rxn["source_paragraph"] = para["index"]
            all_reactions.append(rxn)
        for comp in result.get("compounds", []):
            all_compounds.append(comp)
        if result.get("scaffold_smiles") and not scaffold_smiles:
            scaffold_smiles = result["scaffold_smiles"]
        if result.get("rgroup_table") and not rgroup_table:
            rgroup_table = result["rgroup_table"]
        for proc in result.get("experimental_procedures", []):
            experimental_procedures.append(proc)

    result = _finalize_extraction(
        all_reactions, all_compounds, paragraphs_with_reactions,
        scaffold_smiles, rgroup_table, experimental_procedures,
        paragraphs, metadata,
    )
    logger.info(f"[ReactionLens] Extraction complete: {len(all_reactions)} reactions from "
                f"{len(paragraphs_with_reactions)}/{len(paragraphs)} paragraphs")
    return result


async def extract_with_reactionlens_async(
    pdf_path: str,
    provider: str = "deepseek",
    api_key: str = "",
    model: str = "deepseek-chat",
    max_pages: int = 50,
    min_paragraph_length: int = RL_MIN_PARAGRAPH_LENGTH,
    **kwargs,
) -> Dict[str, Any]:
    """Extract chemical reactions from a PDF using ReactionLens (async)."""
    logger.info(f"[ReactionLens Async] Starting text-based extraction from {pdf_path}")

    text, metadata = extract_text_from_pdf(pdf_path)
    if not text or len(text.strip()) < 50:
        return _build_empty_result(metadata)

    paragraphs = segment_into_paragraphs(text, min_length=min_paragraph_length)
    if not paragraphs:
        return _build_empty_result(metadata)

    async def screen_paragraph(para):
        try:
            return await rl_call_text_async(para["text"], provider, model, api_key, REACTION_DETECTION_PROMPT)
        except Exception as e:
            logger.warning(f"[ReactionLens Async] LLM call failed for paragraph {para['index']}: {e}")
            return None

    batch_size = 5
    all_results = []
    for batch_start in range(0, len(paragraphs), batch_size):
        batch = paragraphs[batch_start:batch_start + batch_size]
        logger.info(f"[ReactionLens Async] Processing batch {batch_start // batch_size + 1}/"
                    f"{(len(paragraphs) + batch_size - 1) // batch_size} ({len(batch)} paragraphs)...")
        tasks = [screen_paragraph(para) for para in batch]
        results = await asyncio.gather(*tasks)
        for para, result in zip(batch, results):
            all_results.append((para, result))

    all_reactions, all_compounds, paragraphs_with_reactions = [], [], []
    scaffold_smiles, rgroup_table, experimental_procedures = None, None, []

    for para, result in all_results:
        if result is None or not result.get("has_reactions", False):
            continue
        paragraphs_with_reactions.append({
            "paragraph_index": para["index"], "char_start": para["char_start"],
            "char_end": para["char_end"],
            "text_preview": para["text"][:300] + ("..." if len(para["text"]) > 300 else ""),
            "reactions_found": len(result.get("reactions", [])),
            "compounds_found": len(result.get("compounds", [])),
        })
        for rxn in result.get("reactions", []):
            rxn["source"] = "reactionlens_text"
            rxn["source_paragraph"] = para["index"]
            all_reactions.append(rxn)
        for comp in result.get("compounds", []):
            all_compounds.append(comp)
        if result.get("scaffold_smiles") and not scaffold_smiles:
            scaffold_smiles = result["scaffold_smiles"]
        if result.get("rgroup_table") and not rgroup_table:
            rgroup_table = result["rgroup_table"]
        for proc in result.get("experimental_procedures", []):
            experimental_procedures.append(proc)

    result = _finalize_extraction(
        all_reactions, all_compounds, paragraphs_with_reactions,
        scaffold_smiles, rgroup_table, experimental_procedures,
        paragraphs, metadata,
    )
    logger.info(f"[ReactionLens Async] Extraction complete: {len(all_reactions)} reactions from "
                f"{len(paragraphs_with_reactions)}/{len(paragraphs)} paragraphs")
    return result


# ---------------------------------------------------------------------------
# Convenience class for backwards compatibility
# ---------------------------------------------------------------------------

class ReactionLens:
    """ReactionLens extractor (text-based reaction detection from papers)."""

    def __init__(self, provider="deepseek", api_key="", model="deepseek-chat", max_pages=50, **kwargs):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_pages = max_pages

    def extract_from_pdf(self, pdf_path):
        return extract_with_reactionlens(
            pdf_path, provider=self.provider, api_key=self.api_key,
            model=self.model, max_pages=self.max_pages,
        )

    async def extract_from_pdf_async(self, pdf_path):
        return await extract_with_reactionlens_async(
            pdf_path, provider=self.provider, api_key=self.api_key,
            model=self.model, max_pages=self.max_pages,
        )

    def format_reaction_schemes(self, extraction_result):
        try:
            from chemextract.reaction_formatter import format_reaction_schemes
            return format_reaction_schemes(extraction_result)
        except Exception:
            pass
        formatted = []
        for rxn in extraction_result.get("reactions", []):
            reactants = [r.get("smiles") or r.get("name", "?") for r in rxn.get("reactants", [])
                        if r.get("smiles") or r.get("name")]
            products = [p.get("smiles") or p.get("name", "?") for p in rxn.get("products", [])
                       if p.get("smiles") or p.get("name")]
            if reactants and products:
                formatted.append({
                    "scheme": f"{'.'.join(reactants)}>>{'.'.join(products)}",
                    "reactants_smiles": reactants, "products_smiles": products,
                    "reaction_id": rxn.get("id", ""), "type": rxn.get("type", "unknown"),
                    "conditions": rxn.get("conditions", {}),
                    "yield": rxn.get("outcomes", {}).get("yield") if isinstance(rxn.get("outcomes"), dict) else None,
                })
        return formatted
