"""
Post-processing + result-merging mixin and helpers for ChemExtractAI.

Two kinds of code live here:

  1. **Result-merging methods** (mixed into ``ChemExtractAI``) —
     ``_merge_figure_result`` and ``_merge_vision_results`` translate the
     two different LLM output formats (structured ``reaction_schemes``
     vs flat ``reactants``/``products``) into the unified result dict
     shape that the frontend expects.

  2. **Post-processing methods + module-level functions** —
     ``_post_process`` runs the full cleanup pipeline (pseudo-SMILES
     stripping → dedup → normalize → R-group assembly). The individual
     steps are also exposed as module-level functions
     (``clean_reaction_smiles``, ``deduplicate_compounds``, etc.) so they
     can be unit-tested without instantiating ``ChemExtractAI``.

Pseudo-SMILES handling
-----------------------
LLMs sometimes emit placeholder SMILES like ``R1-I``, ``ArCH2Cl``, or
``RC(O)Cl`` instead of real, RDKit-parsable SMILES. The cleanup step
detects these via ``smiles_utils._is_pseudo_smiles`` and moves them to
a ``general_form`` field, setting ``smiles`` to None. This prevents
downstream RDKit parsing failures.

Deduplication
-------------
Compounds are deduplicated by lowercased name-or-SMILES.
Reactions are deduplicated by Jaccard overlap of reactant + product
name sets — if both overlap > 80% with an existing reaction, the new
one is dropped. This catches cases where the vision and text pipelines
both extract the same reaction.

Normalization
-------------
Reactions: ensure ``id``, ``type``, ``conditions``, ``reactants``,
``products`` keys exist; hoist ``outcomes.yield`` to top-level ``yield``;
sort conditions dict for stable output.
Compounds: ensure ``name``, ``smiles``, ``role``, ``formula`` keys exist;
strip whitespace from SMILES.
"""

import logging
from typing import Any, Dict, List

from .smiles_utils import _is_pseudo_smiles, assemble_rgroup_reactions

logger = logging.getLogger(__name__)


# =============================================================================
# Module-level pure functions (independently testable)
# =============================================================================

def clean_reaction_smiles(reactions: List[dict]) -> List[dict]:
    """Strip pseudo-SMILES from reaction entities.

    For each reaction's ``reactants``, ``products``, ``catalysts``,
    ``ligands`` lists: if an entity's ``smiles`` field is a pseudo-SMILES
    placeholder (e.g. ``R1-I``), move it to ``general_form`` and set
    ``smiles`` to None. This prevents downstream RDKit parsing failures.

    Returns a new list of reaction dicts; the input list is not mutated.
    """
    cleaned: List[dict] = []
    pseudo_count = 0

    for reaction in reactions:
        if not isinstance(reaction, dict):
            cleaned.append(reaction)
            continue

        new_reaction = dict(reaction)

        # Reactants + products: preserve general_form
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

        # Catalysts + ligands: just null out pseudo-SMILES (no general_form)
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
        logger.info(
            f"[ChemExtract] Cleaned {pseudo_count} pseudo-SMILES from "
            f"{len(reactions)} reactions"
        )
    return cleaned


def clean_compound_smiles(compounds: List[dict]) -> List[dict]:
    """Strip pseudo-SMILES from compound entries.

    Like ``clean_reaction_smiles`` but for the top-level compounds list.
    Pseudo-SMILES are moved to ``general_form`` and ``smiles`` is set to None.
    """
    cleaned: List[dict] = []
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
        logger.info(
            f"[ChemExtract] Cleaned {pseudo_count} pseudo-SMILES from "
            f"{len(compounds)} compounds"
        )
    return cleaned


def deduplicate_compounds(compounds: List[dict]) -> List[dict]:
    """Deduplicate compounds by lowercased name-or-SMILES.

    The first occurrence wins; subsequent duplicates are dropped. Compounds
    with neither name nor SMILES are kept as-is (not deduplicated).
    """
    seen: set = set()
    unique: List[dict] = []
    for compound in compounds:
        key = (
            compound.get("name", "")
            or compound.get("smiles", "")
            or ""
        ).lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(compound)
    return unique


def deduplicate_reactions(reactions: List[dict]) -> List[dict]:
    """Deduplicate reactions by Jaccard overlap of reactant + product names.

    A reaction is considered a duplicate of an existing one if BOTH the
    reactant name set and product name set overlap by > 80% (Jaccard
    similarity). Reactions with no named reactants or products are kept
    as-is (not deduplicated).
    """
    def _name_set(reaction: dict, key: str) -> set:
        entities = reaction.get(key, [])
        if not isinstance(entities, list):
            return set()
        return {
            (e.get("name", "") or "").lower().strip()
            for e in entities
            if isinstance(e, dict) and e.get("name")
        }

    unique: List[dict] = []
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
    logger.info(
        f"[ChemExtract] Deduplicated {len(reactions)} reactions -> {len(unique)}"
    )
    return unique


def normalize_reactions(reactions: List[dict]) -> List[dict]:
    """Ensure all reactions have the expected key set + hoist outcomes.yield.

    - Sets defaults: id, type, conditions, reactants, products.
    - If ``yield`` is None but ``outcomes.yield`` exists, hoist it.
    - If ``entry`` exists but ``entry_id`` doesn't, derive entry_id from entry.
    - Sorts the conditions dict for stable output.
    """
    normalized: List[dict] = []
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


def normalize_compounds(compounds: List[dict]) -> List[dict]:
    """Ensure all compounds have the expected key set + strip SMILES whitespace."""
    normalized: List[dict] = []
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


# =============================================================================
# Post-process mixin (orchestrates the pure functions above)
# =============================================================================

class PostProcessMixin:
    """Result-merging + post-processing methods, mixed into ChemExtractAI.

    The merging methods (_merge_figure_result, _merge_vision_results) are
    instance methods because they need to mutate the shared ``result`` dict
    and read the existing compounds set for dedup-at-merge-time.

    The post-processing methods (_post_process, _clean_*, _deduplicate_*,
    _normalize_*) delegate to the module-level pure functions so they can
    be tested in isolation.
    """

    # ------------------------------------------------------------------
    # Top-level post-process orchestrator
    # ------------------------------------------------------------------

    def _post_process(self, result: dict):
        """Clean, deduplicate, and normalize extraction results.

        Runs in this order:
          1. Strip pseudo-SMILES from reactions + compounds.
          2. Deduplicate compounds (by name-or-SMILES).
          3. Deduplicate reactions (by Jaccard overlap of reactant/product names).
          4. Normalize reactions (ensure key set, hoist outcomes.yield).
          5. Normalize compounds (ensure key set, strip SMILES whitespace).
          6. Assemble R-group reactions (fills in ``assembled_smiles`` on
             reactions that have ``rgroup_values``).
        """
        result["reactions"] = clean_reaction_smiles(result.get("reactions", []))
        result["compounds"] = clean_compound_smiles(result.get("compounds", []))
        result["compounds"] = deduplicate_compounds(result.get("compounds", []))
        result["reactions"] = deduplicate_reactions(result.get("reactions", []))
        result["reactions"] = normalize_reactions(result["reactions"])
        result["compounds"] = normalize_compounds(result["compounds"])
        assemble_rgroup_reactions(result)

    # Backwards-compat aliases — the original ChemExtractAI had these as
    # instance methods. We keep them as thin wrappers so any external code
    # (or tests) that calls instance._clean_reaction_smiles() still works.
    def _clean_reaction_smiles(self, reactions):
        return clean_reaction_smiles(reactions)

    def _clean_compound_smiles(self, compounds):
        return clean_compound_smiles(compounds)

    def _deduplicate_compounds(self, compounds):
        return deduplicate_compounds(compounds)

    def _deduplicate_reactions(self, reactions):
        return deduplicate_reactions(reactions)

    def _normalize_reactions(self, reactions):
        return normalize_reactions(reactions)

    def _normalize_compounds(self, compounds):
        return normalize_compounds(compounds)

    # ------------------------------------------------------------------
    # Result-merging methods (translate LLM output → unified result dict)
    # ------------------------------------------------------------------

    def _merge_figure_result(
        self, result: dict, vision_data: dict, page_num: int, source: str = "embedded",
    ):
        """Merge results from an individual embedded figure analysis.

        The SYSTEM_PROMPT_FIGURE_ANALYSIS prompt produces a structured
        output with a top-level ``reaction_schemes`` key (each scheme has
        its own ``reactants``/``products``/``conditions``/etc.). This
        method translates that into the unified result dict shape.

        If the LLM returned the flat format instead (top-level
        ``reactants``/``products``), we delegate to ``_merge_vision_results``.
        """
        reaction_schemes = vision_data.get("reaction_schemes", [])
        if not reaction_schemes and (vision_data.get("reactants") or vision_data.get("products")):
            # Fallback: flat format (same as _merge_vision_results).
            self._merge_vision_results(result, vision_data, page_num)
            return

        for scheme in reaction_schemes:
            reaction = {
                "id": f"{source}_page{page_num}_{len(result['reactions']) + 1}",
                "source": source,
                "page": page_num,
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

        # Handle table_data from figure analysis.
        table_data = vision_data.get("table_data", [])
        if table_data:
            result["tables"].append({
                "page": page_num,
                "source": source,
                "data": table_data,
                "columns": vision_data.get("table_columns", []),
            })

        # Extract compounds from reaction schemes + top-level compounds.
        existing_names = {
            c.get("name", "").lower()
            for c in result.get("compounds", [])
            if c.get("name")
        }
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

        # Also merge compounds from top-level "compounds" in figure output.
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

        # Record figure metadata.
        fig_type = vision_data.get("figure_type", "unknown")
        description = vision_data.get("description", "")
        result["figures"].append({
            "page": page_num,
            "type": fig_type,
            "source": source,
            "description": description,
            "notes": vision_data.get("notes", ""),
        })

        # Pull scaffold/rgroup data from figure analysis.
        for key in ("scaffold_smiles", "rgroup_table", "rgroup_attachment_map"):
            if key in vision_data and key not in result:
                result[key] = vision_data[key]

    def _merge_vision_results(self, result: dict, vision_data: dict, page_num: int):
        """Merge results from full-page or scheme-page vision analysis (legacy format).

        The SYSTEM_PROMPT_VISION prompt produces either:
          - Structured format with ``reaction_schemes`` key, OR
          - Flat format with top-level ``reactants``/``products`` lists.

        For the flat format, we wrap the lists into a single reaction_scheme
        entry so the rest of the merging logic can treat both formats uniformly.
        """
        reaction_schemes = vision_data.get("reaction_schemes", [])
        if not reaction_schemes and (vision_data.get("reactants") or vision_data.get("products")):
            # Translate flat format → structured format.
            flat_reactants = vision_data.get("reactants", [])
            flat_products = vision_data.get("products", [])
            if isinstance(flat_reactants, list) and isinstance(flat_products, list):
                reaction_schemes = [{
                    "entry": 1,
                    "reactants": [
                        {"name": r, "smiles": None} if isinstance(r, str) else r
                        for r in flat_reactants
                    ],
                    "products": [
                        {"name": p, "smiles": None} if isinstance(p, str) else p
                        for p in flat_products
                    ],
                    "reagents": vision_data.get("reagents", []),
                    "conditions": vision_data.get("conditions", {}),
                    "yield": None,
                    "catalyst": (
                        vision_data.get("catalysts", [None])[0]
                        if vision_data.get("catalysts") else None
                    ),
                    "ligand": (
                        vision_data.get("ligands", [None])[0]
                        if vision_data.get("ligands") else None
                    ),
                }]

        for scheme in reaction_schemes:
            reaction = {
                "id": f"vision_page{page_num}_{len(result['reactions']) + 1}",
                "source": "vision",
                "page": page_num,
                "type": scheme.get("reactionType", "unknown"),
                "reactants": scheme.get("reactants", []),
                "products": scheme.get("products", []),
                "reagents": scheme.get("reagents", []),
                "catalyst": scheme.get("catalyst"),
                "ligand": scheme.get("ligand"),
                "conditions": scheme.get("conditions", {}),
                "yield": scheme.get("yield"),
                "notes": scheme.get("notes", ""),
            }
            result["reactions"].append(reaction)

        if vision_data.get("table_data"):
            result["tables"].append({
                "page": page_num,
                "source": "vision",
                "data": vision_data.get("table_data", []),
                "columns": vision_data.get("table_columns", []),
            })

        existing_names = {
            c.get("name", "").lower()
            for c in result.get("compounds", [])
            if c.get("name")
        }
        for scheme in reaction_schemes:
            for reactant in scheme.get("reactants", []):
                name = reactant.get("name", "") if isinstance(reactant, dict) else str(reactant)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({
                        "name": name,
                        "smiles": reactant.get("smiles") if isinstance(reactant, dict) else None,
                        "role": "reactant",
                        "source": "vision",
                    })
                    existing_names.add(name.lower())
            for product in scheme.get("products", []):
                name = product.get("name", "") if isinstance(product, dict) else str(product)
                if name and name.lower() not in existing_names:
                    result["compounds"].append({
                        "name": name,
                        "smiles": product.get("smiles") if isinstance(product, dict) else None,
                        "role": "product",
                        "source": "vision",
                    })
                    existing_names.add(name.lower())

        result["figures"].append({
            "page": page_num,
            "type": "vision_analysis",
            "description": vision_data.get("description", ""),
            "notes": vision_data.get("notes", ""),
        })
