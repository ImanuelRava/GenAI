import logging
from typing import List, Dict, Any, Optional

from .smiles_utils import _extract_smiles

logger = logging.getLogger(__name__)


def format_reaction_schemes(extraction_result, include_metadata=True,
                           fallback_to_name=False, skip_no_smiles=True):
    formatted = []
    reactions = extraction_result.get("reactions", [])
    for reaction in reactions:
        reactants_smiles = []
        products_smiles = []
        has_any_smiles = False
        for r in reaction.get("reactants", []):
            smiles = _extract_smiles(r)
            if smiles:
                if isinstance(r, dict) and r.get("smiles"):
                    has_any_smiles = True
                reactants_smiles.append(smiles)
        for p in reaction.get("products", []):
            smiles = _extract_smiles(p)
            if smiles:
                if isinstance(p, dict) and p.get("smiles"):
                    has_any_smiles = True
                products_smiles.append(smiles)
        if not reactants_smiles or not products_smiles:
            continue
        if skip_no_smiles and not has_any_smiles and not fallback_to_name:
            continue
        reactant_str = ".".join(reactants_smiles)
        product_str = ".".join(products_smiles)
        scheme = f"{reactant_str}>>{product_str}"
        entry = {
            "scheme": scheme,
            "reactants_smiles": reactants_smiles,
            "products_smiles": products_smiles,
        }
        if include_metadata:
            yield_val = reaction.get("yield")
            if yield_val is None:
                outcomes = reaction.get("outcomes")
                if isinstance(outcomes, dict):
                    yield_val = outcomes.get("yield")
            entry["reaction_id"] = reaction.get("id", "")
            entry["type"] = reaction.get("type", "unknown")
            entry["conditions"] = reaction.get("conditions", {})
            entry["yield"] = yield_val
            entry["catalyst"] = reaction.get("catalyst")
            entry["ligand"] = reaction.get("ligand")
            entry["source"] = reaction.get("source", "")
            entry["page"] = reaction.get("page")
            reagents = reaction.get("reagents")
            if reagents:
                entry["reagents"] = reagents
        formatted.append(entry)
    return formatted


def format_reaction_schemes_simple(extraction_result):
    return [entry["scheme"] for entry in format_reaction_schemes(extraction_result, include_metadata=False)]
