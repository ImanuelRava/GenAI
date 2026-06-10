import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_RGROUP_PLACEHOLDER_RE = re.compile(r'\[(\*|(\*:\d+)|(R\d+)|(R\w+))\]', re.IGNORECASE)


def _is_pseudo_smiles(s):
    if not s or len(s) < 2:
        return True
    if re.search(r'\s', s):
        return True
    unbracketed = re.findall(r'(?<!\w)(Ar|R|X|Y|Z)(?=[A-Z0-9\[\(]|$)', s)
    if unbracketed:
        return True
    if re.search(r'(?<!\w)R(?=[A-Z][a-z]|[A-Z]\d)', s) and not re.search(r'\[R', s):
        return True
    if re.search(r'(?<!\w)Ar(?=[A-Z]|[a-z])', s) and not re.search(r'\[Ar', s):
        return True
    return False


def _extract_smiles(entity):
    if isinstance(entity, dict):
        smiles = entity.get("smiles")
        if smiles and smiles.strip() and smiles.upper() != "NONE":
            smiles = smiles.strip()
            if not _is_pseudo_smiles(smiles):
                return smiles
        name = entity.get("name")
        if name and name.strip():
            return name.strip()
        return None
    elif isinstance(entity, str):
        cleaned = entity.strip()
        if cleaned and cleaned.upper() != "NONE":
            return cleaned
        return None
    return None


def assemble_rgroup_smiles(scaffold_smiles, rgroups):
    if not scaffold_smiles or not rgroups:
        return None
    smiles = scaffold_smiles.strip()
    def _normalize(m):
        tag = m.group(1)
        if tag.startswith('R') and len(tag) > 1 and tag[1:].isdigit():
            return f'[*:{tag[1:]}]'
        if tag.startswith('*:'):
            return m.group(0)
        if tag == '*':
            return '[*:1]'
        return m.group(0)
    smiles = _RGROUP_PLACEHOLDER_RE.sub(_normalize, smiles)
    replacements = {}
    for key, frag in rgroups.items():
        if not frag or frag.strip().upper() in ('NONE', 'H', ''):
            tag = key.lstrip('R')
            replacements[f'[*:{tag}]'] = ''
            continue
        tag = key.lstrip('R')
        replacements[f'[*:{tag}]'] = frag.strip()
    if not replacements:
        return smiles
    for pattern, frag in sorted(replacements.items(), key=lambda x: -len(x[0])):
        smiles = smiles.replace(pattern, frag)
    smiles = smiles.replace('[*]', '')
    return smiles if smiles.strip() else None


def _is_valid_assembled_smiles(smiles, scaffold):
    if not smiles or not scaffold:
        return False
    if '[*]' in smiles or re.search(r'\[\*:\d+\]', smiles):
        return False
    return True


def _patch_reaction_with_assembled(reaction, assembled, scaffold_smiles, scaffold_label, rgroups):
    patched = False
    for r in reaction.get("reactants", []):
        if isinstance(r, dict):
            old_smiles = r.get("smiles", "")
            old_name = r.get("name", "")
            is_scaffold_entry = False
            for rg_dict in rgroups.values():
                if old_name.strip() in rg_dict if old_name else False:
                    is_scaffold_entry = True
                    break
            if (not old_smiles or old_smiles == scaffold_smiles
                    or (scaffold_label and old_name == scaffold_label) or is_scaffold_entry):
                r["smiles"] = assembled
                r["assembled"] = True
                patched = True
    if not reaction.get("reactants"):
        entry_label = reaction.get("entry_id", "") or scaffold_label or "scaffold"
        reaction["reactants"] = [{
            "name": f"{scaffold_label or 'scaffold'} ({entry_label})",
            "smiles": assembled,
            "assembled": True,
        }]
        patched = True
    reaction["assembled_from_rgroup"] = True


def assemble_rgroup_reactions(extraction_result):
    rgroup_table = extraction_result.get("rgroup_table")
    if not rgroup_table:
        return extraction_result
    scaffold_smiles = extraction_result.get("scaffold_smiles")
    if not scaffold_smiles:
        return extraction_result
    rgroups = rgroup_table.get("rgroups", {})
    partner_rgroups = rgroup_table.get("partner_rgroups", {})
    if not rgroups and not partner_rgroups:
        return extraction_result
    import copy
    extraction_result["_original_reactions"] = copy.deepcopy(extraction_result.get("reactions", []))
    scaffold_label = rgroup_table.get("scaffold_label", "")
    for reaction in extraction_result.get("reactions", []):
        llm_assembled = reaction.get("assembled_smiles")
        if llm_assembled and _is_valid_assembled_smiles(llm_assembled, scaffold_smiles):
            _patch_reaction_with_assembled(reaction, llm_assembled, scaffold_smiles, scaffold_label, rgroups)
            continue
        entry_id = str(reaction.get("entry_id", "") or reaction.get("entry", ""))
        if not entry_id:
            for r in reaction.get("reactants", []):
                name = (r.get("name") if isinstance(r, dict) else str(r))
                if name and name.strip():
                    for rg_dict in rgroups.values():
                        if name.strip() in rg_dict:
                            entry_id = name.strip()
                            break
                if entry_id:
                    break
        if not entry_id:
            continue
        entry_rgroups = {}
        for rname, variants in rgroups.items():
            if entry_id in variants:
                entry_rgroups[rname] = variants[entry_id]
        for r in reaction.get("reactants", []):
            partner_name = (r.get("name") if isinstance(r, dict) else str(r))
            if partner_name and partner_name.strip():
                for prname, variants in partner_rgroups.items():
                    if partner_name.strip() in variants:
                        entry_rgroups[prname] = variants[partner_name.strip()]
        if not entry_rgroups:
            continue
        assembled = assemble_rgroup_smiles(scaffold_smiles, entry_rgroups)
        if not assembled:
            continue
        _patch_reaction_with_assembled(reaction, assembled, scaffold_smiles, scaffold_label, rgroups)
    extraction_result["rgroup_assembled"] = True
    return extraction_result
