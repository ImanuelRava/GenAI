"""
RAL (Redox-Active Ligand) RAG retrieval module.

Provides ligand and reaction retrieval from a local CSV/JSON database with
cross-class matching so that comparison queries (e.g. "bipyridine vs
bisoxazoline") surface entries for **all** mentioned ligand families rather
than only the highest-scoring match.

Interface consumed by ``chat/redox.py`` and ``chat/helpers.py``:
    rag = get_ral_rag()
    context = rag.retrieve_context(message)
    # context.formatted_context  – str
    # context.ligands           – list[dict]
    # context.reactions         – list[dict]
    # context.detected_class    – str | None
    enhanced = rag.build_enhanced_prompt(message, system_prompt)

Data files
----------
The module looks for data in the following locations (first found wins):
    * ``<PACKAGE_DATA_DIR>/ral_ligands.csv``
    * ``<PACKAGE_DATA_DIR>/ral_reactions.json``
    * Falls back to ``backend/data/ral_ligands.csv`` etc. relative to CWD.

If no data files are found the module still imports successfully but
``retrieve_context()`` returns an empty context object (formatted_context == "").
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synonym / alias map – bridges colloquial names to canonical class keys
# ---------------------------------------------------------------------------

LIGAND_ALIASES: Dict[str, str] = {
    # bipyridine family
    "bpy": "bipyridine",
    "bipy": "bipyridine",
    "2,2'-bipyridine": "bipyridine",
    "2,2'-bipy": "bipyridine",
    "dtbbpy": "bipyridine",
    "dtbpy": "bipyridine",
    "4,4'-di-tert-butyl-2,2'-bipyridine": "bipyridine",
    "phenanthroline": "bipyridine",          # same N^N chelate class
    "phen": "bipyridine",
    "neocuproine": "bipyridine",
    "bathophenanthroline": "bipyridine",
    # bisoxazoline / bioxazoline family
    "bisoxazoline": "bisoxazoline",
    "bioxazoline": "bisoxazoline",           # common misspelling
    "box": "bisoxazoline",
    "pybox": "bisoxazoline",
    "chiral pybox": "bisoxazoline",
    "oxazoline": "bisoxazoline",
    "i-pr-box": "bisoxazoline",
    "t-butyl-box": "bisoxazoline",
    "ph-box": "bisoxazoline",
    # phenanthroline (if treated as separate class, uncomment)
    # "phenanthroline": "phenanthroline",
    # phosphine family
    "phosphine": "phosphine",
    "pph3": "phosphine",
    "triphenylphosphine": "phosphine",
    "pcy3": "phosphine",
    "xphos": "phosphine",
    "sphos": "phosphine",
    "dppf": "diphosphine",
    "diphosphine": "diphosphine",
    "dppe": "diphosphine",
    "dppp": "diphosphine",
    # NHC family
    "nhc": "nhc",
    "n-heterocyclic carbene": "nhc",
    "imidazolylidene": "nhc",
    "sipc": "nhc",
    "imes": "nhc",
    # PDI family (redox-active)
    "pdi": "pdi",
    "bis-iminopyridine": "pdi",
    "bis(imino)pyridine": "pdi",
    "pyridine diimine": "pdi",
    "redox-active ligand": "pdi",
    # catecholate family (redox-active)
    "catecholate": "catecholate",
    "catechol": "catecholate",
    "o-quinone": "catecholate",
    "semiquinone": "catecholate",
    "dithiolene": "dithiolene",
    "dithiolene": "dithiolene",
    # acetylacetonate
    "acac": "acetylacetonate",
    "acetylacetonate": "acetylacetonate",
}

# Reversible map for fast canonical look-up
_CANONICAL: Dict[str, str] = {}
for _alias, _canonical in LIGAND_ALIASES.items():
    _CANONICAL[_alias.lower()] = _canonical

# Reaction-type keywords that help narrow the context
REACTION_KEYWORDS: Dict[str, List[str]] = {
    "reductive coupling": ["reductive coupling", "reductive cross-coupling",
                           "cross-electrophile coupling", "electrophile coupling"],
    "cross-coupling": ["cross-coupling", "suzuki", "heck", "kumada",
                       "negishi", "stille", "sonogashira"],
    "C-O activation": ["c-o activation", "c-o bond activation",
                       "aryl ether", "ester activation"],
    "hydrofunctionalization": ["hydrofunctionalization", "hydroalkylation",
                               "hydroarylation", "hydrosilylation",
                               "hydroboration"],
    "C-H activation": ["c-h activation", "c-h bond activation"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RALContext:
    """Returned by ``retrieve_context()``."""
    formatted_context: str = ""
    ligands: List[Dict[str, Any]] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    detected_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper – text normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text).lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> Set[str]:
    """Simple whitespace + punctuation tokeniser."""
    return set(re.findall(r"[a-z0-9]+", _normalise(text)))


# ---------------------------------------------------------------------------
# Query analysis – extract *all* ligand classes mentioned
# ---------------------------------------------------------------------------

def _extract_ligand_classes(query: str) -> List[str]:
    """Return a **deduplicated, order-preserving** list of canonical ligand
    class names found in *query*.

    This is the key fix for comparison queries: we must extract *every*
    mentioned ligand, not just the top-1 match.
    """
    nq = _normalise(query)
    found: List[str] = []
    seen: Set[str] = set()

    # 1. Multi-word aliases first (longer matches are more specific)
    sorted_aliases = sorted(LIGAND_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        na = _normalise(alias)
        if na in nq:
            canonical = LIGAND_ALIASES[alias]
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    # 2. Single-word token fallback: check each token against canonical names
    tokens = _tokenise(query)
    for token in tokens:
        if token in _CANONICAL:
            canonical = _CANONICAL[token]
            if canonical not in seen:
                found.append(canonical)
                seen.add(canonical)

    return found


def _detect_reaction_type(query: str) -> Optional[str]:
    """Return the most likely reaction type key, or None."""
    nq = _normalise(query)
    best: Optional[str] = None
    best_count = 0
    for rtype, keywords in REACTION_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in nq)
        if count > best_count:
            best = rtype
            best_count = count
    return best


# ---------------------------------------------------------------------------
# CSV / JSON data loader
# ---------------------------------------------------------------------------

def _find_data_dir() -> Optional[Path]:
    """Search for the data directory in several likely locations."""
    candidates = [
        # Package-level data directory (set via env var)
        os.environ.get("RAL_DATA_DIR"),
        # Relative to this file's grand-parent (backend/)
        Path(__file__).resolve().parent.parent / "data",
        # Relative to CWD
        Path.cwd() / "data",
        Path.cwd() / "backend" / "data",
    ]
    for c in candidates:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def _load_ligands(data_dir: Path) -> List[Dict[str, Any]]:
    """Load ligand entries from ``ral_ligands.csv``."""
    csv_path = data_dir / "ral_ligands.csv"
    if not csv_path.exists():
        logger.warning("RAL ligand data not found: %s", csv_path)
        return []
    ligands: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["class"] = _CANONICAL.get(
                _normalise(row.get("ligand_class", row.get("class", ""))),
                _normalise(row.get("ligand_class", row.get("class", ""))),
            )
            ligands.append(row)
    logger.info("Loaded %d ligand entries from %s", len(ligands), csv_path)
    return ligands


def _load_reactions(data_dir: Path) -> List[Dict[str, Any]]:
    """Load reaction entries from ``ral_reactions.json``."""
    json_path = data_dir / "ral_reactions.json"
    if not json_path.exists():
        # Also try CSV
        csv_path = data_dir / "ral_reactions.csv"
        if csv_path.exists():
            return _load_reactions_csv(csv_path)
        logger.warning("RAL reaction data not found: %s", json_path)
        return []
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        reactions = data.get("reactions", data.get("entries", []))
    else:
        reactions = data
    logger.info("Loaded %d reaction entries from %s", len(reactions), json_path)
    return reactions


def _load_reactions_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Fallback: load reactions from CSV."""
    reactions: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Normalise ligand references
            for field in ("ligand", "ligand_class", "optimal_ligand"):
                if field in row:
                    row[f"{field}_norm"] = _CANONICAL.get(
                        _normalise(row[field]), _normalise(row[field])
                    )
            reactions.append(row)
    logger.info("Loaded %d reaction entries from %s", len(reactions), csv_path)
    return reactions


# ---------------------------------------------------------------------------
# Retrieval / scoring
# ---------------------------------------------------------------------------

def _score_ligand(ligand: Dict[str, Any],
                  query_classes: List[str],
                  reaction_type: Optional[str]) -> float:
    """Score a ligand entry against the query.

    A ligand scores higher when:
      - Its class matches one of the query-mentioned classes
      - Its name / aliases appear in the original query
      - It has associated reaction data matching the detected reaction type
    """
    score = 0.0
    lig_class = ligand.get("class", "").lower()
    lig_name = _normalise(ligand.get("name", ligand.get("ligand", "")))

    # Class match (primary signal)
    for qc in query_classes:
        if lig_class == qc.lower():
            score += 10.0
            break
        # Partial / substring match for broader class hits
        if qc.lower() in lig_class or lig_class in qc.lower():
            score += 3.0

    # Name token overlap
    lig_tokens = _tokenise(lig_name)
    query_tokens = _tokenise(" ".join(query_classes))
    if lig_tokens & query_tokens:
        score += 5.0 * len(lig_tokens & query_tokens) / max(len(lig_tokens), 1)

    # Reaction type affinity
    if reaction_type:
        lig_reactions = ligand.get("reaction_types", ligand.get("reactions", ""))
        if isinstance(lig_reactions, str) and reaction_type.lower() in lig_reactions.lower():
            score += 4.0
        elif isinstance(lig_reactions, list):
            if any(reaction_type.lower() in str(r).lower() for r in lig_reactions):
                score += 4.0

    return score


def _score_reaction(reaction: Dict[str, Any],
                    query_classes: List[str],
                    reaction_type: Optional[str]) -> float:
    """Score a reaction entry against the query."""
    score = 0.0

    # Reaction type match
    if reaction_type:
        r_type_str = " ".join([
            reaction.get("reaction_type", ""),
            reaction.get("reaction_name", ""),
            reaction.get("type", ""),
            reaction.get("description", ""),
        ]).lower()
        if reaction_type.lower() in r_type_str:
            score += 8.0

    # Ligand class match
    for field in ("ligand", "ligand_class", "optimal_ligand"):
        val = reaction.get(field, "")
        norm_val = _normalise(val)
        # Check direct match against canonical names
        if norm_val in _CANONICAL:
            canonical = _CANONICAL[norm_val]
            if canonical in [qc.lower() for qc in query_classes]:
                score += 6.0
        # Also check the normalised field directly
        for qc in query_classes:
            if qc.lower() in norm_val or norm_val in qc.lower():
                score += 4.0

    # Check normalised fields (set by CSV loader)
    for field in ("ligand_norm", "ligand_class_norm", "optimal_ligand_norm"):
        norm_val = reaction.get(field, "")
        for qc in query_classes:
            if qc.lower() == norm_val:
                score += 6.0

    return score


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_ligand_entry(lig: Dict[str, Any]) -> str:
    """Format a single ligand as a readable block."""
    parts = []
    name = lig.get("name", lig.get("ligand", "Unknown"))
    lig_class = lig.get("class", lig.get("ligand_class", ""))
    parts.append(f"- {name} (class: {lig_class})")

    # Electronic properties (core RAL metrics)
    for prop in ("HOMO", "LUMO", "Gap", "omega", "homo", "lumo", "gap", "Omega"):
        val = lig.get(prop)
        if val is not None:
            parts.append(f"  {prop}: {val}")

    # General notes
    notes = lig.get("notes", lig.get("description", ""))
    if notes:
        parts.append(f"  Notes: {notes}")

    return "\n".join(parts)


def _format_reaction_entry(rxn: Dict[str, Any]) -> str:
    """Format a single reaction as a readable block."""
    parts = []
    name = rxn.get("reaction_name", rxn.get("name", rxn.get("description", "Unknown reaction")))
    rtype = rxn.get("reaction_type", rxn.get("type", ""))
    ligand = rxn.get("optimal_ligand", rxn.get("ligand", ""))
    doi = rxn.get("DOI", rxn.get("doi", ""))
    yield_val = rxn.get("yield", rxn.get("yield_%", ""))

    line = f"- {name}"
    if rtype:
        line += f" [{rtype}]"
    parts.append(line)
    if ligand:
        parts.append(f"  Optimal ligand: {ligand}")
    if yield_val:
        parts.append(f"  Yield: {yield_val}")
    if doi:
        parts.append(f"  DOI: {doi}")

    return "\n".join(parts)


def _build_formatted_context(ligands: List[Dict[str, Any]],
                              reactions: List[Dict[str, Any]],
                              query_classes: List[str],
                              reaction_type: Optional[str]) -> str:
    """Build the final formatted context string for the LLM."""
    sections = []

    # Ligand data section
    if ligands:
        lig_lines = ["== Ligand Data =="]
        for lig in ligands:
            lig_lines.append(_format_ligand_entry(lig))
        sections.append("\n".join(lig_lines))

    # Reaction data section
    if reactions:
        rxn_lines = ["== Reaction Data =="]
        for rxn in reactions:
            rxn_lines.append(_format_reaction_entry(rxn))
        sections.append("\n".join(rxn_lines))

    # Comparison note when multiple classes detected
    if len(query_classes) > 1:
        sections.append(
            "== Retrieval Note ==\n"
            "The user is comparing multiple ligand classes. "
            "Data for ALL mentioned classes has been retrieved above. "
            "Do NOT claim data is missing for any mentioned ligand class "
            "unless its section above is genuinely empty."
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main RAL RAG class
# ---------------------------------------------------------------------------

class RALRAG:
    """Retrieve-and-rank engine for the RAL database.

    The critical design decision: **comparison queries must retrieve entries
    for every mentioned ligand class**, not just the top-scoring one.
    """

    MAX_LIGANDS_PER_CLASS = 5
    MAX_REACTIONS_PER_CLASS = 5
    MIN_SCORE_THRESHOLD = 1.0

    def __init__(self):
        self.ligands: List[Dict[str, Any]] = []
        self.reactions: List[Dict[str, Any]] = []

        data_dir = _find_data_dir()
        if data_dir:
            self.ligands = _load_ligands(data_dir)
            self.reactions = _load_reactions(data_dir)
            if not self.ligands and not self.reactions:
                logger.warning(
                    "RAL RAG: data dir found (%s) but no ligand/reaction files loaded",
                    data_dir,
                )
        else:
            logger.warning("RAL RAG: no data directory found — retrieval will return empty context")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_context(self, message: str) -> RALContext:
        """Analyse *message* and return relevant ligand + reaction entries.

        **Cross-class retrieval**: every ligand class mentioned in the query
        gets its own retrieval pass, so comparison queries are balanced.
        """
        if not self.ligands and not self.reactions:
            return RALContext()

        # 1. Parse the query
        query_classes = _extract_ligand_classes(message)
        reaction_type = _detect_reaction_type(message)

        if not query_classes:
            # No ligand class detected — fall back to fuzzy token matching
            return self._fuzzy_retrieve(message, reaction_type)

        # 2. Score ALL ligands against ALL query classes
        scored_ligands = [
            (lig, _score_ligand(lig, query_classes, reaction_type))
            for lig in self.ligands
        ]
        scored_ligands.sort(key=lambda x: x[1], reverse=True)

        # 3. Ensure balanced coverage: take top-N per class, not just global top-N
        selected_ligands = self._balanced_select(
            scored_ligands, query_classes, self.MAX_LIGANDS_PER_CLASS
        )

        # 4. Score reactions
        scored_reactions = [
            (rxn, _score_reaction(rxn, query_classes, reaction_type))
            for rxn in self.reactions
        ]
        scored_reactions.sort(key=lambda x: x[1], reverse=True)

        selected_reactions = self._balanced_select_reactions(
            scored_reactions, query_classes, self.MAX_REACTIONS_PER_CLASS
        )

        # 5. Format
        formatted = _build_formatted_context(
            selected_ligands, selected_reactions, query_classes, reaction_type
        )

        detected = query_classes[0] if query_classes else None
        return RALContext(
            formatted_context=formatted,
            ligands=selected_ligands,
            reactions=selected_reactions,
            detected_class=detected,
        )

    def build_enhanced_prompt(self, message: str, system_prompt: str) -> str:
        """Build a system-prompt-level enhanced prompt.

        Uses the same retrieval logic but injects context into the system
        prompt (legacy path used by ``chat/helpers.py``).
        """
        context = self.retrieve_context(message)
        if not context.formatted_context:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            f"DATABASE-GROUNDED ANSWERING RULES:\n"
            f"1. Use the database context below as your PRIMARY reference point. "
            f"Quote exact HOMO, LUMO, Gap, and omega values from the database when available. "
            f"Cite specific DOIs when discussing reactions.\n"
            f"2. You MAY supplement with your domain knowledge to provide: "
            f"mechanistic explanations, electronic-structure reasoning, "
            f"coordination-chemistry principles, and literature context beyond the database.\n"
            f"3. When comparing ligands, discuss ALL ligand classes mentioned by the user.\n\n"
            f"DATABASE CONTEXT:\n\n"
            f"{context.formatted_context}\n\n"
            f"END DATABASE CONTEXT"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _balanced_select(
        self,
        scored: List[Tuple[Dict, float]],
        query_classes: List[str],
        per_class_limit: int,
    ) -> List[Dict[str, Any]]:
        """Select results ensuring every query class is represented.

        Algorithm:
          1. For each query class, take the top *per_class_limit* scoring
             ligands whose class matches.
          2. Fill remaining slots with the highest-scoring entries overall.
          3. Deduplicate by primary key (name or first field).
        """
        selected: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        def _key(entry: Dict) -> str:
            return _normalise(entry.get("name", entry.get("ligand", "")))

        # Pass 1: per-class top-N
        for qc in query_classes:
            class_matches = [
                (entry, sc) for entry, sc in scored
                if entry.get("class", "").lower() == qc.lower()
                and sc >= self.MIN_SCORE_THRESHOLD
            ]
            for entry, _ in class_matches[:per_class_limit]:
                k = _key(entry)
                if k and k not in seen_keys:
                    selected.append(entry)
                    seen_keys.add(k)

        # Pass 2: fill with global top-N (for classes not in the database
        # but that may have related entries)
        for entry, sc in scored:
            if sc >= self.MIN_SCORE_THRESHOLD:
                k = _key(entry)
                if k and k not in seen_keys:
                    selected.append(entry)
                    seen_keys.add(k)
            if len(selected) >= per_class_limit * len(query_classes) * 2:
                break

        return selected

    def _balanced_select_reactions(
        self,
        scored: List[Tuple[Dict, float]],
        query_classes: List[str],
        per_class_limit: int,
    ) -> List[Dict[str, Any]]:
        """Select reactions ensuring coverage for each query class."""
        selected: List[Dict[str, Any]] = []
        seen_keys: Set[str] = set()

        def _key(entry: Dict) -> str:
            doi = entry.get("DOI", entry.get("doi", ""))
            name = entry.get("reaction_name", entry.get("name", ""))
            return _normalise(f"{doi}-{name}")

        # Per-class selection
        for qc in query_classes:
            class_matches = [
                (entry, sc) for entry, sc in scored
                if sc >= self.MIN_SCORE_THRESHOLD
                and self._reaction_mentions_class(entry, qc)
            ]
            for entry, _ in class_matches[:per_class_limit]:
                k = _key(entry)
                if k not in seen_keys:
                    selected.append(entry)
                    seen_keys.add(k)

        # Global top-up
        for entry, sc in scored:
            if sc >= self.MIN_SCORE_THRESHOLD:
                k = _key(entry)
                if k not in seen_keys:
                    selected.append(entry)
                    seen_keys.add(k)
            if len(selected) >= per_class_limit * len(query_classes) * 2:
                break

        return selected

    @staticmethod
    def _reaction_mentions_class(reaction: Dict, query_class: str) -> bool:
        """Check if a reaction entry references a given ligand class."""
        qc = query_class.lower()
        for field in ("ligand", "ligand_class", "optimal_ligand",
                       "ligand_norm", "ligand_class_norm", "optimal_ligand_norm"):
            val = _normalise(reaction.get(field, ""))
            if val == qc or qc in val or val in qc:
                return True
        # Also check description
        desc = _normalise(reaction.get("description", ""))
        if qc in desc:
            return True
        return False

    def _fuzzy_retrieve(self, message: str,
                        reaction_type: Optional[str]) -> RALContext:
        """Fallback when no specific ligand class is detected.

        Uses token overlap between the query and all ligand/reaction text
        fields to find the most relevant entries.
        """
        query_tokens = _tokenise(message)
        if not query_tokens:
            return RALContext()

        # Score ligands by token overlap
        scored_ligands = []
        for lig in self.ligands:
            text = " ".join(str(v) for v in lig.values())
            lig_tokens = _tokenise(text)
            overlap = len(query_tokens & lig_tokens)
            if overlap > 0:
                scored_ligands.append((lig, float(overlap)))

        scored_ligands.sort(key=lambda x: x[1], reverse=True)
        selected_ligands = [l for l, _ in scored_ligands[:10]]

        # Score reactions similarly
        scored_reactions = []
        for rxn in self.reactions:
            text = " ".join(str(v) for v in rxn.values())
            rxn_tokens = _tokenise(text)
            overlap = len(query_tokens & rxn_tokens)
            bonus = 4.0 if reaction_type and reaction_type.lower() in text.lower() else 0.0
            if overlap > 0 or bonus > 0:
                scored_reactions.append((rxn, float(overlap) + bonus))

        scored_reactions.sort(key=lambda x: x[1], reverse=True)
        selected_reactions = [r for r, _ in scored_reactions[:10]]

        formatted = _build_formatted_context(
            selected_ligands, selected_reactions, [], reaction_type
        )

        return RALContext(
            formatted_context=formatted,
            ligands=selected_ligands,
            reactions=selected_reactions,
            detected_class=None,
        )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_rag_instance: Optional[RALRAG] = None


def get_ral_rag() -> RALRAG:
    """Return the module-level RALRAG singleton (lazy-init)."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RALRAG()
    return _rag_instance


def reset_ral_rag() -> None:
    """Force re-creation on next ``get_ral_rag()`` call (useful in tests)."""
    global _rag_instance
    _rag_instance = None