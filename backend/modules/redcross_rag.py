"""
RedCross (Reductive Coupling) RAG retrieval module.

Bridges the RedCross database service (``redcross_database.py``) with the chat
system (``chat/redcross.py`` and ``chat/helpers.py``).  Delegates all data
loading and search to ``RedCrossDatabase.search_combined()``, then formats
the results for LLM prompt injection.

The underlying database service uses the real research datasets:
  - Grand_Data.xlsx  — 238 ligands with DFT-computed electronic descriptors
  - DOI List.xlsx    — 49 curated reductive coupling reactions

Interface consumed by ``chat/redcross.py`` and ``chat/helpers.py``:
    rag = get_redcross_rag()
    context = rag.retrieve_context(message)
    # context.formatted_context  – str
    # context.ligands           – list[dict]
    # context.reactions         – list[dict]
    # context.detected_class    – str | None
    enhanced = rag.build_enhanced_prompt(message, system_prompt)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RedCrossContext:
    """Returned by ``retrieve_context()``."""
    formatted_context: str = ""
    ligands: List[Dict[str, Any]] = field(default_factory=list)
    reactions: List[Dict[str, Any]] = field(default_factory=list)
    detected_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_ligand(lig: Dict[str, Any]) -> str:
    """Format a single ligand entry for LLM consumption."""
    parts = [f"- {lig['name']} (class: {lig['class']})"]
    parts.append(f"  HOMO: {lig.get('HOMO_eV', 'N/A')} eV")
    parts.append(f"  LUMO: {lig.get('LUMO_eV', 'N/A')} eV")
    parts.append(f"  Gap:  {lig.get('Gap_eV', 'N/A')} eV")
    parts.append(f"  omega: {lig.get('omega_eV', 'N/A')} eV")
    parts.append(f"  I_min: {lig.get('I_min_eV', 'N/A')} eV")
    parts.append(f"  V_min: {lig.get('V_min_eV', 'N/A')} eV")
    parts.append(f"  R1-HOMA: {lig.get('R1_HOMA', 'N/A')}")
    parts.append(f"  R2-HOMA: {lig.get('R2_HOMA', 'N/A')}")
    return "\n".join(parts)


def _format_reaction(rxn: Dict[str, Any]) -> str:
    """Format a single reaction entry for LLM consumption."""
    parts = [f"- {rxn.get('title', 'Untitled reaction')}"]
    if rxn.get('optimum_ligand'):
        parts.append(f"  Optimum ligand: {rxn['optimum_ligand']}")
    if rxn.get('coupling_partner'):
        parts.append(f"  Coupling partner: {rxn['coupling_partner']}")
    if rxn.get('ligand_knowledge'):
        parts.append(f"  Ligand knowledge: {rxn['ligand_knowledge']}")
    if rxn.get('doi'):
        parts.append(f"  DOI: {rxn['doi']}")
    return "\n".join(parts)


def _format_class_comparison(comparison: Dict[str, Any]) -> str:
    """Format a class comparison dict for LLM consumption.

    Input structure from RedCrossDatabase.compare_ligand_classes():
        {'homo': {'Bpy': -5.8, 'BiOX': -6.1, 'diff': 0.3}, ...}
    """
    if not comparison:
        return ""

    # Extract class names from any non-'count' key
    sample_key = next((k for k in comparison if k != 'count'), None)
    if not sample_key:
        return ""
    class_names = [k for k in comparison[sample_key] if k != 'diff']

    display_map = {
        'homo': 'HOMO (eV)', 'lumo': 'LUMO (eV)', 'gap': 'Gap (eV)',
        'omega': 'omega (eV)', 'i_min': 'I_min (eV)', 'v_min': 'V_min (eV)',
        'r1_homa': 'R1-HOMA', 'r2_homa': 'R2-HOMA',
    }

    lines = ["== Class Comparison (Average Electronic Descriptors) =="]
    for attr, display in display_map.items():
        if attr in comparison:
            vals = comparison[attr]
            cls_vals = [f"{cls}: {vals[cls]}" for cls in class_names if cls in vals]
            diff = vals.get('diff', '')
            diff_str = f" (diff: {diff})" if diff else ""
            lines.append(f"  {display}: {' | '.join(cls_vals)}{diff_str}")

    if 'count' in comparison:
        count_info = comparison['count']
        count_str = " | ".join(
            f"{cls}: {count_info[cls]} ligands" for cls in class_names if cls in count_info
        )
        lines.append(f"  Ligand counts: {count_str}")

    return "\n".join(lines)


def _build_formatted_context(
    ligands: List[Dict[str, Any]],
    reactions: List[Dict[str, Any]],
    detected_classes: List[str],
    class_comparison: Optional[Dict[str, Any]],
) -> str:
    """Build the final formatted context string for the LLM."""
    sections = []

    if ligands:
        lig_lines = ["== Ligand Electronic Properties (from Grand_Data) =="]
        # Group by class for readability
        by_class: Dict[str, List[Dict]] = {}
        for l in ligands:
            by_class.setdefault(l.get('class', 'unknown'), []).append(l)
        for cls in detected_classes or list(by_class.keys()):
            if cls in by_class:
                lig_lines.append(f"\n--- {cls} ---")
                for l in by_class[cls]:
                    lig_lines.append(_format_ligand(l))
        # Add any classes not in detected_classes
        for cls, cls_ligands in by_class.items():
            if cls not in (detected_classes or []):
                lig_lines.append(f"\n--- {cls} ---")
                for l in cls_ligands:
                    lig_lines.append(_format_ligand(l))
        sections.append("\n".join(lig_lines))

    if reactions:
        rxn_lines = ["== Reaction Literature (from DOI List) =="]
        for r in reactions:
            rxn_lines.append(_format_reaction(r))
        sections.append("\n".join(rxn_lines))

    if class_comparison:
        sections.append(_format_class_comparison(class_comparison))

    # Comparison note for multi-class queries
    if len(detected_classes or []) > 1:
        class_list = ", ".join(detected_classes)
        sections.append(
            "== Retrieval Note ==\n"
            f"The user is comparing multiple ligand classes: {class_list}. "
            "Data for ALL mentioned classes has been retrieved above. "
            "Do NOT claim data is missing for any mentioned ligand class "
            "unless its section above is genuinely empty."
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Main RedCross RAG class
# ---------------------------------------------------------------------------

class RedCrossRAG:
    """RAG bridge between the RedCross database and the chat system.

    Delegates all data loading/search to ``RedCrossDatabase`` via
    ``search_combined()``, then formats the structured results into
    a text context suitable for LLM prompt injection.
    """

    MAX_LIGANDS = 10
    MAX_REACTIONS = 10

    def retrieve_context(self, message: str) -> RedCrossContext:
        """Retrieve relevant ligand and reaction data for a user message.

        Returns a ``RedCrossContext`` with formatted context and structured data.
        Never raises — returns empty context on any error.
        """
        try:
            from .redcross_database import get_redcross_database
            db = get_redcross_database()
        except Exception as e:
            logger.warning("RedCross RAG: cannot load database: %s", e)
            return RedCrossContext()

        if not db._loaded:
            if not db.load():
                logger.warning("RedCross RAG: database load failed")
                return RedCrossContext()

        try:
            result = db.search_combined(
                message,
                max_ligands=self.MAX_LIGANDS,
                max_reactions=self.MAX_REACTIONS,
            )
        except Exception as e:
            logger.warning("RedCross RAG: search_combined error: %s", e)
            return RedCrossContext()

        if not result['ligands'] and not result['reactions']:
            return RedCrossContext()

        detected_classes = result.get('detected_classes') or []
        if result.get('detected_class') and not detected_classes:
            detected_classes = [result['detected_class']]

        class_comparison = result.get('class_comparison')

        formatted = _build_formatted_context(
            result['ligands'],
            result['reactions'],
            detected_classes,
            class_comparison,
        )

        logger.info(
            "RedCross RAG: %d ligands, %d reactions, classes=%s",
            len(result['ligands']),
            len(result['reactions']),
            detected_classes,
        )

        return RedCrossContext(
            formatted_context=formatted,
            ligands=result['ligands'],
            reactions=result['reactions'],
            detected_class=result.get('detected_class'),
        )

    def build_enhanced_prompt(self, message: str, system_prompt: str) -> str:
        """Build a system-prompt-level enhanced prompt.

        Injects database context into the system prompt (legacy path
        used by ``chat/helpers.py``).
        """
        context = self.retrieve_context(message)
        if not context.formatted_context:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            f"DATABASE-ANSWERING RULES:\n"
            f"RULE 1 — DATABASE USAGE IS MANDATORY: "
            f"You MUST base every factual claim (electronic parameters, redox potentials, "
            f"reaction yields, catalytic performance, etc.) on the database context provided "
            f"in the user message. Start your answer with database-sourced data and "
            f"present it prominently.\n\n"
            f"RULE 2 — EXPLICITLY ATTRIBUTE DATABASE DATA: "
            f"Whenever you use a value, fact, or finding from the database, you MUST "
            f"explicitly state that it comes from the database. Use phrasing such as:\n"
            f'  - "According to the database, …"\n'
            f'  - "The database reports a HOMO value of …"\n'
            f'  - "Database records indicate … (DOI: …)"\n'
            f"Do NOT present database-sourced information as if it were your own knowledge.\n\n"
            f"RULE 3 — FALLBACK TO LLM KNOWLEDGE IS A LAST RESORT: "
            f"You may ONLY fall back to your general / training-data knowledge when:\n"
            f"  (a) the specific information is genuinely NOT available in the database context, AND\n"
            f"  (b) you have already exhausted what the database provides.\n"
            f"In that case you MUST explicitly label the information as LLM knowledge, "
            f"using phrasing such as:\n"
            f'  - "(Note: the following is based on general knowledge from LLM training, not from the database.)"\n'
            f'  - "Beyond the database, from general chemistry knowledge: …"\n'
            f"You may NEVER silently substitute LLM knowledge for database data.\n\n"
            f"RULE 4 — COMPARISON QUERIES: "
            f"When the user compares ligands or reaction types, you MUST discuss ALL classes "
            f"mentioned. Use the database data for each class. If the database has data for "
            f"one class but not another, present the available data normally and state clearly "
            f"for the other: 'The database does not contain entries for [class X].'\n"
            f"Do NOT fabricate 'no data' claims — only say this when the database context "
            f"for that class is genuinely empty.\n\n"
            f"RULE 5 — DOIs AND CITATIONS: "
            f"When the database includes DOIs, always cite them. Format: (DOI: 10.xxxx/…).\n\n"
            f"DATABASE CONTEXT:\n\n"
            f"{context.formatted_context}\n\n"
            f"END DATABASE CONTEXT"
        )


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_redcross_rag_instance: Optional[RedCrossRAG] = None


def get_redcross_rag() -> RedCrossRAG:
    """Return the module-level RedCrossRAG singleton (lazy-init)."""
    global _redcross_rag_instance
    if _redcross_rag_instance is None:
        _redcross_rag_instance = RedCrossRAG()
    return _redcross_rag_instance


def reset_redcross_rag() -> None:
    """Force re-creation on next ``get_redcross_rag()`` call (useful in tests)."""
    global _redcross_rag_instance
    _redcross_rag_instance = None