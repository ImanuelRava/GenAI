"""
RAL (Redox-Active Ligand) RAG retrieval module.

Bridges the RAL database service (``ral_database.py``) with the chat
system (``chat/redox.py`` and ``chat/helpers.py``).  Delegates all data
loading and search to ``RALDatabase.search_combined()``, then formats
the results for LLM prompt injection.

The underlying database service uses the real research datasets:
  - Grand_Data.xlsx  — 238 ligands with DFT-computed electronic descriptors
  - DOI List.xlsx    — 49 curated reductive coupling reactions

Interface consumed by ``chat/redox.py`` and ``chat/helpers.py``:
    rag = get_ral_rag()
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
class RALContext:
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

    Input structure from RALDatabase.compare_ligand_classes():
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
# Main RAL RAG class
# ---------------------------------------------------------------------------

class RALRAG:
    """RAG bridge between the RAL database and the chat system.

    Delegates all data loading/search to ``RALDatabase`` via
    ``search_combined()``, then formats the structured results into
    a text context suitable for LLM prompt injection.
    """

    MAX_LIGANDS = 10
    MAX_REACTIONS = 10

    def retrieve_context(self, message: str) -> RALContext:
        """Retrieve relevant ligand and reaction data for a user message.

        Returns a ``RALContext`` with formatted context and structured data.
        Never raises — returns empty context on any error.
        """
        try:
            from .ral_database import get_ral_database
            db = get_ral_database()
        except Exception as e:
            logger.warning("RAL RAG: cannot load database: %s", e)
            return RALContext()

        if not db._loaded:
            if not db.load():
                logger.warning("RAL RAG: database load failed")
                return RALContext()

        try:
            result = db.search_combined(
                message,
                max_ligands=self.MAX_LIGANDS,
                max_reactions=self.MAX_REACTIONS,
            )
        except Exception as e:
            logger.warning("RAL RAG: search_combined error: %s", e)
            return RALContext()

        if not result['ligands'] and not result['reactions']:
            return RALContext()

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
            "RAL RAG: %d ligands, %d reactions, classes=%s",
            len(result['ligands']),
            len(result['reactions']),
            detected_classes,
        )

        return RALContext(
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