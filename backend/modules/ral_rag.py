"""
RAL (Redox-Active Ligands) RAG Service

Retrieval-Augmented Generation layer that sits on top of RALDatabase.
Analyses incoming chat queries, pulls relevant ligand properties and
reaction-ligand literature, and formats everything for LLM prompt injection.

Design mirrors ``modules/nicobot_rag.py`` but is tailored for the
reductive-coupling / ligand-electronic-property domain.
"""

import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .ral_database import get_ral_database, RALDatabase

logger = logging.getLogger(__name__)


@dataclass
class RALRAGContext:
    """Context retrieved from the RAL database for RAG."""
    ligands: List[Dict[str, Any]]
    reactions: List[Dict[str, Any]]
    detected_class: str
    ligand_class_info: Optional[Dict[str, Any]]
    formatted_context: str


class RALRAG:
    """
    RAG service for the Redox-Active Ligands chatbot (RAL-Bot).

    Provides keyword-aware retrieval across:
      - Grand_Data: DFT electronic descriptors (HOMO, LUMO, gap, omega, etc.)
      - DOI List:   Literature-curated reaction-ligand knowledge
    """

    # Keywords that signal the user is asking about ligand properties
    LIGAND_PROPERTY_KEYWORDS = [
        'homo', 'lumo', 'gap', 'electrophilicity', 'omega', 'ionization',
        'electron affinity', 'homa', 'aromaticity', 'electronic',
        'descriptor', 'property', 'properties', 'dft', 'computed',
        'energy', 'orbital', 'frontier', 'hardness', 'softness',
    ]

    # Keywords that signal the user is asking about reductive coupling reactions
    REACTION_KEYWORDS = [
        'reductive coupling', 'cross-electrophile', 'cross electrophile',
        'reaction', 'coupling', 'nickel', 'ni-', 'catalyzed',
        'electrophile', 'nucleophile', 'oxidative addition',
        'reductive elimination', 'radical', 'single-electron',
        'set', 'cross-coupling', 'c-o activation', 'c-o bond',
        'alkyl', 'aryl', 'vinyl', 'allyl', 'triflate', 'bromide',
        'chloride', 'iodide', 'halide', 'ketone', 'ester', 'amide',
        'boronate', 'grignard', 'zinc', 'manganese', 'zinc',
    ]

    # Keywords about literature / papers / DOI
    LITERATURE_KEYWORDS = [
        'paper', 'publication', 'doi', 'article', 'study', 'reference',
        'literature', 'cite', 'cited', 'author', 'journal',
        'published', 'report',
    ]

    # Keywords that signal a recommendation / suggestion request
    RECOMMENDATION_KEYWORDS = [
        'recommend', 'suggest', 'alternative', 'similar', 'instead of',
        'substitute', 'replacement', 'what ligand', 'which ligand',
        'best ligand', 'good ligand', 'suitable ligand', 'choose',
        'selection', 'pick', 'option', 'comparable', 'equivalent',
        'compare', 'versus', ' vs ', 'difference between',
    ]

    # Keywords about specific ligand classes / families
    LIGAND_CLASS_KEYWORDS = [
        'bipyridine', 'bpy', 'dtbpy', 'dtbbpy', 'phenanthroline', 'phen',
        'bathophenanthroline', 'bphen', 'terpyridine', 'neocuproine',
        'bis(oxazoline)', 'box', 'bisoxazoline', 'chiral box',
        'bis(imidazoline)', 'biim', 'biimidazoline',
        'pyrox', 'pyrim', 'pycam', 'pyridine', 'imidazole',
        'amidine', 'pybcam',
        'redox-active', 'non-innocent', 'redox active', 'non innocent',
    ]

    def __init__(self):
        self.db: Optional[RALDatabase] = None

    def _ensure_db(self):
        """Ensure the RAL database is loaded."""
        if self.db is None:
            self.db = get_ral_database()
            if not self.db._loaded:
                self.db.load()

    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Classify the query intent by scoring keyword categories.

        Returns a dict with normalised scores (0-1) for each category.
        """
        query_lower = query.lower()
        scores = {
            'ligand_properties': 0.0,
            'reactions': 0.0,
            'literature': 0.0,
            'ligand_class': 0.0,
            'recommendation': 0.0,
        }

        for kw in self.LIGAND_PROPERTY_KEYWORDS:
            if kw in query_lower:
                scores['ligand_properties'] += 1.0

        for kw in self.REACTION_KEYWORDS:
            if kw in query_lower:
                scores['reactions'] += 1.0

        for kw in self.LITERATURE_KEYWORDS:
            if kw in query_lower:
                scores['literature'] += 1.0

        for kw in self.LIGAND_CLASS_KEYWORDS:
            if kw in query_lower:
                scores['ligand_class'] += 1.0

        for kw in self.RECOMMENDATION_KEYWORDS:
            if kw in query_lower:
                scores['recommendation'] += 1.0

        # Normalise to 0-1
        total = sum(scores.values())
        if total > 0:
            for k in scores:
                scores[k] /= total

        return scores

    def retrieve_context(self, query: str,
                         max_ligands: int = 5,
                         max_reactions: int = 5) -> RALRAGContext:
        """
        Main retrieval method.  Combines results from both datasets
        and returns formatted context ready for prompt injection.
        """
        self._ensure_db()

        scores = self.analyze_query(query)

        # Decide retrieval priorities based on query intent
        # If heavy on properties, fetch more ligands; if heavy on
        # reactions/literature, fetch more reactions.
        if scores['ligand_properties'] > 0.3:
            max_ligands = max(max_ligands, 8)
        if scores['reactions'] > 0.3 or scores['literature'] > 0.3:
            max_reactions = max(max_reactions, 8)

        combined = self.db.search_combined(query,
                                           max_ligands=max_ligands,
                                           max_reactions=max_reactions)

        ligands = combined['ligands']
        reactions = combined['reactions']
        detected_class = combined['detected_class']

        # If a class was detected, also pull class-level summary
        class_info = None
        if detected_class:
            class_info = self.db.get_ligand_classes().get(detected_class)

        # --- Similarity-based recommendations ---
        similar_ligands = []
        if scores['recommendation'] > 0.15:
            similar_ligands = self._get_similar_recommendations(
                query, reactions, ligands, detected_class
            )

        formatted = self._format_context(ligands, reactions,
                                          detected_class, class_info,
                                          similar_ligands=similar_ligands)

        return RALRAGContext(
            ligands=ligands,
            reactions=reactions,
            detected_class=detected_class or '',
            ligand_class_info=class_info,
            formatted_context=formatted,
        )

    def _get_similar_recommendations(
        self,
        query: str,
        reactions: List[Dict],
        ligands: List[Dict],
        detected_class: str,
    ) -> List[Dict[str, Any]]:
        """
        Use the similarity engine to recommend ligands based on
        what the literature says works for similar reactions.

        Strategy:
          1. If reactions were found, take the optimum ligand name
             and find its electronic-profile neighbours.
          2. If a ligand name was mentioned in the query, find
             similar ligands to that one.
          3. If a class was detected, find the most representative
             ligand and recommend from the same cluster.

        The similarity engine uses UMAP (preferred) or PCA (fallback)
        for dimensionality reduction, then KNN in the embedding space.
        """
        try:
            from .ral_similarity import get_similarity_engine
        except ImportError:
            return []

        try:
            engine = get_similarity_engine()
            if not engine._fitted:
                return []
        except Exception as e:
            logger.warning(f"Similarity engine not available: {e}")
            return []

        # Store method name for context formatting
        self._sim_method = engine._method

        results = []
        query_lower = query.lower()

        # Strategy 1: Use optimum ligand from reaction results
        if reactions:
            opt_ligand = reactions[0].get('optimum_ligand', '')
            if opt_ligand:
                # Try to find a matching ligand in Grand_Data
                # The DOI list uses full names; try key fragments
                for kw in self.LIGAND_CLASS_KEYWORDS + ['bpy', 'phen', 'pyrox', 'pyrim', 'pycam']:
                    if kw in opt_ligand.lower():
                        # Search for a ligand of this class
                        class_matches = self.db.search_ligands(
                            kw, limit=1
                        )
                        if class_matches:
                            similar = engine.recommend_for_ligand(
                                class_matches[0]['name'], k=5
                            )
                            results.extend([
                                self._similarity_result_to_dict(s)
                                for s in similar
                            ])
                            break

        # Strategy 2: User mentioned a specific ligand name
        if not results:
            for lp in ligands:
                if lp['name'].lower() in query_lower:
                    similar = engine.recommend_for_ligand(
                        lp['name'], k=5
                    )
                    results.extend([
                        self._similarity_result_to_dict(s)
                        for s in similar
                    ])
                    break

        # Strategy 3: Class detected → pick class representative
        if not results and detected_class:
            class_ligands = self.db.get_ligands_by_class(detected_class)
            if class_ligands:
                similar = engine.recommend_for_ligand(
                    class_ligands[0]['name'], k=5
                )
                results.extend([
                    self._similarity_result_to_dict(s)
                    for s in similar
                ])

        # Deduplicate by name
        seen = set()
        unique = []
        for r in results:
            if r['name'] not in seen:
                seen.add(r['name'])
                unique.append(r)

        return unique[:5]

    @staticmethod
    def _similarity_result_to_dict(s) -> Dict[str, Any]:
        """Convert a SimilarityResult to a plain dict."""
        return {
            'name': s.name,
            'class': s.ligand_class,
            'distance': round(s.distance, 4),
            'cosine_similarity': round(s.cosine_similarity, 4),
            'embedding_coords': [round(c, 4) for c in s.embedding_coords],
            'pca_coords': [round(c, 4) for c in s.pca_coords],
            'features': s.features,
        }

    def _format_context(
        self,
        ligands: List[Dict],
        reactions: List[Dict],
        detected_class: str,
        class_info: Optional[Dict],
        similar_ligands: List[Dict] = None,
    ) -> str:
        """Format retrieved data into a Markdown block for LLM prompt."""
        parts = []

        # --- Ligand properties section ---
        if ligands:
            parts.append("### Ligand Electronic Properties (from DFT database):")
            for lp in ligands[:8]:
                line = (f"- **{lp['name']}** ({lp['class']}): "
                        f"HOMO={lp['HOMO_eV']:.3f} eV, "
                        f"LUMO={lp['LUMO_eV']:.3f} eV, "
                        f"Gap={lp['Gap_eV']:.3f} eV, "
                        f"\u03c9={lp['omega_eV']:.3f} eV")
                parts.append(line)

            # If class detected, add class summary
            if class_info:
                ci = class_info
                parts.append(
                    f"\n**{detected_class} class summary** ({ci['count']} ligands): "
                    f"HOMO range [{ci['HOMO_range'][0]:.3f}, {ci['HOMO_range'][1]:.3f}] eV, "
                    f"LUMO range [{ci['LUMO_range'][0]:.3f}, {ci['LUMO_range'][1]:.3f}] eV, "
                    f"Gap range [{ci['Gap_range'][0]:.3f}, {ci['Gap_range'][1]:.3f}] eV, "
                    f"\u03c9 range [{ci['omega_range'][0]:.3f}, {ci['omega_range'][1]:.3f}] eV"
                )

        # --- Reaction-ligand literature section ---
        if reactions:
            parts.append("\n### Relevant Reductive Coupling Reactions (from literature database):")
            for r in reactions[:5]:
                parts.append(f"- **{r['title']}**")
                parts.append(f"  - DOI: {r['doi']}")
                if r['optimum_ligand']:
                    parts.append(f"  - Optimum Ligand: {r['optimum_ligand']}")
                if r['coupling_partner']:
                    parts.append(f"  - Coupling Partner: {r['coupling_partner']}")
                if r['mapped_class']:
                    parts.append(f"  - Ligand Class: {r['mapped_class']}")
                # Include ligand knowledge (truncated for prompt brevity)
                if r['ligand_knowledge']:
                    knowledge = r['ligand_knowledge']
                    if len(knowledge) > 600:
                        knowledge = knowledge[:597] + "..."
                    parts.append(f"  - Ligand Knowledge: {knowledge}")
                parts.append("")

        # --- Similarity-based recommendations section ---
        if similar_ligands:
            method_label = getattr(self, '_sim_method', 'UMAP').upper()
            parts.append(
                f"\n### Similar Ligand Recommendations ({method_label}-based similarity):"
            )
            parts.append(
                "The following ligands have similar electronic profiles "
                "(based on " + method_label + " dimensionality reduction of "
                "HOMO, LUMO, Gap, ω, I_min, V_min, R1-HOMA, R2-HOMA):"
            )
            for sl in similar_ligands[:5]:
                f = sl['features']
                line = (
                    f"- **{sl['name']}** ({sl['class']}): "
                    f"HOMO={f.get('HOMO (eV)', 0):.3f}, "
                    f"LUMO={f.get('LUMO (eV)', 0):.3f}, "
                    f"Gap={f.get('Gap (eV)', 0):.3f}, "
                    f"ω={f.get('ω (eV)', 0):.3f} "
                    f"[distance={sl['distance']:.3f}, "
                    f"cosine={sl['cosine_similarity']:.3f}]"
                )
                parts.append(line)
            parts.append(
                "These ligands are recommended as alternatives because they "
                "share similar electronic properties in the "
                + method_label + "-reduced descriptor space."
            )

        return "\n".join(parts) if parts else ""

    def build_enhanced_prompt(
        self,
        user_message: str,
        system_prompt: str = None,
    ) -> str:
        """
        Build a RAG-enhanced system prompt for the RAL-Bot.
        """
        context = self.retrieve_context(user_message)

        base_prompt = system_prompt or (
            "You are a specialized AI assistant for Redox-Active Ligands "
            "chemistry and reductive cross-coupling reactions."
        )

        if context.formatted_context:
            enhanced = (
                f"{base_prompt}\n\n"
                f"The following data has been retrieved from the RAL "
                f"research database. Use this to provide specific, "
                f"data-backed answers:\n\n"
                f"{context.formatted_context}\n\n"
                "When answering:\n"
                "- Reference specific ligand names, electronic descriptors, "
                "and DOI-backed literature when relevant.\n"
                "- If the user asks about ligand properties, compare values "
                "across classes using the HOMO/LUMO/Gap/\u03c9 data.\n"
                "- If the user asks about which ligand to use for a reaction, "
                "cite the relevant papers and their optimum ligand findings.\n"
                "- If no database match is found, answer from general "
                "knowledge but note that no database match was found."
            )
            return enhanced

        return base_prompt

    def get_ligand_info_response(self, ligand_name: str) -> Optional[str]:
        """Get a formatted string about a specific ligand's properties."""
        self._ensure_db()
        ligands = self.db.search_ligands(ligand_name, limit=3)
        if not ligands:
            return None

        response = "### Ligand Properties:\n\n"
        for lp in ligands:
            response += f"**{lp['name']}** ({lp['class']})\n"
            response += f"- HOMO: {lp['HOMO_eV']:.3f} eV\n"
            response += f"- LUMO: {lp['LUMO_eV']:.3f} eV\n"
            response += f"- HOMO-LUMO Gap: {lp['Gap_eV']:.3f} eV\n"
            response += f"- Electrophilicity (\u03c9): {lp['omega_eV']:.3f} eV\n"
            response += f"- Ionization Potential (I_min): {lp['I_min_eV']:.3f} eV\n"
            response += f"- Electron Affinity (V_min): {lp['V_min_eV']:.3f} eV\n"
            response += f"- R1 Aromaticity (HOMA): {lp['R1_HOMA']:.3f}\n"
            response += f"- R2 Aromaticity (HOMA): {lp['R2_HOMA']:.3f}\n\n"

        return response

    def get_database_stats_response(self) -> str:
        """Get a formatted response about RAL database statistics."""
        self._ensure_db()
        stats = self.db.get_statistics()
        classes = self.db.get_ligand_classes()

        response = "### RAL Database Statistics\n\n"
        response += f"The RAL database contains:\n\n"
        response += f"- **{stats['ligands']}** ligands across **{stats['ligand_classes']}** classes\n"
        for cls_name, info in classes.items():
            response += f"  - {cls_name}: {info['count']} ligands\n"
        response += f"\n- **{stats['reactions_curated']}** curated reductive coupling reactions"
        response += f" (with ligand knowledge)\n"
        response += f"- **{stats['reactions_doi_only']}** additional DOIs awaiting curation\n"
        response += "\nI can search this database to provide specific ligand electronic properties, "
        response += "literature-backed ligand recommendations, and reaction-ligand comparisons."

        return response


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_rag_instance: Optional[RALRAG] = None


def get_ral_rag() -> RALRAG:
    """Get or create the global RAL RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RALRAG()
    return _rag_instance


def enhance_redox_prompt(user_message: str,
                         system_prompt: str = None) -> str:
    """Convenience function to enhance a RAL-Bot prompt with database context."""
    rag = get_ral_rag()
    return rag.build_enhanced_prompt(user_message, system_prompt)