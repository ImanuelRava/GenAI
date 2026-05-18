"""
NiCOBot RAG (Retrieval Augmented Generation) Service
Integrates database context with LLM for enhanced responses.
"""

import os
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .nicobot_database import get_database, NiCOBotDatabase

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Context retrieved from database for RAG."""
    compounds: List[Dict[str, Any]]
    papers: List[Dict[str, Any]]
    reactions: List[str]
    general_info: str
    formatted_context: str


class NiCOBotRAG:
    """
    RAG service for NiCOBot.
    Retrieves relevant context from database and enhances LLM prompts.
    """

    COMPOUND_KEYWORDS = [
        'smiles', 'structure', 'compound', 'molecule', 'leaving group',
        'electrophile', 'nucleophile', 'boronic', 'triflate', 'tosylate',
        'mesylate', 'acetate', 'grignard', 'organozinc', 'stannane',
        'phenyl', 'aryl', 'alkyl', 'benzyl', 'vinyl', 'ester', 'carbonate',
        'phosphate', 'carbamate', 'pivalate', 'benzoate', 'sulfonate',
        'boron', 'magnesium', 'zinc', 'tin', 'silane'
    ]

    PAPER_KEYWORDS = [
        'paper', 'publication', 'article', 'study', 'research',
        'author', 'doi', 'citation', 'reference', 'literature',
        'published', 'journal', 'jacs', 'angewandte', 'organic letters'
    ]

    REACTION_KEYWORDS = [
        'suzuki', 'heck', 'sonogashira', 'stille', 'kumada', 'negishi',
        'hiyama', 'cross-coupling', 'coupling reaction', 'c-o activation',
        'oxidative addition', 'transmetalation', 'reductive elimination',
        'cross coupling', 'catalysis', 'catalyst', 'reaction', 'coupling',
        'aryl', 'alkylation', 'arylation', 'vinylation'
    ]

    MECHANISM_KEYWORDS = [
        'mechanism', 'catalytic cycle', 'oxidative addition', 'transmetalation',
        'reductive elimination', 'ligand', 'catalyst', 'nickel', 'palladium',
        'activation', 'bond cleavage', 'insertion', 'migration'
    ]

    def __init__(self):
        self.db: Optional[NiCOBotDatabase] = None

    def _ensure_db(self):
        """Ensure database is loaded."""
        if self.db is None:
            self.db = get_database()
            if not self.db._loaded:
                self.db.load()

    def analyze_query(self, query: str) -> Dict[str, float]:
        """
        Analyze query to determine what type of information is needed.
        Returns scores for each category.
        """
        query_lower = query.lower()
        scores = {
            'compounds': 0.0,
            'papers': 0.0,
            'reactions': 0.0,
            'mechanism': 0.0
        }

        for kw in self.COMPOUND_KEYWORDS:
            if kw in query_lower:
                scores['compounds'] += 1.0

        for kw in self.PAPER_KEYWORDS:
            if kw in query_lower:
                scores['papers'] += 1.0

        for kw in self.REACTION_KEYWORDS:
            if kw in query_lower:
                scores['reactions'] += 1.0

        for kw in self.MECHANISM_KEYWORDS:
            if kw in query_lower:
                scores['mechanism'] += 1.0

        max_score = max(sum(scores.values()), 1.0)
        for k in scores:
            scores[k] = scores[k] / max_score

        return scores

    def retrieve_context(self, query: str, max_results: int = 5) -> RAGContext:
        """
        Retrieve relevant context from database for the query.
        """
        self._ensure_db()

        scores = self.analyze_query(query)

        compounds = []
        papers = []
        reactions = []
        general_info = ""

        query_words = re.findall(r'\w+', query.lower())

        for word in query_words:
            if len(word) > 3:
                found = self.db.search_compounds(word, limit=2)
                for c in found:
                    if c not in compounds:
                        compounds.append(c)

        found = self.db.search_compounds(query, limit=max_results)
        for c in found:
            if c not in compounds:
                compounds.append(c)

        compounds = compounds[:max_results]

        papers = self.db.search_papers(query, limit=max_results)

        query_lower = query.lower()
        for rxn in self.db.reactions.values():
            if rxn.name.lower() in query_lower:
                reactions.append(rxn.name)

        if scores['mechanism'] > 0.1 or scores['reactions'] > 0.1:
            general_info = self.db.get_cross_coupling_info()

        formatted = self._format_context(compounds, papers, reactions, general_info)

        return RAGContext(
            compounds=compounds,
            papers=papers,
            reactions=reactions,
            general_info=general_info,
            formatted_context=formatted
        )

    def _format_context(
        self,
        compounds: List[Dict],
        papers: List[Dict],
        reactions: List[str],
        general_info: str
    ) -> str:
        """Format retrieved context for inclusion in prompt."""
        parts = []

        if general_info:
            parts.append(general_info)

        if compounds:
            parts.append("\n### Relevant Compounds from Database:")
            for c in compounds[:5]:
                parts.append(f"- **{c['name']}** ({c['category']})")
                parts.append(f"  - SMILES: `{c['smiles']}`")
                if c.get('leaving_group'):
                    parts.append(f"  - Leaving Group: {c['leaving_group']}")
                if c.get('type'):
                    parts.append(f"  - Type: {c['type']}")

        if papers:
            parts.append("\n### Relevant Publications from Database:")
            for p in papers[:5]:
                parts.append(f"- **{p['title']}**")
                parts.append(f"  - DOI: {p['doi']}")
                parts.append(f"  - Published: {p['published_date']}")
                if p.get('authors'):
                    parts.append(f"  - Author: {p['authors']}")
                if p.get('reaction_type'):
                    parts.append(f"  - Reaction Type: {p['reaction_type']}")

        if reactions:
            parts.append("\n### Mentioned Reaction Types:")
            for rxn in reactions[:5]:
                parts.append(f"- {rxn}")

        return "\n".join(parts) if parts else ""

    def build_enhanced_prompt(
        self,
        user_message: str,
        system_prompt: str = None
    ) -> str:
        """
        Build an enhanced system prompt with database context.
        """
        context = self.retrieve_context(user_message)

        base_prompt = system_prompt or """You are NiCOBot, a specialized AI assistant for Nickel-catalyzed cross-coupling reactions and C-O bond activation chemistry.
Provide accurate, helpful responses about:
- Nickel catalysis mechanisms and applications
- C-O bond activation strategies
- Cross-coupling reactions (Suzuki, Heck, Kumada, etc.)
- Ligand design for transition metal catalysis
- Comparison of Ni vs Pd catalysis

Keep responses concise but informative. Use proper chemical nomenclature."""

        if context.formatted_context:
            enhanced_prompt = f"""{base_prompt}

## Database Context
The following information has been retrieved from the NiCOBot chemical database and should be used to enhance your response:

{context.formatted_context}

When answering, reference specific compounds, papers, or data from the database when relevant. If the user asks about a specific compound or reaction, provide the SMILES notation and any relevant publication references from the database."""
        else:
            enhanced_prompt = base_prompt

        return enhanced_prompt

    def get_compound_info_response(self, compound_name: str) -> Optional[str]:
        """
        Get detailed information about a specific compound.
        Returns a formatted response or None if not found.
        """
        self._ensure_db()
        compounds = self.db.search_compounds(compound_name, limit=1)

        if compounds:
            c = compounds[0]
            response = f"**{c['name']}**\n\n"
            response += f"- **Category**: {c['category']}\n"
            response += f"- **SMILES**: `{c['smiles']}`\n"
            if c.get('leaving_group'):
                response += f"- **Leaving Group**: {c['leaving_group']}\n"
            if c.get('type'):
                response += f"- **Type**: {c['type']}\n"
            return response

        return None

    def get_paper_info_response(self, query: str) -> Optional[str]:
        """
        Get information about papers matching a query.
        Returns a formatted response or None if not found.
        """
        self._ensure_db()
        papers = self.db.search_papers(query, limit=3)

        if papers:
            response = "### Relevant Publications:\n\n"
            for i, p in enumerate(papers, 1):
                response += f"{i}. **{p['title']}**\n"
                response += f"   - DOI: [{p['doi']}](https://doi.org/{p['doi']})\n"
                response += f"   - Published: {p['published_date']}\n"
                if p.get('authors'):
                    response += f"   - Author: {p['authors']}\n"
                response += "\n"
            return response

        return None

    def get_database_stats_response(self) -> str:
        """Get a formatted response about database statistics."""
        self._ensure_db()
        stats = self.db.get_statistics()

        response = "### NiCOBot Database Statistics\n\n"
        response += f"The NiCOBot database contains:\n\n"
        response += f"- **{stats['electrophiles']}** electrophile compounds\n"
        response += f"- **{stats['nucleophiles']}** nucleophile compounds\n"
        response += f"- **{stats['papers']}** research papers\n"
        response += f"- **{stats['reactions']}** reaction types\n\n"
        response += "I can search this database to provide specific compound information, "
        response += "publication references, and reaction data in my responses."

        return response


_rag_instance: Optional[NiCOBotRAG] = None


def get_rag() -> NiCOBotRAG:
    """Get or create the global RAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = NiCOBotRAG()
    return _rag_instance


def enhance_prompt_with_context(user_message: str, system_prompt: str = None) -> str:
    """Convenience function to enhance a prompt with database context."""
    rag = get_rag()
    return rag.build_enhanced_prompt(user_message, system_prompt)


def get_rag_context(query: str) -> RAGContext:
    """Convenience function to get RAG context for a query."""
    rag = get_rag()
    return rag.retrieve_context(query)
