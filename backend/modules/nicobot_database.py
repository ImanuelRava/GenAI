"""
NiCOBot Database Service
Provides access to chemical reaction data and compounds.
Integrates with LLM for context-aware responses.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CompoundInfo:
    """Information about a chemical compound."""
    smiles: str
    name: str
    leaving_group: str = ""
    compound_type: str = ""
    category: str = ""


@dataclass
class PaperInfo:
    """Information about a research paper."""
    doi: str
    title: str
    published_date: str
    authors: str = ""
    references: List[str] = field(default_factory=list)
    reaction_type: str = ""
    strength: str = ""


@dataclass
class ReactionInfo:
    """Information about a reaction type."""
    reaction_id: str
    name: str
    category: str = ""


class NiCOBotDatabase:
    """
    Database service for NiCOBot.
    Loads and indexes chemical data for intelligent retrieval.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nicobot_data')
        self.data_dir = Path(data_dir)

        self.electrophiles: Dict[str, CompoundInfo] = {}
        self.nucleophiles: Dict[str, CompoundInfo] = {}
        self.papers: Dict[str, PaperInfo] = {}
        self.reactions: Dict[str, ReactionInfo] = {}

        self._name_index: Dict[str, List[str]] = {}
        self._smiles_index: Dict[str, str] = {}
        self._keyword_index: Dict[str, List[str]] = {}

        self._loaded = False

    def load(self) -> bool:
        """Load all data from files."""
        if self._loaded:
            return True

        try:
            self._load_electrophiles()
            self._load_nucleophiles()
            self._load_papers()
            self._load_reactions()
            self._build_indices()
            self._loaded = True
            logger.info(f"NiCOBot Database loaded: {len(self.electrophiles)} electrophiles, "
                       f"{len(self.nucleophiles)} nucleophiles, {len(self.papers)} papers")
            return True
        except Exception as e:
            logger.error(f"Failed to load NiCOBot database: {e}")
            return False

    def _load_electrophiles(self):
        """Load electrophile data."""
        lvg_path = self.data_dir / 'E_LVG.json'
        if lvg_path.exists():
            with open(lvg_path) as f:
                lvg_data = json.load(f)

        major_path = self.data_dir / 'E_Major.json'
        if major_path.exists():
            with open(major_path) as f:
                major_data = json.load(f)

        type_path = self.data_dir / 'E_Type.json'
        if type_path.exists():
            with open(type_path) as f:
                type_data = json.load(f)

        smiles_path = self.data_dir / 'E_LVG_name_smiles.json'
        if smiles_path.exists():
            with open(smiles_path) as f:
                smiles_data = json.load(f)

            for smiles, info in smiles_data.items():
                if isinstance(info, list) and len(info) >= 2:
                    name = info[0]
                    leaving_group_smiles = info[1] if len(info) > 1 else ""

                    lg_type = self._infer_leaving_group(name)

                    self.electrophiles[smiles] = CompoundInfo(
                        smiles=smiles,
                        name=name,
                        leaving_group=lg_type,
                        category='electrophile'
                    )

    def _load_nucleophiles(self):
        """Load nucleophile data."""
        lvg_path = self.data_dir / 'Nu_LVG.json'
        if lvg_path.exists():
            with open(lvg_path) as f:
                lvg_data = json.load(f)

        major_path = self.data_dir / 'Nu_Major.json'
        if major_path.exists():
            with open(major_path) as f:
                major_data = json.load(f)

        type_path = self.data_dir / 'Nu_Type.json'
        if type_path.exists():
            with open(type_path) as f:
                type_data = json.load(f)

        smiles_path = self.data_dir / 'Nu_LVG_name_smiles.json'
        if smiles_path.exists():
            with open(smiles_path) as f:
                smiles_data = json.load(f)

            for smiles, info in smiles_data.items():
                if isinstance(info, list) and len(info) >= 2:
                    name = info[0]
                    leaving_group_smiles = info[1] if len(info) > 1 else ""

                    nucl_type = self._infer_nucleophile_type(name)

                    self.nucleophiles[smiles] = CompoundInfo(
                        smiles=smiles,
                        name=name,
                        leaving_group=leaving_group_smiles,
                        compound_type=nucl_type,
                        category='nucleophile'
                    )

    def _load_papers(self):
        """Load paper metadata and citation data."""
        results_path = self.data_dir / 'results_modify_add.json'
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)

            for doi, info in data.items():
                title = info.get('title', '')
                published_date = info.get('published_date', '')
                references = [ref[0] for ref in info.get('references', []) if ref]

                self.papers[doi] = PaperInfo(
                    doi=doi,
                    title=title,
                    published_date=published_date,
                    references=references
                )

        nodes_path = self.data_dir / 'citation-network-nodes.csv'
        if nodes_path.exists():
            try:
                import csv
                with open(nodes_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        doi = row.get('global_reference', '')
                        if doi and doi in self.papers:
                            self.papers[doi].authors = row.get('last_author', '')
                            self.papers[doi].reaction_type = row.get('Reaction Type', '')
                            self.papers[doi].strength = row.get('Strength', '')
            except Exception as e:
                logger.warning(f"Could not load citation nodes: {e}")

    def _load_reactions(self):
        """Load reaction type information."""
        reactions_path = self.data_dir / 'schneider_50k.json'
        if reactions_path.exists():
            with open(reactions_path) as f:
                data = json.load(f)

            for rxn_id, name in data.items():
                self.reactions[rxn_id] = ReactionInfo(
                    reaction_id=rxn_id,
                    name=name
                )

    def _build_indices(self):
        """Build search indices for fast lookup."""
        for key, compound in self.electrophiles.items():
            name_lower = compound.name.lower()
            if name_lower not in self._name_index:
                self._name_index[name_lower] = []
            self._name_index[name_lower].append(f"e:{key}")
            self._smiles_index[key] = f"e:{key}"

        for key, compound in self.nucleophiles.items():
            name_lower = compound.name.lower()
            if name_lower not in self._name_index:
                self._name_index[name_lower] = []
            self._name_index[name_lower].append(f"n:{key}")
            self._smiles_index[key] = f"n:{key}"

        for doi, paper in self.papers.items():
            title_words = re.findall(r'\w+', paper.title.lower())
            for word in title_words:
                if len(word) > 3:
                    if word not in self._keyword_index:
                        self._keyword_index[word] = []
                    self._keyword_index[word].append(doi)

    def _infer_leaving_group(self, name: str) -> str:
        """Infer leaving group type from compound name."""
        name_lower = name.lower()
        if 'triflate' in name_lower:
            return 'triflate'
        elif 'tosylate' in name_lower:
            return 'tosylate'
        elif 'mesylate' in name_lower:
            return 'mesylate'
        elif 'acetate' in name_lower:
            return 'acetate'
        elif 'benzoate' in name_lower:
            return 'benzoate'
        elif 'phosphate' in name_lower:
            return 'phosphate'
        elif 'carbamate' in name_lower:
            return 'carbamate'
        elif 'pivalate' in name_lower:
            return 'pivalate'
        else:
            return 'other'

    def _infer_nucleophile_type(self, name: str) -> str:
        """Infer nucleophile type from compound name."""
        name_lower = name.lower()
        if 'boronic' in name_lower or 'boron' in name_lower:
            return 'boron'
        elif 'magnesium' in name_lower or 'grignard' in name_lower:
            return 'grignard'
        elif 'zinc' in name_lower:
            return 'zinc'
        elif 'tin' in name_lower or 'stannane' in name_lower:
            return 'tin'
        elif 'silane' in name_lower or 'silyl' in name_lower:
            return 'silicon'
        else:
            return 'other'

    def search_compounds(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for compounds by name or SMILES."""
        query_lower = query.lower()
        results = []

        for name, keys in self._name_index.items():
            if query_lower in name:
                for key in keys:
                    category, smiles = key.split(':')
                    if category == 'e':
                        compound = self.electrophiles.get(smiles)
                    else:
                        compound = self.nucleophiles.get(smiles)

                    if compound:
                        results.append({
                            'name': compound.name,
                            'smiles': compound.smiles,
                            'category': compound.category,
                            'leaving_group': compound.leaving_group,
                            'type': compound.compound_type
                        })

                    if len(results) >= limit:
                        return results

        return results

    def search_papers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for papers by title or keyword."""
        query_words = re.findall(r'\w+', query.lower())
        doi_scores: Dict[str, int] = {}

        for word in query_words:
            if len(word) > 3 and word in self._keyword_index:
                for doi in self._keyword_index[word]:
                    doi_scores[doi] = doi_scores.get(doi, 0) + 1

        sorted_dois = sorted(doi_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for doi, score in sorted_dois:
            paper = self.papers.get(doi)
            if paper:
                results.append({
                    'doi': doi,
                    'title': paper.title,
                    'published_date': paper.published_date,
                    'authors': paper.authors,
                    'reaction_type': paper.reaction_type,
                    'relevance_score': score
                })

        return results

    def get_compound_by_smiles(self, smiles: str) -> Optional[CompoundInfo]:
        """Get compound information by SMILES."""
        if smiles in self.electrophiles:
            return self.electrophiles[smiles]
        if smiles in self.nucleophiles:
            return self.nucleophiles[smiles]
        return None

    def get_paper_by_doi(self, doi: str) -> Optional[PaperInfo]:
        """Get paper information by DOI."""
        return self.papers.get(doi)

    def get_leaving_groups(self) -> Dict[str, List[str]]:
        """Get all available leaving groups for electrophiles and nucleophiles."""
        e_lvg_path = self.data_dir / 'E_LVG.json'
        nu_lvg_path = self.data_dir / 'Nu_LVG.json'

        result = {'electrophiles': [], 'nucleophiles': []}

        if e_lvg_path.exists():
            with open(e_lvg_path) as f:
                result['electrophiles'] = list(json.load(f).values())

        if nu_lvg_path.exists():
            with open(nu_lvg_path) as f:
                result['nucleophiles'] = list(json.load(f).values())

        return result

    def get_reaction_types(self) -> List[Dict[str, str]]:
        """Get all available reaction types."""
        return [
            {'id': rxn.reaction_id, 'name': rxn.name}
            for rxn in self.reactions.values()
        ]

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        return {
            'electrophiles': len(self.electrophiles),
            'nucleophiles': len(self.nucleophiles),
            'papers': len(self.papers),
            'reactions': len(self.reactions)
        }

    def search_for_context(self, query: str, max_results: int = 5) -> str:
        """
        Search database for relevant context to include in LLM prompt.
        Returns a formatted string with relevant information.
        """
        context_parts = []

        compounds = self.search_compounds(query, limit=max_results)
        if compounds:
            context_parts.append("### Relevant Compounds:")
            for c in compounds:
                context_parts.append(f"- {c['name']} ({c['category']}): SMILES={c['smiles']}, Leaving Group={c['leaving_group']}")

        papers = self.search_papers(query, limit=max_results)
        if papers:
            context_parts.append("\n### Relevant Publications:")
            for p in papers:
                context_parts.append(f"- {p['title']} ({p['published_date']}) - DOI: {p['doi']}")
                if p['reaction_type']:
                    context_parts.append(f"  Reaction Type: {p['reaction_type']}")

        query_lower = query.lower()
        for rxn in self.reactions.values():
            if rxn.name.lower() in query_lower:
                context_parts.append(f"\n### Reaction Type: {rxn.name}")

        if context_parts:
            return "\n".join(context_parts)
        return ""

    def get_cross_coupling_info(self) -> str:
        """Get general information about cross-coupling reactions."""
        return """
### Cross-Coupling Reaction Types in Database:
- Suzuki coupling: Organoboron + Organic halide (Pd/Ni catalyst)
- Heck reaction: Aryl halide + Alkene (Pd catalyst)
- Sonogashira coupling: Aryl halide + Alkyne (Pd/Cu catalyst)
- Stille coupling: Organotin + Organic halide (Pd catalyst)
- Kumada coupling: Grignard reagent + Organic halide (Ni/Pd catalyst)
- Negishi coupling: Organozinc + Organic halide (Pd/Ni catalyst)
- Hiyama coupling: Organosilane + Organic halide (Pd catalyst)

### Common Electrophile Leaving Groups:
- Triflate (OTf): Excellent leaving group, very reactive
- Tosylate (OTs): Good leaving group, stable
- Mesylate (OMs): Good leaving group
- Acetate (OAc): Moderate leaving group
- Phenolates: Can be activated with Ni catalysts

### Common Nucleophile Types:
- Boronic acids/esters: Suzuki coupling
- Grignard reagents: Kumada coupling
- Organozinc compounds: Negishi coupling
- Organotin compounds: Stille coupling
"""


_database: Optional[NiCOBotDatabase] = None


def get_database() -> NiCOBotDatabase:
    """Get or create the global database instance."""
    global _database
    if _database is None:
        _database = NiCOBotDatabase()
        _database.load()
    return _database


def search_database_for_context(query: str) -> str:
    """Convenience function to get database context for a query."""
    db = get_database()
    if not db._loaded:
        db.load()
    return db.search_for_context(query)
