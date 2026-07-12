"""
NiCOBot Database Service
Provides access to chemical reaction data and compounds.
Integrates with LLM for context-aware responses.

Data sources (all loaded from pre-converted JSON — no pandas/openpyxl at
runtime to avoid C-extension segfaults):

  1. E_LVG_name_smiles.json   — 96 electrophiles  (SMILES -> [name, LG_SMILES])
  2. Nu_LVG_name_smiles.json  — 31 nucleophiles   (SMILES -> [name, LG_SMILES])
  3. results_modify_add.json  — 238 papers         (DOI -> {title, date, refs})
  4. schneider_50k.json       — 50 reaction names  (id -> name)
  5. paper_links.json         — 238 paper metadata (DOI -> {journal, rxn_type, ...})
  6. paper_abstracts.json     — 32 papers w/ abstracts (doi -> {title, abstract, ...})
  7. reaction_data.json       — 431 reactions w/ metadata (R1, R2, Base, E_LVG, ...)
  8. grand_database.json      — 1126 reactions basic (R1, R2, Base)
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
    journal: str = ""
    abstract: str = ""
    electrophile_lvg: str = ""
    in_situ_activation: str = ""


@dataclass
class ReactionInfo:
    """Information about a reaction type."""
    reaction_id: str
    name: str
    category: str = ""


@dataclass
class ReactionEntry:
    """A specific reaction from the grand database."""
    r1_smiles: str = ""
    r2_smiles: str = ""
    base: str = ""
    e_type: str = ""
    e_major: str = ""
    e_lvg: str = ""
    nu_type: str = ""
    nu_major: str = ""
    nu_lvg: str = ""
    base_class: str = ""
    reaction_name: str = ""


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

        # New: rich reaction entries from Grand_Database
        self.reaction_entries: List[ReactionEntry] = []

        self._name_index: Dict[str, List[str]] = {}
        self._smiles_index: Dict[str, str] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        self._reaction_keyword_index: Dict[str, List[int]] = {}

        self._loaded = False

    def load(self) -> bool:
        """Load all data from files."""
        if self._loaded:
            return True

        try:
            self._load_electrophiles()
            self._load_nucleophiles()
            self._load_papers()
            self._enrich_papers_from_links()
            self._enrich_papers_from_abstracts()
            self._load_reactions()
            self._load_reaction_entries()
            self._build_indices()
            self._loaded = True
            logger.info(
                f"NiCOBot Database loaded: {len(self.electrophiles)} electrophiles, "
                f"{len(self.nucleophiles)} nucleophiles, {len(self.papers)} papers, "
                f"{len(self.reaction_entries)} reaction entries"
            )
            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to load NiCOBot database: {e}")
            return False

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def _load_electrophiles(self):
        """Load electrophile data from E_LVG_name_smiles.json."""
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
        """Load nucleophile data from Nu_LVG_name_smiles.json."""
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
        """Load paper metadata from results_modify_add.json."""
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

    def _enrich_papers_from_links(self):
        """Enrich paper metadata from paper_links.json (journal, reaction_type, LVG)."""
        links_path = self.data_dir / 'paper_links.json'
        if not links_path.exists():
            return
        try:
            with open(links_path) as f:
                links = json.load(f)
            for entry in links:
                doi = entry.get('DOI', '')
                if doi in self.papers:
                    p = self.papers[doi]
                    if not p.journal:
                        p.journal = entry.get('JOURNAL ISO', '')
                    if not p.reaction_type:
                        p.reaction_type = entry.get('REACTION TYPE', '')
                    if not p.electrophile_lvg:
                        p.electrophile_lvg = entry.get('MAIN ELECTROPHILE LVG', '')
                    if not p.in_situ_activation:
                        p.in_situ_activation = entry.get('IN SITU ACTIVATION', '')
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load paper_links.json: {e}")

    def _enrich_papers_from_abstracts(self):
        """Enrich papers with abstracts and author data from paper_abstracts.json."""
        abstracts_path = self.data_dir / 'paper_abstracts.json'
        if not abstracts_path.exists():
            return
        try:
            with open(abstracts_path) as f:
                abstracts = json.load(f)
            for entry in abstracts:
                doi = entry.get('doi', '')
                # Match by DOI — paper_abstracts uses 'name' column which has DOI
                if doi in self.papers:
                    p = self.papers[doi]
                    if not p.abstract:
                        p.abstract = entry.get('abstract', '')
                    if not p.authors:
                        p.authors = entry.get('authors', '')
                    if not p.journal:
                        p.journal = entry.get('journal', '') or entry.get('Journal', '')
                    if not p.reaction_type:
                        p.reaction_type = entry.get('reaction_type', '')
                    if not p.strength:
                        p.strength = entry.get('strength', '')
                    if not p.title:
                        p.title = entry.get('title', '')
                else:
                    # Paper not in results_modify_add.json — add it from abstracts
                    self.papers[doi] = PaperInfo(
                        doi=doi,
                        title=entry.get('title', ''),
                        published_date=entry.get('publication_year', ''),
                        authors=entry.get('authors', ''),
                        abstract=entry.get('abstract', ''),
                        journal=entry.get('journal', '') or entry.get('Journal', ''),
                        reaction_type=entry.get('reaction_type', ''),
                        strength=entry.get('strength', ''),
                    )
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load paper_abstracts.json: {e}")

    def _load_reactions(self):
        """Load reaction type information from schneider_50k.json."""
        reactions_path = self.data_dir / 'schneider_50k.json'
        if reactions_path.exists():
            with open(reactions_path) as f:
                data = json.load(f)

            for rxn_id, name in data.items():
                self.reactions[rxn_id] = ReactionInfo(
                    reaction_id=rxn_id,
                    name=name
                )

    def _load_reaction_entries(self):
        """Load detailed reaction entries from reaction_data.json (431 reactions).

        Falls back to grand_database.json (1126 reactions, less metadata) if
        reaction_data.json is not available.
        """
        details_path = self.data_dir / 'reaction_data.json'
        basic_path = self.data_dir / 'grand_database.json'

        source = details_path if details_path.exists() else basic_path
        if not source.exists():
            return

        try:
            with open(source) as f:
                data = json.load(f)

            for entry in data:
                r = ReactionEntry(
                    r1_smiles=str(entry.get('R1', '')),
                    r2_smiles=str(entry.get('R2', '')),
                    base=str(entry.get('Base', '')),
                    e_type=str(entry.get('E_Type', '')),
                    e_major=str(entry.get('E_Major', '')),
                    e_lvg=str(entry.get('E_LVG', '')),
                    nu_type=str(entry.get('Nu_Type', '')),
                    nu_major=str(entry.get('Nu_Major', '')),
                    nu_lvg=str(entry.get('Nu_LVG', '')),
                    base_class=str(entry.get('Base class', '')),
                    reaction_name=str(entry.get('Reaction name', '')),
                )
                self.reaction_entries.append(r)

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Could not load reaction entries: {e}")

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

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

        # Index paper titles + abstracts for keyword search
        for doi, paper in self.papers.items():
            text = f"{paper.title} {paper.abstract} {paper.reaction_type} {paper.journal}".lower()
            words = re.findall(r'\w+', text)
            for word in words:
                if len(word) > 3:
                    if word not in self._keyword_index:
                        self._keyword_index[word] = []
                    self._keyword_index[word].append(doi)

        # Index reaction entries by keyword
        for idx, r in enumerate(self.reaction_entries):
            text = (
                f"{r.reaction_name} {r.e_lvg} {r.nu_lvg} {r.e_major} "
                f"{r.nu_major} {r.e_type} {r.nu_type} {r.base_class} {r.base}"
            ).lower()
            words = re.findall(r'\w+', text)
            for word in words:
                if len(word) > 2:
                    if word not in self._reaction_keyword_index:
                        self._reaction_keyword_index[word] = []
                    self._reaction_keyword_index[word].append(idx)

    # ------------------------------------------------------------------
    # Type inference helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------

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
        """Search for papers by title, abstract, or keyword."""
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
                    'journal': paper.journal,
                    'abstract': paper.abstract,
                    'strength': paper.strength,
                    'electrophile_lvg': paper.electrophile_lvg,
                    'relevance_score': score
                })

        return results

    def search_reactions(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search reaction entries by keyword matching.

        Returns matching reactions with their full metadata.
        """
        query_words = re.findall(r'\w+', query.lower())
        entry_scores: Dict[int, int] = {}

        for word in query_words:
            if len(word) > 2 and word in self._reaction_keyword_index:
                for idx in self._reaction_keyword_index[word]:
                    entry_scores[idx] = entry_scores.get(idx, 0) + 1

        sorted_indices = sorted(entry_scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        results = []
        for idx, score in sorted_indices:
            r = self.reaction_entries[idx]
            results.append({
                'r1_smiles': r.r1_smiles,
                'r2_smiles': r.r2_smiles,
                'base': r.base,
                'e_type': r.e_type,
                'e_major': r.e_major,
                'e_lvg': r.e_lvg,
                'nu_type': r.nu_type,
                'nu_major': r.nu_major,
                'nu_lvg': r.nu_lvg,
                'base_class': r.base_class,
                'reaction_name': r.reaction_name,
                'relevance_score': score,
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
            'reactions': len(self.reactions),
            'reaction_entries': len(self.reaction_entries),
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
### Common Cross-Coupling Reactions

- Suzuki coupling: Organoboron + Organic halide (Pd/Ni catalyst)
- Heck reaction: Aryl halide + Alkene (Pd catalyst)
- Sonogashira coupling: Aryl halide + Alkyne (Pd/Cu catalyst)
- Stille coupling: Organotin + Organic halide (Pd catalyst)
- Kumada coupling: Grignard reagent + Organic halide (Ni/Pd catalyst)
- Negishi coupling: Organozinc + Organic halide (Pd/Ni catalyst)
- Hiyama coupling: Organosilane + Organic halide (Pd catalyst)

### Common Leaving Groups

- Triflate (OTf): Excellent leaving group, very reactive
- Tosylate (OTs): Good leaving group, stable
- Mesylate (OMs): Good leaving group
- Acetate (OAc): Moderate leaving group
- Phenolates: Can be activated with Ni catalysts

### Common Nucleophiles

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