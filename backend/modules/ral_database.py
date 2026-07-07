"""
RAL (Redox-Active Ligands) Database Service

Loads and indexes two complementary datasets:
  1. Grand_Data.xlsx  — 238 ligands with DFT-computed electronic descriptors
     across 7 classes (Phen, Bipy, PyrOx, PyrIm, PyCam, BiOX, BiIM).
  2. DOI List.xlsx    — 49 fully-curated reductive coupling reactions with
     optimum ligand, coupling partner, and detailed ligand knowledge entries
     (plus 182 DOI-only rows for future curation).
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ligand-class alias mapping
# DOI List uses IUPAC / common names; Grand_Data uses short class codes.
# This table bridges the two naming conventions.
# ---------------------------------------------------------------------------

CLASS_ALIASES: Dict[str, str] = {
    # Bipyridine family
    'bipyridine': 'Bipy', 'bpy': 'Bipy', 'bpy)': 'Bipy',
    'dtbbpy': 'Bipy', 'dtbpy': 'Bipy', 'dtbbpy)': 'Bipy', 'dtbpy)': 'Bipy',
    'dtbbpy)': 'Bipy', "4,4'-di-tert-butyl-2,2'-bipyridine": 'Bipy',
    "4,4'-dimethyl-2,2'-bipyridine": 'Bipy',
    "4,4'-dimethoxy-2,2'-bipyridine": 'Bipy',
    "6,6'-dimethyl-2,2'-bipyridine": 'Bipy',
    'dmbpy': 'Bipy', 'dm bpy': 'Bipy',
    # Phenanthroline family
    'phenanthroline': 'Phen', 'phen': 'Phen', 'phen)': 'Phen',
    'bathophenanthroline': 'Phen', 'bphen': 'Phen',
    'neocuproine': 'Phen', 'me4phen': 'Phen',
    "3,4,7,8-tetramethyl-1,10-phenanthroline": 'Phen',
    "4,7-diphenyl-1,10-phenanthroline": 'Phen',
    # Terpyridine (mapped to Phen as closest class)
    'terpyridine': 'Phen', "2,2':6',2''-terpyridine": 'Phen',
    # Bis(oxazoline) / Box family
    'bis(oxazoline)': 'BiOX', 'box': 'BiOX',
    'bisoxazoline': 'BiOX', 'bis(oxazoline)': 'BiOX',
    # Bis(imidazoline) / BiIm family
    'bis(imidazoline)': 'BiIM', 'biimidazoline': 'BiIM',
    'bisimidazoline': 'BiIM', 'bis(imidazoline)': 'BiIM',
    # Pyridine mono-dentate (map to Bipy as closest)
    'pyridine': 'Bipy', 'dmap': 'Bipy',
    # Amidine / PyBCam (map to PyrIm as closest redox-active class)
    'amidine': 'PyrIm', 'pybcam': 'PyrIm', 'pyridinedicarboxamidine': 'PyrIm',
    # Imidazole (map to PyrIm as closest)
    'imidazole': 'PyrIm',
    # Pyrazolyl
    'pyrazolyl': 'PyrIm',
}


@dataclass
class LigandProperty:
    """DFT-computed electronic descriptors for a single ligand."""
    name: str
    ligand_class: str
    homo: float
    lumo: float
    gap: float
    omega: float       # electrophilicity index
    i_min: float       # ionization potential
    v_min: float       # electron affinity
    r1_homa: float     # aromaticity index, ring 1
    r2_homa: float     # aromaticity index, ring 2


@dataclass
class ReactionLigandEntry:
    """A literature entry linking a reaction to its optimum ligand."""
    doi: str
    title: str = ""
    optimum_ligand: str = ""
    coupling_partner: str = ""
    ligand_knowledge: str = ""
    mapped_class: str = ""  # resolved Grand_Data class name


class RALDatabase:
    """
    Database service for the Redox-Active Ligands subsystem.

    Loads two Excel sources, builds text search indices, and provides
    retrieval methods for both ligand properties and reaction-ligand
    literature data.
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'ral_data'
            )
        self.data_dir = Path(data_dir)

        # Ligand electronic properties (from Grand_Data.xlsx)
        self.ligands: List[LigandProperty] = []
        self._ligand_by_name: Dict[str, LigandProperty] = {}
        self._ligands_by_class: Dict[str, List[LigandProperty]] = {}

        # Reaction-ligand literature (from DOI List.xlsx)
        self.reactions: List[ReactionLigandEntry] = []
        self._reactions_by_doi: Dict[str, ReactionLigandEntry] = {}

        # Search indices
        self._name_index: Dict[str, List[int]] = {}       # ligand name -> indices
        self._class_index: Dict[str, List[int]] = {}      # class -> indices
        self._doi_index: Dict[str, int] = {}              # doi -> reaction index
        self._keyword_index: Dict[str, List[int]] = {}    # word -> reaction indices
        self._ligand_kw_index: Dict[str, List[int]] = {}  # word -> ligand indices

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load all data from Excel files and build indices."""
        if self._loaded:
            return True

        try:
            self._load_grand_data()
            self._load_doi_list()
            self._build_indices()
            self._loaded = True
            logger.info(
                f"RAL Database loaded: {len(self.ligands)} ligands, "
                f"{len(self.reactions)} reactions "
                f"({sum(1 for r in self.reactions if r.title)} curated)"
            )
            return True
        except (OSError, ValueError, KeyError, TypeError, ImportError) as e:
            # OSError: missing file.  ValueError: bad data.  KeyError:
            # missing column.  TypeError: wrong type.  ImportError:
            # openpyxl not installed.
            logger.error(f"Failed to load RAL database: {e}")
            return False

    def _load_grand_data(self):
        """Load ligand electronic properties from Grand_Data.xlsx."""
        path = self.data_dir / 'Grand_Data.xlsx'
        if not path.exists():
            logger.warning(f"Grand_Data.xlsx not found at {path}")
            return

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas/openpyxl required for Grand_Data.xlsx")
            return

        df = pd.read_excel(path)
        for _, row in df.iterrows():
            lp = LigandProperty(
                name=str(row['Ligand']).strip(),
                ligand_class=str(row['Class']).strip(),
                homo=float(row['HOMO (eV)']),
                lumo=float(row['LUMO (eV)']),
                gap=float(row['Gap (eV)']),
                omega=float(row['ω (eV)']),
                i_min=float(row['I_min (eV)']),
                v_min=float(row['V_min (eV)']),
                r1_homa=float(row['R1-HOMA']),
                r2_homa=float(row['R2-HOMA']),
            )
            idx = len(self.ligands)
            self.ligands.append(lp)
            self._ligand_by_name[lp.name.lower()] = lp
            self._ligands_by_class.setdefault(lp.ligand_class, []).append(lp)

    def _load_doi_list(self):
        """Load reaction-ligand literature from DOI List.xlsx."""
        path = self.data_dir / 'DOI List.xlsx'
        if not path.exists():
            logger.warning(f"DOI List.xlsx not found at {path}")
            return

        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas/openpyxl required for DOI List.xlsx")
            return

        df = pd.read_excel(path)
        for _, row in df.iterrows():
            doi = str(row.get('DOI', '')).strip()
            if not doi or doi.lower() in ('nan', 'none', ''):
                continue

            title = self._safe_str(row.get('Title'))
            opt_lig = self._safe_str(row.get('Optimum Ligand'))
            partner = self._safe_str(row.get('Coupling Partner'))
            knowledge = self._safe_str(row.get('Ligand Knowledge'))

            # Map ligand name to Grand_Data class
            mapped_class = self._resolve_ligand_class(opt_lig)

            entry = ReactionLigandEntry(
                doi=doi,
                title=title,
                optimum_ligand=opt_lig,
                coupling_partner=partner,
                ligand_knowledge=knowledge,
                mapped_class=mapped_class,
            )
            idx = len(self.reactions)
            self.reactions.append(entry)
            if doi:
                self._reactions_by_doi[doi] = entry

    @staticmethod
    def _safe_str(val) -> str:
        """Convert a value to string, treating NaN/None as empty."""
        if val is None:
            return ''
        s = str(val).strip()
        return '' if s.lower() in ('nan', 'none', 'nat') else s

    def _resolve_ligand_class(self, ligand_name: str) -> str:
        """Map a DOI-List ligand name to the closest Grand_Data class."""
        if not ligand_name:
            return ''
        name_lower = ligand_name.lower()

        # Direct alias lookup
        for alias, cls in CLASS_ALIASES.items():
            if alias in name_lower:
                return cls

        # Fallback: check if any Grand_Data class name appears directly
        for cls in self._ligands_by_class:
            if cls.lower() in name_lower:
                return cls

        # Check for chiral Box/BiIM patterns
        if 'box' in name_lower or 'oxazoline' in name_lower:
            return 'BiOX'
        if 'biim' in name_lower or 'imidazoline' in name_lower:
            return 'BiIM'

        return 'unknown'

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_indices(self):
        """Build keyword search indices over both datasets."""
        # Ligand name index (word-level)
        for i, lp in enumerate(self.ligands):
            name_lower = lp.name.lower()
            self._name_index.setdefault(name_lower, []).append(i)
            # Also index individual meaningful tokens
            for token in re.findall(r'[a-z]{2,}', name_lower):
                if len(token) >= 2:
                    self._ligand_kw_index.setdefault(token, []).append(i)

        # Ligand class index
        for i, lp in enumerate(self.ligands):
            self._class_index.setdefault(lp.ligand_class, []).append(i)

        # Reaction indices
        for i, entry in enumerate(self.reactions):
            if entry.doi:
                self._doi_index[entry.doi] = i
            # Index keywords from title, ligand name, coupling partner
            text_fields = [entry.title, entry.optimum_ligand,
                           entry.coupling_partner, entry.ligand_knowledge]
            for field in text_fields:
                for word in re.findall(r'\w+', field.lower()):
                    if len(word) > 2:
                        self._keyword_index.setdefault(word, []).append(i)

    # ------------------------------------------------------------------
    # Search methods — Ligand properties
    # ------------------------------------------------------------------

    def search_ligands(self, query: str, limit: int = 10,
                       ligand_class: str = None) -> List[Dict[str, Any]]:
        """Search ligands by name, class, or arbitrary keyword."""
        query_lower = query.lower()
        results: List[Tuple[int, int]] = []  # (index, score)
        seen: set = set()

        def _add(idx, score):
            if idx not in seen:
                seen.add(idx)
                results.append((idx, score))

        # 1. Exact name match (highest score)
        for name, indices in self._name_index.items():
            if query_lower == name:
                for idx in indices:
                    _add(idx, 100)

        # 2. Substring name match
        for name, indices in self._name_index.items():
            if query_lower in name or name in query_lower:
                for idx in indices:
                    _add(idx, 50)

        # 3. Class name match (e.g. query "Bipy" matches all Bipy-class ligands)
        for cls, indices in self._class_index.items():
            if query_lower in cls.lower() or cls.lower() in query_lower:
                for idx in indices:
                    _add(idx, 40)

        # 4. Alias match — check if query matches a CLASS_ALIASES key,
        #    then return ligands of the mapped class
        for alias, cls in CLASS_ALIASES.items():
            if alias in query_lower or query_lower in alias:
                for idx in self._class_index.get(cls, []):
                    _add(idx, 40)

        # 5. Keyword match on individual tokens
        for token in re.findall(r'[a-z]{2,}', query_lower):
            if token in self._ligand_kw_index:
                for idx in self._ligand_kw_index[token]:
                    _add(idx, 10)

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Filter by class if requested
        output = []
        for idx, score in results:
            lp = self.ligands[idx]
            if ligand_class and lp.ligand_class != ligand_class:
                continue
            output.append(self._ligand_to_dict(lp, score))
            if len(output) >= limit:
                break

        return output

    def get_ligand_classes(self) -> Dict[str, Any]:
        """Return all ligand classes with counts and descriptor ranges."""
        classes = {}
        for cls, ligands in self._ligands_by_class.items():
            homos = [l.homo for l in ligands]
            lumos = [l.lumo for l in ligands]
            gaps = [l.gap for l in ligands]
            omegas = [l.omega for l in ligands]
            classes[cls] = {
                'count': len(ligands),
                'ligand_names': [l.name for l in ligands],
                'HOMO_range': [min(homos), max(homos)],
                'LUMO_range': [min(lumos), max(lumos)],
                'Gap_range': [min(gaps), max(gaps)],
                'omega_range': [min(omegas), max(omegas)],
            }
        return classes

    def get_ligands_by_class(self, ligand_class: str) -> List[Dict[str, Any]]:
        """Return all ligands in a given class with full descriptors."""
        ligands = self._ligands_by_class.get(ligand_class, [])
        return [self._ligand_to_dict(lp) for lp in ligands]

    def get_ligand_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Look up a ligand by exact name (case-insensitive)."""
        lp = self._ligand_by_name.get(name.lower())
        if lp:
            return self._ligand_to_dict(lp)
        return None

    def compare_ligand_classes(self, class_a: str, class_b: str) -> Optional[Dict[str, Any]]:
        """Compare average electronic descriptors between two ligand classes."""
        a = self._ligands_by_class.get(class_a, [])
        b = self._ligands_by_class.get(class_b, [])
        if not a or not b:
            return None

        def _avg(lst, attr):
            vals = [getattr(x, attr) for x in lst]
            return round(sum(vals) / len(vals), 3)

        attrs = ['homo', 'lumo', 'gap', 'omega', 'i_min', 'v_min',
                 'r1_homa', 'r2_homa']
        comparison = {}
        for attr in attrs:
            comparison[attr] = {
                class_a: _avg(a, attr),
                class_b: _avg(b, attr),
                'diff': round(_avg(a, attr) - _avg(b, attr), 3),
            }
        comparison['count'] = {class_a: len(a), class_b: len(b)}
        return comparison

    # ------------------------------------------------------------------
    # Search methods — Reaction-Ligand literature
    # ------------------------------------------------------------------

    def search_reactions(self, query: str, limit: int = 10,
                         ligand_class: str = None) -> List[Dict[str, Any]]:
        """Search reaction-ligand entries by keyword (title, ligand, partner, knowledge)."""
        query_lower = query.lower()
        query_words = re.findall(r'\w+', query_lower)
        scores: Dict[int, int] = {}

        # Score by word overlap
        for word in query_words:
            if len(word) > 2 and word in self._keyword_index:
                for idx in self._keyword_index[word]:
                    scores[idx] = scores.get(idx, 0) + 1

        # Also do full-query substring on key fields
        for i, entry in enumerate(self.reactions):
            for field in [entry.title, entry.optimum_ligand,
                          entry.coupling_partner]:
                if query_lower in field.lower():
                    scores[i] = scores.get(i, 0) + 5

        # Sort by score
        sorted_indices = sorted(scores.items(), key=lambda x: x[1],
                                reverse=True)

        output = []
        for idx, score in sorted_indices:
            entry = self.reactions[idx]
            # Skip entries without curated content (DOI-only rows)
            if not entry.title:
                continue
            if ligand_class and entry.mapped_class != ligand_class:
                continue
            output.append(self._reaction_to_dict(entry, score))
            if len(output) >= limit:
                break

        return output

    def get_reaction_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Look up a reaction entry by DOI."""
        entry = self._reactions_by_doi.get(doi)
        if entry:
            return self._reaction_to_dict(entry)
        return None

    def get_reactions_by_class(self, ligand_class: str) -> List[Dict[str, Any]]:
        """Get all curated reactions whose optimum ligand maps to a given class."""
        results = []
        for entry in self.reactions:
            if entry.mapped_class == ligand_class and entry.title:
                results.append(self._reaction_to_dict(entry))
        return results

    # ------------------------------------------------------------------
    # Cross-database search (the key RAG enabler)
    # ------------------------------------------------------------------

    def search_combined(self, query: str, max_ligands: int = 5,
                        max_reactions: int = 5) -> Dict[str, Any]:
        """
        Search both datasets simultaneously and return structured context.

        When a ligand class is detected in the query, this method:
        1. Pulls the top ligands from Grand_Data (electronic properties).
        2. Pulls matching reactions from DOI List (literature evidence).
        3. Returns both together for RAG prompt injection.
        """
        query_lower = query.lower()

        # Detect if any known class name is mentioned
        detected_class = None
        for cls in self._ligands_by_class:
            if cls.lower() in query_lower:
                detected_class = cls
                break
        # Also check aliases
        if not detected_class:
            for alias, cls in CLASS_ALIASES.items():
                if alias in query_lower:
                    detected_class = cls
                    break

        # Search ligands
        ligands = self.search_ligands(query, limit=max_ligands,
                                       ligand_class=detected_class)

        # Search reactions
        reactions = self.search_reactions(query, limit=max_reactions,
                                           ligand_class=detected_class)

        # If class detected but few reactions, supplement by class
        if detected_class and len(reactions) < 2:
            class_reactions = self.get_reactions_by_class(detected_class)
            existing_dois = {r['doi'] for r in reactions}
            for r in class_reactions:
                if r['doi'] not in existing_dois:
                    reactions.append(r)
                if len(reactions) >= max_reactions:
                    break

        # If class detected but few ligands, supplement by class
        if detected_class and len(ligands) < 3:
            class_ligands = self.get_ligands_by_class(detected_class)
            existing_names = {l['name'] for l in ligands}
            for l in class_ligands:
                if l['name'] not in existing_names:
                    ligands.append(l)
                if len(ligands) >= max_ligands:
                    break

        return {
            'ligands': ligands,
            'reactions': reactions,
            'detected_class': detected_class,
            'ligand_count': len(ligands),
            'reaction_count': len(reactions),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics for both datasets."""
        curated = sum(1 for r in self.reactions if r.title)
        return {
            'ligands': len(self.ligands),
            'ligand_classes': len(self._ligands_by_class),
            'class_names': list(self._ligands_by_class.keys()),
            'reactions_total': len(self.reactions),
            'reactions_curated': curated,
            'reactions_doi_only': len(self.reactions) - curated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ligand_to_dict(lp: LigandProperty, score: int = 0) -> Dict[str, Any]:
        return {
            'name': lp.name,
            'class': lp.ligand_class,
            'HOMO_eV': lp.homo,
            'LUMO_eV': lp.lumo,
            'Gap_eV': lp.gap,
            'omega_eV': lp.omega,
            'I_min_eV': lp.i_min,
            'V_min_eV': lp.v_min,
            'R1_HOMA': lp.r1_homa,
            'R2_HOMA': lp.r2_homa,
            'relevance_score': score,
        }

    @staticmethod
    def _reaction_to_dict(entry: ReactionLigandEntry,
                          score: int = 0) -> Dict[str, Any]:
        d = {
            'doi': entry.doi,
            'title': entry.title,
            'optimum_ligand': entry.optimum_ligand,
            'coupling_partner': entry.coupling_partner,
            'ligand_knowledge': entry.ligand_knowledge,
            'mapped_class': entry.mapped_class,
            'relevance_score': score,
        }
        return d


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_ral_database: Optional[RALDatabase] = None


def get_ral_database() -> RALDatabase:
    """Get or create the global RAL database instance."""
    global _ral_database
    if _ral_database is None:
        _ral_database = RALDatabase()
        _ral_database.load()
    return _ral_database